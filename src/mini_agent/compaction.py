"""Token-aware context windowing. Disk session.messages are never mutated or deleted."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mini_agent.cost import estimate_tokens_from_text
from mini_agent.events import CompactionNotice
from mini_agent.llm import LLMClient
from mini_agent.messages import (
    CompactionPart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UnpairedToolError,
    UsagePart,
    assert_pairing,
    to_chat_messages,
)
from mini_agent.models import ToolResult
from mini_agent.prompt import COMPACTION_SUMMARY_PROMPT


@dataclass(frozen=True)
class CompactionConfig:
    context_window_tokens: int = 128_000
    buffer_tokens: int = 8_000
    keep_recent_tokens: int = 24_000
    prune_tool_chars: int = 500
    auto_summarize: bool = True


def _budget(config: CompactionConfig) -> int:
    return max(0, config.context_window_tokens - config.buffer_tokens)


def _result_envelope(part: ToolResultPart) -> str:
    return ToolResult(
        ok=part.ok,
        content=part.content,
        error=part.error,
        metadata=part.metadata,
    ).model_dump_json()


def estimate_message_tokens(
    messages: list[Message],
    tool_schemas: list[dict[str, Any]] | None = None,
) -> int:
    """Skip UsagePart; count call arguments, result envelopes, compaction summaries."""
    total = 0
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, UsagePart):
                continue
            if isinstance(part, TextPart):
                total += estimate_tokens_from_text(part.text)
            elif isinstance(part, ToolCallPart):
                total += estimate_tokens_from_text(part.arguments)
            elif isinstance(part, ToolResultPart):
                total += estimate_tokens_from_text(_result_envelope(part))
            elif isinstance(part, CompactionPart):
                total += estimate_tokens_from_text(part.summary)
    if tool_schemas:
        total += estimate_tokens_from_text(json.dumps(tool_schemas, ensure_ascii=False))
    return total


def _system_prefix_len(messages: list[Message]) -> int:
    n = 0
    for msg in messages:
        if msg.role != "system":
            break
        n += 1
    return n


def _keep_start_index(messages: list[Message], keep_recent_tokens: int) -> int:
    start = _system_prefix_len(messages)
    if start >= len(messages):
        return start
    acc = 0
    keep_start = len(messages)
    for idx in range(len(messages) - 1, start - 1, -1):
        acc += estimate_message_tokens([messages[idx]])
        keep_start = idx
        if acc >= keep_recent_tokens:
            break
    return keep_start


def _fold_arguments(name: str, arguments: str) -> str:
    n = len(arguments)
    folded = f"[工具 {name} 参数已折叠，原长 {n} 字符]"
    try:
        data = json.loads(arguments)
    except (json.JSONDecodeError, TypeError, ValueError):
        return folded
    if isinstance(data, dict) and "path" in data:
        return json.dumps({"path": data["path"], "_folded": folded}, ensure_ascii=False)
    return folded


def _prune_zone(messages: list[Message], zone_start: int, zone_end: int, limit: int) -> None:
    for i in range(zone_start, zone_end):
        new_parts: list[Any] = []
        changed = False
        for part in messages[i].parts:
            if isinstance(part, ToolResultPart) and part.ok and len(part.content) > limit:
                n = len(part.content)
                new_parts.append(
                    part.model_copy(
                        update={"content": f"[工具 {part.name} 成功，输出已折叠，原长 {n} 字符]"}
                    )
                )
                changed = True
            elif isinstance(part, ToolCallPart) and len(part.arguments) > limit:
                folded = _fold_arguments(part.name, part.arguments)
                new_parts.append(part.model_copy(update={"arguments": folded}))
                changed = True
            else:
                new_parts.append(part)
        if changed:
            messages[i] = messages[i].model_copy(update={"parts": new_parts})


def _dedup_read_file(messages: list[Message], zone_start: int, zone_end: int) -> None:
    by_path: dict[str, list[tuple[int, int]]] = {}
    for i in range(zone_start, zone_end):
        for j, part in enumerate(messages[i].parts):
            if not isinstance(part, ToolResultPart) or part.name != "read_file" or not part.ok:
                continue
            path = part.metadata.get("path") if part.metadata else None
            if not isinstance(path, str) or not path:
                continue
            by_path.setdefault(path, []).append((i, j))
    for path, locs in by_path.items():
        if len(locs) < 2:
            continue
        for i, j in locs[:-1]:
            part = messages[i].parts[j]
            new_parts = list(messages[i].parts)
            new_parts[j] = part.model_copy(update={"content": f"[已过时的 read_file: {path}]"})
            messages[i] = messages[i].model_copy(update={"parts": new_parts})


def _user_turns(messages: list[Message]) -> list[list[Message]]:
    turns: list[list[Message]] = []
    current: list[Message] = []
    for msg in messages[_system_prefix_len(messages) :]:
        if msg.role == "user" and current:
            turns.append(current)
            current = [msg]
        elif msg.role == "user":
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)
    return turns


def _flatten(turns: list[list[Message]]) -> list[Message]:
    return [msg for turn in turns for msg in turn]


def _tool_ids(messages: list[Message]) -> tuple[set[str], set[str]]:
    calls: set[str] = set()
    results: set[str] = set()
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                calls.add(part.call_id)
            elif isinstance(part, ToolResultPart):
                results.add(part.call_id)
    return calls, results


def _ids_of(msg: Message) -> set[str]:
    return {part.call_id for part in msg.parts if isinstance(part, ToolCallPart | ToolResultPart)}


def _call_result_messages(
    messages: list[Message],
) -> tuple[dict[str, Message], dict[str, Message]]:
    calls: dict[str, Message] = {}
    results: dict[str, Message] = {}
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                calls[part.call_id] = msg
            elif isinstance(part, ToolResultPart):
                results[part.call_id] = msg
    return calls, results


def _turn_message_ids(original: list[Message], cids: set[str]) -> set[str]:
    drop: set[str] = set()
    for turn in _user_turns(original):
        turn_cids: set[str] = set()
        for msg in turn:
            turn_cids.update(_ids_of(msg))
        if turn_cids & cids:
            drop.update(msg.id for msg in turn)
    return drop


def _assert_chat_tool_pairing(messages: list[Message]) -> None:
    open_ids: set[str] = set()
    for row in to_chat_messages(messages):
        if row.get("role") == "tool":
            cid = row.get("tool_call_id")
            if cid not in open_ids:
                raise UnpairedToolError("orphan tool row without a matching preceding tool_call")
            continue
        if row.get("role") == "assistant" and row.get("tool_calls"):
            open_ids = {tc["id"] for tc in row["tool_calls"]}
        else:
            open_ids = set()


def repair_pairing(
    window: list[Message],
    original: list[Message],
    *,
    budget: int | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> list[Message]:
    """Restore missing call/result messages; drop the whole user-turn if still over budget."""
    orig_index = {msg.id: i for i, msg in enumerate(original)}
    call_of, result_of = _call_result_messages(original)
    versions: dict[str, Message] = {msg.id: msg for msg in original}
    for msg in window:
        if msg.id in orig_index:
            versions[msg.id] = msg

    selected = {msg.id for msg in window if msg.id in orig_index}
    systems = [msg for msg in window if msg.role == "system"]
    synthetics = [msg for msg in window if msg.id not in orig_index and msg.role != "system"]

    present_calls, present_results = _tool_ids(window)
    unpaired = present_calls.symmetric_difference(present_results)

    for cid in unpaired:
        src = result_of.get(cid) if cid in present_calls else call_of.get(cid)
        if src is not None:
            selected.add(src.id)

    def assemble(ids: set[str]) -> list[Message]:
        ordered = [versions[msg.id] for msg in original if msg.id in ids and msg.role != "system"]
        return [*systems, *synthetics, *ordered]

    out = assemble(selected)

    def over_budget(msgs: list[Message]) -> bool:
        return budget is not None and estimate_message_tokens(msgs, tool_schemas) > budget

    if over_budget(out) and unpaired:
        selected -= _turn_message_ids(original, unpaired)
        out = assemble(selected)

    present_calls, present_results = _tool_ids(out)
    still = present_calls.symmetric_difference(present_results)
    if still:
        selected -= _turn_message_ids(original, still)
        still_parts = still
        out = [
            msg
            for msg in assemble(selected)
            if msg.role == "system" or not (_ids_of(msg) & still_parts)
        ]
    return out


def _hard_trim(
    messages: list[Message],
    budget: int,
    tool_schemas: list[dict[str, Any]] | None,
) -> tuple[list[Message], int]:
    prefix_n = _system_prefix_len(messages)
    prefix = messages[:prefix_n]
    turns = _user_turns(messages)
    dropped = 0
    assembled = prefix + _flatten(turns)
    while len(turns) > 1 and estimate_message_tokens(assembled, tool_schemas) > budget:
        dropped += len(turns[0])
        turns.pop(0)
        assembled = prefix + _flatten(turns)
    return assembled, dropped


def _try_summarize(
    window: list[Message],
    system_end: int,
    keep_start: int,
    llm: LLMClient,
    model: str,
) -> tuple[list[Message], CompactionNotice] | None:
    prefix = window[:system_end]
    pre_keep = window[system_end:keep_start]
    keep = window[keep_start:]
    if not pre_keep:
        return None
    prompt_messages = [
        Message(role="system", parts=[TextPart(text=COMPACTION_SUMMARY_PROMPT)]),
        *pre_keep,
    ]
    try:
        response = llm.create_response(prompt_messages, tools=[], model=model)
    except Exception:
        return None
    text = (response.text or "").strip() if response is not None else ""
    if not text:
        return None
    summary_msg = Message(
        role="user",
        parts=[CompactionPart(summary=text, dropped_count=len(pre_keep))],
    )
    notice = CompactionNotice(summary=text, dropped_count=len(pre_keep))
    return prefix + [summary_msg] + keep, notice


def compact_for_model(
    messages: list[Message],
    config: CompactionConfig,
    tool_schemas: list[dict[str, Any]],
    llm: LLMClient | None,
    model: str,
) -> tuple[list[Message], CompactionNotice | None]:
    """Build an in-memory window. Does not modify or delete disk messages."""
    budget = _budget(config)
    if estimate_message_tokens(messages, tool_schemas) <= budget:
        assert_pairing(messages)
        _assert_chat_tool_pairing(messages)
        return messages, None

    window = [msg.model_copy(deep=True) for msg in messages]
    system_end = _system_prefix_len(window)
    keep_start = _keep_start_index(window, config.keep_recent_tokens)
    _prune_zone(window, system_end, keep_start, config.prune_tool_chars)
    _dedup_read_file(window, system_end, keep_start)

    def finalize(
        msgs: list[Message], notice: CompactionNotice | None
    ) -> tuple[list[Message], CompactionNotice | None]:
        repaired = repair_pairing(msgs, messages, budget=budget, tool_schemas=tool_schemas)
        if estimate_message_tokens(repaired, tool_schemas) > budget:
            repaired, extra = _hard_trim(repaired, budget, tool_schemas)
            repaired = repair_pairing(repaired, messages, budget=budget, tool_schemas=tool_schemas)
            if extra and notice is not None:
                notice = CompactionNotice(
                    summary=notice.summary,
                    dropped_count=notice.dropped_count + extra,
                )
            elif extra:
                notice = CompactionNotice(dropped_count=extra)
        _assert_chat_tool_pairing(repaired)
        return repaired, notice

    if estimate_message_tokens(window, tool_schemas) <= budget:
        return finalize(window, None)

    notice: CompactionNotice | None = None
    if config.auto_summarize and llm is not None:
        summarized = _try_summarize(window, system_end, keep_start, llm, model)
        if summarized is not None:
            window, notice = summarized
            if estimate_message_tokens(window, tool_schemas) <= budget:
                return finalize(window, notice)

    window, dropped = _hard_trim(window, budget, tool_schemas)
    if dropped:
        if notice is None:
            notice = CompactionNotice(dropped_count=dropped)
        else:
            notice = CompactionNotice(
                summary=notice.summary,
                dropped_count=notice.dropped_count + dropped,
            )
    return finalize(window, notice)
