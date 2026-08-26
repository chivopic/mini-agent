"""Token-aware compaction: prune, dedup, pairing repair, optional summarize, turn trim."""

from __future__ import annotations

import json
import threading
from typing import Any, Literal

from mini_agent.agent import Agent
from mini_agent.compaction import (
    CompactionConfig,
    compact_for_model,
    estimate_message_tokens,
    repair_pairing,
)
from mini_agent.cost import estimate_tokens_from_text
from mini_agent.llm import LLMResponse
from mini_agent.messages import (
    CompactionPart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UsagePart,
    to_chat_messages,
)
from mini_agent.models import AgentConfig, ToolResult
from mini_agent.session import SessionData, SessionMeta, load_session
from mini_agent.tools.registry import ToolRegistry


class RecordingLLM:
    def __init__(self, text: str = "先前讨论了若干文件") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []
        self.exc: Exception | None = None

    def create_response(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        on_token: Any = None,
        cancel: Any = None,
    ) -> LLMResponse:
        _assert_no_orphan_tools(messages)
        self.calls.append({"messages": list(messages), "tools": list(tools), "model": model})
        if self.exc is not None:
            raise self.exc
        if self.text and on_token is not None:
            on_token(self.text)
        return LLMResponse(text=self.text)


def _text(role: Literal["system", "user", "assistant"], text: str) -> Message:
    return Message(role=role, parts=[TextPart(text=text)])


def _call(call_id: str, name: str, arguments: str) -> Message:
    return Message(
        role="assistant",
        parts=[ToolCallPart(call_id=call_id, name=name, arguments=arguments)],
    )


def _result(
    call_id: str,
    name: str,
    content: str,
    *,
    ok: bool = True,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    return Message(
        role="assistant",
        parts=[
            ToolResultPart(
                call_id=call_id,
                name=name,
                ok=ok,
                content=content,
                error=error,
                metadata=metadata or {},
            )
        ],
    )


def _fillers(n: int, size: int = 400) -> list[Message]:
    return [_text("user", f"{i} " + "X" * size) for i in range(n)]


def _over_config(
    messages: list[Message],
    tool_schemas: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> CompactionConfig:
    total = estimate_message_tokens(messages, tool_schemas)
    window = max(total, 2)
    params: dict[str, Any] = {
        "context_window_tokens": window,
        "buffer_tokens": 1,
        "keep_recent_tokens": 1,
        "auto_summarize": False,
    }
    params.update(kwargs)
    return CompactionConfig(**params)


def _assert_no_orphan_tools(window: list[Message]) -> None:
    open_ids: set[str] = set()
    for row in to_chat_messages(window):
        if row.get("role") == "tool":
            assert row.get("tool_call_id") in open_ids
            open_ids.remove(row["tool_call_id"])
            continue
        assert not open_ids
        if row.get("role") == "assistant" and row.get("tool_calls"):
            open_ids = {tc["id"] for tc in row["tool_calls"]}
        else:
            open_ids = set()
    assert not open_ids


def _keep_recent_starting_at(messages: list[Message], idx: int) -> int:
    """Smallest keep_recent_tokens so the tail starts at idx (message-atomic)."""
    after = estimate_message_tokens(messages[idx + 1 :])
    return after + 1


def test_estimate_skips_usage_counts_envelope_and_schemas() -> None:
    envelope = ToolResult(
        ok=True, content="hello", error=None, metadata={"path": "a.py"}
    ).model_dump_json()
    msgs = [
        Message(
            role="assistant",
            parts=[
                TextPart(text="hi"),
                UsagePart(prompt_tokens=9999, total_tokens=9999),
                ToolResultPart(
                    call_id="c1",
                    name="read_file",
                    ok=True,
                    content="hello",
                    metadata={"path": "a.py"},
                ),
            ],
        )
    ]
    got = estimate_message_tokens(msgs)
    assert got == estimate_tokens_from_text("hi") + estimate_tokens_from_text(envelope)

    call_msg = _call("c2", "write_file", '{"path": "a.py"}')
    assert estimate_message_tokens([call_msg]) == estimate_tokens_from_text('{"path": "a.py"}')

    summary = Message(role="user", parts=[CompactionPart(summary="先前做了重构")])
    assert estimate_message_tokens([summary]) == estimate_tokens_from_text("先前做了重构")

    schemas = [{"type": "function", "function": {"name": "read_file"}}]
    assert estimate_message_tokens(msgs, schemas) > got


def test_under_budget_returns_original() -> None:
    messages = [_text("system", "sys"), _text("user", "hello"), _text("assistant", "hi")]
    window, notice = compact_for_model(messages, CompactionConfig(), [], None, "gpt")
    assert window is messages
    assert notice is None


def test_prune_folds_old_success_keeps_failure_and_original() -> None:
    huge = "A" * 8000
    fail_body = "E" * 600
    messages = [
        _text("system", "sys"),
        _text("user", "read"),
        _call("c1", "read_file", "{}"),
        _result("c1", "read_file", huge, metadata={"path": "a.py"}),
        _text("user", "run"),
        _call("c2", "run_shell", '{"command": "true"}'),
        _result("c2", "run_shell", fail_body, ok=False, error="boom"),
        _text("user", "recent"),
    ]
    original_huge = messages[3].parts[0].content
    config = _over_config(messages)
    window, _notice = compact_for_model(messages, config, [], None, "gpt")
    _assert_no_orphan_tools(window)
    assert messages[3].parts[0].content == original_huge
    assert len(messages) == 8

    folded = window[3].parts[0]
    assert isinstance(folded, ToolResultPart)
    assert "已折叠" in folded.content
    assert "8000" in folded.content
    assert huge not in folded.content

    failed = window[6].parts[0]
    assert isinstance(failed, ToolResultPart)
    assert failed.error == "boom"
    assert failed.content == fail_body


def test_prune_folds_large_arguments_keeps_path() -> None:
    args = json.dumps({"path": "src/a.py", "content": "Z" * 2000}, ensure_ascii=False)
    messages = [
        _text("system", "sys"),
        _text("user", "write"),
        _call("c1", "write_file", args),
        _result("c1", "write_file", "ok", metadata={"path": "src/a.py"}),
        _text("user", "recent"),
    ]
    config = _over_config(messages)
    window, _notice = compact_for_model(messages, config, [], None, "gpt")
    _assert_no_orphan_tools(window)
    assert messages[2].parts[0].arguments == args
    folded_args = window[2].parts[0].arguments
    assert isinstance(window[2].parts[0], ToolCallPart)
    parsed = json.loads(folded_args)
    assert parsed["path"] == "src/a.py"
    assert "已折叠" in parsed["_folded"]
    assert "Z" * 50 not in folded_args


def test_dedup_read_file_in_prune_zone_keeps_last_body() -> None:
    huge = "H" * 8000
    messages = [
        _text("system", "sys"),
        _text("user", "r1"),
        _call("c1", "read_file", '{"path": "a.py"}'),
        _result("c1", "read_file", "first body", metadata={"path": "a.py"}),
        _text("user", "r2"),
        _call("c2", "read_file", '{"path": "a.py"}'),
        _result("c2", "read_file", "second body", metadata={"path": "a.py"}),
        _text("user", "r3"),
        _call("c3", "read_file", '{"path": "b.py"}'),
        _result("c3", "read_file", "other", metadata={"path": "b.py"}),
        _text("user", "r4"),
        _call("c4", "read_file", "{}"),
        _result("c4", "read_file", "no path"),
        _text("user", "inflate"),
        _call("c5", "read_file", "{}"),
        _result("c5", "read_file", huge, metadata={"path": "z.py"}),
        _text("user", "recent"),
    ]
    config = _over_config(messages)
    window, _notice = compact_for_model(messages, config, [], None, "gpt")
    _assert_no_orphan_tools(window)
    assert window[3].parts[0].content == "[已过时的 read_file: a.py]"
    assert window[6].parts[0].content == "second body"
    assert window[9].parts[0].content == "other"
    assert window[12].parts[0].content == "no path"
    assert messages[3].parts[0].content == "first body"


def test_keep_zone_not_pruned() -> None:
    huge = "K" * 8000
    keep_body = "R" * 600
    messages = [
        _text("system", "sys"),
        _text("user", "old"),
        _call("c1", "read_file", "{}"),
        _result("c1", "read_file", huge, metadata={"path": "old.py"}),
        _text("user", "now"),
        _call("c2", "read_file", "{}"),
        _result("c2", "read_file", keep_body, metadata={"path": "new.py"}),
    ]
    config = _over_config(messages, keep_recent_tokens=1)
    window, _notice = compact_for_model(messages, config, [], None, "gpt")
    _assert_no_orphan_tools(window)
    assert "已折叠" in window[3].parts[0].content
    assert window[-1].parts[0].content == keep_body


def test_auto_summarize_false_does_not_call_llm() -> None:
    llm = RecordingLLM()
    messages = [_text("system", "sys"), *_fillers(8, 500), _text("user", "recent")]
    config = _over_config(messages, auto_summarize=False)
    window, _notice = compact_for_model(messages, config, [], llm, "gpt")
    assert llm.calls == []
    _assert_no_orphan_tools(window)


def test_llm_none_does_not_summarize() -> None:
    messages = [_text("system", "sys"), *_fillers(8, 500), _text("user", "recent")]
    config = _over_config(messages, auto_summarize=True)
    window, _notice = compact_for_model(messages, config, [], None, "gpt")
    _assert_no_orphan_tools(window)
    assert not any(isinstance(part, CompactionPart) for msg in window for part in msg.parts)


def test_summarize_inserts_compaction_part_after_system() -> None:
    llm = RecordingLLM("用户在改 a.py")
    messages = [
        _text("system", "live system"),
        *_fillers(8, 500),
        _text("user", "recent"),
        _text("assistant", "done"),
    ]
    system_text = messages[0].parts[0].text
    config = _over_config(messages, auto_summarize=True, keep_recent_tokens=1)
    window, notice = compact_for_model(messages, config, [], llm, "gpt")
    assert llm.calls, "summarize should call the llm"
    assert llm.calls[0]["tools"] == []
    assert window[0].parts[0].text == system_text
    assert isinstance(window[1].parts[0], CompactionPart)
    assert window[1].parts[0].summary == "用户在改 a.py"
    wire = to_chat_messages(window)
    assert wire[0] == {"role": "system", "content": "live system"}
    assert wire[1]["content"].startswith("[上下文摘要]\n")
    assert notice is not None
    assert notice.summary == "用户在改 a.py"
    assert messages[0].parts[0].text == system_text
    _assert_no_orphan_tools(window)


def test_summarize_failure_falls_back_to_hard_trim() -> None:
    llm = RecordingLLM("")
    llm.exc = RuntimeError("boom")
    messages = [_text("system", "sys"), *_fillers(8, 500), _text("user", "recent-keep")]
    config = _over_config(messages, auto_summarize=True)
    window, _notice = compact_for_model(messages, config, [], llm, "gpt")
    _assert_no_orphan_tools(window)
    assert not any(isinstance(part, CompactionPart) for msg in window for part in msg.parts)
    assert any(
        isinstance(part, TextPart) and part.text == "recent-keep"
        for msg in window
        for part in msg.parts
    )


def test_token_window_never_produces_orphan_tool_when_keep_splits_pair() -> None:
    huge_args = json.dumps({"path": "a.py", "content": "Z" * 4000}, ensure_ascii=False)
    fillers = _fillers(6, 400)
    call = _call("c1", "write_file", huge_args)
    result = _result("c1", "write_file", "wrote", metadata={"path": "a.py"})
    recent_u = _text("user", "next")
    recent_a = _text("assistant", "ok")
    messages = [
        _text("system", "sys"),
        *fillers,
        _text("user", "write it"),
        call,
        result,
        recent_u,
        recent_a,
    ]
    result_idx = messages.index(result)
    keep_recent = _keep_recent_starting_at(messages, result_idx)
    llm = RecordingLLM("摘要")
    config = CompactionConfig(
        context_window_tokens=120,
        buffer_tokens=20,
        keep_recent_tokens=keep_recent,
        auto_summarize=True,
        prune_tool_chars=500,
    )
    assert estimate_message_tokens(messages) > config.context_window_tokens - config.buffer_tokens
    window, _notice = compact_for_model(messages, config, [], llm, "gpt")
    _assert_no_orphan_tools(window)
    assert llm.calls
    _assert_no_orphan_tools(llm.calls[0]["messages"])
    assert huge_args == call.parts[0].arguments


def test_summarize_split_pair_restores_pruned_call_not_disk_args() -> None:
    huge_args = json.dumps({"path": "a.py", "content": "Z" * 4000}, ensure_ascii=False)
    call = _call("c1", "write_file", huge_args)
    result = _result("c1", "write_file", "wrote", metadata={"path": "a.py"})
    recent_u = _text("user", "next")
    recent_a = _text("assistant", "ok")
    messages = [
        _text("system", "sys"),
        *_fillers(6, 400),
        _text("user", "write it"),
        call,
        result,
        recent_u,
        recent_a,
    ]
    result_idx = messages.index(result)
    keep_recent = _keep_recent_starting_at(messages, result_idx)
    llm = RecordingLLM("摘要")
    config = CompactionConfig(
        context_window_tokens=400,
        buffer_tokens=20,
        keep_recent_tokens=keep_recent,
        auto_summarize=True,
    )
    window, _notice = compact_for_model(messages, config, [], llm, "gpt")
    _assert_no_orphan_tools(window)
    assert llm.calls
    _assert_no_orphan_tools(llm.calls[0]["messages"])

    call_parts = [p for m in window for p in m.parts if isinstance(p, ToolCallPart)]
    assert [p.call_id for p in call_parts] == ["c1"]
    parsed = json.loads(call_parts[0].arguments)
    assert parsed["path"] == "a.py"
    assert "已折叠" in parsed["_folded"]
    assert "Z" * 50 not in call_parts[0].arguments
    assert call.parts[0].arguments == huge_args


def test_over_budget_drops_whole_turn_not_half_pair() -> None:
    huge_keep = "R" * 4000
    call = _call("c1", "read_file", "{}")
    result = _result("c1", "read_file", huge_keep, metadata={"path": "a.py"})
    recent_u = _text("user", "next")
    recent_a = _text("assistant", "ok")
    messages = [
        _text("system", "sys"),
        *_fillers(6, 400),
        _text("user", "read it"),
        call,
        result,
        recent_u,
        recent_a,
    ]
    result_idx = messages.index(result)
    keep_recent = _keep_recent_starting_at(messages, result_idx)
    llm = RecordingLLM("摘要")
    config = CompactionConfig(
        context_window_tokens=80,
        buffer_tokens=20,
        keep_recent_tokens=keep_recent,
        auto_summarize=True,
    )
    window, _notice = compact_for_model(messages, config, [], llm, "gpt")
    _assert_no_orphan_tools(window)
    assert llm.calls
    _assert_no_orphan_tools(llm.calls[0]["messages"])

    rows = to_chat_messages(window)
    tool_ids = [row["tool_call_id"] for row in rows if row.get("role") == "tool"]
    call_ids = [
        tc["id"]
        for row in rows
        if row.get("role") == "assistant"
        for tc in row.get("tool_calls") or []
    ]
    assert "c1" not in tool_ids
    assert "c1" not in call_ids
    assert any(row.get("content") == "next" for row in rows)
    assert messages[messages.index(result)].parts[0].content == huge_keep


def test_repair_pairing_inserts_missing_call() -> None:
    sys = _text("system", "s")
    user = _text("user", "u")
    call = _call("c1", "read_file", "{}")
    result = _result("c1", "read_file", "body")
    original = [sys, user, call, result]
    window = [sys, result]
    fixed = repair_pairing(window, original)
    _assert_no_orphan_tools(fixed)
    call_ids = [p.call_id for m in fixed for p in m.parts if isinstance(p, ToolCallPart)]
    result_ids = [p.call_id for m in fixed for p in m.parts if isinstance(p, ToolResultPart)]
    assert call_ids == ["c1"]
    assert result_ids == ["c1"]


def test_repair_pairing_over_budget_drops_whole_turn() -> None:
    sys = _text("system", "s")
    user = _text("user", "u")
    call = _call("c1", "write_file", json.dumps({"path": "a.py", "content": "Z" * 4000}))
    result = _result("c1", "write_file", "ok")
    recent = _text("user", "next")
    original = [sys, user, call, result, recent]
    window = [sys, result, recent]
    fixed = repair_pairing(window, original, budget=20, tool_schemas=[])
    _assert_no_orphan_tools(fixed)
    ids = {
        p.call_id for m in fixed for p in m.parts if isinstance(p, ToolCallPart | ToolResultPart)
    }
    assert "c1" not in ids
    assert any(isinstance(p, TextPart) and p.text == "next" for m in fixed for p in m.parts)


def test_last_resort_prunes_keep_zone_when_last_turn_still_over() -> None:
    args = json.dumps({"path": "a.py", "content": "Z" * 5000}, ensure_ascii=False)
    messages = [
        _text("system", "sys"),
        _text("user", "write"),
        _call("c1", "write_file", args),
        _result("c1", "write_file", "ok", metadata={"path": "a.py"}),
    ]
    total = estimate_message_tokens(messages)
    config = CompactionConfig(
        context_window_tokens=50,
        buffer_tokens=10,
        keep_recent_tokens=total + 10,
        auto_summarize=False,
        prune_tool_chars=500,
    )
    window, _notice = compact_for_model(messages, config, [], None, "gpt")
    _assert_no_orphan_tools(window)
    call_part = next(p for m in window for p in m.parts if isinstance(p, ToolCallPart))
    parsed = json.loads(call_part.arguments)
    assert parsed["path"] == "a.py"
    assert "已折叠" in parsed["_folded"]
    assert "Z" * 50 not in call_part.arguments
    assert messages[2].parts[0].arguments == args


def test_cancel_skips_summarize() -> None:
    llm = RecordingLLM()
    messages = [_text("system", "sys"), *_fillers(8, 500), _text("user", "recent")]
    config = _over_config(messages, auto_summarize=True)
    cancel = threading.Event()
    cancel.set()
    window, _notice = compact_for_model(messages, config, [], llm, "gpt", cancel=cancel)
    assert llm.calls == []
    _assert_no_orphan_tools(window)


def test_hard_trim_drops_oldest_user_turns() -> None:
    messages = [_text("system", "sys"), *_fillers(10, 400), _text("user", "latest")]
    config = _over_config(messages, auto_summarize=False, keep_recent_tokens=8)
    window, notice = compact_for_model(messages, config, [], None, "gpt")
    _assert_no_orphan_tools(window)
    assert window[0].role == "system"
    assert any(isinstance(p, TextPart) and p.text == "latest" for m in window for p in m.parts)
    assert len(window) < len(messages)
    assert notice is not None
    assert notice.dropped_count >= 1
    assert len(messages) == 12


def test_session_v2_without_compacted_at_still_loads(tmp_path: Any) -> None:
    sid = "no_compacted_at"
    payload = {
        "schema_version": 2,
        "meta": {
            "session_id": sid,
            "workspace_root": tmp_path.as_posix(),
            "created_at": "2026-08-18T10:00:00",
            "updated_at": "2026-08-18T10:00:00",
            "model": "gpt-4o",
            "title": "旧",
            "turn_count": 0,
        },
        "messages": [
            {
                "id": "sys1",
                "role": "system",
                "parts": [{"type": "text", "text": "sys"}],
                "created_at": "2026-08-18T10:00:00",
            }
        ],
        "permission_memory": [],
    }
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_session(sid, sessions_dir=sessions_dir)
    assert loaded is not None
    assert loaded.meta.compacted_at is None


def test_agent_compact_does_not_delete_disk_messages(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("MINI_AGENT_SESSIONS_DIR", str(tmp_path / "sess"))
    huge = "H" * 4000
    now = "2026-08-24T00:00:00"
    messages = [
        _text("system", "sys"),
        _text("user", "see"),
        _call("c1", "read_file", "{}"),
        _result("c1", "read_file", huge, metadata={"path": "a.py"}),
    ]
    session = SessionData(
        meta=SessionMeta(
            session_id="compact_disk",
            workspace_root=tmp_path.as_posix(),
            created_at=now,
            updated_at=now,
            model="gpt-4o-mini",
        ),
        messages=messages,
    )
    total = estimate_message_tokens(messages)
    llm = RecordingLLM("ok")
    agent = Agent(
        config=AgentConfig(workspace_root=tmp_path),
        llm_client=llm,  # type: ignore[arg-type]
        session=session,
        registry=ToolRegistry(),
        compaction_config=CompactionConfig(
            context_window_tokens=max(total, 8),
            buffer_tokens=max(total // 2, 1),
            keep_recent_tokens=1,
            auto_summarize=False,
        ),
    )
    answer = agent.step("continue")
    assert answer == "ok"
    disk_contents = [
        p.content
        for m in agent.messages
        for p in m.parts
        if isinstance(p, ToolResultPart) and p.call_id == "c1"
    ]
    assert disk_contents == [huge]
    window = llm.calls[0]["messages"]
    folded = [
        p.content
        for m in window
        for p in m.parts
        if isinstance(p, ToolResultPart) and p.call_id == "c1"
    ]
    assert folded and "已折叠" in folded[0]
