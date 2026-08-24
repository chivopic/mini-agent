"""Typed conversation messages, v1 history migration, and Chat Completions projection."""

import json
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from mini_agent.models import ToolResult


def _now_iso() -> str:
    return datetime.now().isoformat()


class UnpairedToolError(Exception):
    """A tool result could not be paired with an open tool call in the current round."""


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: str


class ToolResultPart(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    ok: bool
    content: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsagePart(BaseModel):
    type: Literal["usage"] = "usage"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_cny: float = 0.0
    estimated: bool = False


class CompactionPart(BaseModel):
    type: Literal["compaction"] = "compaction"
    summary: str
    dropped_count: int = 0


Part = Annotated[
    TextPart | ToolCallPart | ToolResultPart | UsagePart | CompactionPart,
    Field(discriminator="type"),
]


class Message(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    role: Literal["system", "user", "assistant"]
    parts: list[Part] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)


def _tool_result_from_part(part: ToolResultPart) -> ToolResult:
    return ToolResult(
        ok=part.ok,
        content=part.content,
        error=part.error,
        metadata=part.metadata,
    )


def _parse_tool_result(raw: Any) -> ToolResult:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return ToolResult.model_validate(data)
    except Exception:
        return ToolResult(ok=False, content=str(raw))


def assert_pairing(messages: list[Message]) -> None:
    """Require each tool-call assistant message to be followed by matching results."""
    for i, msg in enumerate(messages):
        call_ids = [p.call_id for p in msg.parts if isinstance(p, ToolCallPart)]
        if not call_ids:
            continue
        if msg.role != "assistant":
            raise UnpairedToolError("ToolCallPart is only valid on assistant messages")
        if i + 1 >= len(messages):
            raise UnpairedToolError("Assistant tool call is missing a following result message")
        nxt = messages[i + 1]
        result_parts = [p for p in nxt.parts if isinstance(p, ToolResultPart)]
        result_ids = [p.call_id for p in result_parts]
        if (
            nxt.role != "assistant"
            or len(result_parts) != len(nxt.parts)
            or set(result_ids) != set(call_ids)
        ):
            raise UnpairedToolError("Tool call ids do not match the following ToolResultParts")


def history_v1_to_messages(items: list[dict[str, Any]], created_at: str) -> list[Message]:
    """Migrate 0.2 history dicts into typed messages using the current-round pairing machine."""
    open_calls: dict[str, str] = {}
    pending_results: list[ToolResultPart] = []
    out: list[Message] = []

    def flush_results(*, eof: bool) -> None:
        if not pending_results and not open_calls:
            return
        if eof and not pending_results and open_calls:
            # Crash-truncated v1: trailing tool_calls with no results — drop the hanging call.
            assert out and any(isinstance(p, ToolCallPart) for p in out[-1].parts)
            out.pop()
            open_calls.clear()
            return
        if {p.call_id for p in pending_results} != set(open_calls):
            raise UnpairedToolError("Tool results do not match the open tool calls")
        ordered = [next(p for p in pending_results if p.call_id == cid) for cid in open_calls]
        out.append(Message(role="assistant", parts=ordered, created_at=created_at))
        pending_results.clear()
        open_calls.clear()

    for item in items:
        role, typ = item.get("role"), item.get("type")
        if typ == "function_call_output" or role == "tool":
            cid = item.get("call_id") or item.get("tool_call_id")
            if cid not in open_calls:
                raise UnpairedToolError("Tool result call_id is not in the current open round")
            name = open_calls[cid]
            raw = item.get("output") or item.get("content") or ""
            tr = _parse_tool_result(raw)
            pending_results.append(
                ToolResultPart(
                    call_id=cid,
                    name=name,
                    ok=tr.ok,
                    content=tr.content,
                    error=tr.error,
                    metadata=tr.metadata,
                )
            )
            continue

        flush_results(eof=False)

        if role == "system" or role == "user":
            out.append(
                Message(
                    role=role,
                    parts=[TextPart(text=item.get("content") or "")],
                    created_at=created_at,
                )
            )
        elif role == "assistant" and item.get("tool_calls"):
            parts: list[Part] = []
            if item.get("content"):
                parts.append(TextPart(text=item["content"]))
            for tc in item["tool_calls"]:
                cid = tc["id"]
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                parts.append(ToolCallPart(call_id=cid, name=name, arguments=args))
                open_calls[cid] = name
            out.append(Message(role="assistant", parts=parts, created_at=created_at))
        elif role == "assistant" or typ == "message":
            out.append(
                Message(
                    role="assistant",
                    parts=[TextPart(text=item.get("content") or "")],
                    created_at=created_at,
                )
            )

    flush_results(eof=True)
    return out


def messages_to_v1_history(messages: list[Message]) -> list[dict[str, Any]]:
    """Project typed messages back to 0.2 history dicts (skips usage and compaction)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        texts = [p for p in msg.parts if isinstance(p, TextPart)]
        calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
        results = [p for p in msg.parts if isinstance(p, ToolResultPart)]

        if results:
            for part in results:
                out.append(
                    {
                        "type": "function_call_output",
                        "call_id": part.call_id,
                        "output": _tool_result_from_part(part).model_dump_json(),
                    }
                )
            continue

        if calls:
            item: dict[str, Any] = {"role": "assistant"}
            if texts and texts[0].text:
                item["content"] = texts[0].text
            item["tool_calls"] = [
                {
                    "id": part.call_id,
                    "type": "function",
                    "function": {"name": part.name, "arguments": part.arguments},
                }
                for part in calls
            ]
            out.append(item)
            continue

        if msg.role in ("system", "user"):
            if not texts:
                continue
            out.append({"role": msg.role, "content": texts[0].text})
            continue

        if texts:
            out.append({"role": "assistant", "content": texts[0].text})

    return out


def to_chat_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Expand typed messages into Chat Completions wire dicts."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role in ("system", "user"):
            chunks: list[str] = []
            for part in msg.parts:
                if isinstance(part, CompactionPart):
                    chunks.append("[上下文摘要]\n" + part.summary)
                elif isinstance(part, TextPart):
                    chunks.append(part.text)
            if chunks:
                out.append({"role": msg.role, "content": "\n\n".join(chunks)})
            continue

        results = [p for p in msg.parts if isinstance(p, ToolResultPart)]
        if results:
            for part in results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.call_id,
                        "content": _tool_result_from_part(part).model_dump_json(),
                    }
                )
            continue

        texts = [p for p in msg.parts if isinstance(p, TextPart)]
        calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
        if calls:
            item: dict[str, Any] = {"role": "assistant"}
            if texts and texts[0].text:
                item["content"] = texts[0].text
            item["tool_calls"] = [
                {
                    "id": part.call_id,
                    "type": "function",
                    "function": {"name": part.name, "arguments": part.arguments},
                }
                for part in calls
            ]
            out.append(item)
            continue

        if texts:
            out.append({"role": "assistant", "content": texts[0].text})

    return out
