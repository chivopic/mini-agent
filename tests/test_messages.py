"""Unit tests for typed messages, v1 migration, and Chat Completions projection."""

import json
from pathlib import Path

import pytest

from mini_agent.llm import OpenAIChatCompletionsClient
from mini_agent.messages import (
    CompactionPart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UnpairedToolError,
    UsagePart,
    history_v1_to_messages,
    to_chat_messages,
)
from mini_agent.session import SessionData, SessionMeta, load_session

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "session_v1_tool_round.json"
GOLDEN_CREATED_AT = "2026-08-18T10:00:00"


def _golden_v1() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _meta(session_id: str, workspace_root: str) -> dict:
    return {
        "session_id": session_id,
        "workspace_root": workspace_root,
        "created_at": GOLDEN_CREATED_AT,
        "updated_at": GOLDEN_CREATED_AT,
        "model": "gpt-4o",
        "title": "测试",
        "turn_count": 1,
    }


def test_golden_v1_migrates_to_five_messages() -> None:
    msgs = history_v1_to_messages(_golden_v1(), created_at=GOLDEN_CREATED_AT)
    assert len(msgs) == 5
    assert [m.role for m in msgs] == ["system", "user", "assistant", "assistant", "assistant"]
    assert all(m.created_at == GOLDEN_CREATED_AT for m in msgs)

    assert isinstance(msgs[0].parts[0], TextPart)
    assert msgs[0].parts[0].text == "You are assistant."
    assert isinstance(msgs[1].parts[0], TextPart)
    assert msgs[1].parts[0].text == "读 a.txt 和 b.txt"

    call_ids = [p.call_id for p in msgs[2].parts if isinstance(p, ToolCallPart)]
    assert call_ids == ["call_read_a", "call_read_b"]
    assert all(isinstance(p, ToolCallPart) for p in msgs[2].parts)

    result_ids = [p.call_id for p in msgs[3].parts if isinstance(p, ToolResultPart)]
    assert result_ids == ["call_read_a", "call_read_b"]
    assert all(isinstance(p, ToolResultPart) for p in msgs[3].parts)
    assert msgs[3].parts[0].ok is True
    assert msgs[3].parts[0].content == "AAA"
    assert msgs[3].parts[1].content == "BBB"

    assert isinstance(msgs[4].parts[0], TextPart)
    assert "AAA" in msgs[4].parts[0].text


def test_to_chat_messages_matches_convert_messages() -> None:
    golden = _golden_v1()
    migrated = history_v1_to_messages(golden, created_at=GOLDEN_CREATED_AT)
    converted = OpenAIChatCompletionsClient(api_key="fake-key")._convert_messages(golden)
    assert to_chat_messages(migrated) == converted


def test_unpaired_result_raises() -> None:
    items = [
        {"role": "system", "content": "You are assistant."},
        {
            "type": "function_call_output",
            "call_id": "orphan",
            "output": '{"ok": false}',
        },
    ]
    with pytest.raises(UnpairedToolError):
        history_v1_to_messages(items, created_at=GOLDEN_CREATED_AT)


def test_unpaired_v1_session_load_returns_none(tmp_path: Path) -> None:
    sid = "unpaired_v1"
    payload = {
        "meta": _meta(sid, tmp_path.as_posix()),
        "history": [
            {"role": "system", "content": "You are assistant."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "not_open",
                "output": '{"ok": true, "content": "x"}',
            },
        ],
    }
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_session(sid, sessions_dir=sessions_dir) is None


def test_hanging_tool_calls_at_eof_are_dropped(tmp_path: Path) -> None:
    sid = "hanging_v1"
    payload = {
        "meta": _meta(sid, tmp_path.as_posix()),
        "history": [
            {"role": "system", "content": "You are assistant."},
            {"role": "user", "content": "读文件"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_hang",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
        ],
    }
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_session(sid, sessions_dir=sessions_dir)
    assert loaded is not None
    assert [m.role for m in loaded.messages] == ["system", "user"]
    assert not any(isinstance(p, ToolCallPart) for m in loaded.messages for p in m.parts)


def test_to_chat_messages_skips_usage_part() -> None:
    msgs = [
        Message(role="system", parts=[TextPart(text="sys")]),
        Message(
            role="assistant",
            parts=[
                TextPart(text="hello"),
                UsagePart(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            ],
        ),
        Message(
            role="assistant",
            parts=[UsagePart(prompt_tokens=1, completion_tokens=1, total_tokens=2)],
        ),
    ]
    wire = to_chat_messages(msgs)
    assert wire == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "hello"},
    ]


def test_compaction_part_encodes_as_user_summary() -> None:
    msgs = [
        Message(role="system", parts=[TextPart(text="sys")]),
        Message(role="user", parts=[CompactionPart(summary="先前讨论了排序")]),
    ]
    wire = to_chat_messages(msgs)
    assert wire == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "[上下文摘要]\n先前讨论了排序"},
    ]


def test_compaction_part_then_text_joined() -> None:
    msgs = [
        Message(
            role="user",
            parts=[CompactionPart(summary="摘要"), TextPart(text="继续")],
        )
    ]
    wire = to_chat_messages(msgs)
    assert wire == [{"role": "user", "content": "[上下文摘要]\n摘要\n\n继续"}]


def test_unpaired_v2_session_load_returns_none(tmp_path: Path) -> None:
    sid = "unpaired_v2"
    session = SessionData(
        schema_version=2,
        meta=SessionMeta.model_validate(_meta(sid, tmp_path.as_posix())),
        messages=[
            Message(
                role="assistant",
                parts=[
                    ToolCallPart(call_id="c1", name="read_file", arguments="{}"),
                ],
                created_at=GOLDEN_CREATED_AT,
            )
        ],
    )
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / f"{sid}.json").write_text(session.model_dump_json(), encoding="utf-8")
    assert load_session(sid, sessions_dir=sessions_dir) is None
