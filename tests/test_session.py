"""Unit tests for session persistence and resume manager."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from mini_agent.messages import (
    Message,
    TextPart,
    ToolCallPart,
    UnpairedToolError,
    history_v1_to_messages,
)
from mini_agent.session import (
    SessionData,
    SessionMeta,
    delete_session,
    generate_session_id,
    get_default_sessions_dir,
    get_latest_session,
    list_sessions,
    load_session,
    save_session,
)


def test_generate_session_id() -> None:
    sid1 = generate_session_id()
    sid2 = generate_session_id()
    assert len(sid1) > 10
    assert sid1 != sid2


def test_save_and_load_session(tmp_path: Path) -> None:
    sid = "20260818_120000_abc123"
    meta = SessionMeta(
        session_id=sid,
        workspace_root=tmp_path.as_posix(),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        model="deepseek-chat",
        title="测试会话",
        turn_count=2,
    )
    history = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    session = SessionData(
        meta=meta,
        messages=history_v1_to_messages(history, created_at=meta.created_at),
    )

    saved_path = save_session(session, sessions_dir=tmp_path)
    assert saved_path.exists()

    loaded = load_session(sid, sessions_dir=tmp_path)
    assert loaded is not None
    assert loaded.schema_version == 2
    assert loaded.meta.session_id == sid
    assert loaded.meta.title == "测试会话"
    assert loaded.meta.turn_count == 2
    assert len(loaded.messages) == 3
    assert len(loaded.history) == 3
    assert isinstance(loaded.messages[0].parts[0], TextPart)

    raw = json.loads(saved_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    assert "messages" in raw
    assert "history" not in raw
    assert "permission_memory" in raw


def test_load_non_existent_session(tmp_path: Path) -> None:
    loaded = load_session("non_existent_id", sessions_dir=tmp_path)
    assert loaded is None


def test_list_sessions_and_filter(tmp_path: Path) -> None:
    ws1 = tmp_path / "ws1"
    ws2 = tmp_path / "ws2"
    ws1.mkdir()
    ws2.mkdir()

    sessions_dir = tmp_path / "sessions"

    # Create session 1 for ws1
    s1 = SessionData(
        meta=SessionMeta(
            session_id="s1",
            workspace_root=ws1.resolve().as_posix(),
            created_at="2026-08-18T10:00:00",
            updated_at="2026-08-18T10:00:00",
            model="gpt-4o",
            title="会话 1",
            turn_count=1,
        )
    )
    save_session(s1, sessions_dir=sessions_dir)

    # Create session 2 for ws2
    s2 = SessionData(
        meta=SessionMeta(
            session_id="s2",
            workspace_root=ws2.resolve().as_posix(),
            created_at="2026-08-18T11:00:00",
            updated_at="2026-08-18T11:00:00",
            model="deepseek-chat",
            title="会话 2",
            turn_count=3,
        )
    )
    save_session(s2, sessions_dir=sessions_dir)

    # List all
    all_sessions = list_sessions(sessions_dir=sessions_dir)
    assert len(all_sessions) == 2
    assert all_sessions[0].session_id == "s2"  # updated later

    # Filter ws1
    ws1_sessions = list_sessions(workspace_root=ws1, sessions_dir=sessions_dir)
    assert len(ws1_sessions) == 1
    assert ws1_sessions[0].session_id == "s1"

    # Filter ws2
    ws2_sessions = list_sessions(workspace_root=ws2, sessions_dir=sessions_dir)
    assert len(ws2_sessions) == 1
    assert ws2_sessions[0].session_id == "s2"


def test_get_latest_session(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    ws = tmp_path / "my_project"
    ws.mkdir()

    assert get_latest_session(workspace_root=ws, sessions_dir=sessions_dir) is None

    s1 = SessionData(
        meta=SessionMeta(
            session_id="first",
            workspace_root=ws.resolve().as_posix(),
            created_at="2026-08-18T09:00:00",
            updated_at="2026-08-18T09:00:00",
            model="gpt-4o",
            title="旧对话",
            turn_count=1,
        )
    )
    save_session(s1, sessions_dir=sessions_dir)

    s2 = SessionData(
        meta=SessionMeta(
            session_id="second",
            workspace_root=ws.resolve().as_posix(),
            created_at="2026-08-18T12:00:00",
            updated_at="2026-08-18T12:00:00",
            model="gpt-4o",
            title="新对话",
            turn_count=2,
        )
    )
    save_session(s2, sessions_dir=sessions_dir)

    latest = get_latest_session(workspace_root=ws, sessions_dir=sessions_dir)
    assert latest is not None
    assert latest.meta.session_id == "second"
    assert latest.meta.title == "新对话"


def test_delete_session(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    s = SessionData(
        meta=SessionMeta(
            session_id="to_delete",
            workspace_root=tmp_path.as_posix(),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            model="gpt-4o",
            title="将被删除",
            turn_count=1,
        )
    )
    save_session(s, sessions_dir=sessions_dir)

    assert delete_session("to_delete", sessions_dir=sessions_dir) is True
    assert load_session("to_delete", sessions_dir=sessions_dir) is None
    assert delete_session("to_delete", sessions_dir=sessions_dir) is False


def test_corrupt_session_file_ignored(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    corrupt_file = sessions_dir / "bad.json"
    corrupt_file.write_text("invalid json content", encoding="utf-8")

    assert load_session("bad", sessions_dir=sessions_dir) is None
    assert list_sessions(sessions_dir=sessions_dir) == []


def test_load_v1_session_migrates_and_resave_is_v2(tmp_path: Path) -> None:
    sid = "v1_tool_round"
    fixture = Path(__file__).parent / "fixtures" / "session_v1_tool_round.json"
    v1 = {
        "meta": {
            "session_id": sid,
            "workspace_root": tmp_path.as_posix(),
            "created_at": "2026-08-18T10:00:00",
            "updated_at": "2026-08-18T10:00:00",
            "model": "gpt-4o",
            "title": "旧会话",
            "turn_count": 1,
        },
        "history": json.loads(fixture.read_text(encoding="utf-8")),
    }
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    path = sessions_dir / f"{sid}.json"
    path.write_text(json.dumps(v1), encoding="utf-8")

    loaded = load_session(sid, sessions_dir=sessions_dir)
    assert loaded is not None
    assert loaded.schema_version == 2
    assert len(loaded.messages) == 5
    assert loaded.meta.title == "旧会话"
    assert loaded.history[0]["role"] == "system"
    assert loaded.history[3]["type"] == "function_call_output"

    save_session(loaded, sessions_dir=sessions_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    assert "messages" in raw
    assert "history" not in raw
    assert len(raw["messages"]) == 5

    reloaded = load_session(sid, sessions_dir=sessions_dir)
    assert reloaded is not None
    assert reloaded.schema_version == 2
    assert len(reloaded.messages) == 5


def test_list_sessions_reads_v1_and_v2_meta(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    ws = tmp_path.resolve().as_posix()

    v1 = {
        "meta": {
            "session_id": "old_v1",
            "workspace_root": ws,
            "created_at": "2026-08-18T09:00:00",
            "updated_at": "2026-08-18T09:00:00",
            "model": "gpt-4o",
            "title": "v1 会话",
            "turn_count": 1,
        },
        "history": [{"role": "system", "content": "sys"}],
    }
    (sessions_dir / "old_v1.json").write_text(json.dumps(v1), encoding="utf-8")

    s2 = SessionData(
        meta=SessionMeta(
            session_id="new_v2",
            workspace_root=ws,
            created_at="2026-08-18T12:00:00",
            updated_at="2026-08-18T12:00:00",
            model="gpt-4o",
            title="v2 会话",
            turn_count=2,
        )
    )
    save_session(s2, sessions_dir=sessions_dir)

    listed = list_sessions(sessions_dir=sessions_dir)
    ids = {m.session_id for m in listed}
    assert ids == {"old_v1", "new_v2"}


def test_default_sessions_dir_chmod_mini_agent_home(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.delenv("MINI_AGENT_SESSIONS_DIR", raising=False)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "mini_agent.session.Path.home",
        classmethod(lambda cls: tmp_path),
    )

    sessions_dir = get_default_sessions_dir()
    home_dir = tmp_path / ".mini-agent"
    assert sessions_dir == home_dir / "sessions"
    assert home_dir.is_dir()
    assert (home_dir.stat().st_mode & 0o777) == 0o700


def _v1_meta(session_id: str, workspace_root: str) -> dict:
    return {
        "session_id": session_id,
        "workspace_root": workspace_root,
        "created_at": "2026-08-18T10:00:00",
        "updated_at": "2026-08-18T10:00:00",
        "model": "gpt-4o",
        "title": "坏文件",
        "turn_count": 1,
    }


def test_load_v1_malformed_history_returns_none(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    ws = tmp_path.as_posix()

    cases: list[tuple[str, object]] = [
        ("hist_str", "oops"),
        ("hist_null", [None]),
    ]
    for sid, history in cases:
        payload = {"meta": _v1_meta(sid, ws), "history": history}
        (sessions_dir / f"{sid}.json").write_text(json.dumps(payload), encoding="utf-8")
        assert load_session(sid, sessions_dir=sessions_dir) is None


def test_save_session_rejects_unpaired_tool_calls(tmp_path: Path) -> None:
    sid = "unpaired_save"
    session = SessionData(
        meta=SessionMeta(
            session_id=sid,
            workspace_root=tmp_path.as_posix(),
            created_at="2026-08-18T10:00:00",
            updated_at="2026-08-18T10:00:00",
            model="gpt-4o",
            title="未配对",
            turn_count=1,
        ),
        messages=[
            Message(
                role="assistant",
                parts=[ToolCallPart(call_id="c1", name="read_file", arguments="{}")],
                created_at="2026-08-18T10:00:00",
            )
        ],
    )
    with pytest.raises(UnpairedToolError):
        save_session(session, sessions_dir=tmp_path)
    assert not (tmp_path / f"{sid}.json").exists()
    assert not (tmp_path / f"{sid}.tmp").exists()
    assert load_session(sid, sessions_dir=tmp_path) is None


def test_list_sessions_skips_unloadable_files(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    ws = tmp_path.resolve().as_posix()

    good = SessionData(
        meta=SessionMeta(
            session_id="good",
            workspace_root=ws,
            created_at="2026-08-18T12:00:00",
            updated_at="2026-08-18T12:00:00",
            model="gpt-4o",
            title="可恢复",
            turn_count=1,
        )
    )
    save_session(good, sessions_dir=sessions_dir)

    unpaired = SessionData(
        schema_version=2,
        meta=SessionMeta.model_validate(_v1_meta("unpaired_list", ws)),
        messages=[
            Message(
                role="assistant",
                parts=[ToolCallPart(call_id="c1", name="read_file", arguments="{}")],
                created_at="2026-08-18T10:00:00",
            )
        ],
    )
    (sessions_dir / "unpaired_list.json").write_text(unpaired.model_dump_json(), encoding="utf-8")

    bad_v1 = {"meta": _v1_meta("hist_str_list", ws), "history": "oops"}
    (sessions_dir / "hist_str_list.json").write_text(json.dumps(bad_v1), encoding="utf-8")

    listed = list_sessions(sessions_dir=sessions_dir)
    ids = {m.session_id for m in listed}
    assert ids == {"good"}
    assert load_session("unpaired_list", sessions_dir=sessions_dir) is None
    assert load_session("hist_str_list", sessions_dir=sessions_dir) is None
