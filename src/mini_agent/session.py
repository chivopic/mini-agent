"""Session persistence and resume manager for mini-agent."""

import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class SessionMeta(BaseModel):
    """Metadata summary of a saved session."""

    session_id: str
    workspace_root: str
    created_at: str
    updated_at: str
    model: str
    title: str = "新对话"
    turn_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_cny: float = 0.0


class SessionData(BaseModel):
    """Complete session data including conversation history."""

    meta: SessionMeta
    history: list[dict[str, Any]] = Field(default_factory=list)


def get_default_sessions_dir() -> Path:
    """Return default sessions directory under ~/.mini-agent/sessions."""
    custom_dir = os.environ.get("MINI_AGENT_SESSIONS_DIR")
    if custom_dir:
        path = Path(custom_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    home_dir = Path.home() / ".mini-agent"
    home_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(home_dir, 0o700)
    path = home_dir / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)
    return path


def generate_session_id() -> str:
    """Generate a unique timestamp-based session ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid.uuid4().hex[:6]
    return f"{timestamp}_{random_suffix}"


def is_valid_session_id(session_id: str) -> bool:
    """Return whether a session ID is safe to use as a single filename stem."""
    return bool(SESSION_ID_PATTERN.fullmatch(session_id))


def _resolve_session_file(target_dir: Path, session_id: str) -> Path | None:
    """Resolve a session path without allowing traversal or symlink escape."""
    if not is_valid_session_id(session_id):
        return None

    resolved_dir = target_dir.resolve()
    candidate = resolved_dir / f"{session_id}.json"
    try:
        resolved_candidate = candidate.resolve()
    except OSError:
        return None
    if resolved_candidate.parent != resolved_dir:
        return None
    return candidate


def save_session(session: SessionData, sessions_dir: Path | None = None) -> Path:
    """Save session data atomically."""
    target_dir = sessions_dir or get_default_sessions_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = target_dir.resolve()

    file_path = _resolve_session_file(target_dir, session.meta.session_id)
    if file_path is None:
        raise ValueError(f"非法会话 ID: '{session.meta.session_id}'")

    session.meta.updated_at = datetime.now().isoformat()
    json_str = session.model_dump_json(indent=2)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            prefix=f".{session.meta.session_id}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            f.write(json_str)
            f.flush()
            os.fsync(f.fileno())
            temp_path = Path(f.name)
        temp_path.replace(file_path)
        if os.name == "posix":
            os.chmod(file_path, 0o600)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return file_path


def load_session(session_id: str, sessions_dir: Path | None = None) -> SessionData | None:
    """Load a session by its ID."""
    target_dir = sessions_dir or get_default_sessions_dir()
    file_path = _resolve_session_file(target_dir, session_id)
    if file_path is None or not file_path.is_file():
        return None

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        session = SessionData.model_validate(data)
        if session.meta.session_id != session_id:
            return None
        return session
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def list_sessions(
    workspace_root: Path | None = None,
    sessions_dir: Path | None = None,
) -> list[SessionMeta]:
    """List all saved session metadata sorted by updated_at descending."""
    target_dir = sessions_dir or get_default_sessions_dir()
    if not target_dir.is_dir():
        return []

    sessions: list[SessionMeta] = []
    resolved_ws = workspace_root.resolve().as_posix() if workspace_root else None

    for file_path in target_dir.glob("*.json"):
        loaded = load_session(file_path.stem, sessions_dir=target_dir)
        if loaded is None:
            continue
        meta = loaded.meta
        if resolved_ws is None or meta.workspace_root == resolved_ws:
            sessions.append(meta)

    sessions.sort(key=lambda s: s.updated_at, reverse=True)
    return sessions


def get_latest_session(
    workspace_root: Path | None = None,
    sessions_dir: Path | None = None,
) -> SessionData | None:
    """Get the most recent session for the given workspace."""
    all_sessions = list_sessions(workspace_root=workspace_root, sessions_dir=sessions_dir)
    if not all_sessions:
        return None
    latest_meta = all_sessions[0]
    return load_session(latest_meta.session_id, sessions_dir=sessions_dir)


def delete_session(session_id: str, sessions_dir: Path | None = None) -> bool:
    """Delete a session file by its ID."""
    target_dir = sessions_dir or get_default_sessions_dir()
    file_path = _resolve_session_file(target_dir, session_id)
    if file_path is None:
        return False
    if file_path.is_file():
        try:
            file_path.unlink()
            return True
        except OSError:
            return False
    return False
