"""Permission evaluation: allow / deny / ask, with session-only always memory."""

from __future__ import annotations

import shlex
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

from mini_agent.models import PermissionClass
from mini_agent.tools.shell import check_command_safety

_NEVER_ALWAYS_INTERPRETERS = frozenset({"python", "python3", "bash", "sh", "zsh"})


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class Reply(StrEnum):
    ONCE = "once"
    ALWAYS = "always"
    REJECT = "reject"


class PermissionRequest(BaseModel):
    cls: PermissionClass
    tool: str
    resource: str
    pattern: str
    reason: str = ""


class PermissionMemoryEntry(BaseModel):
    cls: PermissionClass
    pattern: str = Field(min_length=1)
    effect: Decision

    @field_validator("effect")
    @classmethod
    def effect_must_be_allow_or_deny(cls, value: Decision) -> Decision:
        if value not in (Decision.ALLOW, Decision.DENY):
            raise ValueError("permission memory effect must be allow or deny")
        return value


class PermissionService(Protocol):
    def check(self, req: PermissionRequest) -> Decision: ...

    def remember(self, req: PermissionRequest, reply: Reply) -> None: ...

    def snapshot(self) -> list[dict[str, str]]: ...


def is_sensitive_write_path(rel_posix: str) -> bool:
    """Return True for secrets / vcs / key material. Comparison is case-sensitive."""
    if not rel_posix:
        return False
    parts = rel_posix.split("/")
    name = parts[-1]
    if name == ".env" or name.startswith(".env."):
        return True
    if ".git" in parts:
        return True
    if name.endswith(".pem") or name.endswith(".key"):
        return True
    if name == ".mini-agent.toml":
        return True
    return False


def pattern_for_shell(command: str) -> str | None:
    """Normalize a shell command to an always-memory key. None means never always."""
    stripped = command.strip()
    if not stripped:
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None
    if not tokens:
        return None

    if len(tokens) >= 2 and tokens[1] == "-c":
        return None
    if tokens[0] in _NEVER_ALWAYS_INTERPRETERS and "-c" in tokens[1:3]:
        return None

    if len(tokens) >= 2 and tokens[0] == "git" and tokens[1] == "add":
        return " ".join(tokens)

    if tokens[0] == "uv" and len(tokens) >= 3 and tokens[1] == "run":
        return " ".join(tokens[:3]) + "*"

    if tokens[0] == "git" and len(tokens) >= 2:
        return " ".join(tokens[:2]) + "*"

    if tokens[0] == "npx" and len(tokens) >= 2:
        return " ".join(tokens[:2]) + "*"

    return " ".join(tokens)


class DefaultPermissionService:
    """Process-local permission policy. always is never written to TOML."""

    def __init__(
        self,
        *,
        class_defaults: dict[PermissionClass, Decision] | None = None,
        auto_allow_ask: bool = False,
        memory: list[dict[str, Any]] | None = None,
    ) -> None:
        self._defaults = class_defaults or {
            PermissionClass.READ: Decision.ALLOW,
            PermissionClass.EDIT: Decision.ALLOW,
            PermissionClass.SHELL: Decision.ASK,
            PermissionClass.GIT: Decision.ASK,
        }
        self._auto_allow_ask = auto_allow_ask
        self._memory: dict[tuple[PermissionClass, str], Decision] = {}
        if memory:
            self.restore(memory)

    def restore(self, snapshot: list[Any]) -> None:
        if not isinstance(snapshot, list):
            raise TypeError("permission_memory must be a list")
        for i, raw in enumerate(snapshot):
            if not isinstance(raw, dict):
                raise ValueError(f"permission_memory[{i}] must be an object")
            entry = PermissionMemoryEntry.model_validate(raw)
            self._memory[(entry.cls, entry.pattern)] = entry.effect

    def check(self, req: PermissionRequest) -> Decision:
        shell_needs_confirm = True
        if req.cls == PermissionClass.SHELL:
            blocked, shell_needs_confirm, _reason = check_command_safety(req.resource)
            if blocked:
                return Decision.DENY

        if req.pattern:
            remembered = self._memory.get((req.cls, req.pattern))
            if remembered is not None:
                return remembered

        # Sensitive writes skip class-default ALLOW so they stay ASK unless remembered.
        if req.cls == PermissionClass.EDIT and is_sensitive_write_path(req.resource):
            return self._apply_ask_policy(Decision.ASK)

        default = self._defaults.get(req.cls, Decision.ASK)
        if default == Decision.ALLOW:
            return Decision.ALLOW
        if default == Decision.DENY:
            return Decision.DENY

        if req.cls == PermissionClass.SHELL and not shell_needs_confirm:
            return Decision.ALLOW

        return self._apply_ask_policy(Decision.ASK)

    def remember(self, req: PermissionRequest, reply: Reply) -> None:
        if reply != Reply.ALWAYS or not req.pattern:
            return
        self._memory[(req.cls, req.pattern)] = Decision.ALLOW

    def snapshot(self) -> list[dict[str, str]]:
        return [
            {"cls": cls.value, "pattern": pattern, "effect": effect.value}
            for (cls, pattern), effect in self._memory.items()
        ]

    def _apply_ask_policy(self, decision: Decision) -> Decision:
        if decision == Decision.ASK and self._auto_allow_ask:
            return Decision.ALLOW
        return decision
