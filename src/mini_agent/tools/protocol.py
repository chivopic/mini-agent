"""Tool protocol, execution context, and cancellation."""

import threading
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from mini_agent.models import AgentConfig, PermissionClass, ToolResult


class ToolKind(StrEnum):
    READONLY = "readonly"
    MUTATING = "mutating"


class ToolCancelled(Exception):  # noqa: N818
    """Raised by ToolContext.check_cancel when the cancel event is set."""


class ToolContext:
    """Per-call execution context. Authorization happens in Agent before dispatch."""

    def __init__(
        self,
        workspace_root: Path,
        config: AgentConfig,
        cancel: threading.Event,
    ) -> None:
        self.workspace_root = workspace_root
        self.config = config
        self.cancel = cancel

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise ToolCancelled


class Tool(Protocol):
    name: str
    description: str
    permission: PermissionClass
    kind: ToolKind
    input_model: type[BaseModel]

    def execute(self, inp: BaseModel, ctx: ToolContext) -> ToolResult: ...

    def format_call(self, inp: BaseModel) -> str: ...

    def approval_pattern(self, inp: BaseModel) -> str: ...
