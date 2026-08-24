"""Typed agent-loop events. Permission prompts stay on on_permission_ask."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from mini_agent.cost import UsageStats
from mini_agent.models import ToolResult
from mini_agent.permission import PermissionRequest, Reply


@dataclass(frozen=True, kw_only=True)
class TurnStarted:
    user_input: str
    type: Literal["turn_started"] = "turn_started"


@dataclass(frozen=True, kw_only=True)
class ModelStarted:
    type: Literal["model_started"] = "model_started"


@dataclass(frozen=True, kw_only=True)
class TokenDelta:
    token: str
    type: Literal["token_delta"] = "token_delta"


@dataclass(frozen=True, kw_only=True)
class ToolStarted:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""
    type: Literal["tool_started"] = "tool_started"


@dataclass(frozen=True, kw_only=True)
class ToolFinished:
    name: str
    result: ToolResult
    call_id: str = ""
    type: Literal["tool_finished"] = "tool_finished"


@dataclass(frozen=True, kw_only=True)
class UsageReported:
    usage: UsageStats
    cost_cny: float
    model: str
    type: Literal["usage_reported"] = "usage_reported"


@dataclass(frozen=True, kw_only=True)
class TurnFinished:
    response: str
    type: Literal["turn_finished"] = "turn_finished"


@dataclass(frozen=True, kw_only=True)
class TurnCancelled:
    response: str = "已取消当前回合"
    type: Literal["turn_cancelled"] = "turn_cancelled"


@dataclass(frozen=True, kw_only=True)
class TurnFailed:
    error: str
    type: Literal["turn_failed"] = "turn_failed"


@dataclass(frozen=True, kw_only=True)
class CompactionNotice:
    summary: str = ""
    dropped_count: int = 0
    type: Literal["compaction_notice"] = "compaction_notice"


AgentEvent = (
    TurnStarted
    | ModelStarted
    | TokenDelta
    | ToolStarted
    | ToolFinished
    | UsageReported
    | TurnFinished
    | TurnCancelled
    | TurnFailed
    | CompactionNotice
)


class AgentEventListener(Protocol):
    """UI callback surface. PermissionAsk is not an AgentEvent (needs a return value)."""

    def on_event(self, event: AgentEvent) -> None: ...

    def on_permission_ask(self, req: PermissionRequest) -> Reply: ...


class NoopListener:
    """Default listener: ignore events, reject permission prompts."""

    def on_event(self, event: AgentEvent) -> None:
        return

    def on_permission_ask(self, req: PermissionRequest) -> Reply:
        return Reply.REJECT
