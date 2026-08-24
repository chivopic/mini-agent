"""Agent loop, history management, session persistence, usage tracking, and tool dispatching."""

import json
import threading
from datetime import datetime
from typing import Any, Protocol

from pydantic import ValidationError

from mini_agent.context import compact_history
from mini_agent.cost import UsageStats, calculate_cost_cny
from mini_agent.llm import LLMClient
from mini_agent.messages import (
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    history_v1_to_messages,
    messages_to_v1_history,
)
from mini_agent.models import AgentConfig, RunShellInput, ToolResult
from mini_agent.prompt import get_system_prompt
from mini_agent.session import (
    SessionData,
    SessionMeta,
    generate_session_id,
    save_session,
)
from mini_agent.tools import ToolRegistry, default_registry
from mini_agent.tools.protocol import ToolContext
from mini_agent.tools.shell import check_command_safety


class AgentEventListener(Protocol):
    """Event listener protocol for monitoring agent steps."""

    def on_turn_start(self, user_input: str) -> None:
        """Called when a user turn begins."""
        ...

    def on_token(self, token: str) -> None:
        """Called for each streamed token from the model."""
        ...

    def on_model_start(self) -> None:
        """Called before sending request to LLM."""
        ...

    def on_tool_start(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Called before executing a tool."""
        ...

    def on_tool_confirm(self, command: str) -> bool:
        """Prompt user for confirmation when running non-allowlisted shell commands."""
        ...

    def on_tool_finished(self, tool_name: str, result: ToolResult) -> None:
        """Called after tool execution finishes."""
        ...

    def on_usage(self, usage: UsageStats, cost_cny: float, model: str) -> None:
        """Called when token usage and cost for a turn are calculated."""
        ...

    def on_turn_finished(self, response: str) -> None:
        """Called when agent turn is fully completed."""
        ...


class Agent:
    """Core Agent coordinating LLM interactions, streaming, tool execution, and usage tracking."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        listener: AgentEventListener | None = None,
        session: SessionData | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.listener = listener
        self.registry = registry or default_registry()
        self._cancel = threading.Event()
        self.tools = self.registry.json_schemas()

        if session is not None:
            self.session = session
            self.messages: list[Message] = session.messages
        else:
            now_iso = datetime.now().isoformat()
            new_meta = SessionMeta(
                session_id=generate_session_id(),
                workspace_root=self.config.workspace_root.as_posix(),
                created_at=now_iso,
                updated_at=now_iso,
                model=self.config.model,
                title="新对话",
                turn_count=0,
            )
            self.messages = [
                Message(
                    role="system",
                    parts=[
                        TextPart(text=get_system_prompt(self.config.workspace_root, self.registry))
                    ],
                    created_at=now_iso,
                )
            ]
            self.session = SessionData(meta=new_meta, messages=self.messages)

        self.session_usage = UsageStats(
            prompt_tokens=self.session.meta.total_prompt_tokens,
            completion_tokens=self.session.meta.total_completion_tokens,
            total_tokens=self.session.meta.total_prompt_tokens
            + self.session.meta.total_completion_tokens,
        )

    @property
    def history(self) -> list[dict[str, Any]]:
        return messages_to_v1_history(self.messages)

    @history.setter
    def history(self, items: list[dict[str, Any]]) -> None:
        created_at = self.session.meta.created_at
        self.messages = history_v1_to_messages(items, created_at=created_at)
        self.session.messages = self.messages

    def _on_token(self, token: str) -> None:
        """Forward streamed token to listener if present."""
        if self.listener and hasattr(self.listener, "on_token"):
            self.listener.on_token(token)

    def _execute_tool(self, name: str, raw_arguments: str) -> ToolResult:
        """Parse arguments and dispatch execution to the corresponding tool."""
        tool = self.registry.get(name)
        if tool is None:
            return ToolResult(ok=False, content="", error=f"未知的工具名称: '{name}'")

        if name == "run_shell":
            inp = None
            try:
                args = json.loads(raw_arguments) if raw_arguments.strip() else {}
                inp = RunShellInput(**args) if isinstance(args, dict) else None
            except (json.JSONDecodeError, ValidationError):
                inp = None
            if inp is not None:
                is_blocked, req_conf, reason = check_command_safety(inp.command)
                if is_blocked:
                    return ToolResult(
                        ok=False,
                        content="",
                        error=reason,
                        metadata={"blocked": True, "command": inp.command},
                    )
                if req_conf:
                    allowed = bool(
                        self.listener
                        and hasattr(self.listener, "on_tool_confirm")
                        and self.listener.on_tool_confirm(inp.command)
                    )
                    if not allowed:
                        return ToolResult(
                            ok=False,
                            content="",
                            error=f"用户拒绝执行命令: '{inp.command}'",
                            metadata={"user_cancelled": True, "command": inp.command},
                        )

        ctx = ToolContext(self.config.workspace_root, self.config, self._cancel)
        return self.registry.dispatch(name, raw_arguments, ctx)

    def _persist_session(self, user_input: str, turn_usage: UsageStats) -> None:
        """Update metadata, usage counters, and auto-save session."""
        self.session.meta.turn_count += 1
        if self.session.meta.title == "新对话":
            clean_title = user_input.replace("\n", " ").strip()
            self.session.meta.title = clean_title[:40] + ("..." if len(clean_title) > 40 else "")

        cost = calculate_cost_cny(
            turn_usage.prompt_tokens, turn_usage.completion_tokens, self.config.model
        )
        self.session.meta.total_prompt_tokens += turn_usage.prompt_tokens
        self.session.meta.total_completion_tokens += turn_usage.completion_tokens
        self.session.meta.total_cost_cny += cost
        self.session_usage = self.session_usage.add(turn_usage)
        self.session.messages = self.messages

        save_session(self.session)

        if self.listener and hasattr(self.listener, "on_usage"):
            self.listener.on_usage(turn_usage, cost, self.config.model)

    def step(self, user_input: str) -> str:
        """Run a single user turn in the agent loop."""
        cleaned_input = user_input.strip()
        if not cleaned_input:
            return ""

        if self.listener and hasattr(self.listener, "on_turn_start"):
            self.listener.on_turn_start(cleaned_input)

        self.messages.append(Message(role="user", parts=[TextPart(text=cleaned_input)]))
        turn_usage = UsageStats()

        for _ in range(self.config.max_tool_rounds):
            if self.listener and hasattr(self.listener, "on_model_start"):
                self.listener.on_model_start()

            compacted_history = compact_history(self.history)
            response = self.llm_client.create_response(
                compacted_history,
                self.tools,
                model=self.config.model,
                on_token=self._on_token,
            )

            turn_usage = turn_usage.add(response.usage)

            if not response.function_calls:
                if response.text:
                    self.messages.append(
                        Message(role="assistant", parts=[TextPart(text=response.text)])
                    )
                final_answer = response.text or "(模型未返回文本内容)"
                if self.listener and hasattr(self.listener, "on_turn_finished"):
                    self.listener.on_turn_finished(final_answer)
                self._persist_session(cleaned_input, turn_usage)
                return final_answer

            call_parts: list[TextPart | ToolCallPart] = []
            if response.text:
                call_parts.append(TextPart(text=response.text))
            for call in response.function_calls:
                call_parts.append(
                    ToolCallPart(
                        call_id=call.call_id,
                        name=call.name,
                        arguments=call.arguments,
                    )
                )
            # Hold the unpaired call message in memory until results are appended.
            self.messages.append(Message(role="assistant", parts=call_parts))

            result_parts: list[ToolResultPart] = []
            for call in response.function_calls:
                try:
                    args_dict = json.loads(call.arguments) if call.arguments.strip() else {}
                except Exception:
                    args_dict = {"raw": call.arguments}

                if self.listener and hasattr(self.listener, "on_tool_start"):
                    self.listener.on_tool_start(call.name, args_dict)

                result = self._execute_tool(call.name, call.arguments)

                if self.listener and hasattr(self.listener, "on_tool_finished"):
                    self.listener.on_tool_finished(call.name, result)

                result_parts.append(
                    ToolResultPart(
                        call_id=call.call_id,
                        name=call.name,
                        ok=result.ok,
                        content=result.content,
                        error=result.error,
                        metadata=result.metadata,
                    )
                )

            self.messages.append(Message(role="assistant", parts=result_parts))
            self.session.messages = self.messages
            save_session(self.session)

        timeout_msg = "工具调用轮数已达到上限，请缩小任务范围后重试。"
        if self.listener and hasattr(self.listener, "on_turn_finished"):
            self.listener.on_turn_finished(timeout_msg)
        self._persist_session(cleaned_input, turn_usage)
        return timeout_msg
