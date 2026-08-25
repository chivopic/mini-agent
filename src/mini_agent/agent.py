"""Agent loop, history management, session persistence, usage tracking, and tool dispatching."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from mini_agent.context import compact_history
from mini_agent.cost import UsageStats, calculate_cost_cny
from mini_agent.llm import LLMClient, get_system_prompt, get_tool_definitions
from mini_agent.messages import (
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    history_v1_to_messages,
    messages_to_v1_history,
)
from mini_agent.models import (
    AgentConfig,
    EditFileInput,
    GetRepoMapInput,
    ListFilesInput,
    ReadFileInput,
    RunShellInput,
    SearchCodeInput,
    ToolResult,
    WriteFileInput,
)
from mini_agent.repomap import generate_repo_map
from mini_agent.session import (
    SessionData,
    SessionMeta,
    generate_session_id,
    save_session,
)
from mini_agent.tools.filesystem import (
    edit_file,
    list_files,
    read_file,
    resolve_relative_path,
    search_code,
    write_file,
)
from mini_agent.tools.shell import check_command_safety, run_shell


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
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.listener = listener
        self.tools = get_tool_definitions()

        if session is not None:
            self.resume_session(session)
        else:
            self._create_new_session()

    def _create_new_session(self) -> None:
        """Create fresh in-memory session state for the current workspace."""
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
                parts=[TextPart(text=get_system_prompt(self.config.workspace_root))],
                created_at=now_iso,
            )
        ]
        self.session = SessionData(meta=new_meta, messages=self.messages)
        self.session_usage = UsageStats()

    def resume_session(self, session: SessionData) -> None:
        """Attach a same-workspace session and rebuild all derived usage state."""
        session_workspace = Path(session.meta.workspace_root).resolve()
        current_workspace = self.config.workspace_root.resolve()
        if session_workspace != current_workspace:
            raise ValueError(
                f"会话工作区与当前工作区不一致: '{session_workspace}' != '{current_workspace}'"
            )
        self.session = session
        self.messages = session.messages
        self.session_usage = UsageStats(
            prompt_tokens=session.meta.total_prompt_tokens,
            completion_tokens=session.meta.total_completion_tokens,
            total_tokens=session.meta.total_prompt_tokens + session.meta.total_completion_tokens,
        )

    def reset_session(self) -> str:
        """Reset context and return the actual ID of the newly active session."""
        self._create_new_session()
        return self.session.meta.session_id

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
        try:
            args = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolResult(
                ok=False,
                content="",
                error=f"工具参数不是合法的 JSON 字符串: {exc}",
            )

        if not isinstance(args, dict):
            return ToolResult(
                ok=False,
                content="",
                error=f"工具参数必须为 JSON 对象 (dict)，收到: {type(args).__name__}",
            )

        if name == "get_repo_map":
            try:
                inp = GetRepoMapInput(**args)
                target_dir, path_error = resolve_relative_path(self.config.workspace_root, inp.path)
                if path_error or target_dir is None:
                    return ToolResult(
                        ok=False,
                        content="",
                        error=path_error,
                        metadata={"path": inp.path},
                    )
                if not target_dir.exists():
                    return ToolResult(
                        ok=False,
                        content="",
                        error=f"指定的目录不存在: '{inp.path}'",
                        metadata={"path": inp.path},
                    )
                if not target_dir.is_dir():
                    return ToolResult(
                        ok=False,
                        content="",
                        error=f"指定的路径不是目录: '{inp.path}'",
                        metadata={"path": inp.path},
                    )
                repo_map = generate_repo_map(
                    target_dir,
                    boundary_root=self.config.workspace_root,
                )
                return ToolResult(
                    ok=True,
                    content=repo_map if repo_map else "未在当前目录发现有效的代码文件与符号。",
                    metadata={"path": inp.path},
                )
            except ValidationError as exc:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"get_repo_map 参数校验失败: {exc}",
                )

        if name == "search_code":
            try:
                inp = SearchCodeInput(**args)
                return search_code(
                    inp,
                    workspace_root=self.config.workspace_root,
                    max_output_chars=self.config.max_output_chars,
                )
            except ValidationError as exc:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"search_code 参数校验失败: {exc}",
                )

        if name == "read_file":
            try:
                inp = ReadFileInput(**args)
                return read_file(
                    inp,
                    workspace_root=self.config.workspace_root,
                    max_output_chars=self.config.max_output_chars,
                )
            except ValidationError as exc:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"read_file 参数校验失败: {exc}",
                )

        if name == "list_files":
            try:
                inp = ListFilesInput(**args)
                return list_files(
                    inp,
                    workspace_root=self.config.workspace_root,
                    max_output_chars=self.config.max_output_chars,
                )
            except ValidationError as exc:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"list_files 参数校验失败: {exc}",
                )

        if name == "write_file":
            try:
                inp = WriteFileInput(**args)
                return write_file(inp, workspace_root=self.config.workspace_root)
            except ValidationError as exc:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"write_file 参数校验失败: {exc}",
                )

        if name == "edit_file":
            try:
                inp = EditFileInput(**args)
                return edit_file(inp, workspace_root=self.config.workspace_root)
            except ValidationError as exc:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"edit_file 参数校验失败: {exc}",
                )

        if name == "run_shell":
            try:
                inp = RunShellInput(**args)
                is_blocked, req_conf, reason = check_command_safety(inp.command)
                if is_blocked:
                    return ToolResult(
                        ok=False,
                        content="",
                        error=reason,
                        metadata={"blocked": True, "command": inp.command},
                    )

                if req_conf:
                    confirmed = False
                    if self.listener and hasattr(self.listener, "on_tool_confirm"):
                        confirmed = self.listener.on_tool_confirm(inp.command)

                    if not confirmed:
                        return ToolResult(
                            ok=False,
                            content="",
                            error=f"用户拒绝执行命令: '{inp.command}'",
                            metadata={"user_cancelled": True, "command": inp.command},
                        )

                return run_shell(
                    inp,
                    workspace_root=self.config.workspace_root,
                    confirmed=True,
                    timeout_seconds=self.config.shell_timeout_seconds,
                    max_output_chars=self.config.max_output_chars,
                )
            except ValidationError as exc:
                return ToolResult(
                    ok=False,
                    content="",
                    error=f"run_shell 参数校验失败: {exc}",
                )

        return ToolResult(
            ok=False,
            content="",
            error=f"未知的工具名称: '{name}'",
        )

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

    def step(
        self,
        user_input: str,
        *,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> str:
        """Run a single user turn in the agent loop."""
        cleaned_input = user_input.strip()
        if not cleaned_input:
            return ""

        active_tools = self.tools if tool_definitions is None else tool_definitions
        allowed_tool_names = {
            str(tool.get("name"))
            for tool in active_tools
            if isinstance(tool, dict) and tool.get("name")
        }

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
                active_tools,
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

                if call.name not in allowed_tool_names:
                    result = ToolResult(
                        ok=False,
                        content="",
                        error=f"模型请求了本轮未授权的工具: '{call.name}'",
                        metadata={"tool_name": call.name, "not_offered": True},
                    )
                else:
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
