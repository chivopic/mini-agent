"""Agent loop: cancellable turns, parallel readonly tools, typed events, pairing."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mini_agent.compaction import CompactionConfig, compact_for_model
from mini_agent.cost import UsageStats, calculate_cost_cny
from mini_agent.events import (
    AgentEventListener,
    ModelStarted,
    NoopListener,
    TokenDelta,
    ToolFinished,
    ToolStarted,
    TurnCancelled,
    TurnFailed,
    TurnFinished,
    TurnStarted,
    UsageReported,
)
from mini_agent.llm import (
    FunctionCall,
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
)
from mini_agent.messages import (
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from mini_agent.models import AgentConfig, PermissionClass, ToolResult
from mini_agent.permission import (
    Decision,
    DefaultPermissionService,
    PermissionRequest,
    PermissionService,
    Reply,
    pattern_for_shell,
)
from mini_agent.prompt import get_system_prompt
from mini_agent.session import (
    SessionData,
    SessionMeta,
    generate_session_id,
    save_session,
)
from mini_agent.tools import ToolRegistry, default_registry
from mini_agent.tools.filesystem import resolve_relative_path
from mini_agent.tools.protocol import Tool, ToolCancelled, ToolContext, ToolKind
from mini_agent.tools.shell import check_command_safety

_DOOM_EXEMPT_OK = frozenset({"read_file", "list_files", "search_code", "get_repo_map"})
_CANCELLED_MSG = "已取消当前回合"
_ROUNDS_EXCEEDED_MSG = "工具调用轮数已达到上限，请缩小任务范围后重试。"
_DOOM_MSG = "检测到重复工具调用"


class _DoubleSigintError(Exception):
    """Second Ctrl-C within 2s after pairing persist; REPL exits the process."""


def _debug_log(message: str) -> None:
    if os.environ.get("MINI_AGENT_DEBUG") != "1":
        return
    try:
        path = Path.home() / ".mini-agent" / "debug.log"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now().isoformat()} {message}\n")
    except OSError:
        pass


def _cancelled_result() -> ToolResult:
    return ToolResult(
        ok=False,
        content="",
        error="用户取消",
        metadata={"user_cancelled": True},
    )


def _doom_result() -> ToolResult:
    return ToolResult(ok=False, content="", error="检测到重复工具调用")


class Agent:
    """Core Agent coordinating LLM interactions, streaming, tool execution, and usage tracking."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        listener: AgentEventListener | None = None,
        session: SessionData | None = None,
        registry: ToolRegistry | None = None,
        permission: PermissionService | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        compaction_config: CompactionConfig | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.listener: AgentEventListener = listener or NoopListener()
        self.registry = registry or default_registry()
        self.compaction_config = compaction_config or CompactionConfig()
        self._cancel = threading.Event()
        self._monotonic = monotonic
        self._last_sigint_at: float | None = None
        self._double_sigint = False
        self._retry_backoff: tuple[float, ...] = (1.0, 2.0, 4.0)
        self.tools = self.registry.json_schemas()
        self.permission = permission or DefaultPermissionService()
        self._doom_key: tuple[str, str] | None = None
        self._doom_streak = 0
        self._doom_results: list[ToolResult] = []
        self._doom_fired = False
        self._stream_text = ""

        if session is not None:
            self.session = session
            self.messages: list[Message] = session.messages
            restore = getattr(self.permission, "restore", None)
            if restore is not None:
                restore(session.permission_memory or [])
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

    def reset_session(self) -> str:
        """Start a new SessionData, persist it, and return the stored session_id."""
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
                parts=[TextPart(text=get_system_prompt(self.config.workspace_root, self.registry))],
                created_at=now_iso,
            )
        ]
        self.session = SessionData(meta=new_meta, messages=self.messages)
        restore = getattr(self.permission, "restore", None)
        if restore is not None:
            restore([])
        self.session_usage = UsageStats()
        self._cancel.clear()
        save_session(self.session)
        return self.session.meta.session_id

    def request_cancel(self) -> None:
        now = self._monotonic()
        last = self._last_sigint_at
        self._double_sigint = last is not None and (now - last) < 2.0
        self._last_sigint_at = now
        self._cancel.set()

    def _on_token(self, token: str) -> None:
        self._stream_text += token
        self.listener.on_event(TokenDelta(token=token))

    def _tools_for_turn(self, extra_tools: list[str] | None) -> list[dict[str, Any]]:
        if extra_tools is None:
            return self.tools
        allowed = set(extra_tools)
        return [
            schema for schema in self.tools if (schema.get("function") or {}).get("name") in allowed
        ]

    def _posix_rel(self, user_path: str) -> str:
        resolved, _error = resolve_relative_path(self.config.workspace_root, user_path)
        if resolved is None:
            return user_path.replace("\\", "/")
        root = self.config.workspace_root.resolve()
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            return user_path.replace("\\", "/")
        posix = rel.as_posix()
        return posix if posix else "."

    def _permission_request(self, tool: Tool, inp: Any) -> PermissionRequest:
        cls = tool.permission
        if cls == PermissionClass.SHELL:
            command = getattr(inp, "command", "")
            pattern = pattern_for_shell(command) or ""
            _blocked, _needs_conf, reason = check_command_safety(command)
            return PermissionRequest(
                cls=cls,
                tool=tool.name,
                resource=command,
                pattern=pattern,
                reason=reason or "",
            )
        if cls == PermissionClass.EDIT:
            path = getattr(inp, "path", "")
            resource = self._posix_rel(path)
            return PermissionRequest(
                cls=cls,
                tool=tool.name,
                resource=resource,
                pattern=f"edit:{resource}",
            )
        path = getattr(inp, "path", "")
        return PermissionRequest(
            cls=cls,
            tool=tool.name,
            resource=path,
            pattern=tool.approval_pattern(inp),
        )

    def _denied_result(self, name: str, inp: Any, req: PermissionRequest) -> ToolResult:
        if name == "run_shell":
            command = getattr(inp, "command", req.resource)
            is_blocked, _needs_conf, reason = check_command_safety(command)
            return ToolResult(
                ok=False,
                content="",
                error=reason,
                metadata={"blocked": is_blocked, "command": command},
            )
        return ToolResult(
            ok=False,
            content="",
            error=req.reason or f"操作被拒绝: {name}",
            metadata={"denied": True},
        )

    def _rejected_result(self, name: str, inp: Any, req: PermissionRequest) -> ToolResult:
        if name == "run_shell":
            command = getattr(inp, "command", req.resource)
            return ToolResult(
                ok=False,
                content="",
                error=f"用户拒绝执行命令: '{command}'",
                metadata={"user_cancelled": True, "command": command},
            )
        resource = req.resource or name
        return ToolResult(
            ok=False,
            content="",
            error=f"用户拒绝执行 {name}: '{resource}'",
            metadata={"user_cancelled": True},
        )

    def _sync_permission_memory(self) -> None:
        snapshot = getattr(self.permission, "snapshot", None)
        if snapshot is not None:
            self.session.permission_memory = snapshot()

    def _try_parse(self, tool: Tool, raw_arguments: str) -> Any | None:
        try:
            args = json.loads(raw_arguments) if raw_arguments.strip() else {}
            if isinstance(args, dict):
                return tool.input_model.model_validate(args)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            return None
        return None

    def _args_preview(self, raw_arguments: str) -> dict[str, Any]:
        try:
            args = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except Exception:
            return {"raw": raw_arguments}
        return args if isinstance(args, dict) else {"raw": raw_arguments}

    def _execute_tool(self, name: str, raw_arguments: str) -> ToolResult:
        """Authorize, then parse+dispatch. Invalid JSON skips the permission peek."""
        tool = self.registry.get(name)
        if tool is None:
            return ToolResult(ok=False, content="", error=f"未知的工具名称: '{name}'")

        inp = self._try_parse(tool, raw_arguments)
        if inp is not None:
            req = self._permission_request(tool, inp)
            decision = self.permission.check(req)
            if decision == Decision.DENY:
                return self._denied_result(name, inp, req)
            if decision == Decision.ASK:
                reply = self.listener.on_permission_ask(req)
                if reply == Reply.ALWAYS:
                    self.permission.remember(req, reply)
                    self._sync_permission_memory()
                elif reply != Reply.ONCE:
                    return self._rejected_result(name, inp, req)

        ctx = ToolContext(self.config.workspace_root, self.config, self._cancel)
        try:
            return self.registry.dispatch(name, raw_arguments, ctx)
        except ToolCancelled:
            return _cancelled_result()

    def _save_paired(self) -> None:
        self.session.messages = self.messages
        self._sync_permission_memory()
        save_session(self.session)

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
        self._save_paired()
        self.listener.on_event(
            UsageReported(usage=turn_usage, cost_cny=cost, model=self.config.model)
        )

    def _compact_window(self, tool_schemas: list[dict[str, Any]]) -> list[Message]:
        window, notice = compact_for_model(
            self.messages,
            self.compaction_config,
            tool_schemas,
            self.llm_client,
            self.config.model,
            cancel=self._cancel,
        )
        if notice is not None:
            self.listener.on_event(notice)
            if notice.summary:
                self.session.meta.compacted_at = datetime.now().isoformat()
        return window

    def _reset_doom(self) -> None:
        self._doom_key = None
        self._doom_streak = 0
        self._doom_results = []
        self._doom_fired = False

    def _observe_doom(self, name: str, arguments: str, result: ToolResult) -> bool:
        key = (name, arguments)
        if key == self._doom_key:
            self._doom_streak += 1
            self._doom_results.append(result)
        else:
            self._doom_key = key
            self._doom_streak = 1
            self._doom_results = [result]
        if self._doom_streak < 3:
            return False
        last3 = self._doom_results[-3:]
        if name in _DOOM_EXEMPT_OK and all(item.ok for item in last3):
            return False
        all_failed = all(not item.ok for item in last3)
        dumps = [
            json.dumps(item.model_dump(), sort_keys=True, ensure_ascii=False) for item in last3
        ]
        identical = dumps[0] == dumps[1] == dumps[2]
        if all_failed or identical:
            self._doom_fired = True
            _debug_log(f"doom-loop name={name}")
            return True
        return False

    def _should_run_parallel(self, calls: list[FunctionCall]) -> bool:
        if len(calls) < 2:
            return False
        for call in calls:
            tool = self.registry.get(call.name)
            if tool is None or tool.kind != ToolKind.READONLY:
                return False
            inp = self._try_parse(tool, call.arguments)
            if inp is None:
                return False
            req = self._permission_request(tool, inp)
            if self.permission.check(req) != Decision.ALLOW:
                return False
        return True

    def _emit_tool_started(self, call: FunctionCall) -> None:
        self.listener.on_event(
            ToolStarted(
                name=call.name,
                arguments=self._args_preview(call.arguments),
                call_id=call.call_id,
            )
        )

    def _emit_tool_finished(self, call: FunctionCall, result: ToolResult) -> None:
        self.listener.on_event(ToolFinished(name=call.name, result=result, call_id=call.call_id))

    def _result_parts(
        self, calls: list[FunctionCall], results: list[ToolResult]
    ) -> list[ToolResultPart]:
        return [
            ToolResultPart(
                call_id=call.call_id,
                name=call.name,
                ok=result.ok,
                content=result.content,
                error=result.error,
                metadata=result.metadata,
            )
            for call, result in zip(calls, results, strict=True)
        ]

    def _apply_doom_to_tail(
        self, calls: list[FunctionCall], results: list[ToolResult]
    ) -> list[ToolResult]:
        committed: list[ToolResult] = []
        doom = False
        for call, result in zip(calls, results, strict=True):
            if doom:
                committed.append(_doom_result())
                continue
            committed.append(result)
            if self._observe_doom(call.name, call.arguments, result):
                doom = True
        return committed

    def _run_serial(self, calls: list[FunctionCall], *, tools_disabled: bool) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            self._emit_tool_started(call)
            if self._doom_fired:
                result = _doom_result()
            elif self._cancel.is_set():
                result = _cancelled_result()
            elif tools_disabled:
                result = ToolResult(
                    ok=False,
                    content="",
                    error=f"本轮未启用工具，拒绝执行: '{call.name}'",
                    metadata={"tools_disabled": True, "name": call.name},
                )
            else:
                try:
                    result = self._execute_tool(call.name, call.arguments)
                except KeyboardInterrupt:
                    self.request_cancel()
                    result = _cancelled_result()
            self._emit_tool_finished(call, result)
            results.append(result)
            self._observe_doom(call.name, call.arguments, result)
        return results

    def _dispatch_readonly(self, call: FunctionCall) -> ToolResult:
        if self._cancel.is_set():
            return _cancelled_result()
        ctx = ToolContext(self.config.workspace_root, self.config, self._cancel)
        try:
            return self.registry.dispatch(call.name, call.arguments, ctx)
        except ToolCancelled:
            return _cancelled_result()
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

    def _take_finished_future(
        self,
        fut: Future[ToolResult],
        collected: list[ToolResult | None],
        idx: int,
    ) -> None:
        if collected[idx] is not None:
            return
        if fut.done() and not fut.cancelled():
            try:
                collected[idx] = fut.result()
            except Exception as exc:
                collected[idx] = ToolResult(ok=False, content="", error=str(exc))

    def _run_parallel(self, calls: list[FunctionCall]) -> list[ToolResult]:
        for call in calls:
            self._emit_tool_started(call)

        collected: list[ToolResult | None] = [None] * len(calls)
        workers = min(len(calls), self.config.max_parallel_readonly)
        executor = ThreadPoolExecutor(max_workers=workers)
        future_to_idx: dict[Future[ToolResult], int] = {}
        try:
            future_to_idx = {
                executor.submit(self._dispatch_readonly, call): i for i, call in enumerate(calls)
            }
            pending = set(future_to_idx)
            try:
                while pending:
                    if self._cancel.is_set():
                        break
                    done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                    for fut in done:
                        self._take_finished_future(fut, collected, future_to_idx[fut])
            except KeyboardInterrupt:
                self.request_cancel()
        finally:
            for fut, idx in future_to_idx.items():
                self._take_finished_future(fut, collected, idx)
            executor.shutdown(wait=True, cancel_futures=True)
            for fut, idx in future_to_idx.items():
                self._take_finished_future(fut, collected, idx)

        raw: list[ToolResult] = [
            item if item is not None else _cancelled_result() for item in collected
        ]
        ordered = self._apply_doom_to_tail(calls, raw)
        for call, result in zip(calls, ordered, strict=True):
            self._emit_tool_finished(call, result)
        return ordered

    def _run_calls(self, calls: list[FunctionCall], *, tools_disabled: bool) -> list[ToolResult]:
        if tools_disabled or not self._should_run_parallel(calls):
            return self._run_serial(calls, tools_disabled=tools_disabled)
        return self._run_parallel(calls)

    def _create_response(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> LLMResponse | None:
        self._stream_text = ""
        attempts = 1 + len(self._retry_backoff)
        last_exc: LLMError | None = None
        for attempt in range(attempts):
            if self._cancel.is_set():
                return None
            try:
                return self.llm_client.create_response(
                    messages,
                    tools,
                    model=self.config.model,
                    on_token=self._on_token,
                    cancel=self._cancel,
                )
            except (LLMConnectionError, LLMRateLimitError) as exc:
                last_exc = exc
                if attempt >= attempts - 1:
                    raise
                delay = self._retry_backoff[attempt]
                if self._cancel.wait(timeout=delay):
                    return None
            except KeyboardInterrupt:
                self.request_cancel()
                return LLMResponse(text=self._stream_text or None)
        if last_exc is not None:
            raise last_exc
        return None

    def _drop_unpaired_tool_calls(self) -> None:
        if not self.messages:
            return
        last = self.messages[-1]
        calls = [part for part in last.parts if isinstance(part, ToolCallPart)]
        if not calls:
            return
        texts = [part for part in last.parts if isinstance(part, TextPart)]
        if texts:
            self.messages[-1] = Message(
                role="assistant",
                parts=list(texts),
                id=last.id,
                created_at=last.created_at,
            )
        else:
            self.messages.pop()

    def _end_cancelled(self, user_input: str, turn_usage: UsageStats) -> str:
        self._drop_unpaired_tool_calls()
        self.listener.on_event(TurnCancelled(response=_CANCELLED_MSG))
        self._persist_session(user_input, turn_usage)
        return _CANCELLED_MSG

    def _finish_cancel(self, user_input: str, turn_usage: UsageStats) -> str:
        msg = self._end_cancelled(user_input, turn_usage)
        if self._double_sigint:
            raise _DoubleSigintError
        return msg

    def _end_finished(self, user_input: str, turn_usage: UsageStats, response: str) -> str:
        self.listener.on_event(TurnFinished(response=response))
        self._persist_session(user_input, turn_usage)
        return response

    def _append_text_if_any(self, text: str | None) -> None:
        if text:
            self.messages.append(Message(role="assistant", parts=[TextPart(text=text)]))

    def step(self, user_input: str, extra_tools: list[str] | None = None) -> str:
        """Run a single user turn in the agent loop.

        extra_tools=None uses the full registry. extra_tools=[] sends no tools.
        """
        cleaned_input = user_input.strip()
        if not cleaned_input:
            return ""

        self._cancel.clear()
        self._reset_doom()
        self.listener.on_event(TurnStarted(user_input=cleaned_input))
        self.messages.append(Message(role="user", parts=[TextPart(text=cleaned_input)]))
        turn_usage = UsageStats()
        turn_tools = self._tools_for_turn(extra_tools)
        tools_disabled = extra_tools == []
        pending_calls: list[FunctionCall] = []

        try:
            for _ in range(self.config.max_tool_rounds):
                if self._cancel.is_set():
                    return self._finish_cancel(cleaned_input, turn_usage)
                if self._doom_fired:
                    return self._end_finished(cleaned_input, turn_usage, _DOOM_MSG)

                self.listener.on_event(ModelStarted())
                window = self._compact_window(turn_tools)
                response = self._create_response(window, turn_tools)
                if response is None or self._cancel.is_set():
                    text = response.text if response is not None else None
                    self._append_text_if_any(text)
                    return self._finish_cancel(cleaned_input, turn_usage)

                turn_usage = turn_usage.add(response.usage)

                if not response.function_calls:
                    if response.text:
                        self.messages.append(
                            Message(role="assistant", parts=[TextPart(text=response.text)])
                        )
                    final_answer = response.text or "(模型未返回文本内容)"
                    return self._end_finished(cleaned_input, turn_usage, final_answer)

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
                pending_calls = list(response.function_calls)
                self.messages.append(Message(role="assistant", parts=call_parts))

                results = self._run_calls(pending_calls, tools_disabled=tools_disabled)
                self.messages.append(
                    Message(role="assistant", parts=self._result_parts(pending_calls, results))
                )
                pending_calls = []
                self._save_paired()

                if self._cancel.is_set():
                    return self._finish_cancel(cleaned_input, turn_usage)
                if self._doom_fired:
                    return self._end_finished(cleaned_input, turn_usage, _DOOM_MSG)

            if pending_calls:
                results = [_cancelled_result() for _ in pending_calls]
                self.messages.append(
                    Message(role="assistant", parts=self._result_parts(pending_calls, results))
                )
                pending_calls = []
                self._save_paired()
                return self._finish_cancel(cleaned_input, turn_usage)

            return self._end_finished(cleaned_input, turn_usage, _ROUNDS_EXCEEDED_MSG)
        except KeyboardInterrupt:
            self.request_cancel()
            if pending_calls:
                results = [_cancelled_result() for _ in pending_calls]
                self.messages.append(
                    Message(role="assistant", parts=self._result_parts(pending_calls, results))
                )
                self._save_paired()
            msg = self._end_cancelled(cleaned_input, turn_usage)
            if self._double_sigint:
                raise
            return msg
        except _DoubleSigintError:
            raise KeyboardInterrupt from None
        except LLMError as exc:
            self.listener.on_event(TurnFailed(error=str(exc)))
            self._drop_unpaired_tool_calls()
            raise
