"""Unit tests for Agent loop and Fake LLM client integration."""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from mini_agent.agent import Agent
from mini_agent.events import AgentEvent, AgentEventListener, TokenDelta
from mini_agent.llm import (
    FunctionCall,
    LLMClient,
    LLMConnectionError,
    LLMResponse,
    OpenAIChatCompletionsClient,
    _function_calls_from_accumulator,
)
from mini_agent.messages import Message, ToolResultPart, assert_pairing, to_chat_messages
from mini_agent.models import AgentConfig, PermissionClass, ToolResult
from mini_agent.permission import PermissionRequest, Reply
from mini_agent.session import load_session
from mini_agent.tools import default_registry
from mini_agent.tools.protocol import ToolCancelled, ToolContext, ToolKind


class FakeLLMClient(LLMClient):
    """Deterministic Fake LLM Client for testing."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.call_history: list[list[Message]] = []

    def create_response(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        on_token: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> LLMResponse:
        self.call_history.append(list(messages))
        if not self.responses:
            return LLMResponse(text="[FakeLLM: No more responses configured]")
        resp = self.responses.pop(0)
        if resp.text and on_token:
            for char in resp.text:
                if cancel is not None and cancel.is_set():
                    break
                on_token(char)
        return resp


class RecordingEventListener(AgentEventListener):
    """Event listener that records event.type values."""

    def __init__(self, confirm_decision: bool = True) -> None:
        self.events: list[AgentEvent] = []
        self.types: list[str] = []
        self.tokens: list[str] = []
        self.asks: list[PermissionRequest] = []
        self.confirm_decision = confirm_decision

    def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)
        self.types.append(event.type)
        if isinstance(event, TokenDelta):
            self.tokens.append(event.token)

    def on_permission_ask(self, req: PermissionRequest) -> Reply:
        self.asks.append(req)
        return Reply.ONCE if self.confirm_decision else Reply.REJECT


def _result_parts(messages: list[Message]) -> list[ToolResultPart]:
    return [p for msg in messages for p in msg.parts if isinstance(p, ToolResultPart)]


class TestOpenAIChatCompletionsAdapter:
    """Test message and tool conversion in OpenAIChatCompletionsClient."""

    def test_message_conversion(self) -> None:
        client = OpenAIChatCompletionsClient(api_key="fake-key")
        history = [
            {"role": "system", "content": "You are assistant."},
            {"role": "user", "content": "Hello"},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"ok": true}'},
            {"role": "assistant", "content": "Done"},
        ]
        converted = client._convert_messages(history)
        assert len(converted) == 4
        assert converted[0] == {"role": "system", "content": "You are assistant."}
        assert converted[1] == {"role": "user", "content": "Hello"}
        assert converted[2] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}
        assert converted[3] == {"role": "assistant", "content": "Done"}

    def test_tools_conversion(self) -> None:
        client = OpenAIChatCompletionsClient(api_key="fake-key")
        raw_tools = [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read file",
                "parameters": {"type": "object"},
            }
        ]
        converted = client._convert_tools(raw_tools)
        assert len(converted) == 1
        assert "function" in converted[0]
        assert converted[0]["function"]["name"] == "read_file"


class TestAgentLoop:
    """Test full agent loop scenarios with Fake LLM."""

    def test_direct_text_response_no_tools(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient([LLMResponse(text="你好！我是助手。")])
        config = AgentConfig(workspace_root=tmp_path)
        listener = RecordingEventListener()
        agent = Agent(config=config, llm_client=fake_llm, listener=listener)

        answer = agent.step("你好")
        assert answer == "你好！我是助手。"
        assert len(fake_llm.call_history) == 1
        assert len(fake_llm.call_history[0]) == 2
        wire = to_chat_messages(fake_llm.call_history[0])
        assert wire[1]["content"] == "你好"
        assert "".join(listener.tokens) == "你好！我是助手。"
        assert "turn_started" in listener.types
        assert "turn_finished" in listener.types

    def test_list_files_then_final_answer(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")

        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="list_files",
                            call_id="call_list_1",
                            arguments='{"path": "."}',
                        )
                    ]
                ),
                LLMResponse(text="项目包含入口文件 main.py。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        listener = RecordingEventListener()
        agent = Agent(config=config, llm_client=fake_llm, listener=listener)

        answer = agent.step("项目中有什么文件？")
        assert answer == "项目包含入口文件 main.py。"
        assert len(fake_llm.call_history) == 2

        results = _result_parts(fake_llm.call_history[1])
        assert len(results) == 1
        assert results[0].call_id == "call_list_1"
        assert "main.py" in results[0].content

    def test_multiple_tool_calls_in_single_turn(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
        (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")

        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="read_file",
                            call_id="call_read_a",
                            arguments='{"path": "a.txt"}',
                        ),
                        FunctionCall(
                            name="read_file",
                            call_id="call_read_b",
                            arguments='{"path": "b.txt"}',
                        ),
                    ]
                ),
                LLMResponse(text="文件 a 的内容是 AAA，文件 b 的内容是 BBB。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        agent = Agent(config=config, llm_client=fake_llm)

        answer = agent.step("读 a.txt 和 b.txt")
        assert "AAA" in answer and "BBB" in answer

        results = _result_parts(fake_llm.call_history[1])
        assert len(results) == 2
        assert results[0].call_id == "call_read_a"
        assert results[1].call_id == "call_read_b"

    def test_tool_failure_handled_gracefully(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="read_file",
                            call_id="call_fail_1",
                            arguments='{"path": "non_existent.py"}',
                        )
                    ]
                ),
                LLMResponse(text="文件不存在，我无法读取。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        agent = Agent(config=config, llm_client=fake_llm)

        answer = agent.step("读不存在的文件")
        assert answer == "文件不存在，我无法读取。"

        results = _result_parts(fake_llm.call_history[1])
        assert "不存在" in results[0].error or "不存在" in results[0].content

    def test_invalid_json_arguments(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="read_file",
                            call_id="call_bad_json",
                            arguments="not valid json",
                        )
                    ]
                ),
                LLMResponse(text="参数格式错误。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        agent = Agent(config=config, llm_client=fake_llm)

        answer = agent.step("测试坏参数")
        assert answer == "参数格式错误。"

    def test_unknown_tool_name(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="non_existent_tool",
                            call_id="call_unknown",
                            arguments="{}",
                        )
                    ]
                ),
                LLMResponse(text="未知工具已处理。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        agent = Agent(config=config, llm_client=fake_llm)

        answer = agent.step("调用未知工具")
        assert answer == "未知工具已处理。"

    def test_max_tool_rounds_exceeded(self, tmp_path: Path) -> None:
        infinite_responses = [
            LLMResponse(
                function_calls=[
                    FunctionCall(
                        name="list_files",
                        call_id=f"call_{i}",
                        arguments='{"path": "."}',
                    )
                ]
            )
            for i in range(10)
        ]
        fake_llm = FakeLLMClient(infinite_responses)
        config = AgentConfig(workspace_root=tmp_path, max_tool_rounds=3)
        agent = Agent(config=config, llm_client=fake_llm)

        answer = agent.step("无限循环工具")
        assert "上限" in answer
        assert len(fake_llm.call_history) == 3

    def test_empty_user_input(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient([])
        config = AgentConfig(workspace_root=tmp_path)
        agent = Agent(config=config, llm_client=fake_llm)

        answer = agent.step("   ")
        assert answer == ""
        assert len(fake_llm.call_history) == 0

    def test_non_allowlisted_command_user_confirmed(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="run_shell",
                            call_id="call_shell_ok",
                            arguments='{"command": "touch allowed.txt"}',
                        )
                    ]
                ),
                LLMResponse(text="已创建 allowed.txt。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        listener = RecordingEventListener(confirm_decision=True)
        agent = Agent(config=config, llm_client=fake_llm, listener=listener)

        answer = agent.step("创建文件")
        assert "allowed.txt" in answer
        assert (tmp_path / "allowed.txt").exists()
        assert listener.asks

    def test_dangerous_command_user_rejected(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="run_shell",
                            call_id="call_shell_1",
                            arguments='{"command": "touch dangerous.txt"}',
                        )
                    ]
                ),
                LLMResponse(text="既然您拒绝了，我不会执行此命令。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        listener = RecordingEventListener(confirm_decision=False)
        agent = Agent(config=config, llm_client=fake_llm, listener=listener)

        answer = agent.step("创建文件")
        assert "拒绝" in answer
        assert not (tmp_path / "dangerous.txt").exists()

    def test_write_file_turn(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="write_file",
                            call_id="call_write_1",
                            arguments='{"path": "hello.py", "content": "print(\'hello\')"}',
                        )
                    ]
                ),
                LLMResponse(text="已成功创建 hello.py 文件。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        agent = Agent(config=config, llm_client=fake_llm)

        answer = agent.step("创建 hello.py")
        assert "创建" in answer
        assert (tmp_path / "hello.py").exists()
        assert (tmp_path / "hello.py").read_text(encoding="utf-8") == "print('hello')"

    def test_edit_file_turn(self, tmp_path: Path) -> None:
        (tmp_path / "calc.py").write_text("def sub(a, b):\n    return a - b\n", encoding="utf-8")

        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="edit_file",
                            call_id="call_edit_1",
                            arguments=(
                                '{"path": "calc.py", "target_content": "def sub", '
                                '"replacement_content": "def add"}'
                            ),
                        )
                    ]
                ),
                LLMResponse(text="已将函数名修改为 add。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        agent = Agent(config=config, llm_client=fake_llm)

        answer = agent.step("修改函数名")
        assert "修改" in answer
        assert "def add" in (tmp_path / "calc.py").read_text(encoding="utf-8")

    def test_session_auto_persistence_and_resume(self, tmp_path: Path, monkeypatch: Any) -> None:
        sessions_dir = tmp_path / "sessions"
        monkeypatch.setenv("MINI_AGENT_SESSIONS_DIR", str(sessions_dir))

        fake_llm = FakeLLMClient([LLMResponse(text="第一轮回答")])
        config = AgentConfig(workspace_root=tmp_path)
        agent1 = Agent(config=config, llm_client=fake_llm)

        session_id = agent1.session.meta.session_id
        agent1.step("我的名字是 Alice")

        # Verify session file was automatically saved
        saved = load_session(session_id, sessions_dir=sessions_dir)
        assert saved is not None
        assert saved.meta.turn_count == 1
        assert "Alice" in saved.meta.title

        # Resume session with new Agent instance
        fake_llm2 = FakeLLMClient([LLMResponse(text="你好 Alice！")])
        agent2 = Agent(config=config, llm_client=fake_llm2, session=saved)
        assert len(agent2.messages) == len(saved.messages)

        answer = agent2.step("你还记得我的名字吗？")
        assert "Alice" in answer
        assert agent2.session.meta.turn_count == 2

    def test_agent_executes_search_code(self, tmp_path: Path) -> None:
        (tmp_path / "service.py").write_text(
            "def find_user_by_id(uid):\n    pass\n", encoding="utf-8"
        )
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="search_code",
                            call_id="call_search_1",
                            arguments='{"pattern": "find_user"}',
                        )
                    ]
                ),
                LLMResponse(text="找到函数 find_user_by_id 在 service.py 中。"),
            ]
        )
        config = AgentConfig(workspace_root=tmp_path)
        agent = Agent(config=config, llm_client=fake_llm)

        answer = agent.step("查找用户函数在哪里")
        assert "find_user_by_id" in answer
        assert len(agent.messages) >= 4

    def test_extra_tools_empty_sends_no_tools(self, tmp_path: Path) -> None:
        class ToolsLLM(FakeLLMClient):
            def __init__(self) -> None:
                super().__init__([LLMResponse(text="feat: x")])
                self.tools_args: list[list[dict[str, Any]]] = []

            def create_response(
                self,
                messages: list[Message],
                tools: list[dict[str, Any]],
                model: str = "gpt-4o-mini",
                on_token: Callable[[str], None] | None = None,
                cancel: threading.Event | None = None,
            ) -> LLMResponse:
                self.tools_args.append(tools)
                return super().create_response(messages, tools, model, on_token, cancel)

        llm = ToolsLLM()
        agent = Agent(config=AgentConfig(workspace_root=tmp_path), llm_client=llm)
        answer = agent.step("生成提交说明", extra_tools=[])
        assert answer == "feat: x"
        assert llm.tools_args == [[]]

    def test_extra_tools_empty_does_not_dispatch_function_calls(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="write_file",
                            call_id="call_hallucinated",
                            arguments='{"path": "pwned.py", "content": "x"}',
                        )
                    ]
                ),
                LLMResponse(text="feat: no tools"),
            ]
        )
        agent = Agent(config=AgentConfig(workspace_root=tmp_path), llm_client=fake_llm)
        answer = agent.step("生成提交说明", extra_tools=[])
        assert answer == "feat: no tools"
        assert not (tmp_path / "pwned.py").exists()
        outputs = _result_parts(agent.messages)
        assert len(outputs) == 1
        assert outputs[0].call_id == "call_hallucinated"
        assert outputs[0].error is not None
        assert "未启用工具" in outputs[0].error


class SleepInput(BaseModel):
    n: int = Field(default=1)


class SleepTool:
    name = "sleep_tool"
    description = "阻塞直到取消。"
    permission = PermissionClass.READ
    kind = ToolKind.READONLY
    input_model = SleepInput

    def __init__(self, started: threading.Event) -> None:
        self.started = started

    def execute(self, inp: SleepInput, ctx: ToolContext) -> ToolResult:
        self.started.set()
        if ctx.cancel.wait(timeout=5):
            raise ToolCancelled
        return ToolResult(ok=True, content="slept")

    def format_call(self, inp: SleepInput) -> str:
        return "sleep_tool"

    def approval_pattern(self, inp: SleepInput) -> str:
        return "read:sleep"


class FastInput(BaseModel):
    x: int = 0


class FastTool:
    name = "fast_tool"
    description = "立即返回。"
    permission = PermissionClass.READ
    kind = ToolKind.READONLY
    input_model = FastInput

    def execute(self, inp: FastInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, content="fast-ok")

    def format_call(self, inp: FastInput) -> str:
        return "fast_tool"

    def approval_pattern(self, inp: FastInput) -> str:
        return "read:fast"


class GateTool:
    name = "gate_tool"
    description = "等待取消。"
    permission = PermissionClass.READ
    kind = ToolKind.READONLY
    input_model = FastInput

    def __init__(self, started: threading.Event) -> None:
        self.started = started

    def execute(self, inp: FastInput, ctx: ToolContext) -> ToolResult:
        self.started.set()
        if ctx.cancel.wait(timeout=5):
            raise ToolCancelled
        return ToolResult(ok=True, content="gate-ok")

    def format_call(self, inp: FastInput) -> str:
        return "gate_tool"

    def approval_pattern(self, inp: FastInput) -> str:
        return "read:gate"


class TestAgentPr4Loop:
    def test_parallel_read_file_persists_in_call_order(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
        (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")
        finish_order: list[str] = []
        b_started = threading.Event()
        a_can_finish = threading.Event()

        from mini_agent.tools.filesystem import ReadFileTool, read_file

        def slow_execute(self: Any, inp: Any, ctx: ToolContext) -> ToolResult:
            if inp.path == "b.txt":
                b_started.set()
                result = read_file(inp, workspace_root=ctx.workspace_root)
                finish_order.append("b.txt")
                a_can_finish.set()
                return result
            b_started.wait(timeout=2)
            a_can_finish.wait(timeout=2)
            result = read_file(inp, workspace_root=ctx.workspace_root)
            finish_order.append("a.txt")
            return result

        monkeypatch.setattr(ReadFileTool, "execute", slow_execute)
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="read_file",
                            call_id="call_read_a",
                            arguments='{"path": "a.txt"}',
                        ),
                        FunctionCall(
                            name="read_file",
                            call_id="call_read_b",
                            arguments='{"path": "b.txt"}',
                        ),
                    ]
                ),
                LLMResponse(text="ok"),
            ]
        )
        listener = RecordingEventListener()
        agent = Agent(
            config=AgentConfig(workspace_root=tmp_path),
            llm_client=fake_llm,
            listener=listener,
        )
        agent.step("读 a 和 b")
        assert finish_order == ["b.txt", "a.txt"]
        persisted = _result_parts(agent.messages)
        assert [part.call_id for part in persisted] == ["call_read_a", "call_read_b"]
        assert persisted[0].content.startswith("AAA") or "AAA" in persisted[0].content
        assert "BBB" in persisted[1].content
        started = [e for e in listener.events if e.type == "tool_started"]
        finished = [e for e in listener.events if e.type == "tool_finished"]
        assert [e.name for e in started] == ["read_file", "read_file"]
        assert [e.call_id for e in started] == ["call_read_a", "call_read_b"]
        assert [e.call_id for e in finished] == ["call_read_a", "call_read_b"]

    def test_cancel_during_sleep_keeps_pairing(self, tmp_path: Path, monkeypatch: Any) -> None:
        sessions_dir = tmp_path / "sessions"
        monkeypatch.setenv("MINI_AGENT_SESSIONS_DIR", str(sessions_dir))
        started = threading.Event()
        registry = default_registry()
        registry.register(SleepTool(started))
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(name="sleep_tool", call_id="call_sleep", arguments="{}")
                    ]
                ),
                LLMResponse(text="should not run"),
            ]
        )
        agent = Agent(
            config=AgentConfig(workspace_root=tmp_path),
            llm_client=fake_llm,
            registry=registry,
        )

        def cancel_soon() -> None:
            started.wait(timeout=2)
            agent.request_cancel()

        thread = threading.Thread(target=cancel_soon)
        thread.start()
        answer = agent.step("sleep")
        thread.join(timeout=2)
        assert "取消" in answer
        saved = load_session(agent.session.meta.session_id, sessions_dir=sessions_dir)
        assert saved is not None
        assert_pairing(saved.messages)
        results = _result_parts(saved.messages)
        assert results
        assert results[0].ok is False
        assert results[0].error == "用户取消"

    def test_double_ctrl_c_with_fake_monotonic(self, tmp_path: Path) -> None:
        clock = {"t": 100.0}
        agent = Agent(
            config=AgentConfig(workspace_root=tmp_path),
            llm_client=FakeLLMClient([]),
            monotonic=lambda: clock["t"],
        )
        agent.request_cancel()
        assert agent._cancel.is_set()
        assert agent._double_sigint is False
        clock["t"] = 101.5
        agent.request_cancel()
        assert agent._double_sigint is True
        clock["t"] = 110.0
        agent.request_cancel()
        assert agent._double_sigint is False

    def test_llm_keyboard_interrupt_counts_sigint(self, tmp_path: Path) -> None:
        class InterruptLLM(FakeLLMClient):
            def create_response(
                self,
                messages: list[Message],
                tools: list[dict[str, Any]],
                model: str = "gpt-4o-mini",
                on_token: Callable[[str], None] | None = None,
                cancel: threading.Event | None = None,
            ) -> LLMResponse:
                raise KeyboardInterrupt

        clock = {"t": 50.0}
        agent = Agent(
            config=AgentConfig(workspace_root=tmp_path),
            llm_client=InterruptLLM([]),
            monotonic=lambda: clock["t"],
        )
        answer = agent.step("hi")
        assert "取消" in answer
        assert agent._last_sigint_at == 50.0
        assert agent._double_sigint is False
        clock["t"] = 51.0
        with pytest.raises(KeyboardInterrupt):
            agent.step("again")

    def test_parallel_cancel_keeps_finished_results(self, tmp_path: Path) -> None:
        started = threading.Event()
        registry = default_registry()
        registry.register(FastTool())
        registry.register(GateTool(started))
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(name="fast_tool", call_id="call_fast", arguments="{}"),
                        FunctionCall(name="gate_tool", call_id="call_gate", arguments="{}"),
                    ]
                ),
                LLMResponse(text="should not run"),
            ]
        )
        agent = Agent(
            config=AgentConfig(workspace_root=tmp_path),
            llm_client=fake_llm,
            registry=registry,
        )

        def cancel_soon() -> None:
            started.wait(timeout=2)
            agent.request_cancel()

        thread = threading.Thread(target=cancel_soon)
        thread.start()
        agent.step("go")
        thread.join(timeout=2)
        results = _result_parts(agent.messages)
        by_id = {part.call_id: part for part in results}
        assert by_id["call_fast"].ok is True
        assert "fast-ok" in by_id["call_fast"].content
        assert by_id["call_gate"].ok is False
        assert by_id["call_gate"].error == "用户取消"

    def test_doom_does_not_fire_on_three_successful_list_files(self, tmp_path: Path) -> None:
        infinite_responses = [
            LLMResponse(
                function_calls=[
                    FunctionCall(
                        name="list_files",
                        call_id=f"call_{i}",
                        arguments='{"path": "."}',
                    )
                ]
            )
            for i in range(10)
        ]
        fake_llm = FakeLLMClient(infinite_responses)
        config = AgentConfig(workspace_root=tmp_path, max_tool_rounds=3)
        agent = Agent(config=config, llm_client=fake_llm)
        answer = agent.step("无限循环工具")
        assert "上限" in answer
        assert "重复" not in answer
        assert len(fake_llm.call_history) == 3

    def test_doom_fires_on_three_identical_failures(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="run_shell",
                            call_id=f"call_{i}",
                            arguments='{"command": "rm -rf /"}',
                        )
                        for i in range(4)
                    ]
                ),
                LLMResponse(text="should not run"),
            ]
        )
        agent = Agent(config=AgentConfig(workspace_root=tmp_path), llm_client=fake_llm)
        answer = agent.step("删")
        assert "重复" in answer
        assert len(fake_llm.call_history) == 1
        results = _result_parts(agent.messages)
        assert len(results) == 4
        assert results[3].error == "检测到重复工具调用"

    def test_max_tool_rounds_default_is_32(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace_root=tmp_path)
        assert config.max_tool_rounds == 32

    def test_reset_session_returns_stored_id(self, tmp_path: Path, monkeypatch: Any) -> None:
        sessions_dir = tmp_path / "sessions"
        monkeypatch.setenv("MINI_AGENT_SESSIONS_DIR", str(sessions_dir))
        agent = Agent(config=AgentConfig(workspace_root=tmp_path), llm_client=FakeLLMClient([]))
        old_id = agent.session.meta.session_id
        new_id = agent.reset_session()
        assert new_id != old_id
        assert new_id == agent.session.meta.session_id
        saved = load_session(new_id, sessions_dir=sessions_dir)
        assert saved is not None
        assert saved.meta.session_id == new_id

    def test_streamed_tool_calls_sorted_by_index(self) -> None:
        acc = {
            1: {"id": "b", "name": "read_file", "arguments": "{}"},
            0: {"id": "a", "name": "list_files", "arguments": "{}"},
        }
        calls = _function_calls_from_accumulator(acc)
        assert [c.call_id for c in calls] == ["a", "b"]
        empty_id = _function_calls_from_accumulator({2: {"id": "", "name": "x", "arguments": "{}"}})
        assert empty_id[0].call_id == "invalid_2"

    def test_llm_connection_error_retries(self, tmp_path: Path) -> None:
        class Flaky(FakeLLMClient):
            def __init__(self) -> None:
                super().__init__([LLMResponse(text="recovered")])
                self.attempts = 0

            def create_response(
                self,
                messages: list[Message],
                tools: list[dict[str, Any]],
                model: str = "gpt-4o-mini",
                on_token: Callable[[str], None] | None = None,
                cancel: threading.Event | None = None,
            ) -> LLMResponse:
                self.attempts += 1
                if self.attempts < 3:
                    raise LLMConnectionError("boom")
                return super().create_response(messages, tools, model, on_token, cancel)

        llm = Flaky()
        agent = Agent(config=AgentConfig(workspace_root=tmp_path), llm_client=llm)
        agent._retry_backoff = (0.0, 0.0, 0.0)
        assert agent.step("hi") == "recovered"
        assert llm.attempts == 3
