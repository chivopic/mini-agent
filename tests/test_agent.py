"""Unit tests for Agent loop and Fake LLM client integration."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mini_agent.agent import Agent, AgentEventListener
from mini_agent.llm import (
    FunctionCall,
    LLMClient,
    LLMResponse,
    OpenAIChatCompletionsClient,
)
from mini_agent.models import AgentConfig, ToolResult
from mini_agent.permission import PermissionRequest, Reply
from mini_agent.session import load_session


class FakeLLMClient(LLMClient):
    """Deterministic Fake LLM Client for testing."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.call_history: list[list[dict[str, Any]]] = []

    def create_response(
        self,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        self.call_history.append(list(history))
        if not self.responses:
            return LLMResponse(text="[FakeLLM: No more responses configured]")
        resp = self.responses.pop(0)
        if resp.text and on_token:
            for char in resp.text:
                on_token(char)
        return resp


class RecordingEventListener(AgentEventListener):
    """Event listener that records all events."""

    def __init__(self, confirm_decision: bool = True) -> None:
        self.events: list[tuple[str, Any]] = []
        self.tokens: list[str] = []
        self.confirm_decision = confirm_decision

    def on_turn_start(self, user_input: str) -> None:
        self.events.append(("turn_start", user_input))

    def on_token(self, token: str) -> None:
        self.tokens.append(token)

    def on_model_start(self) -> None:
        self.events.append(("model_start", None))

    def on_tool_start(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self.events.append(("tool_start", (tool_name, arguments)))

    def on_permission_ask(self, req: PermissionRequest) -> Reply:
        self.events.append(("permission_ask", req))
        return Reply.ONCE if self.confirm_decision else Reply.REJECT

    def on_tool_finished(self, tool_name: str, result: ToolResult) -> None:
        self.events.append(("tool_finished", (tool_name, result)))

    def on_turn_finished(self, response: str) -> None:
        self.events.append(("turn_finished", response))


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
        assert fake_llm.call_history[0][1]["content"] == "你好"
        assert "".join(listener.tokens) == "你好！我是助手。"

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

        second_history = fake_llm.call_history[1]
        tool_outputs = [
            item for item in second_history if item.get("type") == "function_call_output"
        ]
        assert len(tool_outputs) == 1
        assert tool_outputs[0]["call_id"] == "call_list_1"
        assert "main.py" in tool_outputs[0]["output"]

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

        second_history = fake_llm.call_history[1]
        tool_outputs = [
            item for item in second_history if item.get("type") == "function_call_output"
        ]
        assert len(tool_outputs) == 2
        assert tool_outputs[0]["call_id"] == "call_read_a"
        assert tool_outputs[1]["call_id"] == "call_read_b"

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

        second_history = fake_llm.call_history[1]
        tool_outputs = [
            item for item in second_history if item.get("type") == "function_call_output"
        ]
        assert "不存在" in tool_outputs[0]["output"]

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
        assert any(event[0] == "permission_ask" for event in listener.events)

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
        assert len(agent2.history) == len(saved.history)

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
        assert len(agent.history) >= 4

    def test_extra_tools_empty_sends_no_tools(self, tmp_path: Path) -> None:
        class ToolsLLM(FakeLLMClient):
            def __init__(self) -> None:
                super().__init__([LLMResponse(text="feat: x")])
                self.tools_args: list[list[dict[str, Any]]] = []

            def create_response(
                self,
                history: list[dict[str, Any]],
                tools: list[dict[str, Any]],
                model: str = "gpt-4o-mini",
                on_token: Callable[[str], None] | None = None,
            ) -> LLMResponse:
                self.tools_args.append(tools)
                return super().create_response(history, tools, model, on_token)

        llm = ToolsLLM()
        agent = Agent(config=AgentConfig(workspace_root=tmp_path), llm_client=llm)
        answer = agent.step("生成提交说明", extra_tools=[])
        assert answer == "feat: x"
        assert llm.tools_args == [[]]
