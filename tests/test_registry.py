"""Tests for the tool registry, schema generation, and Agent dispatch wiring."""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mini_agent.agent import Agent
from mini_agent.llm import (
    FunctionCall,
    LLMClient,
    LLMResponse,
    OpenAIChatCompletionsClient,
    get_system_prompt,
)
from mini_agent.models import AgentConfig, PermissionClass, ToolResult
from mini_agent.permission import PermissionRequest, Reply
from mini_agent.tools import default_registry
from mini_agent.tools.protocol import ToolContext, ToolKind


class FakeLLMClient(LLMClient):
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
        return self.responses.pop(0)


class ConfirmListener:
    def __init__(self) -> None:
        self.confirm_calls: list[str] = []

    def on_permission_ask(self, req: PermissionRequest) -> Reply:
        self.confirm_calls.append(req.resource)
        return Reply.ONCE


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace, AgentConfig(workspace_root=workspace), threading.Event())


class EchoInput(BaseModel):
    message: str = Field(description="Text to echo back.")


class EchoTool:
    name = "echo_tool"
    description = "回显输入消息。"
    permission = PermissionClass.READ
    kind = ToolKind.READONLY
    input_model = EchoInput

    def execute(self, inp: EchoInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, content=inp.message)

    def format_call(self, inp: EchoInput) -> str:
        return f"echo_tool message={inp.message}"

    def approval_pattern(self, inp: EchoInput) -> str:
        return f"read:{inp.message}"


class TestRegistryDispatch:
    """Registry.dispatch error strings stay compatible with 0.2."""

    def test_invalid_json(self, tmp_path: Path) -> None:
        result = default_registry().dispatch("read_file", "not valid json", _ctx(tmp_path))
        assert result.ok is False
        assert (result.error or "").startswith("工具参数不是合法的 JSON 字符串")

    def test_non_object_json(self, tmp_path: Path) -> None:
        result = default_registry().dispatch("read_file", "[1, 2]", _ctx(tmp_path))
        assert result.ok is False
        assert (result.error or "").startswith("工具参数必须为 JSON 对象")

    def test_unknown_tool_name(self, tmp_path: Path) -> None:
        result = default_registry().dispatch("non_existent_tool", "{}", _ctx(tmp_path))
        assert result.ok is False
        assert "未知的工具名称" in (result.error or "")

    def test_empty_tool_name(self, tmp_path: Path) -> None:
        result = default_registry().dispatch("", "{}", _ctx(tmp_path))
        assert result.ok is False
        assert "未知的工具名称" in (result.error or "")


class TestJsonSchemas:
    """Schema post-process: nested Chat Completions format, no strict, extra fields closed."""

    def test_search_code_includes_max_results_and_no_strict(self) -> None:
        schemas = default_registry().json_schemas()
        by_name = {item["function"]["name"]: item for item in schemas}
        assert "search_code" in by_name
        params = by_name["search_code"]["function"]["parameters"]
        assert "max_results" in params["properties"]
        for item in schemas:
            assert item["type"] == "function"
            assert "function" in item
            assert "strict" not in item
            assert "strict" not in item["function"]
            assert item["function"]["parameters"]["additionalProperties"] is False

    def test_nested_schemas_pass_through_convert_tools(self) -> None:
        client = OpenAIChatCompletionsClient(api_key="fake-key")
        schemas = default_registry().json_schemas()
        assert client._convert_tools(schemas) == schemas

    def test_send_strict_opt_in(self) -> None:
        schemas = default_registry().json_schemas(send_strict=True)
        for item in schemas:
            assert item["function"]["strict"] is True
            assert "strict" not in item


class TestAgentRegistryWiring:
    """Agent uses the registry and keeps shell confirmation in the loop."""

    def test_invalid_json_run_shell_does_not_confirm(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="run_shell",
                            call_id="call_bad_json",
                            arguments="not valid json",
                        )
                    ]
                ),
                LLMResponse(text="参数格式错误。"),
            ]
        )
        listener = ConfirmListener()
        agent = Agent(
            config=AgentConfig(workspace_root=tmp_path),
            llm_client=fake_llm,
            listener=listener,
        )
        answer = agent.step("执行坏参数命令")
        assert answer == "参数格式错误。"
        assert listener.confirm_calls == []
        tool_outputs = [
            item for item in fake_llm.call_history[1] if item.get("type") == "function_call_output"
        ]
        assert "工具参数不是合法的 JSON 字符串" in tool_outputs[0]["output"]

    def test_non_allowlisted_shell_confirmed_executes(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="run_shell",
                            call_id="call_shell_ok",
                            arguments='{"command": "touch confirmed.txt"}',
                        )
                    ]
                ),
                LLMResponse(text="已创建 confirmed.txt。"),
            ]
        )
        listener = ConfirmListener()
        agent = Agent(
            config=AgentConfig(workspace_root=tmp_path),
            llm_client=fake_llm,
            listener=listener,
        )
        answer = agent.step("创建 confirmed.txt")
        assert "confirmed.txt" in answer
        assert (tmp_path / "confirmed.txt").exists()
        assert listener.confirm_calls == ["touch confirmed.txt"]

    def test_registered_extra_tool_invoked_by_fake_llm(self, tmp_path: Path) -> None:
        registry = default_registry()
        registry.register(EchoTool())
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="echo_tool",
                            call_id="call_echo",
                            arguments='{"message": "pong"}',
                        )
                    ]
                ),
                LLMResponse(text="pong 已收到。"),
            ]
        )
        agent = Agent(
            config=AgentConfig(workspace_root=tmp_path),
            llm_client=fake_llm,
            registry=registry,
        )
        answer = agent.step("echo")
        assert answer == "pong 已收到。"
        tool_outputs = [
            item for item in fake_llm.call_history[1] if item.get("type") == "function_call_output"
        ]
        assert len(tool_outputs) == 1
        assert "pong" in tool_outputs[0]["output"]

    def test_llm_reexports_get_system_prompt(self, tmp_path: Path) -> None:
        text = get_system_prompt(tmp_path)
        assert "请严格遵守以下开发准则：" in text
        assert "`get_repo_map`" in text
        assert "`read_file` 的 metadata 含行号" in text
        assert "不要带行号前缀" in text
        assert "uv run pytest" in text
