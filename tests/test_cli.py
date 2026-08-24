"""Unit tests for Typer CLI and REPL interface."""

import re
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any
from unittest.mock import patch

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from mini_agent.agent import Agent
from mini_agent.cli import (
    RichAgentEventListener,
    app,
    render_banner,
    render_help,
    render_sessions_table,
    run_cli,
)
from mini_agent.llm import FunctionCall, LLMClient, LLMResponse
from mini_agent.messages import Message
from mini_agent.models import AgentConfig, ToolResult
from mini_agent.permission import DefaultPermissionService, Reply
from mini_agent.session import SessionData, SessionMeta, save_session

runner = CliRunner()


class DummyLLM(LLMClient):
    """Dummy LLM for CLI testing."""

    def __init__(self, answer: str = "这是回答") -> None:
        self.answer = answer

    def create_response(
        self,
        messages: list[Message],
        tools: list[dict[str, object]],
        model: str = "gpt-4o-mini",
        on_token: object = None,
        cancel: object = None,
    ) -> LLMResponse:
        return LLMResponse(text=self.answer)


class TestCliCommands:
    """Test CLI options, arguments, and validation."""

    def test_cli_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Rich may wrap `--` with ANSI when FORCE_COLOR is set.
        out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        assert "--workspace" in out
        assert "--model" in out
        assert "--base-url" in out
        assert "--continue" in out
        assert "--session" in out
        assert "--verbose" in out
        assert "--yes" in out
        assert "--config" in out

    def test_cli_invalid_workspace(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "not_found_dir"
        result = runner.invoke(app, ["--workspace", str(non_existent)])
        assert result.exit_code != 0
        assert "不存在" in result.stdout

    def test_cli_missing_api_key_exits_non_zero(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["--workspace", str(tmp_path)])
            assert result.exit_code != 0
            assert "OPENAI_API_KEY" in result.stdout

    def test_cli_render_help_content(self) -> None:
        test_console = Console(record=True)
        render_help(test_console)
        out = test_console.export_text()
        assert "/help" in out
        assert "/exit" in out
        assert "/clear" in out
        assert "/model" in out
        assert "/sessions" in out
        assert "/resume" in out
        assert "/new" in out
        assert "list_files" in out
        assert "read_file" in out
        assert "run_shell" in out
        assert "edit_file" in out
        assert "write_file" in out

    def test_cli_render_banner(self, tmp_path: Path) -> None:
        test_console = Console(record=True)
        render_banner(test_console, tmp_path, "deepseek-chat", session_id="test_sess_123")
        out = test_console.export_text()
        assert "MINI-AGENT" in out
        assert "deepseek-chat" in out
        assert "test_sess_123" in out

    def test_render_sessions_table(self, tmp_path: Path, monkeypatch: object) -> None:
        sessions_dir = tmp_path / "sessions"
        monkeypatch.setenv("MINI_AGENT_SESSIONS_DIR", str(sessions_dir))  # type: ignore[attr-defined]

        test_console = Console(record=True)
        render_sessions_table(test_console, tmp_path)
        assert "暂无历史会话" in test_console.export_text()

        # Save a session
        s = SessionData(
            meta=SessionMeta(
                session_id="s_123",
                workspace_root=tmp_path.resolve().as_posix(),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                model="deepseek-chat",
                title="写一个快速排序",
                turn_count=2,
            )
        )
        save_session(s, sessions_dir=sessions_dir)

        test_console2 = Console(record=True, width=140)
        render_sessions_table(test_console2, tmp_path)
        out2 = test_console2.export_text()
        assert "s_123" in out2
        assert "写一个快速排序" in out2


class TestCliReplExecution:
    """Test interactive REPL input handling and commands."""

    def test_repl_exit_command(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/exit\n",
                )
                assert result.exit_code == 0
                assert "再见" in result.stdout

    def test_repl_help_command(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/help\n/exit\n",
                )
                assert result.exit_code == 0
                assert "快捷指令" in result.stdout

    def test_repl_model_command(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/model deepseek-reasoner\n/exit\n",
                )
                assert result.exit_code == 0
                assert "deepseek-reasoner" in result.stdout

    def test_repl_resume_replaces_permission_memory(self, tmp_path: Path, monkeypatch: Any) -> None:
        sessions_dir = tmp_path / "sessions"
        monkeypatch.setenv("MINI_AGENT_SESSIONS_DIR", str(sessions_dir))
        now = datetime.now().isoformat()
        ws = tmp_path.resolve().as_posix()
        s_old = SessionData(
            meta=SessionMeta(
                session_id="s_old",
                workspace_root=ws,
                created_at=now,
                updated_at=now,
                model="gpt-4o-mini",
                title="旧会话",
                turn_count=1,
            ),
            permission_memory=[{"cls": "shell", "pattern": "git add -u", "effect": "allow"}],
        )
        s_new = SessionData(
            meta=SessionMeta(
                session_id="s_new",
                workspace_root=ws,
                created_at=now,
                updated_at=now,
                model="gpt-4o-mini",
                title="新会话",
                turn_count=1,
            ),
            permission_memory=[{"cls": "shell", "pattern": "uv run pytest*", "effect": "allow"}],
        )
        save_session(s_old, sessions_dir=sessions_dir)
        save_session(s_new, sessions_dir=sessions_dir)

        captured: dict[str, Agent] = {}

        def factory(config: AgentConfig, client: LLMClient, listener: object) -> Agent:
            agent = Agent(
                config=config,
                llm_client=client,
                listener=listener,  # type: ignore[arg-type]
                session=s_old,
                permission=DefaultPermissionService(),
            )
            captured["agent"] = agent
            return agent

        dummy_llm = DummyLLM("ok")
        with patch("rich.prompt.Prompt.ask", side_effect=["/resume s_new", "/exit"]):
            run_cli(workspace=tmp_path, llm_client=dummy_llm, agent_factory=factory)

        agent = captured["agent"]
        assert agent.session.meta.session_id == "s_new"
        patterns = {entry["pattern"] for entry in agent.permission.snapshot()}
        assert patterns == {"uv run pytest*"}

    def test_repl_sessions_and_new_commands(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/sessions\n/new\n/exit\n",
                )
                assert result.exit_code == 0
                assert "暂无历史会话" in result.stdout
                assert "全新会话" in result.stdout

    def test_repl_provider_command(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/provider\n/provider deepseek-r1\n/exit\n",
                )
                assert result.exit_code == 0
                assert "大模型服务商预设列表" in result.stdout
                assert "deepseek-v4-reasoner" in result.stdout

    def test_repl_provider_failure_visible_and_keeps_model(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient") as mock_client:
            mock_client.side_effect = [dummy_llm, RuntimeError("boom")]
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/provider deepseek\n/model\n/exit\n",
                )
                assert result.exit_code == 0
                assert "失败" in result.stdout
                assert "boom" in result.stdout
                assert "已成功切换" not in result.stdout
                assert "gpt-4o-mini" in result.stdout
                assert "deepseek-v4" not in result.stdout.split("/model")[-1]

    def test_repl_commit_uses_git_add_u(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("feat: add tests")

        def fake_git(argv: list[str], **kwargs: Any) -> CompletedProcess[str]:
            if argv[:3] == ["git", "diff", "HEAD"]:
                return CompletedProcess(argv, 0, stdout="diff --git a/foo.py b/foo.py\n", stderr="")
            if argv[:3] == ["git", "status", "--porcelain"]:
                return CompletedProcess(argv, 0, stdout="?? scratch.py\n", stderr="")
            return CompletedProcess(argv, 0, stdout="", stderr="")

        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                with patch("mini_agent.gitutil.subprocess.run", side_effect=fake_git) as mock_run:
                    with patch.object(
                        RichAgentEventListener, "on_permission_ask", return_value=Reply.ONCE
                    ):
                        result = runner.invoke(
                            app,
                            ["--workspace", str(tmp_path)],
                            input="/commit\n/exit\n",
                        )
        assert result.exit_code == 0
        assert "Git 提交成功" in result.stdout
        assert "feat: add tests" in result.stdout
        argv_calls = [call.args[0] for call in mock_run.call_args_list]
        assert ["git", "add", "-u"] in argv_calls
        assert ["git", "commit", "-m", "feat: add tests"] in argv_calls
        assert ["git", "add", "."] not in argv_calls
        for argv in argv_calls:
            assert argv[:2] != ["git", "add"] or argv == ["git", "add", "-u"]

    def test_repl_commit_message_gen_passes_empty_tools(self, tmp_path: Path) -> None:
        class RecordingLLM(DummyLLM):
            def __init__(self) -> None:
                super().__init__("feat: recorded")
                self.tools_seen: list[list[dict[str, object]]] = []

            def create_response(
                self,
                messages: list[Message],
                tools: list[dict[str, object]],
                model: str = "gpt-4o-mini",
                on_token: object = None,
                cancel: object = None,
            ) -> LLMResponse:
                self.tools_seen.append(tools)
                return super().create_response(messages, tools, model, on_token, cancel)

        llm = RecordingLLM()
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                with patch(
                    "mini_agent.gitutil.subprocess.run",
                    return_value=CompletedProcess(
                        ["git"], 0, stdout="diff --git a/foo.py\n", stderr=""
                    ),
                ):
                    with patch.object(
                        RichAgentEventListener, "on_permission_ask", return_value=Reply.ONCE
                    ):
                        result = runner.invoke(
                            app,
                            ["--workspace", str(tmp_path)],
                            input="/commit\n/exit\n",
                        )
        assert result.exit_code == 0
        assert llm.tools_seen
        assert llm.tools_seen[0] == []

    def test_repl_diff_command(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/diff\n/exit\n",
                )
                assert result.exit_code == 0

    def test_repl_cost_command(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/cost\n/exit\n",
                )
                assert result.exit_code == 0
                assert "当前会话 Token 用量与累计费用统计" in result.stdout

    def test_one_shot_prompt_flag(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("这是单次执行的回答")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path), "-p", "请问今天天气怎么样？"],
                )
                assert result.exit_code == 0
                assert "这是单次执行的回答" in result.stdout

    def test_rich_listener_events_and_streaming(self) -> None:
        test_console = Console(record=True)
        listener = RichAgentEventListener(console=test_console, verbose=True)

        listener.on_turn_start("hello")
        listener.on_token("你")
        listener.on_token("好")
        listener.on_tool_start("read_file", {"path": "main.py"})
        listener.on_tool_finished(
            "read_file", ToolResult(ok=True, content="code", metadata={"size_bytes": 100})
        )
        listener.on_tool_finished("run_shell", ToolResult(ok=False, error="command failed"))
        listener.on_turn_finished("完整回答")

        out = test_console.export_text()
        assert "你好" in out
        assert "read_file" in out
        assert "成功" in out
        assert "失败" in out

    def test_run_cli_with_injected_client(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("你好，这是测试！")
        with patch("rich.prompt.Prompt.ask", side_effect=["你好", "/exit"]):
            run_cli(workspace=tmp_path, llm_client=dummy_llm)

    def test_repl_double_ctrl_c_exits(self, tmp_path: Path) -> None:
        clock = {"t": 10.0}

        def factory(config: AgentConfig, client: LLMClient, listener: object) -> Agent:
            return Agent(
                config=config,
                llm_client=client,
                listener=listener,  # type: ignore[arg-type]
                monotonic=lambda: clock["t"],
            )

        n = {"i": 0}

        def fake_ask(*args: object, **kwargs: object) -> str:
            n["i"] += 1
            if n["i"] == 1:
                raise KeyboardInterrupt
            if n["i"] == 2:
                clock["t"] = 11.0
                raise KeyboardInterrupt
            return "/exit"

        with patch("rich.prompt.Prompt.ask", side_effect=fake_ask):
            with pytest.raises(typer.Exit) as exc_info:
                run_cli(
                    workspace=tmp_path,
                    llm_client=DummyLLM("x"),
                    agent_factory=factory,
                )
        assert exc_info.value.exit_code == 0

    def test_oneshot_non_tty_ask_exits_2(self, tmp_path: Path, monkeypatch: Any) -> None:
        class ShellLLM(LLMClient):
            def create_response(
                self,
                messages: list[Message],
                tools: list[dict[str, object]],
                model: str = "gpt-4o-mini",
                on_token: object = None,
                cancel: object = None,
            ) -> LLMResponse:
                return LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="run_shell",
                            call_id="call_shell",
                            arguments='{"command": "touch pwned.txt"}',
                        )
                    ]
                )

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(typer.Exit) as exc_info:
            run_cli(workspace=tmp_path, prompt="创建文件", llm_client=ShellLLM())
        assert exc_info.value.exit_code == 2
        assert not (tmp_path / "pwned.txt").exists()

    def test_oneshot_yes_allows_ask(self, tmp_path: Path, monkeypatch: Any) -> None:
        class ShellThenText(LLMClient):
            def __init__(self) -> None:
                self.calls = 0

            def create_response(
                self,
                messages: list[Message],
                tools: list[dict[str, object]],
                model: str = "gpt-4o-mini",
                on_token: object = None,
                cancel: object = None,
            ) -> LLMResponse:
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        function_calls=[
                            FunctionCall(
                                name="run_shell",
                                call_id="call_shell",
                                arguments='{"command": "touch allowed.txt"}',
                            )
                        ]
                    )
                return LLMResponse(text="已创建 allowed.txt")

        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        run_cli(workspace=tmp_path, prompt="创建文件", llm_client=ShellThenText(), yes=True)
        assert (tmp_path / "allowed.txt").exists()
