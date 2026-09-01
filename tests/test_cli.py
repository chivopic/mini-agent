"""Unit tests for Typer CLI and REPL interface."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from typer.testing import CliRunner

from mini_agent.cli import (
    RichAgentEventListener,
    app,
    render_banner,
    render_help,
    render_sessions_table,
    run_cli,
)
from mini_agent.llm import LLMClient, LLMResponse
from mini_agent.models import ToolResult
from mini_agent.session import SessionData, SessionMeta, save_session

runner = CliRunner()


class DummyLLM(LLMClient):
    """Dummy LLM for CLI testing."""

    def __init__(self, answer: str = "这是回答") -> None:
        self.answer = answer

    def create_response(
        self,
        history: list[dict[str, object]],
        tools: list[dict[str, object]],
        model: str = "gpt-4o-mini",
        on_token: object = None,
    ) -> LLMResponse:
        return LLMResponse(text=self.answer)


class TestCliCommands:
    """Test CLI options, arguments, and validation."""

    def test_cli_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Typer/Rich may collapse or reflow individual option names depending on terminal width
        # and version. Test stable user-facing help markers here; option behavior is covered by
        # the functional CLI tests below.
        assert "mini-agent" in result.stdout.lower()
        assert "--help" in result.stdout

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
                assert "deepseek-v4-pro" in result.stdout

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
