"""Tests for the slash command registry and prompt_toolkit history helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from mini_agent.agent import Agent
from mini_agent.commands import (
    CommandContext,
    DispatchResult,
    SlashCommand,
    apply_loaded_session,
    build_command_registry,
    cmd_cancel,
    cmd_config,
    cmd_new,
    help_rows,
    redact_secrets,
)
from mini_agent.config import defaults
from mini_agent.llm import LLMClient, LLMResponse
from mini_agent.messages import Message
from mini_agent.models import AgentConfig
from mini_agent.render import format_tool_call, render_help_tables
from mini_agent.repl import (
    create_prompt_session,
    default_history_path,
    normalize_repl_input,
    read_repl_line,
)
from mini_agent.session import SessionData, SessionMeta
from mini_agent.tools import default_registry


class DummyLLM(LLMClient):
    def create_response(
        self,
        messages: list[Message],
        tools: list[dict[str, object]],
        model: str = "gpt-4o-mini",
        on_token: object = None,
        cancel: object = None,
    ) -> LLMResponse:
        return LLMResponse(text="ok")


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        config=AgentConfig(workspace_root=tmp_path),
        llm_client=DummyLLM(),
    )


def _ctx(tmp_path: Path, agent: Agent | None = None) -> CommandContext:
    registry = build_command_registry()
    return CommandContext(
        agent=agent or _agent(tmp_path),
        console=Console(record=True),
        app_config=defaults(),
        commands=registry,
        make_client=lambda **kwargs: DummyLLM(),
    )


def test_help_rows_generated_from_registry() -> None:
    registry = build_command_registry()
    registry.register(
        SlashCommand(
            name="ping",
            usage="/ping",
            description="unique-help-ping-command",
            handler=lambda ctx, args: DispatchResult.CONTINUE,
        )
    )
    rows = help_rows(registry)
    usages = [usage for usage, _desc in rows]
    descriptions = [desc for _usage, desc in rows]
    assert "/help" in usages
    assert "/cancel" in usages
    assert "/config" in usages
    assert "/ping" in usages
    assert "unique-help-ping-command" in descriptions

    console = Console(record=True)
    render_help_tables(console, rows, [("list_files", "列出文件")])
    out = console.export_text()
    assert "/ping" in out
    assert "unique-help-ping-command" in out
    assert "list_files" in out


def test_unknown_slash_command_does_not_crash(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = ctx.commands.dispatch("/not-a-real-command", ctx)
    assert result is DispatchResult.CONTINUE
    assert "未知指令" in ctx.console.export_text()


def test_redact_secrets_masks_api_keys() -> None:
    payload = {
        "provider": {"model": "gpt-4o-mini", "api_key": "sk-live"},
        "openai_api_key": "sk-secret-value",
        "limits": {"max_tool_rounds": 32},
    }
    redacted = redact_secrets(payload)
    assert redacted["provider"]["api_key"] == "********"
    assert redacted["openai_api_key"] == "********"
    assert redacted["provider"]["model"] == "gpt-4o-mini"
    assert redacted["limits"]["max_tool_rounds"] == 32
    assert "sk-live" not in str(redacted)
    assert "sk-secret-value" not in str(redacted)


def test_cmd_config_redacts_env_api_key(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    ctx = _ctx(tmp_path)
    assert cmd_config(ctx, "") is DispatchResult.CONTINUE
    out = ctx.console.export_text()
    assert "sk-super-secret" not in out
    assert "********" in out
    assert "gpt-4o-mini" in out or "max_tool_rounds" in out


def test_apply_loaded_session_reloads_usage(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    now = datetime.now().isoformat()
    loaded = SessionData(
        meta=SessionMeta(
            session_id="s_usage",
            workspace_root=tmp_path.resolve().as_posix(),
            created_at=now,
            updated_at=now,
            model="gpt-4o-mini",
            title="用量",
            turn_count=3,
            total_prompt_tokens=40,
            total_completion_tokens=15,
        )
    )
    apply_loaded_session(agent, loaded)
    assert agent.session.meta.session_id == "s_usage"
    assert agent.session_usage.prompt_tokens == 40
    assert agent.session_usage.completion_tokens == 15
    assert agent.session_usage.total_tokens == 55


def test_cmd_new_prints_stored_session_id(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert cmd_new(ctx, "") is DispatchResult.CONTINUE
    stored = ctx.agent.session.meta.session_id
    assert stored in ctx.console.export_text()


def test_cmd_cancel_idle_does_not_request_cancel(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert ctx.agent._last_sigint_at is None
    assert not ctx.agent._cancel.is_set()
    assert cmd_cancel(ctx, "") is DispatchResult.CONTINUE
    assert ctx.agent._last_sigint_at is None
    assert not ctx.agent._cancel.is_set()
    assert not ctx.agent._double_sigint
    out = ctx.console.export_text()
    assert "当前没有正在执行的回合" in out
    assert "Ctrl-C" in out


def test_format_call_known_and_unknown_tools() -> None:
    registry = default_registry()
    assert format_tool_call("read_file", {"path": "main.py"}, registry) == "read_file path=main.py"
    generic = format_tool_call("mystery_tool", {"foo": "bar"}, registry)
    assert generic == "mystery_tool foo='bar'"


def test_default_history_path_is_under_mini_agent_home() -> None:
    assert default_history_path() == Path.home() / ".mini-agent" / "history"


def test_create_prompt_session_uses_file_history(tmp_path: Path) -> None:
    history_file = tmp_path / "history"
    session = create_prompt_session(history_file=history_file)
    assert isinstance(session.history, FileHistory)


def test_normalize_repl_input_keeps_indent_drops_surrounding_newlines() -> None:
    pasted = "    def foo():\n        pass\n"
    assert normalize_repl_input(pasted) == "    def foo():\n        pass"
    assert normalize_repl_input("   \n\t  ") == ""
    assert normalize_repl_input("\nhello\n") == "hello"


def test_prompt_toolkit_multiline_paste_is_one_turn(tmp_path: Path) -> None:
    with create_pipe_input() as pipe:
        session = create_prompt_session(
            history_file=tmp_path / "history",
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text("\x1b[200~    def foo():\n        pass\x1b[201~\n")
        result = read_repl_line(Console(), session)
        assert result == "    def foo():\n        pass"


def test_prompt_toolkit_up_arrow_recalls_history(tmp_path: Path) -> None:
    history_file = tmp_path / "history"
    history_file.write_text("# 2020-01-01 00:00:00.000000\n+previous command\n", encoding="utf-8")
    with create_pipe_input() as pipe:
        session = create_prompt_session(
            history_file=history_file,
            input=pipe,
            output=DummyOutput(),
        )
        pipe.send_text("\x1b[A\n")
        result = read_repl_line(Console(), session)
        assert result == "previous command"
