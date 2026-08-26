"""prompt_toolkit input loop: history, SIGINT, slash dispatch or agent.step."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.prompt import Prompt

from mini_agent.agent import Agent
from mini_agent.commands import (
    CommandContext,
    DispatchResult,
    build_command_registry,
)
from mini_agent.config import AppConfig
from mini_agent.llm import LLMClient, LLMError
from mini_agent.render import NonInteractiveAskError, render_banner


def default_history_path() -> Path:
    return Path.home() / ".mini-agent" / "history"


def use_prompt_toolkit() -> bool:
    # pytest 下 stdin 仍可能 isatty；避免抢真实终端并写入 ~/.mini-agent/history。
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def create_prompt_session(
    *,
    history_file: Path | None = None,
    **session_kwargs: Any,
) -> PromptSession[str]:
    """File-backed history; Enter submits. Bracketed paste keeps newlines as one turn."""
    path = history_file if history_file is not None else default_history_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if history_file is None:
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
    return PromptSession[str](
        history=FileHistory(str(path)),
        enable_history_search=True,
        multiline=False,
        **session_kwargs,
    )


def normalize_repl_input(raw: str) -> str:
    """Keep interior indent; only drop surrounding newlines. Whitespace-only is empty."""
    text = raw.strip("\n\r")
    if not text.strip():
        return ""
    return text


def read_repl_line(
    console: Console,
    session: PromptSession[str] | None = None,
) -> str:
    """Read one user turn. prompt_toolkit when a session is provided; Rich Prompt otherwise."""
    if session is not None:
        raw = str(session.prompt(HTML("<ansicyan><b>&gt; </b></ansicyan>")))
    else:
        raw = Prompt.ask("[bold cyan]>[/bold cyan]", console=console)
    return normalize_repl_input(raw)


def _handle_keyboard_interrupt(agent: Agent, console: Console, *, during_step: bool) -> None:
    agent.request_cancel()
    if agent._double_sigint:
        console.print("\n[dim]👋 再见！[/dim]")
        raise typer.Exit(code=0) from None
    if during_step:
        console.print("\n[yellow]已取消当前回合[/yellow]")
    else:
        console.print("\n[yellow]已取消当前输入[/yellow]")


def repl_loop(
    agent: Agent,
    console: Console,
    *,
    app_config: AppConfig,
    make_client: Callable[..., LLMClient],
) -> None:
    """Main interactive REPL loop with Antigravity styling."""
    render_banner(
        console,
        agent.config.workspace_root,
        agent.config.model,
        session_id=agent.session.meta.session_id,
    )

    commands = build_command_registry()
    ctx = CommandContext(
        agent=agent,
        console=console,
        app_config=app_config,
        commands=commands,
        make_client=make_client,
    )

    session: PromptSession[str] | None = None
    if use_prompt_toolkit():
        try:
            session = create_prompt_session()
        except OSError:
            session = None

    while True:
        try:
            user_input = read_repl_line(console, session)
        except KeyboardInterrupt:
            _handle_keyboard_interrupt(agent, console, during_step=False)
            continue
        except EOFError:
            console.print("\n[dim]👋 再见！[/dim]")
            break

        if not user_input:
            continue

        dispatched = commands.dispatch(user_input, ctx)
        if dispatched is DispatchResult.EXIT:
            break
        if dispatched is DispatchResult.CONTINUE:
            continue

        try:
            agent.step(user_input)
        except KeyboardInterrupt:
            _handle_keyboard_interrupt(agent, console, during_step=True)
        except NonInteractiveAskError as exc:
            console.print(
                "\n[bold red]需要确认才能执行该操作。非交互模式请传递 -y/--yes。[/bold red]\n"
            )
            raise typer.Exit(code=2) from exc
        except LLMError as exc:
            console.print(f"\n[bold red]LLM 错误[/bold red]: {exc}\n")
        except Exception as exc:
            console.print(f"\n[bold red]执行错误[/bold red]: {exc}\n")
