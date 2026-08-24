"""Typer CLI: assembly, one-shot vs REPL entry."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from mini_agent.agent import Agent
from mini_agent.commands import build_command_registry, help_rows
from mini_agent.config import ConfigError, load_app_config
from mini_agent.events import AgentEventListener
from mini_agent.llm import LLMClient, OpenAIChatCompletionsClient
from mini_agent.models import AgentConfig
from mini_agent.permission import DefaultPermissionService
from mini_agent.render import (
    NonInteractiveAskError,
    RichAgentEventListener,
    render_banner,
    render_help_tables,
    render_sessions_table,
)
from mini_agent.repl import repl_loop
from mini_agent.session import SessionData, get_latest_session, load_session
from mini_agent.tools import default_registry

app = typer.Typer(
    name="mini-agent",
    help="mini-agent: 本地终端 AI 编程助手",
    add_completion=False,
)
console = Console()

__all__ = [
    "NonInteractiveAskError",
    "RichAgentEventListener",
    "app",
    "console",
    "main",
    "render_banner",
    "render_help",
    "render_sessions_table",
    "run_cli",
]


def render_help(console: Console) -> None:
    """Render slash-command and tool help from the default registries."""
    commands = build_command_registry()
    tool_rows = [(tool.name, tool.description) for tool in default_registry().list()]
    render_help_tables(console, help_rows(commands), tool_rows)


def _make_llm_client(**kwargs: Any) -> LLMClient:
    return OpenAIChatCompletionsClient(**kwargs)


def run_cli(
    workspace: Path | None = None,
    model: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
    continue_session: bool = False,
    session_id: str | None = None,
    verbose: bool = False,
    yes: bool = False,
    config_path: Path | None = None,
    agent_factory: Callable[[AgentConfig, LLMClient, AgentEventListener], Agent] | None = None,
    llm_client: LLMClient | None = None,
) -> None:
    """Core logic to run the CLI in interactive or one-shot mode."""
    target_workspace = (workspace or Path.cwd()).resolve()
    if not target_workspace.exists():
        console.print(f"[bold red]错误[/bold red]: 指定的工作区路径不存在: '{target_workspace}'")
        raise typer.Exit(code=1)
    if not target_workspace.is_dir():
        console.print(f"[bold red]错误[/bold red]: 指定的工作区路径不是目录: '{target_workspace}'")
        raise typer.Exit(code=1)

    try:
        app_config, config_warnings = load_app_config(
            target_workspace,
            cli_model=model,
            cli_base_url=base_url,
            user_config_path=config_path,
            user_config_required=config_path is not None,
        )
    except ConfigError as exc:
        console.print(f"[bold red]错误[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc

    for warning in config_warnings:
        console.print(f"[yellow]{warning}[/yellow]")

    effective_base_url = app_config.provider.base_url

    # Verify API key if default client is used. TOML api_key keys are ignored.
    if llm_client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            console.print(
                "[bold red]错误[/bold red]: 未检测到 OPENAI_API_KEY 环境变量。\n"
                "请先设置您的 API Key，例如在终端执行：\n"
                "  [cyan]export OPENAI_API_KEY='sk-...'[/cyan]\n"
                "若使用 DeepSeek，可同时配置：\n"
                "  [cyan]export OPENAI_BASE_URL='https://api.deepseek.com'[/cyan]\n"
                "  [cyan]export MINI_AGENT_MODEL='deepseek-v4'[/cyan]"
            )
            raise typer.Exit(code=1)
        client: LLMClient = OpenAIChatCompletionsClient(
            api_key=api_key,
            base_url=effective_base_url,
        )
    else:
        client = llm_client

    config = app_config.to_agent_config(target_workspace)
    listener = RichAgentEventListener(
        console=console,
        verbose=verbose,
        interactive=sys.stdin.isatty(),
    )

    # Handle session loading
    loaded_session: SessionData | None = None
    if session_id:
        loaded_session = load_session(session_id)
        if not loaded_session:
            console.print(f"[bold red]错误[/bold red]: 未找到指定的会话 ID: '{session_id}'")
            raise typer.Exit(code=1)
    elif continue_session:
        loaded_session = get_latest_session(target_workspace)
        if loaded_session:
            console.print(
                f"[dim]已自动恢复上一次会话: {loaded_session.meta.session_id} "
                f"({loaded_session.meta.title})[/dim]"
            )

    if agent_factory is not None:
        agent = agent_factory(config, client, listener)
    else:
        agent = Agent(
            config=config,
            llm_client=client,
            listener=listener,
            session=loaded_session,
            permission=DefaultPermissionService(
                class_defaults=app_config.permission_class_defaults(),
                auto_allow_ask=yes,
            ),
        )

    registry = getattr(agent, "registry", None)
    if registry is not None:
        listener.registry = registry

    # One-shot non-interactive execution
    if prompt:
        try:
            agent.step(prompt)
        except NonInteractiveAskError as exc:
            console.print(
                "[bold red]需要确认才能执行该操作。非交互模式请传递 -y/--yes。[/bold red]"
            )
            raise typer.Exit(code=2) from exc
        except Exception as exc:
            console.print(f"[bold red]执行失败[/bold red]: {exc}")
            raise typer.Exit(code=1) from exc
        return

    # Interactive REPL mode
    repl_loop(agent, console=console, app_config=app_config, make_client=_make_llm_client)


@app.command()
def main(
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help="目标工作区根目录路径（缺省为当前工作目录）",
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            "-p",
            help="单次非交互执行模式：直接执行指定任务并退出",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="覆盖本次会话的模型名称（如 deepseek-v4 或 gpt-4o）",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            "-b",
            help="自定义 API Base URL（如 https://api.deepseek.com）",
        ),
    ] = None,
    continue_session: Annotated[
        bool,
        typer.Option(
            "--continue",
            "-c",
            help="恢复当前工作区的最近一次历史会话",
        ),
    ] = False,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session",
            "-s",
            help="指定要恢复的历史会话 ID",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="显示诊断与执行详细信息",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="将权限询问视为允许（黑名单命令仍会拒绝）",
        ),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="用户配置文件路径（默认 ~/.mini-agent/config.toml）",
        ),
    ] = None,
) -> None:
    """启动 mini-agent 交互式 REPL 或执行单次任务。"""
    run_cli(
        workspace=workspace,
        prompt=prompt,
        model=model,
        base_url=base_url,
        continue_session=continue_session,
        session_id=session_id,
        verbose=verbose,
        yes=yes,
        config_path=config_path,
    )
