"""Slash command registry for the REPL."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from mini_agent.agent import Agent
from mini_agent.config import AppConfig
from mini_agent.cost import UsageStats, set_custom_pricing
from mini_agent.gitutil import git_add_u, git_commit, git_diff, git_diff_head, list_untracked
from mini_agent.llm import LLMClient
from mini_agent.models import PermissionClass
from mini_agent.permission import Decision, PermissionRequest, Reply
from mini_agent.providers import get_provider_preset
from mini_agent.render import (
    NonInteractiveAskError,
    render_banner,
    render_cost_table,
    render_help_tables,
    render_pricing_list_table,
    render_providers_table,
    render_sessions_table,
)
from mini_agent.session import SessionData, load_session, save_session


class DispatchResult(StrEnum):
    CONTINUE = "continue"
    EXIT = "exit"


@dataclass(frozen=True)
class SlashCommand:
    name: str
    usage: str
    description: str
    handler: Callable[[CommandContext, str], DispatchResult]
    aliases: tuple[str, ...] = ()


@dataclass
class CommandContext:
    agent: Agent
    console: Console
    app_config: AppConfig
    commands: CommandRegistry
    make_client: Callable[..., LLMClient]


class CommandRegistry:
    """Named slash commands. /help is generated from registration order."""

    def __init__(self) -> None:
        self._by_name: dict[str, SlashCommand] = {}
        self._order: list[str] = []

    def register(self, command: SlashCommand) -> None:
        primary = command.name.lstrip("/").lower()
        self._by_name[primary] = command
        if primary not in self._order:
            self._order.append(primary)
        for alias in command.aliases:
            self._by_name[alias.lstrip("/").lower()] = command

    def get(self, name: str) -> SlashCommand | None:
        return self._by_name.get(name.lstrip("/").lower())

    def list(self) -> list[SlashCommand]:
        return [self._by_name[key] for key in self._order]

    def dispatch(self, user_input: str, ctx: CommandContext) -> DispatchResult | None:
        if not user_input.startswith("/"):
            return None
        token, _, rest = user_input.partition(" ")
        name = token[1:].lower()
        if not name:
            ctx.console.print("[yellow]未知指令。输入 /help 查看可用指令。[/yellow]\n")
            return DispatchResult.CONTINUE
        command = self.get(name)
        if command is None:
            ctx.console.print(f"[yellow]未知指令: /{name}。输入 /help 查看可用指令。[/yellow]\n")
            return DispatchResult.CONTINUE
        return command.handler(ctx, rest.strip())


def help_rows(registry: CommandRegistry) -> list[tuple[str, str]]:
    return [(command.usage, command.description) for command in registry.list()]


def apply_loaded_session(agent: Agent, loaded: SessionData) -> None:
    """Swap session/messages and rebuild session_usage from meta.total_* tokens."""
    restore = getattr(agent.permission, "restore", None)
    if restore is not None:
        restore(loaded.permission_memory or [])
    agent.session = loaded
    agent.messages = loaded.messages
    agent.session_usage = UsageStats(
        prompt_tokens=loaded.meta.total_prompt_tokens,
        completion_tokens=loaded.meta.total_completion_tokens,
        total_tokens=loaded.meta.total_prompt_tokens + loaded.meta.total_completion_tokens,
    )


def redact_secrets(value: object) -> object:
    """Replace API keys and similar secrets with ********. Empty values become (unset)."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            if _is_secret_key(str(key)):
                redacted[key] = "********" if nested not in (None, "") else "(unset)"
            else:
                redacted[key] = redact_secrets(nested)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return "api_key" in lowered or lowered.endswith("_secret") or lowered.endswith("_password")


def _confirm_permission(agent: Agent, req: PermissionRequest) -> bool:
    """Evaluate GIT/slash permission; one panel via on_permission_ask when ASK."""
    decision = agent.permission.check(req)
    if decision == Decision.ALLOW:
        return True
    if decision == Decision.DENY:
        return False
    reply = Reply.REJECT
    listener = agent.listener
    if listener is not None:
        reply = listener.on_permission_ask(req)
    if reply == Reply.ALWAYS:
        agent.permission.remember(req, reply)
        snapshot = getattr(agent.permission, "snapshot", None)
        if snapshot is not None:
            agent.session.permission_memory = snapshot()
            save_session(agent.session)
        return True
    return reply == Reply.ONCE


def cmd_help(ctx: CommandContext, args: str) -> DispatchResult:
    tool_rows = [(tool.name, tool.description) for tool in ctx.agent.registry.list()]
    render_help_tables(ctx.console, help_rows(ctx.commands), tool_rows)
    return DispatchResult.CONTINUE


def cmd_provider(ctx: CommandContext, args: str) -> DispatchResult:
    if args:
        pname = args.strip()
        preset = get_provider_preset(pname)
        if preset:
            try:
                new_client = ctx.make_client(base_url=preset.base_url)
            except Exception as exc:
                ctx.console.print(f"[red]✗ 切换服务商失败，模型未更改: {exc}[/red]\n")
                return DispatchResult.CONTINUE
            ctx.agent.llm_client = new_client
            ctx.agent.config.model = preset.default_model
            ctx.agent.session.meta.model = preset.default_model
            ctx.console.print(
                f"[green]✔ 已成功切换服务商:[/green] "
                f"[bold cyan]{preset.display_name}[/bold cyan] "
                f"[dim](模型: {preset.default_model})[/dim]\n"
            )
        else:
            ctx.console.print(f"[red]✗ 未知服务商预设: '{pname}'[/red]")
            render_providers_table(ctx.console)
    else:
        render_providers_table(ctx.console)
    return DispatchResult.CONTINUE


def cmd_cost(ctx: CommandContext, args: str) -> DispatchResult:
    cost_parts = args.split()
    if not cost_parts:
        render_cost_table(ctx.console, ctx.agent)
    elif cost_parts[0] == "list":
        render_pricing_list_table(ctx.console)
    elif cost_parts[0] == "set" and len(cost_parts) >= 4:
        target_m = cost_parts[1]
        try:
            p_in = float(cost_parts[2])
            p_out = float(cost_parts[3])
            set_custom_pricing(target_m, p_in, p_out)
            ctx.console.print(
                f"[green]✔ 已成功更新模型 '{target_m}' 费率:[/green] "
                f"输入 ¥{p_in}/M | 输出 ¥{p_out}/M\n"
            )
        except ValueError:
            ctx.console.print(
                "[red]✗ 价格格式错误。用法: /cost set <模型> <输入价> <输出价>[/red]\n"
            )
    else:
        ctx.console.print(
            "[yellow]用法:\n"
            "  /cost      - 查看当前会话用量与费用看板\n"
            "  /cost list - 查看所有已配置模型的费率表\n"
            "  /cost set <模型> <输入价> <输出价> - 自定义模型费率 (元/1M)[/yellow]\n"
        )
    return DispatchResult.CONTINUE


def cmd_diff(ctx: CommandContext, args: str) -> DispatchResult:
    try:
        res = git_diff(ctx.agent.config.workspace_root)
        if res.stdout.strip():
            ctx.console.print(
                Syntax(
                    res.stdout,
                    "diff",
                    theme="monokai",
                    line_numbers=False,
                )
            )
        else:
            ctx.console.print("[dim green]✔ 工作区代码干净，无未暂存的代码改动。[/dim green]\n")
    except Exception as exc:
        ctx.console.print(f"[red]✗ 执行 git diff 失败: {exc}[/red]\n")
    return DispatchResult.CONTINUE


def cmd_commit(ctx: CommandContext, args: str) -> DispatchResult:
    msg = args.strip()
    if not msg:
        try:
            diff_res = git_diff_head(ctx.agent.config.workspace_root)
            diff_text = diff_res.stdout.strip()
            if not diff_text:
                ctx.console.print("[yellow]当前没有代码变更可提交。[/yellow]\n")
                return DispatchResult.CONTINUE
            gen_prompt = (
                "请根据以下 git diff 生成一行标准规范的 Conventional Commit 信息"
                "（例如 feat: ... 或 fix: ...），仅直接返回 Commit 文本本身：\n"
                f"```diff\n{diff_text[:8000]}\n```"
            )
            msg = ctx.agent.step(gen_prompt, extra_tools=[]).strip().strip("`'\"")
            if not msg:
                ctx.console.print("[red]✗ 生成提交信息失败: 模型未返回文本[/red]\n")
                return DispatchResult.CONTINUE
        except Exception as exc:
            ctx.console.print(f"[red]✗ 生成提交信息失败: {exc}[/red]\n")
            return DispatchResult.CONTINUE

    if msg:
        try:
            untracked = list_untracked(ctx.agent.config.workspace_root)
            untracked_block = (
                "\n".join(f"  {path}" for path in untracked) if untracked else "  （无）"
            )
            req = PermissionRequest(
                cls=PermissionClass.GIT,
                tool="/commit",
                resource=msg,
                pattern="git:commit",
                reason=(
                    "将执行：\n"
                    "  git add -u\n"
                    f"  git commit -m {msg!r}\n"
                    "未跟踪文件（不会被加入）：\n"
                    f"{untracked_block}"
                ),
            )
            if not _confirm_permission(ctx.agent, req):
                ctx.console.print("[yellow]已取消 Git 提交。[/yellow]\n")
                return DispatchResult.CONTINUE
            add_res = git_add_u(ctx.agent.config.workspace_root)
            if add_res.returncode != 0:
                err = (add_res.stderr or add_res.stdout).strip() or add_res.returncode
                ctx.console.print(f"[red]✗ Git 提交失败: {err}[/red]\n")
                return DispatchResult.CONTINUE
            commit_res = git_commit(ctx.agent.config.workspace_root, msg)
            if commit_res.returncode != 0:
                err = (commit_res.stderr or commit_res.stdout).strip()
                err = err or commit_res.returncode
                ctx.console.print(f"[red]✗ Git 提交失败: {err}[/red]\n")
                return DispatchResult.CONTINUE
            ctx.console.print(f"[green]✔ Git 提交成功:[/green] [bold cyan]{msg}[/bold cyan]\n")
        except NonInteractiveAskError as exc:
            ctx.console.print("[red]✗ 需要确认才能提交。非交互模式请传递 -y/--yes。[/red]\n")
            raise typer.Exit(code=2) from exc
        except Exception as exc:
            ctx.console.print(f"[red]✗ Git 提交失败: {exc}[/red]\n")
    return DispatchResult.CONTINUE


def cmd_sessions(ctx: CommandContext, args: str) -> DispatchResult:
    render_sessions_table(ctx.console, ctx.agent.config.workspace_root)
    return DispatchResult.CONTINUE


def cmd_resume(ctx: CommandContext, args: str) -> DispatchResult:
    if args:
        target_id = args.strip()
        loaded = load_session(target_id)
        if loaded:
            apply_loaded_session(ctx.agent, loaded)
            ctx.console.print(
                f"[green]✔ 已成功恢复会话:[/green] [bold cyan]{target_id}[/bold cyan] "
                f"[dim]({loaded.meta.title}, {len(loaded.messages)} 条记录)[/dim]\n"
            )
        else:
            ctx.console.print(f"[red]✗ 未找到指定的会话 ID: '{target_id}'[/red]\n")
    else:
        ctx.console.print(
            "[yellow]用法: /resume <Session_ID> (可通过 /sessions 查看 ID)[/yellow]\n"
        )
    return DispatchResult.CONTINUE


def cmd_new(ctx: CommandContext, args: str) -> DispatchResult:
    new_id = ctx.agent.reset_session()
    ctx.console.print(
        f"[green]✔ 已重置上下文，开启全新会话:[/green] [bold cyan]{new_id}[/bold cyan]\n"
    )
    return DispatchResult.CONTINUE


def cmd_clear(ctx: CommandContext, args: str) -> DispatchResult:
    ctx.console.clear()
    render_banner(
        ctx.console,
        ctx.agent.config.workspace_root,
        ctx.agent.config.model,
        session_id=ctx.agent.session.meta.session_id,
    )
    return DispatchResult.CONTINUE


def cmd_model(ctx: CommandContext, args: str) -> DispatchResult:
    if args:
        new_model = args.strip()
        ctx.agent.config.model = new_model
        ctx.agent.session.meta.model = new_model
        ctx.console.print(
            f"[green]✔ 已切换当前模型为:[/green] [bold cyan]{new_model}[/bold cyan]\n"
        )
    else:
        ctx.console.print(
            f"[dim]当前会话模型:[/dim] [bold cyan]{ctx.agent.config.model}[/bold cyan]\n"
        )
    return DispatchResult.CONTINUE


def cmd_config(ctx: CommandContext, args: str) -> DispatchResult:
    payload: dict[str, Any] = {
        "workspace": ctx.agent.config.workspace_root.as_posix(),
        "provider": ctx.app_config.provider.model_dump(mode="json"),
        "limits": ctx.app_config.limits.model_dump(mode="json"),
        "permission": ctx.app_config.permission.model_dump(mode="json"),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    }
    text = json.dumps(redact_secrets(payload), indent=2, ensure_ascii=False)
    ctx.console.print(
        Panel(
            Syntax(text, "json", theme="monokai", word_wrap=True),
            title="当前配置（密钥已打码）",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )
    ctx.console.print()
    return DispatchResult.CONTINUE


def cmd_cancel(ctx: CommandContext, args: str) -> DispatchResult:
    # Idle only: calling request_cancel() here would set _last_sigint_at and
    # make the next Ctrl-C within 2s exit the process.
    ctx.console.print(
        "[yellow]当前没有正在执行的回合。[/yellow] "
        "[dim]模型回复或工具执行期间请按 Ctrl-C 取消。[/dim]\n"
    )
    return DispatchResult.CONTINUE


def cmd_exit(ctx: CommandContext, args: str) -> DispatchResult:
    ctx.console.print("[dim]👋 再见！[/dim]")
    return DispatchResult.EXIT


def build_command_registry() -> CommandRegistry:
    registry = CommandRegistry()
    for command in (
        SlashCommand("help", "/help", "显示快捷指令与 Agent 工具能力说明", cmd_help),
        SlashCommand(
            "provider",
            "/provider [name]",
            "切换或查看各大模型服务商预设 (DeepSeek V4, Ollama 等)",
            cmd_provider,
        ),
        SlashCommand(
            "cost",
            "/cost [set/list]",
            "查看 Token 消耗看板，或自定义/查看模型费率表",
            cmd_cost,
        ),
        SlashCommand("diff", "/diff", "查看当前工作区的所有 Git 代码改动", cmd_diff),
        SlashCommand("commit", "/commit [msg]", "智能生成或执行 Git 提交", cmd_commit),
        SlashCommand("sessions", "/sessions", "查看当前工作区的所有历史会话", cmd_sessions),
        SlashCommand("resume", "/resume <id>", "切换并恢复指定历史会话", cmd_resume),
        SlashCommand("new", "/new", "重置并开启全新会话", cmd_new),
        SlashCommand("clear", "/clear", "清屏并重新展示顶部状态 Banner", cmd_clear),
        SlashCommand(
            "model",
            "/model [name]",
            "查看或临时切换当前模型 (如 /model deepseek-v4-flash)",
            cmd_model,
        ),
        SlashCommand("config", "/config", "显示当前合并后的配置（密钥已打码）", cmd_config),
        SlashCommand(
            "cancel",
            "/cancel",
            "取消当前回合（执行中请使用 Ctrl-C）",
            cmd_cancel,
        ),
        SlashCommand(
            "exit",
            "/exit, /quit",
            "退出当前 mini-agent 会话",
            cmd_exit,
            aliases=("quit",),
        ),
    ):
        registry.register(command)
    return registry
