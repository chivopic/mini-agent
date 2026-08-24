"""Rich rendering for banners, tables, and agent events."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from mini_agent import __version__
from mini_agent.cost import (
    UsageStats,
    format_cost_cny,
    get_model_pricing,
    load_pricing_table,
)
from mini_agent.events import (
    AgentEvent,
    AgentEventListener,
    CompactionNotice,
    ModelStarted,
    TokenDelta,
    ToolFinished,
    ToolStarted,
    TurnCancelled,
    TurnFailed,
    TurnFinished,
    TurnStarted,
    UsageReported,
)
from mini_agent.models import ToolResult
from mini_agent.permission import PermissionRequest, Reply
from mini_agent.providers import list_provider_presets
from mini_agent.session import list_sessions
from mini_agent.tools.registry import ToolRegistry


class NonInteractiveAskError(Exception):
    """ASK in non-interactive mode without --yes; CLI exits with code 2."""


def format_tool_call(
    name: str,
    arguments: dict[str, Any],
    registry: ToolRegistry | None = None,
) -> str:
    """UI summary for a tool invocation. Known tools use format_call; others stay generic."""
    tool = registry.get(name) if registry is not None else None
    if tool is not None:
        try:
            inp = tool.input_model.model_validate(arguments)
            return tool.format_call(inp)
        except Exception:
            pass
    if not arguments:
        return name
    preview = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{name} {preview}"


def render_banner(
    console: Console,
    workspace: Any,
    model: str,
    session_id: str | None = None,
) -> None:
    """Render sleek Antigravity-style header banner."""
    content = Text()
    content.append("✦ ", style="bold cyan")
    content.append("MINI-AGENT", style="bold white")
    content.append(f"  v{__version__}\n", style="dim")
    content.append("📁 工作区: ", style="bold bright_black")
    content.append(f"{workspace.as_posix()}\n", style="white")
    content.append("⚡ 模型:   ", style="bold bright_black")
    content.append(f"{model}\n", style="bright_cyan")
    if session_id:
        content.append("💬 会话:   ", style="bold bright_black")
        content.append(f"{session_id}\n", style="dim")
    content.append("💡 提示:   ", style="bold bright_black")
    content.append("输入问题开始协作，使用 ", style="dim")
    content.append("/help", style="bold cyan")
    content.append(" 查看指令，", style="dim")
    content.append("/provider", style="bold cyan")
    content.append(" 切换模型，", style="dim")
    content.append("/cost", style="bold cyan")
    content.append(" 费用看板，", style="dim")
    content.append("/exit", style="bold cyan")
    content.append(" 退出", style="dim")

    console.print(
        Panel(
            content,
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2),
        )
    )


class RichAgentEventListener(AgentEventListener):
    """Rich terminal event listener with structured step cards and token streaming."""

    def __init__(
        self,
        console: Console,
        verbose: bool = False,
        *,
        interactive: bool = True,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.console = console
        self.verbose = verbose
        self.interactive = interactive
        self.registry = registry
        self._streamed_any = False

    def on_event(self, event: AgentEvent) -> None:
        if isinstance(event, TurnStarted):
            self.on_turn_start(event.user_input)
        elif isinstance(event, TokenDelta):
            self.on_token(event.token)
        elif isinstance(event, ModelStarted):
            self.on_model_start()
        elif isinstance(event, ToolStarted):
            self.on_tool_start(event.name, event.arguments)
        elif isinstance(event, ToolFinished):
            self.on_tool_finished(event.name, event.result)
        elif isinstance(event, UsageReported):
            self.on_usage(event.usage, event.cost_cny, event.model)
        elif isinstance(event, TurnFinished):
            self.on_turn_finished(event.response)
        elif isinstance(event, TurnCancelled):
            if self._streamed_any:
                self.console.print("\n")
                self._streamed_any = False
            self.console.print(f"[yellow]{event.response}[/yellow]\n")
        elif isinstance(event, TurnFailed):
            self.console.print(f"\n[bold red]执行错误[/bold red]: {event.error}\n")
        elif isinstance(event, CompactionNotice) and self.verbose:
            self.console.print(f"  [dim]上下文已压缩（丢弃 {event.dropped_count}）[/dim]")

    def on_turn_start(self, user_input: str) -> None:
        self._streamed_any = False

    def on_token(self, token: str) -> None:
        if not self._streamed_any:
            self.console.print("\n[bold cyan]🤖 Mini-Agent[/bold cyan]")
            self._streamed_any = True
        self.console.print(token, end="", markup=False)

    def on_model_start(self) -> None:
        if self.verbose and not self._streamed_any:
            self.console.print("  [dim cyan]⏺ 正在请求模型思考...[/dim cyan]")

    def on_tool_start(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if self._streamed_any:
            self.console.print()
            self._streamed_any = False
        summary = format_tool_call(tool_name, arguments, self.registry)
        self.console.print(f"  [bold cyan]⚡ Tool: {summary}[/bold cyan]")

    def on_permission_ask(self, req: PermissionRequest) -> Reply:
        if not self.interactive:
            raise NonInteractiveAskError(req)
        details = req.reason or f"{req.tool}: {req.resource}"
        self.console.print(
            Panel(
                f"[yellow]需要授权才能继续：[/yellow]\n\n"
                f"{details}\n\n"
                f"[dim]y=一次 / a=本会话 always / n=拒绝（默认拒绝）[/dim]",
                title="[bold yellow]⚠️  安全确认 (Security Confirmation)[/bold yellow]",
                box=box.ROUNDED,
                border_style="yellow",
                padding=(0, 2),
            )
        )
        answer = (
            Prompt.ask(
                "允许执行？",
                default="n",
                console=self.console,
            )
            .strip()
            .lower()
        )
        if answer in ("y", "yes"):
            return Reply.ONCE
        if answer in ("a", "always"):
            return Reply.ALWAYS
        return Reply.REJECT

    def on_tool_finished(self, tool_name: str, result: ToolResult) -> None:
        if result.ok:
            metadata_parts: list[str] = []
            if "total_matches" in result.metadata:
                metadata_parts.append(f"{result.metadata['total_matches']} 处匹配")
            if "files_searched" in result.metadata:
                metadata_parts.append(f"检索 {result.metadata['files_searched']} 个文件")
            if "size_bytes" in result.metadata:
                metadata_parts.append(f"{result.metadata['size_bytes']} 字节")
            if "bytes_written" in result.metadata:
                metadata_parts.append(f"{result.metadata['bytes_written']} 字节写入")
            if "total_entries" in result.metadata:
                metadata_parts.append(f"{result.metadata['total_entries']} 项")
            if "exit_code" in result.metadata and self.verbose:
                metadata_parts.append(f"返回码: {result.metadata['exit_code']}")

            extra_info = f" [dim]({', '.join(metadata_parts)})[/dim]" if metadata_parts else ""
            self.console.print(f"  [bold green]✔ 执行成功[/bold green]{extra_info}")
        else:
            reason = result.error or "未知错误"
            self.console.print(f"  [bold red]✗ 执行失败[/bold red]: [red]{reason}[/red]")

    def on_usage(self, usage: UsageStats, cost_cny: float, model: str) -> None:
        cost_str = format_cost_cny(cost_cny)
        self.console.print(
            f"  [dim]📊 本轮消耗: {usage.total_tokens:,} Tokens "
            f"(输入: {usage.prompt_tokens:,} / 输出: {usage.completion_tokens:,}) | "
            f"预估费用: {cost_str}[/dim]\n"
        )

    def on_turn_finished(self, response: str) -> None:
        if self._streamed_any:
            self.console.print("\n")
            self._streamed_any = False
        else:
            self.console.print("\n[bold cyan]🤖 Mini-Agent[/bold cyan]")
            try:
                self.console.print(Markdown(response))
            except Exception:
                self.console.print(response)
            self.console.print()


def render_help_tables(
    console: Console,
    command_rows: list[tuple[str, str]],
    tool_rows: list[tuple[str, str]],
) -> None:
    """Print help tables generated from the slash registry and tool registry."""
    table = Table(box=box.ROUNDED, border_style="cyan", show_header=True, header_style="bold cyan")
    table.add_column("指令", style="bold white", width=22)
    table.add_column("说明与用途", style="white")
    for usage, description in command_rows:
        table.add_row(usage, description)
    console.print(table)

    tools_table = Table(
        box=box.ROUNDED, border_style="dim", show_header=True, header_style="bold green"
    )
    tools_table.add_column("内置工具", style="bold green", width=16)
    tools_table.add_column("功能", style="white")
    for name, description in tool_rows:
        tools_table.add_row(name, description)
    console.print(tools_table)
    console.print("[dim]提示：按 Ctrl-C 取消当前输入或当前回合，按 Ctrl-D 正常退出。[/dim]\n")


def render_providers_table(console: Console) -> None:
    """Render list of available predefined providers."""
    presets = list_provider_presets()
    table = Table(
        title="🌐 大模型服务商预设列表 (输入 /provider <名称> 切换)",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("预设名 (Name)", style="bold white", width=16)
    table.add_column("服务商 / 名称", style="bright_cyan", width=24)
    table.add_column("默认模型", style="magenta", width=24)
    table.add_column("说明与特点", style="dim")

    for p in presets:
        table.add_row(p.name, p.display_name, p.default_model, p.description)

    console.print(table)


def render_cost_table(console: Console, agent: Any) -> None:
    """Render table of token usage and estimated cost for the current session."""
    table = Table(
        title="💰 当前会话 Token 用量与累计费用统计",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("统计项", style="bold white", width=22)
    table.add_column("数值 / 明细", style="bright_cyan", width=26)
    table.add_column("说明", style="dim")

    u = agent.session_usage
    total_cost = agent.session.meta.total_cost_cny
    p_in, p_out = get_model_pricing(agent.config.model)

    table.add_row("活跃模型", agent.config.model, "当前生效的大模型")
    table.add_row(
        "当前模型单价",
        f"输入 ¥{p_in}/M | 输出 ¥{p_out}/M",
        "百万 Tokens 计费单价",
    )
    table.add_row("累计输入 Tokens", f"{u.prompt_tokens:,}", "提示词与上下文所占 Token")
    table.add_row("累计输出 Tokens", f"{u.completion_tokens:,}", "模型回答与思考所占 Token")
    table.add_row("累计总 Tokens", f"{u.total_tokens:,}", "输入 + 输出 Token 总和")
    table.add_row("累计预估费用", format_cost_cny(total_cost), "基于当前模型定价计算")

    console.print(table)
    console.print(
        "[dim]提示：使用 /cost set <模型> <输入价> <输出价> 可自定义任意模型费率。[/dim]\n"
    )


def render_pricing_list_table(console: Console) -> None:
    """Render full active pricing table for all models."""
    table = load_pricing_table()
    rich_table = Table(
        title="📋 活跃模型计费费率表 (单位: 元/百万 Tokens)",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    rich_table.add_column("模型名称 / 前缀", style="bold white", width=24)
    rich_table.add_column("输入单价 (¥/1M)", style="bright_cyan", justify="right", width=18)
    rich_table.add_column("输出单价 (¥/1M)", style="magenta", justify="right", width=18)

    for m_name, (p_in, p_out) in sorted(table.items()):
        rich_table.add_row(m_name, f"¥{p_in:.2f}", f"¥{p_out:.2f}")

    console.print(rich_table)
    console.print("[dim]提示：自定义费率已保存至 ~/.mini-agent/pricing.json。[/dim]\n")


def render_sessions_table(console: Console, workspace: Any) -> None:
    """Render list of saved sessions for workspace."""
    sessions = list_sessions(workspace_root=workspace)
    if not sessions:
        console.print("[dim]当前工作区暂无历史会话。[/dim]\n")
        return

    table = Table(
        title=f"📜 历史会话列表 (工作区: {workspace.name})",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    table.add_column("Session ID", style="bold white", width=24)
    table.add_column("更新时间", style="dim", width=20)
    table.add_column("轮数", style="cyan", justify="right", width=6)
    table.add_column("模型", style="magenta", width=16)
    table.add_column("会话主题 / 摘要", style="white")

    for s in sessions:
        try:
            dt = s.updated_at.split("T")[0] + " " + s.updated_at.split("T")[1][:8]
        except Exception:
            dt = s.updated_at
        table.add_row(s.session_id, dt, str(s.turn_count), s.model, s.title)

    console.print(table)
    console.print("[dim]输入 /resume <Session ID> 即可继续对应历史会话。[/dim]\n")
