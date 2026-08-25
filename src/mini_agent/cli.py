"""Typer CLI interface and Rich REPL implementation (Antigravity Style)."""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from mini_agent import __version__
from mini_agent.agent import Agent, AgentEventListener
from mini_agent.cost import (
    UsageStats,
    format_cost_cny,
    get_model_pricing,
    load_pricing_table,
    set_custom_pricing,
)
from mini_agent.llm import LLMClient, LLMError, OpenAIChatCompletionsClient
from mini_agent.models import AgentConfig, ToolResult
from mini_agent.providers import (
    get_provider_preset,
    list_provider_presets,
)
from mini_agent.session import (
    SessionData,
    get_latest_session,
    list_sessions,
    load_session,
)
from mini_agent.tools.shell import sanitize_environment

app = typer.Typer(
    name="mini-agent",
    help="mini-agent: 本地终端 AI 编程助手",
    add_completion=False,
)
console = Console()

DOTENV_ALLOWED_KEYS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "MINI_AGENT_MODEL",
    "MINI_AGENT_PRICING",
}


def run_git(
    workspace: Path,
    *args: str,
    capture_output: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run Git without exposing API keys or cloud credentials to hooks and helpers."""
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=capture_output,
        text=True,
        check=check,
        env=sanitize_environment(),
    )


def render_banner(
    console: Console,
    workspace: Path,
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

    def __init__(self, console: Console, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self._streamed_any = False

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

        if tool_name == "get_repo_map":
            path = arguments.get("path", ".")
            self.console.print(
                f"  [bold cyan]⚡ Tool: get_repo_map[/bold cyan] "
                f"[dim](路径: {escape(str(path))})[/dim]"
            )
        elif tool_name == "search_code":
            pattern = arguments.get("pattern", "")
            path = arguments.get("path", ".")
            self.console.print(
                f"  [bold cyan]⚡ Tool: search_code[/bold cyan] "
                f"[dim](模式: '{escape(str(pattern))}', 路径: {escape(str(path))})[/dim]"
            )
        elif tool_name == "read_file":
            path = arguments.get("path", "")
            self.console.print(
                f"  [bold cyan]⚡ Tool: read_file[/bold cyan] "
                f"[dim](路径: {escape(str(path))})[/dim]"
            )
        elif tool_name == "list_files":
            path = arguments.get("path", ".")
            depth = arguments.get("max_depth", 2)
            self.console.print(
                f"  [bold cyan]⚡ Tool: list_files[/bold cyan] "
                f"[dim](路径: {escape(str(path))}, 深度: {depth})[/dim]"
            )
        elif tool_name == "write_file":
            path = arguments.get("path", "")
            chars = len(arguments.get("content", ""))
            self.console.print(
                f"  [bold cyan]⚡ Tool: write_file[/bold cyan] "
                f"[dim](写入: {escape(str(path))}, {chars} 字符)[/dim]"
            )
        elif tool_name == "edit_file":
            path = arguments.get("path", "")
            self.console.print(
                f"  [bold cyan]⚡ Tool: edit_file[/bold cyan] "
                f"[dim](修改: {escape(str(path))})[/dim]"
            )
        elif tool_name == "run_shell":
            cmd = arguments.get("command", "")
            self.console.print(
                f"  [bold cyan]⚡ Tool: run_shell[/bold cyan] [dim](命令: {escape(str(cmd))})[/dim]"
            )
        else:
            self.console.print(f"  [bold cyan]⚡ Tool: {escape(tool_name)}[/bold cyan]")

    def on_tool_confirm(self, command: str) -> bool:
        self.console.print(
            Panel(
                f"[yellow]Agent 请求执行以下非只读 Shell 命令：[/yellow]\n\n"
                f"  [bold cyan]{escape(command)}[/bold cyan]\n\n"
                f"[dim]请确认该命令在当前工作区内执行是否安全。[/dim]",
                title="[bold yellow]⚠️  安全确认 (Security Confirmation)[/bold yellow]",
                box=box.ROUNDED,
                border_style="yellow",
                padding=(0, 2),
            )
        )
        return Confirm.ask("是否允许执行该命令？", default=False, console=self.console)

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
            self.console.print(f"  [bold red]✗ 执行失败[/bold red]: [red]{escape(reason)}[/red]")

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


def render_help(console: Console) -> None:
    """Print beautifully formatted help table."""
    table = Table(box=box.ROUNDED, border_style="cyan", show_header=True, header_style="bold cyan")
    table.add_column("指令", style="bold white", width=22)
    table.add_column("说明与用途", style="white")

    table.add_row("/help", "显示快捷指令与 Agent 工具能力说明")
    table.add_row("/provider [name]", "切换或查看模型服务商预设 (DeepSeek V4, Ollama 等)")
    table.add_row("/cost [set/list]", "查看 Token 消耗看板，或自定义/查看模型费率表")
    table.add_row("/diff", "查看当前工作区的所有 Git 代码改动")
    table.add_row("/commit [msg]", "智能生成或执行 Git 提交")
    table.add_row("/sessions", "查看当前工作区的所有历史会话")
    table.add_row("/resume <id>", "切换并恢复指定历史会话")
    table.add_row("/new", "重置并开启全新会话")
    table.add_row("/clear", "清屏并重新展示顶部状态 Banner")
    table.add_row("/model [name]", "查看或临时切换当前模型 (如 /model deepseek-v4-pro)")
    table.add_row("/exit, /quit", "退出当前 mini-agent 会话")

    console.print(table)

    tools_table = Table(
        box=box.ROUNDED, border_style="dim", show_header=True, header_style="bold green"
    )
    tools_table.add_column("内置工具", style="bold green", width=16)
    tools_table.add_column("功能", style="white", width=26)
    tools_table.add_column("安全策略与约束", style="dim")

    tools_table.add_row(
        "get_repo_map",
        "代码骨架地图提取 (Repo Map)",
        "自动解析 Python / JS / TS 类与函数签名，构建项目骨架",
    )
    tools_table.add_row(
        "search_code",
        "全文正则代码检索",
        "递归搜索关键词或正则，自动过滤 .git/.venv 等无关目录",
    )
    tools_table.add_row(
        "list_files",
        "列出工作区目录结构",
        "仅允许工作区内相对路径，深度 1~5，跳过 .git/.venv",
    )
    tools_table.add_row(
        "read_file",
        "读取 UTF-8 文件内容",
        "工作区相对路径沙箱，单文件上限 100 KiB",
    )
    tools_table.add_row(
        "edit_file",
        "精准修改文件代码片段",
        "唯一匹配 target_content 并替换为 replacement_content",
    )
    tools_table.add_row(
        "write_file",
        "创建新文件或覆盖写入",
        "工作区相对路径沙箱，自动创建父级目录",
    )
    tools_table.add_row(
        "run_shell",
        "受控 Shell 命令执行",
        "阻断破坏性高危指令，环境脱敏，非白名单命令弹窗确认",
    )

    console.print(tools_table)
    console.print("[dim]提示：按 Ctrl-C 取消当前行输入，按 Ctrl-D 正常退出。[/dim]\n")


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


def render_cost_table(console: Console, agent: Agent) -> None:
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


def render_sessions_table(console: Console, workspace: Path) -> None:
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


def repl_loop(agent: Agent, console: Console) -> None:
    """Main interactive REPL loop with Antigravity styling."""
    render_banner(
        console,
        agent.config.workspace_root,
        agent.config.model,
        session_id=agent.session.meta.session_id,
    )

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]>[/bold cyan]", console=console).strip()
        except KeyboardInterrupt:
            console.print("\n[yellow]已取消当前输入[/yellow]")
            continue
        except EOFError:
            console.print("\n[dim]👋 再见！[/dim]")
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            console.print("[dim]👋 再见！[/dim]")
            break

        if user_input == "/clear":
            console.clear()
            render_banner(
                console,
                agent.config.workspace_root,
                agent.config.model,
                session_id=agent.session.meta.session_id,
            )
            continue

        if user_input.startswith("/cost"):
            cost_parts = user_input.split()
            if len(cost_parts) == 1:
                render_cost_table(console, agent)
            elif cost_parts[1] == "list":
                render_pricing_list_table(console)
            elif cost_parts[1] == "set" and len(cost_parts) >= 5:
                target_m = cost_parts[2]
                try:
                    p_in = float(cost_parts[3])
                    p_out = float(cost_parts[4])
                    set_custom_pricing(target_m, p_in, p_out)
                    console.print(
                        f"[green]✔ 已成功更新模型 '{target_m}' 费率:[/green] "
                        f"输入 ¥{p_in}/M | 输出 ¥{p_out}/M\n"
                    )
                except ValueError:
                    console.print(
                        "[red]✗ 价格格式错误。用法: /cost set <模型> <输入价> <输出价>[/red]\n"
                    )
            else:
                console.print(
                    "[yellow]用法:\n"
                    "  /cost      - 查看当前会话用量与费用看板\n"
                    "  /cost list - 查看所有已配置模型的费率表\n"
                    "  /cost set <模型> <输入价> <输出价> - 自定义模型费率 (元/1M)[/yellow]\n"
                )
            continue

        if user_input.startswith("/provider"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                pname = parts[1].strip()
                preset = get_provider_preset(pname)
                if preset:
                    try:
                        new_client = OpenAIChatCompletionsClient(
                            base_url=preset.base_url,
                        )
                    except Exception as exc:
                        console.print(f"[red]✗ 切换服务商失败: {escape(str(exc))}[/red]\n")
                    else:
                        agent.llm_client = new_client
                        agent.config.model = preset.default_model
                        agent.session.meta.model = preset.default_model
                        console.print(
                            f"[green]✔ 已成功切换服务商:[/green] "
                            f"[bold cyan]{preset.display_name}[/bold cyan] "
                            f"[dim](模型: {preset.default_model})[/dim]\n"
                        )
                else:
                    console.print(f"[red]✗ 未知服务商预设: '{escape(pname)}'[/red]")
                    render_providers_table(console)
            else:
                render_providers_table(console)
            continue

        if user_input == "/diff":
            try:
                res = run_git(
                    agent.config.workspace_root,
                    "diff",
                    "HEAD",
                    capture_output=True,
                )
                status_res = run_git(
                    agent.config.workspace_root,
                    "status",
                    "--short",
                    capture_output=True,
                )
                if res.returncode != 0 or status_res.returncode != 0:
                    detail = res.stderr or status_res.stderr or "当前目录可能不是 Git 仓库"
                    console.print(f"[red]✗ 执行 git diff 失败: {escape(detail.strip())}[/red]\n")
                    continue
                if res.stdout.strip():
                    console.print(
                        Syntax(
                            res.stdout,
                            "diff",
                            theme="monokai",
                            line_numbers=False,
                        )
                    )
                if status_res.stdout.strip():
                    console.print("[bold cyan]Git 状态（含未跟踪文件）[/bold cyan]")
                    console.print(status_res.stdout, markup=False)
                if not res.stdout.strip() and not status_res.stdout.strip():
                    console.print("[dim green]✔ 工作区代码干净，无待提交改动。[/dim green]\n")
            except Exception as exc:
                console.print(f"[red]✗ 执行 git diff 失败: {escape(str(exc))}[/red]\n")
            continue

        if user_input.startswith("/commit"):
            parts = user_input.split(maxsplit=1)
            msg = parts[1].strip() if len(parts) > 1 else ""
            try:
                status_res = run_git(
                    agent.config.workspace_root,
                    "status",
                    "--short",
                    capture_output=True,
                )
            except Exception as exc:
                console.print(f"[red]✗ 读取 Git 状态失败: {escape(str(exc))}[/red]\n")
                continue
            if status_res.returncode != 0:
                console.print(
                    f"[red]✗ 读取 Git 状态失败: {escape(status_res.stderr.strip())}[/red]\n"
                )
                continue
            status_text = status_res.stdout.strip()
            if not status_text:
                console.print("[yellow]当前没有代码变更可提交。[/yellow]\n")
                continue

            if not msg:
                try:
                    diff_res = run_git(
                        agent.config.workspace_root,
                        "diff",
                        "HEAD",
                        capture_output=True,
                    )
                    diff_text = diff_res.stdout.strip()
                    gen_prompt = (
                        "请根据以下 git 状态与 diff 生成一行标准规范的 Conventional Commit 信息"
                        "（例如 feat: ... 或 fix: ...），仅直接返回 Commit 文本本身：\n"
                        f"```text\n{status_text[:2000]}\n```\n"
                        f"```diff\n{diff_text[:3000]}\n```"
                    )
                    gen_msg = agent.step(gen_prompt, tool_definitions=[]).strip().strip("`'\"")
                    if Confirm.ask(
                        f"是否以此信息提交？\n[bold cyan]{escape(gen_msg)}[/bold cyan]",
                        default=True,
                        console=console,
                    ):
                        msg = gen_msg
                    else:
                        continue
                except Exception as exc:
                    console.print(f"[red]✗ 生成提交信息失败: {escape(str(exc))}[/red]\n")
                    continue

            if msg:
                refreshed_status = run_git(
                    agent.config.workspace_root,
                    "status",
                    "--short",
                    capture_output=True,
                )
                if refreshed_status.returncode != 0:
                    console.print(
                        f"[red]✗ 读取 Git 状态失败: "
                        f"{escape(refreshed_status.stderr.strip())}[/red]\n"
                    )
                    continue
                status_text = refreshed_status.stdout.strip()
                if not status_text:
                    console.print("[yellow]当前没有代码变更可提交。[/yellow]\n")
                    continue
                console.print("[bold cyan]即将暂存并提交以下变更：[/bold cyan]")
                console.print(status_text, markup=False)
                if not Confirm.ask(
                    "是否暂存以上全部当前变更？",
                    default=False,
                    console=console,
                ):
                    console.print("[yellow]已取消 Git 提交。[/yellow]\n")
                    continue
                try:
                    run_git(agent.config.workspace_root, "add", "--all", check=True)
                    staged_diff = run_git(
                        agent.config.workspace_root,
                        "diff",
                        "--cached",
                        "--no-ext-diff",
                        "--no-textconv",
                        capture_output=True,
                        check=True,
                    )
                    console.print("[bold cyan]已暂存、即将提交的精确 diff：[/bold cyan]")
                    console.print(
                        Syntax(
                            staged_diff.stdout or "(无文本 diff)",
                            "diff",
                            theme="monokai",
                            line_numbers=False,
                        )
                    )
                    if not Confirm.ask(
                        f"是否以 [bold cyan]{escape(msg)}[/bold cyan] 提交以上已暂存内容？",
                        default=False,
                        console=console,
                    ):
                        console.print(
                            "[yellow]已取消提交；为避免破坏原有暂存状态，"
                            "以上变更保持 staged。[/yellow]\n"
                        )
                        continue
                    run_git(
                        agent.config.workspace_root,
                        "commit",
                        "-m",
                        msg,
                        check=True,
                    )
                    console.print(
                        f"[green]✔ Git 提交成功:[/green] [bold cyan]{escape(msg)}[/bold cyan]\n"
                    )
                except subprocess.CalledProcessError as exc:
                    console.print(f"[red]✗ Git 提交失败: {escape(str(exc))}[/red]\n")
            continue

        if user_input == "/sessions":
            render_sessions_table(console, agent.config.workspace_root)
            continue

        if user_input.startswith("/resume"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                target_id = parts[1].strip()
                loaded = load_session(target_id)
                if loaded:
                    try:
                        agent.resume_session(loaded)
                    except ValueError as exc:
                        console.print(f"[red]✗ 无法恢复会话: {escape(str(exc))}[/red]\n")
                    else:
                        console.print(
                            f"[green]✔ 已成功恢复会话:[/green] "
                            f"[bold cyan]{escape(target_id)}[/bold cyan] "
                            f"[dim]({loaded.meta.title}, {len(loaded.history)} 条记录)[/dim]\n"
                        )
                else:
                    console.print(f"[red]✗ 未找到指定的会话 ID: '{escape(target_id)}'[/red]\n")
            else:
                console.print(
                    "[yellow]用法: /resume <Session_ID> (可通过 /sessions 查看 ID)[/yellow]\n"
                )
            continue

        if user_input == "/new":
            new_id = agent.reset_session()
            console.print(
                f"[green]✔ 已重置上下文，开启全新会话:[/green] "
                f"[bold cyan]{escape(new_id)}[/bold cyan]\n"
            )
            continue

        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                new_model = parts[1].strip()
                agent.config.model = new_model
                agent.session.meta.model = new_model
                console.print(
                    f"[green]✔ 已切换当前模型为:[/green] "
                    f"[bold cyan]{escape(new_model)}[/bold cyan]\n"
                )
            else:
                console.print(
                    f"[dim]当前会话模型:[/dim] [bold cyan]{agent.config.model}[/bold cyan]\n"
                )
            continue

        if user_input == "/help":
            render_help(console)
            continue

        try:
            agent.step(user_input)
        except LLMError as exc:
            console.print(f"\n[bold red]LLM 错误[/bold red]: {escape(str(exc))}\n")
        except Exception as exc:
            console.print(f"\n[bold red]执行错误[/bold red]: {escape(str(exc))}\n")


def load_dotenv(workspace_root: Path | None = None) -> None:
    """Lightweight loader for .env file within workspace."""
    path = (workspace_root / ".env") if workspace_root else Path(".env")
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, val = stripped.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key in DOTENV_ALLOWED_KEYS and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            pass


def run_cli(
    workspace: Path | None = None,
    model: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
    continue_session: bool = False,
    session_id: str | None = None,
    verbose: bool = False,
    agent_factory: Callable[[AgentConfig, LLMClient, AgentEventListener], Agent] | None = None,
    llm_client: LLMClient | None = None,
) -> None:
    """Core logic to run the CLI in interactive or one-shot mode."""
    target_workspace = (workspace or Path.cwd()).resolve()
    load_dotenv(target_workspace)
    if not target_workspace.exists():
        console.print(
            f"[bold red]错误[/bold red]: 指定的工作区路径不存在: '{escape(str(target_workspace))}'"
        )
        raise typer.Exit(code=1)
    if not target_workspace.is_dir():
        console.print(
            f"[bold red]错误[/bold red]: 指定的工作区路径不是目录: "
            f"'{escape(str(target_workspace))}'"
        )
        raise typer.Exit(code=1)

    effective_base_url = (
        base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    )
    if model:
        resolved_model = model
    elif os.environ.get("MINI_AGENT_MODEL"):
        resolved_model = os.environ["MINI_AGENT_MODEL"]
    elif effective_base_url and "deepseek" in effective_base_url.lower():
        resolved_model = "deepseek-v4-flash"
    else:
        resolved_model = "gpt-4o-mini"

    # Verify API key if default client is used
    if llm_client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            console.print(
                "[bold red]错误[/bold red]: 未检测到 OPENAI_API_KEY 环境变量。\n"
                "请先设置您的 API Key，例如在终端执行：\n"
                "  [cyan]export OPENAI_API_KEY='sk-...'[/cyan]\n"
                "若使用 DeepSeek，可同时配置：\n"
                "  [cyan]export OPENAI_BASE_URL='https://api.deepseek.com'[/cyan]\n"
                "  [cyan]export MINI_AGENT_MODEL='deepseek-v4-flash'[/cyan]"
            )
            raise typer.Exit(code=1)
        client: LLMClient = OpenAIChatCompletionsClient(
            api_key=api_key,
            base_url=effective_base_url,
        )
    else:
        client = llm_client

    config = AgentConfig(
        workspace_root=target_workspace,
        model=resolved_model,
    )
    listener = RichAgentEventListener(console=console, verbose=verbose)

    # Handle session loading
    loaded_session: SessionData | None = None
    if session_id:
        loaded_session = load_session(session_id)
        if not loaded_session:
            console.print(f"[bold red]错误[/bold red]: 未找到指定的会话 ID: '{escape(session_id)}'")
            raise typer.Exit(code=1)
    elif continue_session:
        loaded_session = get_latest_session(target_workspace)
        if loaded_session:
            console.print(
                f"[dim]已自动恢复上一次会话: {loaded_session.meta.session_id} "
                f"({loaded_session.meta.title})[/dim]"
            )

    if loaded_session is not None:
        loaded_workspace = Path(loaded_session.meta.workspace_root).resolve()
        if loaded_workspace != target_workspace:
            console.print(
                "[bold red]错误[/bold red]: 该会话属于其他工作区，已拒绝恢复。\n"
                f"  当前: {escape(str(target_workspace))}\n"
                f"  会话: {escape(str(loaded_workspace))}"
            )
            raise typer.Exit(code=1)

    if agent_factory is not None:
        agent = agent_factory(config, client, listener)
        if loaded_session is not None:
            agent.resume_session(loaded_session)
    else:
        agent = Agent(
            config=config,
            llm_client=client,
            listener=listener,
            session=loaded_session,
        )

    # One-shot non-interactive execution
    if prompt:
        try:
            agent.step(prompt)
        except Exception as exc:
            console.print(f"[bold red]执行失败[/bold red]: {escape(str(exc))}")
            raise typer.Exit(code=1) from exc
        return

    # Interactive REPL mode
    repl_loop(agent, console=console)


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
            help="覆盖本次会话的模型名称（如 deepseek-v4-flash 或 gpt-4o-mini）",
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
    )
