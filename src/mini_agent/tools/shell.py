"""Controlled shell execution tool with safety checks and environment sanitization."""

import os
import re
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any

from mini_agent.models import PermissionClass, RunShellInput, ToolResult
from mini_agent.tools.filesystem import truncate_text
from mini_agent.tools.protocol import ToolContext, ToolKind

# High-risk patterns that must be blocked outright
BLOCKLIST_PATTERNS = [
    # Destructive deletions
    r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)?(\/|~|\.\.|\*|\/\*)(\s|$)",
    r"\brmdir\s+(\/|~|\.\.)",
    # Disk formatting and raw writes
    r"\bmkfs(\.\w+)?\b",
    r"\bfdisk\b",
    r"\bdd\s+if=",
    # System control
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\binit\s+[06]\b",
    # Privilege escalation
    r"\bsudo\b",
    r"\bsu\b",
    r"\bdoas\b",
    r"\bchmod\s+777\b",
    r"\bchown\b",
    # Direct piped execution from network downloads
    r"\b(curl|wget|fetch)\b.*\|\s*(ba|z|k|c)?sh\b",
    r"\b(curl|wget|fetch)\b.*\|\s*python[0-9.]*\b",
    # Dangerous git operations
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+push\s+.*(-f|--force)\b",
    r"\bgit\s+clean\s+-[a-zA-Z]*f\b",
    r"\bgit\s+add\s+\.(?:\s|$)",
    r"\bgit\s+add\s+-[a-zA-Z]*A",
    r"\bgit\s+add\s+--all\b",
    # Secret / credential leak attempts
    r"\b(printenv|env)\b",
    r"\becho\s+\$(OPENAI|ANTHROPIC|AWS|GITHUB|TOKEN|KEY|SECRET|PASSWORD)",
    r"\bcat\s+.*(\/etc\/shadow|\/etc\/passwd|~?\/\.ssh|~?\/\.aws|~?\/\.netrc)",
]

# Low-risk / read-only commands that can be automatically executed without confirmation
ALLOWLIST_COMMANDS = {
    "pwd",
    "ls",
    "dir",
    "find",
    "rg",
    "grep",
    "pytest",
    "git status",
    "git diff",
    "git log",
    "git branch",
    "git show",
    "python -m pytest",
    "python --version",
    "uv run pytest",
    "uv run ruff",
    "uv run python",
    "uv --version",
}

SENSITIVE_ENV_PREFIXES = (
    "OPENAI_",
    "ANTHROPIC_",
    "AWS_",
    "AZURE_",
    "GOOGLE_",
    "GITHUB_",
    "GIT_",
    "SSH_",
)
SENSITIVE_ENV_EXACT = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "SSH_AUTH_SOCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
}


def sanitize_environment(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Create a minimal, sanitized environment dictionary for child processes."""
    source_env = base_env if base_env is not None else os.environ

    # Essential system keys needed to run commands
    allowed_keys = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "TMPDIR",
        "VIRTUAL_ENV",
    }

    clean_env: dict[str, str] = {}
    for key, value in source_env.items():
        if key in allowed_keys or key.startswith("LC_"):
            # Ensure no sensitive tokens leaked even if matching allowed pattern
            if key not in SENSITIVE_ENV_EXACT and not any(
                key.startswith(prefix) for prefix in SENSITIVE_ENV_PREFIXES
            ):
                clean_env[key] = value

    return clean_env


def _is_dangerous_git_add(tokens: list[str]) -> bool:
    """True when `git add` stages everything via `.` / `-A` / `--all`. Does not match `-u`."""
    if len(tokens) < 2 or tokens[0] != "git" or tokens[1] != "add":
        return False
    for tok in tokens[2:]:
        if tok in {".", "-A", "--all"}:
            return True
        if tok.startswith("-") and not tok.startswith("--") and "A" in tok[1:]:
            return True
    return False


def check_command_safety(command: str) -> tuple[bool, bool, str | None]:
    """Inspect command safety.

    Returns:
        tuple[bool, bool, str | None]: (is_blocked, requires_confirmation, reason)
    """
    stripped = command.strip()
    if not stripped:
        return True, False, "命令不能为空"

    # Check blocklist
    for pattern in BLOCKLIST_PATTERNS:
        if re.search(pattern, stripped, re.IGNORECASE):
            return True, False, f"命令包含高危模式，已被安全策略直接阻断: '{stripped}'"

    # Check allowlist (exact match or matching prefix)
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        # If shell syntax fails to split safely, require confirmation
        return False, True, "命令包含复杂或未闭合的 Shell 结构，需要用户确认"

    if not tokens:
        return True, False, "命令不能为空"

    if _is_dangerous_git_add(tokens):
        return True, False, f"命令包含高危模式，已被安全策略直接阻断: '{stripped}'"

    # Check against allowlist
    first_token = tokens[0]
    first_two = " ".join(tokens[:2]) if len(tokens) >= 2 else first_token
    first_three = " ".join(tokens[:3]) if len(tokens) >= 3 else first_two

    if (
        first_token in ALLOWLIST_COMMANDS
        or first_two in ALLOWLIST_COMMANDS
        or first_three in ALLOWLIST_COMMANDS
    ):
        # Even if command base is allowlisted, check for suspicious redirection/pipe
        if any(tok in ("|", ">", ">>", "&", "&&", ";") for tok in tokens):
            return False, True, "命令包含管道或重定向，需要用户确认"
        return False, False, None

    # Default for all non-allowlisted commands: require confirmation
    return False, True, f"命令不在只读白名单内，需要用户确认方可执行: '{stripped}'"


def run_shell(
    input_data: RunShellInput,
    workspace_root: Path,
    confirmed: bool = False,
    timeout_seconds: int = 30,
    max_output_chars: int = 12_000,
    custom_env: dict[str, str] | None = None,
) -> ToolResult:
    """Execute a controlled shell command in the workspace directory."""
    command = input_data.command.strip()
    is_blocked, requires_confirmation, reason = check_command_safety(command)

    if is_blocked:
        return ToolResult(
            ok=False,
            content="",
            error=reason,
            metadata={"command": command, "blocked": True},
        )

    if requires_confirmation and not confirmed:
        return ToolResult(
            ok=False,
            content="",
            error=f"需要确认: {reason}",
            metadata={"command": command, "requires_confirmation": True},
        )

    resolved_root = workspace_root.resolve()
    clean_env = sanitize_environment(custom_env)

    process: subprocess.Popen[str] | None = None
    try:
        # Use start_new_session to create a separate process group for clean timeout termination
        process = subprocess.Popen(
            command,
            cwd=resolved_root,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=clean_env,
            start_new_session=True,
        )

        stdout_data, stderr_data = process.communicate(timeout=timeout_seconds)
        exit_code = process.returncode

    except subprocess.TimeoutExpired:
        if process is not None:
            # Terminate the entire process group
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                stdout_data, stderr_data = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except OSError:
                    pass
                stdout_data, stderr_data = "", ""

        return ToolResult(
            ok=False,
            content="",
            error=f"命令执行超时 (限制 {timeout_seconds} 秒): '{command}'",
            metadata={"command": command, "timeout": True, "exit_code": -1},
        )
    except OSError as exc:
        return ToolResult(
            ok=False,
            content="",
            error=f"启动子进程失败: {exc}",
            metadata={"command": command, "exit_code": -1},
        )

    # Combine output
    combined_output = stdout_data
    if stderr_data:
        if combined_output:
            combined_output = f"{combined_output}\n[stderr]\n{stderr_data}"
        else:
            combined_output = stderr_data

    truncated_content, is_truncated = truncate_text(
        combined_output.strip(), max_chars=max_output_chars
    )

    metadata: dict[str, Any] = {
        "command": command,
        "exit_code": exit_code,
        "truncated": is_truncated,
        "timeout": False,
    }

    if exit_code == 0:
        return ToolResult(
            ok=True,
            content=truncated_content if truncated_content else "(命令成功执行，无输出)",
            error=None,
            metadata=metadata,
        )

    error_msg = f"命令返回非零退出码: {exit_code}"
    if truncated_content:
        error_msg = f"{error_msg}\n{truncated_content}"

    return ToolResult(
        ok=False,
        content=truncated_content,
        error=error_msg,
        metadata=metadata,
    )


class RunShellTool:
    name = "run_shell"
    description = "在工作区根目录下执行受控的 Shell 命令行指令。"
    permission = PermissionClass.SHELL
    kind = ToolKind.MUTATING
    input_model = RunShellInput

    def execute(self, inp: RunShellInput, ctx: ToolContext) -> ToolResult:
        return run_shell(
            inp,
            workspace_root=ctx.workspace_root,
            confirmed=True,
            timeout_seconds=ctx.config.shell_timeout_seconds,
            max_output_chars=ctx.config.max_output_chars,
        )

    def format_call(self, inp: RunShellInput) -> str:
        return f"run_shell command={inp.command}"

    def approval_pattern(self, inp: RunShellInput) -> str:
        return inp.command
