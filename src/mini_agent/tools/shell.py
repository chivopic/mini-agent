"""Controlled shell execution tool with safety checks and environment sanitization."""

import os
import re
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any

from mini_agent.models import RunShellInput, ToolResult
from mini_agent.tools.filesystem import truncate_text

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
    # Secret / credential leak attempts
    r"\b(printenv|env)\b",
    r"\becho\s+\$(OPENAI|ANTHROPIC|AWS|GITHUB|TOKEN|KEY|SECRET|PASSWORD)",
    r"\bcat\s+.*(\/etc\/shadow|\/etc\/passwd|~?\/\.ssh|~?\/\.aws|~?\/\.netrc)",
]

# Only simple, low-risk commands are allowed without confirmation. Interpreters, test runners,
# generic `find`, and Git commands are intentionally excluded because their arguments can execute
# code, invoke helpers, or mutate the repository.
AUTO_ALLOWED_COMMANDS = {"pwd", "ls"}
AUTO_ALLOWED_EXACT = {
    ("python", "--version"),
    ("python3", "--version"),
    ("uv", "--version"),
}

# Shell grammar and expansion are supported only after explicit confirmation. Auto-allowed commands
# are executed with shell=False so the permission decision and execution semantics stay aligned.
SHELL_SYNTAX_PATTERN = re.compile(r"[\r\n|&;<>`$(){}*?\[\]]")

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
        # Windows process startup essentials.
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
    }

    clean_env: dict[str, str] = {}
    for key, value in source_env.items():
        if key in allowed_keys or key.startswith("LC_"):
            if key not in SENSITIVE_ENV_EXACT and not any(
                key.startswith(prefix) for prefix in SENSITIVE_ENV_PREFIXES
            ):
                clean_env[key] = value

    return clean_env


def check_command_safety(command: str) -> tuple[bool, bool, str | None]:
    """Inspect command safety.

    Returns:
        (is_blocked, requires_confirmation, reason)
    """
    stripped = command.strip()
    if not stripped:
        return True, False, "命令不能为空"

    for pattern in BLOCKLIST_PATTERNS:
        if re.search(pattern, stripped, re.IGNORECASE):
            return True, False, f"命令包含高危模式，已被安全策略直接阻断: '{stripped}'"

    if SHELL_SYNTAX_PATTERN.search(stripped):
        return False, True, "命令包含 Shell 组合、重定向或展开语法，需要用户确认"

    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return False, True, "命令包含复杂或未闭合的 Shell 结构，需要用户确认"

    if not tokens:
        return True, False, "命令不能为空"

    normalized = tuple(tokens)
    first_token = tokens[0]

    if normalized in AUTO_ALLOWED_EXACT:
        return False, False, None

    if first_token in AUTO_ALLOWED_COMMANDS:
        return False, False, None

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

    process: subprocess.Popen[str] | subprocess.Popen[bytes] | None = None
    try:
        # Auto-allowed commands are parsed as argv and executed without a shell. Only commands that
        # required confirmation may retain shell-string semantics such as pipes and redirection.
        popen_command: str | list[str]
        use_shell = requires_confirmation
        if use_shell:
            popen_command = command
        else:
            popen_command = shlex.split(command)

        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(
            popen_command,
            cwd=resolved_root,
            shell=use_shell,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=clean_env,
            **popen_kwargs,
        )

        stdout_data, stderr_data = process.communicate(timeout=timeout_seconds)
        exit_code = process.returncode

    except subprocess.TimeoutExpired:
        if process is not None:
            if os.name == "posix":
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except OSError:
                    process.terminate()
            else:
                process.terminate()
            try:
                stdout_data, stderr_data = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except OSError:
                        process.kill()
                else:
                    process.kill()
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
