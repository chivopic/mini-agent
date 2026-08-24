"""Workspace git helpers. Always argv lists, never shell=True."""

from __future__ import annotations

import subprocess
from pathlib import Path

from mini_agent.tools.shell import sanitize_environment


def run_git(argv: list[str], workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run git with a sanitized environment in the workspace."""
    return subprocess.run(
        argv,
        cwd=workspace.resolve(),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        env=sanitize_environment(),
    )


def git_diff(workspace: Path) -> subprocess.CompletedProcess[str]:
    return run_git(["git", "diff"], workspace)


def git_diff_head(workspace: Path) -> subprocess.CompletedProcess[str]:
    return run_git(["git", "diff", "HEAD"], workspace)


def git_status_porcelain(workspace: Path) -> subprocess.CompletedProcess[str]:
    return run_git(["git", "status", "--porcelain"], workspace)


def list_untracked(workspace: Path) -> list[str]:
    result = git_status_porcelain(workspace)
    untracked: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("?? "):
            untracked.append(line[3:])
    return untracked


def git_add_u(workspace: Path) -> subprocess.CompletedProcess[str]:
    return run_git(["git", "add", "-u"], workspace)


def git_commit(workspace: Path, message: str) -> subprocess.CompletedProcess[str]:
    return run_git(["git", "commit", "-m", message], workspace)
