"""Tools package for mini-agent."""

from mini_agent.tools.filesystem import (
    EditFileTool,
    GetRepoMapTool,
    ListFilesTool,
    ReadFileTool,
    SearchCodeTool,
    WriteFileTool,
    edit_file,
    get_repo_map,
    list_files,
    read_file,
    search_code,
    write_file,
)
from mini_agent.tools.registry import ToolRegistry
from mini_agent.tools.shell import RunShellTool, check_command_safety, run_shell


def default_registry() -> ToolRegistry:
    """Return a registry with the built-in tools."""
    registry = ToolRegistry()
    for tool in (
        GetRepoMapTool(),
        SearchCodeTool(),
        ReadFileTool(),
        ListFilesTool(),
        EditFileTool(),
        WriteFileTool(),
        RunShellTool(),
    ):
        registry.register(tool)
    return registry


__all__ = [
    "read_file",
    "list_files",
    "search_code",
    "write_file",
    "edit_file",
    "get_repo_map",
    "run_shell",
    "check_command_safety",
    "default_registry",
    "ToolRegistry",
]
