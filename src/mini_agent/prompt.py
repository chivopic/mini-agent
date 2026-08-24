"""System prompt assembly from the tool registry and frozen development rules."""

from pathlib import Path

from mini_agent.rules import load_project_rules
from mini_agent.tools.registry import ToolRegistry

_DEV_GUIDELINES = (
    "请严格遵守以下开发准则：\n"
    "1. 在分析大型项目时，可优先调用 `get_repo_map` 获取全局代码骨架；\n"
    "2. 在不知道某个函数或变量定义在哪个文件时，优先使用 `search_code` 检索；\n"
    "3. 修改代码前，务必先调用 `read_file` 查看最新代码，确保上下文完全一致；\n"
    "4. 修改现有代码时，优先使用 `edit_file` 进行精准局部替换，提供唯一的 `target_content`；\n"
    "5. 仅在创建全新文件时使用 `write_file`；\n"
    "6. 所有文件路径必须相对于工作区根目录；工具执行完毕后，使用中文清晰解释修改的内容和原因。"
)

_RANGE_EDIT_HINT = (
    "`read_file` 的 metadata 含行号；编辑时 `target_content` 必须是文件正文，不要带行号前缀。"
)
_TEST_AFTER_EDIT_HINT = "修改后如有测试，可通过 `run_shell` 执行 `uv run pytest`，不要自动跑测试。"


def get_system_prompt(workspace_root: Path, registry: ToolRegistry | None = None) -> str:
    """Generate system instructions for the Agent with project rules if present."""
    if registry is None:
        from mini_agent.tools import default_registry

        registry = default_registry()

    bullets = "\n".join(f"- `{tool.name}`：{tool.description}" for tool in registry.list())
    base_prompt = (
        f"你是一个运行在工作区 '{workspace_root.as_posix()}' 的智能开发助手 (mini-agent)。\n"
        "你可以使用以下工具进行开发：\n"
        f"{bullets}\n\n"
        f"{_DEV_GUIDELINES}\n{_RANGE_EDIT_HINT}\n{_TEST_AFTER_EDIT_HINT}"
    )

    project_rules = load_project_rules(workspace_root)
    if project_rules:
        return base_prompt + project_rules
    return base_prompt
