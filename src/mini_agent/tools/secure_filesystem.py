"""Security-hardened filesystem tools used by the agent runtime."""

import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from mini_agent.models import (
    EditFileInput,
    ListFilesInput,
    ReadFileInput,
    SearchCodeInput,
    ToolResult,
    WriteFileInput,
)
from mini_agent.tools.filesystem import (
    BINARY_EXTENSIONS,
    IGNORED_NAMES,
    resolve_relative_path,
    truncate_text,
)
from mini_agent.tools.filesystem import (
    list_files as _list_files,
)

SAFE_ENV_TEMPLATE_NAMES = {
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.dist",
}
SENSITIVE_EXACT_NAMES = {
    ".npmrc",
    ".pypirc",
    ".netrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
SENSITIVE_RELATIVE_PATHS = {
    ".aws/credentials",
    ".docker/config.json",
    ".config/gcloud/application_default_credentials.json",
}


def is_sensitive_path(path: Path, workspace_root: Path) -> bool:
    """Return whether a workspace path is likely to contain credentials or private keys."""
    try:
        relative = path.resolve().relative_to(workspace_root.resolve())
    except (OSError, ValueError):
        return True

    rel_posix = relative.as_posix().lower()
    name = relative.name.lower()

    if name in SAFE_ENV_TEMPLATE_NAMES:
        return False
    if name == ".env" or name.startswith(".env."):
        return True
    if name in SENSITIVE_EXACT_NAMES or relative.suffix.lower() in SENSITIVE_SUFFIXES:
        return True
    if rel_posix in SENSITIVE_RELATIVE_PATHS:
        return True
    return False


def _sensitive_result(path: str, action: str) -> ToolResult:
    return ToolResult(
        ok=False,
        content="",
        error=(
            f"安全策略拒绝{action}疑似敏感文件 '{path}'。"
            "为避免 API Key、凭据或私钥进入模型上下文或被意外修改，请改用模板文件"
            "（如 .env.example）或由用户在 Agent 外部处理该文件。"
        ),
        metadata={"path": path, "sensitive": True},
    )


def _atomic_write_text(target: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file with fsync and same-directory rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_mode: int | None = None
    if target.exists():
        existing_mode = stat.S_IMODE(target.stat().st_mode)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)

        if existing_mode is not None:
            os.chmod(temp_path, existing_mode)
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def list_files(
    input_data: ListFilesInput,
    workspace_root: Path,
    max_output_chars: int = 12_000,
) -> ToolResult:
    """List workspace files using the existing bounded directory traversal."""
    return _list_files(
        input_data,
        workspace_root=workspace_root,
        max_output_chars=max_output_chars,
    )


def read_file(
    input_data: ReadFileInput,
    workspace_root: Path,
    max_file_bytes: int = 100 * 1024,
    max_output_chars: int = 12_000,
) -> ToolResult:
    """Read a non-sensitive UTF-8 file within the workspace."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )
    if is_sensitive_path(resolved_path, workspace_root):
        return _sensitive_result(input_data.path, "读取")
    if not resolved_path.exists():
        return ToolResult(
            ok=False,
            content="",
            error=f"文件不存在: '{input_data.path}'",
            metadata={"path": input_data.path},
        )
    if not resolved_path.is_file():
        return ToolResult(
            ok=False,
            content="",
            error=f"路径不是普通文件: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    try:
        size = resolved_path.stat().st_size
    except OSError as exc:
        return ToolResult(
            ok=False,
            content="",
            error=f"无法获取文件状态: '{input_data.path}' ({exc})",
            metadata={"path": input_data.path},
        )
    if size > max_file_bytes:
        return ToolResult(
            ok=False,
            content="",
            error=(
                f"文件体积过大 ({size} 字节，上限 {max_file_bytes} 字节): '{input_data.path}'，"
                "请指定更小文件。"
            ),
            metadata={"path": input_data.path, "size_bytes": size, "truncated": False},
        )

    try:
        raw_content = resolved_path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return ToolResult(
            ok=False,
            content="",
            error=f"文件不是 UTF-8 编码文本文件: '{input_data.path}'",
            metadata={"path": input_data.path, "size_bytes": size},
        )
    except OSError as exc:
        return ToolResult(
            ok=False,
            content="",
            error=f"无法读取文件 '{input_data.path}': {exc}",
            metadata={"path": input_data.path, "size_bytes": size},
        )

    content, truncated = truncate_text(raw_content, max_chars=max_output_chars)
    return ToolResult(
        ok=True,
        content=content,
        error=None,
        metadata={"path": input_data.path, "size_bytes": size, "truncated": truncated},
    )


def write_file(input_data: WriteFileInput, workspace_root: Path) -> ToolResult:
    """Atomically write a non-sensitive text file within the workspace."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )
    if is_sensitive_path(resolved_path, workspace_root):
        return _sensitive_result(input_data.path, "写入")

    try:
        _atomic_write_text(resolved_path, input_data.content)
    except OSError as exc:
        return ToolResult(
            ok=False,
            content="",
            error=f"写入文件失败 '{input_data.path}': {exc}",
            metadata={"path": input_data.path},
        )

    size = len(input_data.content.encode("utf-8"))
    return ToolResult(
        ok=True,
        content=f"已成功写入文件 '{input_data.path}' ({size} 字节)。",
        error=None,
        metadata={"path": input_data.path, "bytes_written": size, "atomic": True},
    )


def edit_file(input_data: EditFileInput, workspace_root: Path) -> ToolResult:
    """Atomically replace one unique snippet in a non-sensitive UTF-8 file."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )
    if is_sensitive_path(resolved_path, workspace_root):
        return _sensitive_result(input_data.path, "修改")
    if not resolved_path.exists():
        return ToolResult(
            ok=False,
            content="",
            error=f"目标文件不存在: '{input_data.path}'",
            metadata={"path": input_data.path},
        )
    if not resolved_path.is_file():
        return ToolResult(
            ok=False,
            content="",
            error=f"路径不是普通文件: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    try:
        original_content = resolved_path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return ToolResult(
            ok=False,
            content="",
            error=f"文件不是 UTF-8 编码文本: '{input_data.path}'",
            metadata={"path": input_data.path},
        )
    except OSError as exc:
        return ToolResult(
            ok=False,
            content="",
            error=f"无法读取文件 '{input_data.path}': {exc}",
            metadata={"path": input_data.path},
        )

    if input_data.target_content not in original_content:
        return ToolResult(
            ok=False,
            content="",
            error=(
                f"在文件 '{input_data.path}' 中未找到目标代码片段。"
                "请先使用 read_file 重新确认文件当前最新内容。"
            ),
            metadata={"path": input_data.path},
        )

    match_count = original_content.count(input_data.target_content)
    if match_count > 1:
        return ToolResult(
            ok=False,
            content="",
            error=(
                f"在文件 '{input_data.path}' 中找到了 {match_count} 处匹配的目标代码。"
                "匹配不唯一，请包含更多上下文行以确保精准替换。"
            ),
            metadata={"path": input_data.path, "match_count": match_count},
        )

    new_content = original_content.replace(
        input_data.target_content, input_data.replacement_content, 1
    )
    try:
        _atomic_write_text(resolved_path, new_content)
    except OSError as exc:
        return ToolResult(
            ok=False,
            content="",
            error=f"保存修改失败 '{input_data.path}': {exc}",
            metadata={"path": input_data.path},
        )

    return ToolResult(
        ok=True,
        content=f"已成功修改文件 '{input_data.path}'。",
        error=None,
        metadata={"path": input_data.path, "atomic": True},
    )


def search_code(
    input_data: SearchCodeInput,
    workspace_root: Path,
    max_output_chars: int = 12_000,
) -> ToolResult:
    """Search text files while excluding likely credential and private-key files."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error or "路径错误",
            metadata={"path": input_data.path},
        )
    if is_sensitive_path(resolved_path, workspace_root):
        return _sensitive_result(input_data.path, "检索")
    if not resolved_path.exists():
        return ToolResult(
            ok=False,
            content="",
            error=f"指定的搜索路径不存在: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    flags = 0 if input_data.case_sensitive else re.IGNORECASE
    try:
        pattern = (
            re.compile(input_data.pattern, flags)
            if input_data.is_regex
            else re.compile(re.escape(input_data.pattern), flags)
        )
    except re.error as exc:
        return ToolResult(
            ok=False,
            content="",
            error=f"正则表达式格式错误: '{input_data.pattern}' ({exc})",
            metadata={"pattern": input_data.pattern},
        )

    files_to_search: list[Path] = []
    if resolved_path.is_file():
        files_to_search.append(resolved_path)
    else:
        for root, dirs, files in os.walk(resolved_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in IGNORED_NAMES]
            for filename in sorted(files):
                if filename in IGNORED_NAMES:
                    continue
                file_path = Path(root) / filename
                if file_path.suffix.lower() in BINARY_EXTENSIONS:
                    continue
                if is_sensitive_path(file_path, workspace_root):
                    continue
                files_to_search.append(file_path)

    matches: list[str] = []
    files_searched = 0
    resolved_root = workspace_root.resolve()

    for file_path in files_to_search:
        if len(matches) >= input_data.max_results:
            break
        try:
            if file_path.stat().st_size > 1_000_000:
                continue
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            files_searched += 1
            rel_path = file_path.resolve().relative_to(resolved_root).as_posix()
            for line_no, line in enumerate(lines, start=1):
                if pattern.search(line):
                    matches.append(f"{rel_path}:{line_no}: {line}")
                    if len(matches) >= input_data.max_results:
                        break
        except (OSError, ValueError):
            continue

    metadata: dict[str, Any] = {
        "pattern": input_data.pattern,
        "total_matches": len(matches),
        "files_searched": files_searched,
    }
    if not matches:
        return ToolResult(
            ok=True,
            content=(
                f"未找到与模式 '{input_data.pattern}' 匹配的代码内容 "
                f"(已检索 {files_searched} 个文件)。"
            ),
            metadata=metadata,
        )

    result_text = f"找到 {len(matches)} 处匹配代码 (已检索 {files_searched} 个文件):\n" + "\n".join(
        f"- {match}" for match in matches
    )
    content, truncated = truncate_text(result_text, max_chars=max_output_chars)
    metadata["truncated"] = truncated
    return ToolResult(ok=True, content=content, metadata=metadata)


__all__ = [
    "edit_file",
    "is_sensitive_path",
    "list_files",
    "read_file",
    "search_code",
    "write_file",
]
