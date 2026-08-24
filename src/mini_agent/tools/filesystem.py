"""Safe filesystem tools for mini-agent."""

import os
import re
from pathlib import Path
from typing import Any

from mini_agent.models import (
    EditFileInput,
    GetRepoMapInput,
    ListFilesInput,
    PermissionClass,
    ReadFileInput,
    SearchCodeInput,
    ToolResult,
    WriteFileInput,
)
from mini_agent.repomap import generate_repo_map
from mini_agent.syntax_guard import validate_syntax
from mini_agent.tools.protocol import ToolContext, ToolKind

IGNORED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".DS_Store",
    "dist",
    "build",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".whl",
    ".pyc",
    ".wasm",
    ".bin",
    ".dylib",
    ".so",
    ".exe",
    ".dll",
    ".mp4",
    ".mp3",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}

RANGE_MAX_LINES = 400
RANGE_MAX_BYTES = 100 * 1024


def truncate_text(text: str, max_chars: int = 12_000) -> tuple[str, bool]:
    """Truncate text if it exceeds max_chars, keeping head and tail with a marker."""
    if len(text) <= max_chars:
        return text, False

    if max_chars < 50:
        return text[:max_chars] + "...", True

    half = (max_chars - 50) // 2
    omitted = len(text) - (half * 2)
    truncated = f"{text[:half]}\n... [已省略 {omitted} 字符] ...\n{text[-half:]}"
    return truncated, True


def resolve_relative_path(
    workspace_root: Path, user_path: str | Path
) -> tuple[Path | None, str | None]:
    """Safely resolve user-provided relative path within workspace_root.

    Returns:
        tuple[Path | None, str | None]: (resolved_path, error_message)
    """
    raw_str = str(user_path).strip()
    if not raw_str:
        return None, "路径不能为空"

    # Reject absolute paths explicitly
    path_obj = Path(raw_str)
    if path_obj.is_absolute() or raw_str.startswith("/") or raw_str.startswith("\\"):
        return None, f"非法绝对路径: '{raw_str}'。文件工具只允许访问工作区内的相对路径。"

    resolved_root = workspace_root.resolve()
    candidate = resolved_root / path_obj

    # Check for path traversal escaping workspace
    try:
        resolved_path = candidate.resolve()
    except OSError as exc:
        return None, f"路径解析失败: '{raw_str}' ({exc})"

    # Must be workspace root or inside workspace root
    if resolved_path != resolved_root and not resolved_path.is_relative_to(resolved_root):
        return None, f"路径越界被拒绝: '{raw_str}' 指向工作区外部。"

    # Check symlinks for symlink escape
    if candidate.is_symlink():
        try:
            target = candidate.resolve()
            if target != resolved_root and not target.is_relative_to(resolved_root):
                return None, f"符号链接越界被拒绝: '{raw_str}' 指向工作区外部。"
        except OSError as exc:
            return None, f"无法解析符号链接: '{raw_str}' ({exc})"

    return resolved_path, None


_READ_CHUNK = 64 * 1024


def _text_line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _decode_text_line(raw: bytes, *, strict: bool) -> str:
    text = raw.decode("utf-8") if strict else raw.decode("utf-8", errors="ignore")
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _BinLines:
    """Binary line scanner that can unread leftover bytes without seeking."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._buf = b""

    def _read(self, n: int) -> bytes:
        if n <= 0:
            return b""
        if self._buf:
            take = self._buf[:n]
            self._buf = self._buf[n:]
            if len(take) < n:
                take += self._handle.read(n - len(take))
            return take
        return self._handle.read(n)

    def _unread(self, data: bytes) -> None:
        if data:
            self._buf = data + self._buf

    def skip_line(self) -> bool:
        """Skip one line. False if already at EOF."""
        found = False
        while True:
            chunk = self._read(_READ_CHUNK)
            if not chunk:
                return found
            found = True
            idx = chunk.find(b"\n")
            if idx != -1:
                self._unread(chunk[idx + 1 :])
                return True

    def read_line(self, max_bytes: int | None) -> tuple[bytes, bool]:
        """Read one line, optionally capped. ``(b'', False)`` at EOF.

        The bool is True when ``max_bytes`` was hit before a newline.
        """
        if max_bytes is not None and max_bytes <= 0:
            return b"", True
        parts: list[bytes] = []
        got = 0
        while True:
            to_read = _READ_CHUNK if max_bytes is None else min(_READ_CHUNK, max_bytes - got)
            if to_read <= 0:
                more = self._read(1)
                if not more:
                    return b"".join(parts), False
                self._unread(more)
                return b"".join(parts), True
            chunk = self._read(to_read)
            if not chunk:
                return b"".join(parts), False
            idx = chunk.find(b"\n")
            if idx != -1:
                parts.append(chunk[: idx + 1])
                self._unread(chunk[idx + 1 :])
                return b"".join(parts), False
            parts.append(chunk)
            got += len(chunk)

    def count_remaining_lines(self) -> int:
        count = 0
        any_data = False
        ends_with_nl = True
        while True:
            chunk = self._read(_READ_CHUNK)
            if not chunk:
                break
            any_data = True
            count += chunk.count(b"\n")
            ends_with_nl = chunk.endswith(b"\n")
        if any_data and not ends_with_nl:
            count += 1
        return count


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.tmp"
    try:
        with open(temp_path, mode="w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_text_range(
    resolved_path: Path,
    offset: int | None,
    limit: int | None,
    max_lines: int = RANGE_MAX_LINES,
    max_bytes: int = RANGE_MAX_BYTES,
) -> tuple[str, dict[str, Any]]:
    start_line = 1 if offset is None else offset
    line_budget = max_lines if limit is None else min(limit, max_lines)
    collected: list[str] = []
    nbytes = 0
    line_no = 0
    end_line = start_line - 1
    truncated_mid_line = False

    with open(resolved_path, "rb") as raw_handle:
        scanner = _BinLines(raw_handle)
        while line_no < start_line - 1:
            if not scanner.skip_line():
                break
            line_no += 1

        while len(collected) < line_budget:
            remaining = max_bytes - nbytes
            if remaining <= 0:
                break
            raw, hit_cap = scanner.read_line(remaining)
            if not raw:
                break
            line_no += 1
            if hit_cap:
                truncated_mid_line = True
                if not collected:
                    collected.append(_decode_text_line(raw, strict=False))
                    end_line = line_no
                break
            collected.append(_decode_text_line(raw, strict=True))
            nbytes += len(raw)
            end_line = line_no

        if truncated_mid_line:
            scanner.skip_line()
        extra = scanner.count_remaining_lines()

    total_lines = line_no + extra
    if offset is not None and start_line > total_lines:
        return "", {
            "truncated": False,
            "start_line": start_line,
            "total_lines": total_lines,
            "offset_past_eof": True,
        }

    hit_internal_line_cap = (
        extra > 0 and len(collected) >= line_budget and (limit is None or limit > max_lines)
    )
    truncated = truncated_mid_line or hit_internal_line_cap or extra > 0 and nbytes >= max_bytes
    return "".join(collected), {
        "truncated": truncated,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
    }


def read_file(
    input_data: ReadFileInput,
    workspace_root: Path,
    max_file_bytes: int = 100 * 1024,
    max_output_chars: int = 12_000,
) -> ToolResult:
    """Read UTF-8 content of a file within workspace."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )

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

    has_range = input_data.offset is not None or input_data.limit is not None
    if not has_range and size > max_file_bytes:
        return ToolResult(
            ok=False,
            content="",
            error=(
                f"文件体积过大 ({size} 字节，上限 {max_file_bytes} 字节): '{input_data.path}'，"
                "请指定更小文件，或使用 offset/limit 按行区间读取"
                f"（例如 offset=1, limit={RANGE_MAX_LINES}）。"
            ),
            metadata={"path": input_data.path, "size_bytes": size, "truncated": False},
        )

    try:
        if has_range:
            content, range_meta = _read_text_range(
                resolved_path,
                offset=input_data.offset,
                limit=input_data.limit,
            )
            offset_past_eof = range_meta.pop("offset_past_eof", False)
            metadata = {"path": input_data.path, "size_bytes": size, **range_meta}
            if offset_past_eof:
                start = range_meta.get("start_line")
                total = range_meta.get("total_lines", 0)
                return ToolResult(
                    ok=False,
                    content="",
                    error=(f"起始行 {start} 超出文件末尾 (共 {total} 行): '{input_data.path}'。"),
                    metadata=metadata,
                )
            return ToolResult(
                ok=True,
                content=content,
                error=None,
                metadata=metadata,
            )

        with open(resolved_path, encoding="utf-8", errors="strict") as f:
            raw_content = f.read()
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

    total_lines = _text_line_count(raw_content)
    content, truncated = truncate_text(raw_content, max_chars=max_output_chars)
    metadata: dict[str, Any] = {
        "path": input_data.path,
        "size_bytes": size,
        "truncated": truncated,
        "total_lines": total_lines,
    }
    if not truncated:
        metadata["start_line"] = 1
        metadata["end_line"] = total_lines
    return ToolResult(
        ok=True,
        content=content,
        error=None,
        metadata=metadata,
    )


def list_files(
    input_data: ListFilesInput,
    workspace_root: Path,
    max_entries: int = 500,
    max_output_chars: int = 12_000,
) -> ToolResult:
    """List directory contents within workspace up to max_depth."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )

    if not resolved_path.exists():
        return ToolResult(
            ok=False,
            content="",
            error=f"目录不存在: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    if not resolved_path.is_dir():
        return ToolResult(
            ok=False,
            content="",
            error=f"路径不是目录: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    resolved_root = workspace_root.resolve()
    max_depth = max(1, min(input_data.max_depth, 5))
    entries: list[str] = []
    skipped_count = 0
    reached_limit = False

    def _traverse(current_dir: Path, current_depth: int) -> None:
        nonlocal skipped_count, reached_limit
        if current_depth > max_depth or reached_limit:
            return

        try:
            items = sorted(os.scandir(current_dir), key=lambda e: (not e.is_dir(), e.name.lower()))
        except (PermissionError, OSError):
            skipped_count += 1
            return

        for item in items:
            if len(entries) >= max_entries:
                reached_limit = True
                return

            name = item.name
            if name in IGNORED_NAMES:
                continue

            try:
                is_dir = item.is_dir(follow_symlinks=False)
                is_symlink = item.is_symlink()
            except OSError:
                skipped_count += 1
                continue

            # Check for symlinks pointing outside
            if is_symlink:
                try:
                    target = Path(item.path).resolve()
                    if target != resolved_root and not target.is_relative_to(resolved_root):
                        skipped_count += 1
                        continue
                except OSError:
                    skipped_count += 1
                    continue

            # Compute relative path to workspace root
            rel_to_root = Path(item.path).relative_to(resolved_root).as_posix()
            display_path = f"{rel_to_root}/" if is_dir else rel_to_root
            entries.append(display_path)

            if is_dir and not is_symlink:
                _traverse(Path(item.path), current_depth + 1)

    _traverse(resolved_path, current_depth=1)

    metadata: dict[str, Any] = {
        "path": input_data.path,
        "max_depth": max_depth,
        "total_entries": len(entries),
        "skipped_inaccessible": skipped_count,
        "reached_limit": reached_limit,
    }

    if not entries:
        content = "(目录为空)"
    else:
        lines = [f"- {e}" for e in entries]
        if reached_limit:
            lines.append(f"\n[已达到最大条目上限 {max_entries} 项]")
        if skipped_count > 0:
            lines.append(f"[跳过 {skipped_count} 个无法访问或越界条目]")
        content = "\n".join(lines)

    content, truncated = truncate_text(content, max_chars=max_output_chars)
    metadata["truncated"] = truncated

    return ToolResult(
        ok=True,
        content=content,
        error=None,
        metadata=metadata,
    )


def write_file(input_data: WriteFileInput, workspace_root: Path) -> ToolResult:
    """Safely write full content to a file within workspace."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )

    is_valid, syntax_err = validate_syntax(input_data.content, resolved_path)
    if not is_valid:
        return ToolResult(
            ok=False,
            content="",
            error=syntax_err,
            metadata={"path": input_data.path, "syntax_error": True},
        )

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
        metadata={"path": input_data.path, "bytes_written": size},
    )


def edit_file(input_data: EditFileInput, workspace_root: Path) -> ToolResult:
    """Replace target_content with replacement_content; unique match unless replace_all."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )

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
        with open(resolved_path, encoding="utf-8", errors="strict") as f:
            original_content = f.read()
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
    if match_count > 1 and not input_data.replace_all:
        return ToolResult(
            ok=False,
            content="",
            error=(
                f"在文件 '{input_data.path}' 中找到了 {match_count} 处匹配的目标代码。"
                "匹配不唯一，请包含更多上下文行以确保精准替换。"
            ),
            metadata={"path": input_data.path, "match_count": match_count},
        )

    if input_data.replace_all:
        new_content = original_content.replace(
            input_data.target_content, input_data.replacement_content
        )
    else:
        new_content = original_content.replace(
            input_data.target_content, input_data.replacement_content, 1
        )

    is_valid, syntax_err = validate_syntax(new_content, resolved_path)
    if not is_valid:
        return ToolResult(
            ok=False,
            content="",
            error=syntax_err,
            metadata={"path": input_data.path, "syntax_error": True},
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
        metadata={"path": input_data.path, "match_count": match_count},
    )


def search_code(
    input_data: SearchCodeInput,
    workspace_root: Path,
    max_output_chars: int = 12_000,
) -> ToolResult:
    """Recursively search for regex pattern or text across workspace text files."""
    resolved_path, err = resolve_relative_path(workspace_root, input_data.path)
    if err or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=err or "路径错误",
            metadata={"path": input_data.path},
        )

    if not resolved_path.exists():
        return ToolResult(
            ok=False,
            content="",
            error=f"指定的搜索路径不存在: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    flags = 0 if input_data.case_sensitive else re.IGNORECASE
    try:
        if input_data.is_regex:
            pattern = re.compile(input_data.pattern, flags)
        else:
            pattern = re.compile(re.escape(input_data.pattern), flags)
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
            for f in sorted(files):
                if f in IGNORED_NAMES:
                    continue
                file_path = Path(root) / f
                if file_path.suffix.lower() in BINARY_EXTENSIONS:
                    continue
                files_to_search.append(file_path)

    matches: list[str] = []
    files_searched = 0
    resolved_root = workspace_root.resolve()

    for fp in files_to_search:
        if len(matches) >= input_data.max_results:
            break
        try:
            if fp.stat().st_size > 1_000_000:
                continue
            with open(fp, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            files_searched += 1
            rel_p = fp.resolve().relative_to(resolved_root).as_posix()
            for line_no, line in enumerate(lines, start=1):
                if pattern.search(line):
                    clean_line = line.rstrip("\r\n")
                    matches.append(f"{rel_p}:{line_no}: {clean_line}")
                    if len(matches) >= input_data.max_results:
                        break
        except OSError:
            continue

    if not matches:
        return ToolResult(
            ok=True,
            content=(
                f"未找到与模式 '{input_data.pattern}' 匹配的代码内容 "
                f"(已检索 {files_searched} 个文件)。"
            ),
            metadata={
                "pattern": input_data.pattern,
                "total_matches": 0,
                "files_searched": files_searched,
            },
        )

    result_text = f"找到 {len(matches)} 处匹配代码 (已检索 {files_searched} 个文件):\n" + "\n".join(
        f"- {m}" for m in matches
    )
    truncated, is_trunc = truncate_text(result_text, max_chars=max_output_chars)
    return ToolResult(
        ok=True,
        content=truncated,
        metadata={
            "pattern": input_data.pattern,
            "total_matches": len(matches),
            "files_searched": files_searched,
            "truncated": is_trunc,
        },
    )


def get_repo_map(input_data: GetRepoMapInput, workspace_root: Path) -> ToolResult:
    """Build a repo map for a sandboxed relative path inside workspace."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )

    if not resolved_path.exists():
        return ToolResult(
            ok=False,
            content="",
            error=f"指定的目录不存在: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    repo_map = generate_repo_map(resolved_path)
    return ToolResult(
        ok=True,
        content=repo_map if repo_map else "未在当前目录发现有效的代码文件与符号。",
        metadata={"path": input_data.path},
    )


class GetRepoMapTool:
    name = "get_repo_map"
    description = (
        "提取工作区各代码文件的类名、方法名与函数签名，生成全局代码骨架拓扑地图 (Repo Map)。"
    )
    permission = PermissionClass.READ
    kind = ToolKind.READONLY
    input_model = GetRepoMapInput

    def execute(self, inp: GetRepoMapInput, ctx: ToolContext) -> ToolResult:
        return get_repo_map(inp, workspace_root=ctx.workspace_root)

    def format_call(self, inp: GetRepoMapInput) -> str:
        return f"get_repo_map path={inp.path}"

    def approval_pattern(self, inp: GetRepoMapInput) -> str:
        return f"read:{inp.path}"


class SearchCodeTool:
    name = "search_code"
    description = (
        "在工作区文本文件中递归搜索关键词或正则表达式模式，返回匹配的文件路径、行号与代码行。"
    )
    permission = PermissionClass.READ
    kind = ToolKind.READONLY
    input_model = SearchCodeInput

    def execute(self, inp: SearchCodeInput, ctx: ToolContext) -> ToolResult:
        return search_code(
            inp,
            workspace_root=ctx.workspace_root,
            max_output_chars=ctx.config.max_output_chars,
        )

    def format_call(self, inp: SearchCodeInput) -> str:
        return f"search_code pattern={inp.pattern!r} path={inp.path}"

    def approval_pattern(self, inp: SearchCodeInput) -> str:
        return f"read:{inp.path}"


class ListFilesTool:
    name = "list_files"
    description = "列出工作区内指定相对目录的文件和子目录结构。"
    permission = PermissionClass.READ
    kind = ToolKind.READONLY
    input_model = ListFilesInput

    def execute(self, inp: ListFilesInput, ctx: ToolContext) -> ToolResult:
        return list_files(
            inp,
            workspace_root=ctx.workspace_root,
            max_output_chars=ctx.config.max_output_chars,
        )

    def format_call(self, inp: ListFilesInput) -> str:
        return f"list_files path={inp.path} max_depth={inp.max_depth}"

    def approval_pattern(self, inp: ListFilesInput) -> str:
        return f"read:{inp.path}"


class ReadFileTool:
    name = "read_file"
    description = (
        "读取工作区内指定 UTF-8 文本文件的内容。"
        "可通过 offset（从 1 起的行号）与 limit 读取区间；"
        "返回的 content 是原文切片，不含行号前缀。"
    )
    permission = PermissionClass.READ
    kind = ToolKind.READONLY
    input_model = ReadFileInput

    def execute(self, inp: ReadFileInput, ctx: ToolContext) -> ToolResult:
        return read_file(
            inp,
            workspace_root=ctx.workspace_root,
            max_output_chars=ctx.config.max_output_chars,
        )

    def format_call(self, inp: ReadFileInput) -> str:
        extra = ""
        if inp.offset is not None:
            extra += f" offset={inp.offset}"
        if inp.limit is not None:
            extra += f" limit={inp.limit}"
        return f"read_file path={inp.path}{extra}"

    def approval_pattern(self, inp: ReadFileInput) -> str:
        return f"read:{inp.path}"


class EditFileTool:
    name = "edit_file"
    description = (
        "在已有文件中搜索 target_content 并替换为 replacement_content；"
        "默认必须唯一匹配，replace_all=true 时替换全部出现。"
    )
    permission = PermissionClass.EDIT
    kind = ToolKind.MUTATING
    input_model = EditFileInput

    def execute(self, inp: EditFileInput, ctx: ToolContext) -> ToolResult:
        return edit_file(inp, workspace_root=ctx.workspace_root)

    def format_call(self, inp: EditFileInput) -> str:
        extra = " replace_all=true" if inp.replace_all else ""
        return f"edit_file path={inp.path}{extra}"

    def approval_pattern(self, inp: EditFileInput) -> str:
        return f"edit:{inp.path}"


class WriteFileTool:
    name = "write_file"
    description = "创建新文件或覆盖写入完整内容。"
    permission = PermissionClass.EDIT
    kind = ToolKind.MUTATING
    input_model = WriteFileInput

    def execute(self, inp: WriteFileInput, ctx: ToolContext) -> ToolResult:
        return write_file(inp, workspace_root=ctx.workspace_root)

    def format_call(self, inp: WriteFileInput) -> str:
        return f"write_file path={inp.path}"

    def approval_pattern(self, inp: WriteFileInput) -> str:
        return f"edit:{inp.path}"
