"""Repository Map and AST symbol extraction engine (OpenCode-class code intelligence)."""

import ast
import os
import re
from pathlib import Path

IGNORED_DIRS = {
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
    "target",
}


def _format_py_arg(arg: ast.arg) -> str:
    """Format AST argument with type annotation if present."""
    if arg.annotation:
        try:
            return f"{arg.arg}: {ast.unparse(arg.annotation)}"
        except Exception:
            return arg.arg
    return arg.arg


def _format_py_arguments(args: ast.arguments) -> str:
    """Format full Python function arguments list."""
    parts = []
    # posonlyargs
    for a in getattr(args, "posonlyargs", []):
        parts.append(_format_py_arg(a))
    if getattr(args, "posonlyargs", []):
        parts.append("/")

    # regular args
    for a in args.args:
        parts.append(_format_py_arg(a))

    # vararg (*args)
    if args.vararg:
        parts.append(f"*{_format_py_arg(args.vararg)}")

    # kwonlyargs
    if args.kwonlyargs:
        if not args.vararg:
            parts.append("*")
        for a in args.kwonlyargs:
            parts.append(_format_py_arg(a))

    # kwarg (**kwargs)
    if args.kwarg:
        parts.append(f"**{_format_py_arg(args.kwarg)}")

    return ", ".join(parts)


def extract_python_symbols(source_code: str) -> list[str]:
    """Extract classes, methods, and functions from Python source code using AST."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    symbols: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases_str = ""
            if node.bases:
                try:
                    bases_list = [ast.unparse(b) for b in node.bases]
                    bases_str = f"({', '.join(bases_list)})"
                except Exception:
                    pass
            symbols.append(f"class {node.name}{bases_str}:")

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = "async def " if isinstance(item, ast.AsyncFunctionDef) else "def "
                    args_str = _format_py_arguments(item.args)
                    ret_str = ""
                    if item.returns:
                        try:
                            ret_str = f" -> {ast.unparse(item.returns)}"
                        except Exception:
                            pass
                    symbols.append(f"  {prefix}{item.name}({args_str}){ret_str}")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            args_str = _format_py_arguments(node.args)
            ret_str = ""
            if node.returns:
                try:
                    ret_str = f" -> {ast.unparse(node.returns)}"
                except Exception:
                    pass
            symbols.append(f"{prefix}{node.name}({args_str}){ret_str}")

    return symbols


def extract_js_ts_symbols(source_code: str) -> list[str]:
    """Extract exported classes, functions, and interfaces from JS/TS source code."""
    symbols: list[str] = []
    # Pattern for functions, classes, interfaces
    pattern = re.compile(
        r"^(?:export\s+)?(?:default\s+)?(class|interface|type|function|const|let|async\s+function)\s+([a-zA-Z0-9_$]+)",
        re.MULTILINE,
    )
    for match in pattern.finditer(source_code):
        kind, name = match.group(1), match.group(2)
        symbols.append(f"{kind} {name}")
    return symbols


def extract_file_symbols(file_path: Path) -> list[str]:
    """Extract code skeleton symbols for a given file."""
    ext = file_path.suffix.lower()
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    if ext == ".py":
        return extract_python_symbols(content)
    if ext in (".js", ".ts", ".jsx", ".tsx", ".mjs"):
        return extract_js_ts_symbols(content)
    return []


def generate_repo_map(
    workspace_root: Path,
    max_tokens: int = 1500,
    boundary_root: Path | None = None,
) -> str:
    """Generate a condensed Repo Map without following files outside the allowed boundary."""
    resolved_root = workspace_root.resolve()
    resolved_boundary = (boundary_root or resolved_root).resolve()
    if resolved_root != resolved_boundary and not resolved_root.is_relative_to(resolved_boundary):
        return ""
    lines: list[str] = []

    code_files: list[Path] = []
    for root, dirs, files in os.walk(resolved_root, topdown=True):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for f in sorted(files):
            if f in IGNORED_DIRS:
                continue
            fp = Path(root) / f
            try:
                resolved_fp = fp.resolve()
            except OSError:
                continue
            if resolved_fp != resolved_boundary and not resolved_fp.is_relative_to(
                resolved_boundary
            ):
                continue
            if fp.suffix.lower() in (
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".rs",
                ".go",
            ):
                code_files.append(resolved_fp)

    for fp in code_files:
        try:
            rel_path = fp.resolve().relative_to(resolved_root).as_posix()
        except ValueError:
            rel_path = fp.name

        symbols = extract_file_symbols(fp)
        if symbols:
            lines.append(f"{rel_path}:")
            for s in symbols[:20]:  # Cap symbols per file
                lines.append(f"  {s}")

    full_map = "\n".join(lines)
    # Simple character truncation (~4 chars per token)
    max_chars = max_tokens * 4
    if len(full_map) > max_chars:
        return full_map[:max_chars] + "\n... [Repo Map 截断] ..."
    return full_map
