"""Unit tests for Repository Map and AST symbol extraction."""

from pathlib import Path

from mini_agent.repomap import (
    extract_js_ts_symbols,
    extract_python_symbols,
    generate_repo_map,
)


def test_extract_python_symbols() -> None:
    code = """
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    async def fetch_data(self, url: str) -> dict:
        pass

def global_helper(flag: bool = True) -> None:
    pass
"""
    symbols = extract_python_symbols(code)
    assert len(symbols) == 4
    assert symbols[0] == "class Calculator:"
    assert "def add(self, a: int, b: int) -> int" in symbols[1]
    assert "async def fetch_data(self, url: str) -> dict" in symbols[2]
    assert (
        "def global_helper(flag: bool) -> None" in symbols[3] or "def global_helper" in symbols[3]
    )


def test_extract_js_ts_symbols() -> None:
    code = """
export class UserService {
    getUser() {}
}
export function calculateTax(amount) {}
export interface UserConfig {}
"""
    symbols = extract_js_ts_symbols(code)
    assert "class UserService" in symbols
    assert "function calculateTax" in symbols
    assert "interface UserConfig" in symbols


def test_generate_repo_map(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text(
        "class App:\n    def run(self) -> None:\n        pass\n", encoding="utf-8"
    )
    (src / "utils.js").write_text("export function formatString() {}\n", encoding="utf-8")

    repo_map = generate_repo_map(tmp_path)
    assert "src/app.py:" in repo_map
    assert "class App:" in repo_map
    assert "def run(self) -> None" in repo_map
    assert "src/utils.js:" in repo_map
    assert "function formatString" in repo_map


def test_generate_repo_map_skips_external_file_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def outside_secret():\n    pass\n", encoding="utf-8")
    (workspace / "linked.py").symlink_to(outside)

    repo_map = generate_repo_map(workspace, boundary_root=workspace)

    assert "outside_secret" not in repo_map
