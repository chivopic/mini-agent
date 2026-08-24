"""Unit tests for pre-commit syntax validation guard."""

from pathlib import Path

from mini_agent.models import EditFileInput, WriteFileInput
from mini_agent.syntax_guard import validate_syntax
from mini_agent.tools.filesystem import edit_file, write_file


def test_valid_python_syntax() -> None:
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    is_valid, err = validate_syntax(code, "test.py")
    assert is_valid is True
    assert err is None


def test_invalid_python_syntax_missing_colon() -> None:
    code = "def add(a, b)\n    return a + b\n"
    is_valid, err = validate_syntax(code, "test.py")
    assert is_valid is False
    assert err is not None
    assert "语法错误" in err


def test_invalid_python_indentation_error() -> None:
    code = "def add(a, b):\nreturn a + b\n"
    is_valid, err = validate_syntax(code, "test.py")
    assert is_valid is False
    assert err is not None


def test_valid_json_syntax() -> None:
    code = '{"name": "mini-agent", "version": "0.2.0"}'
    is_valid, err = validate_syntax(code, "config.json")
    assert is_valid is True
    assert err is None


def test_invalid_json_syntax() -> None:
    code = '{"name": "mini-agent", "version": }'
    is_valid, err = validate_syntax(code, "config.json")
    assert is_valid is False
    assert err is not None
    assert "JSON 格式错误" in err


def test_write_file_rejects_invalid_python_before_persist(tmp_path: Path) -> None:
    result = write_file(
        WriteFileInput(path="broken.py", content="def oops(\n"),
        workspace_root=tmp_path,
    )
    assert result.ok is False
    assert result.metadata.get("syntax_error") is True
    assert not (tmp_path / "broken.py").exists()
    assert not (tmp_path / ".broken.py.tmp").exists()


def test_edit_file_rejects_invalid_python_before_persist(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    original = "x = 1\n"
    target.write_text(original, encoding="utf-8")
    result = edit_file(
        EditFileInput(
            path="mod.py",
            target_content="x = 1",
            replacement_content="def oops(",
        ),
        workspace_root=tmp_path,
    )
    assert result.ok is False
    assert result.metadata.get("syntax_error") is True
    assert target.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".mod.py.tmp").exists()


def test_write_file_rejects_invalid_json_before_persist(tmp_path: Path) -> None:
    result = write_file(
        WriteFileInput(path="config.json", content='{"name":'),
        workspace_root=tmp_path,
    )
    assert result.ok is False
    assert result.metadata.get("syntax_error") is True
    assert not (tmp_path / "config.json").exists()
