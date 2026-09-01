"""Regression tests for the security-hardened agent filesystem boundary."""

from pathlib import Path

from mini_agent.models import EditFileInput, ReadFileInput, SearchCodeInput, WriteFileInput
from mini_agent.tools.secure_filesystem import edit_file, read_file, search_code, write_file


def test_read_env_file_is_blocked(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=super-secret", encoding="utf-8")

    result = read_file(ReadFileInput(path=".env"), workspace_root=tmp_path)

    assert result.ok is False
    assert result.metadata.get("sensitive") is True
    assert "super-secret" not in result.content


def test_env_template_can_be_read(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=replace-me", encoding="utf-8")

    result = read_file(ReadFileInput(path=".env.example"), workspace_root=tmp_path)

    assert result.ok is True
    assert "replace-me" in result.content


def test_private_key_file_is_blocked(tmp_path: Path) -> None:
    (tmp_path / "deploy.pem").write_text("PRIVATE KEY MATERIAL", encoding="utf-8")

    result = read_file(ReadFileInput(path="deploy.pem"), workspace_root=tmp_path)

    assert result.ok is False
    assert result.metadata.get("sensitive") is True


def test_recursive_search_skips_sensitive_files(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("needle=secret", encoding="utf-8")
    (tmp_path / "app.py").write_text("needle = 'public'\n", encoding="utf-8")

    result = search_code(SearchCodeInput(pattern="needle"), workspace_root=tmp_path)

    assert result.ok is True
    assert "app.py:1:" in result.content
    assert ".env" not in result.content
    assert "secret" not in result.content
    assert result.metadata["total_matches"] == 1


def test_direct_search_of_sensitive_file_is_blocked(tmp_path: Path) -> None:
    (tmp_path / ".npmrc").write_text("//registry/:_authToken=secret", encoding="utf-8")

    result = search_code(
        SearchCodeInput(pattern="secret", path=".npmrc"),
        workspace_root=tmp_path,
    )

    assert result.ok is False
    assert result.metadata.get("sensitive") is True


def test_write_file_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    result = write_file(
        WriteFileInput(path="src/new.txt", content="new content"),
        workspace_root=tmp_path,
    )

    assert result.ok is True
    assert result.metadata.get("atomic") is True
    assert (tmp_path / "src" / "new.txt").read_text(encoding="utf-8") == "new content"
    assert list((tmp_path / "src").glob(".new.txt.*.tmp")) == []


def test_edit_file_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("answer = 41\n", encoding="utf-8")

    result = edit_file(
        EditFileInput(
            path="app.py",
            target_content="answer = 41",
            replacement_content="answer = 42",
        ),
        workspace_root=tmp_path,
    )

    assert result.ok is True
    assert result.metadata.get("atomic") is True
    assert target.read_text(encoding="utf-8") == "answer = 42\n"
    assert list(tmp_path.glob(".app.py.*.tmp")) == []


def test_agent_cannot_write_sensitive_file(tmp_path: Path) -> None:
    result = write_file(
        WriteFileInput(path=".env", content="TOKEN=generated-secret"),
        workspace_root=tmp_path,
    )

    assert result.ok is False
    assert result.metadata.get("sensitive") is True
    assert not (tmp_path / ".env").exists()
