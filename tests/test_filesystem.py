"""Unit tests for filesystem tools and data models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_agent.models import (
    AgentConfig,
    EditFileInput,
    GetRepoMapInput,
    ListFilesInput,
    ReadFileInput,
    SearchCodeInput,
    ToolResult,
    WriteFileInput,
)
from mini_agent.tools.filesystem import (
    edit_file,
    get_repo_map,
    list_files,
    read_file,
    resolve_relative_path,
    search_code,
    truncate_text,
    write_file,
)


class TestDataModels:
    """Test data models and configuration validation."""

    def test_agent_config_valid(self, tmp_path: Path) -> None:
        config = AgentConfig(workspace_root=tmp_path, model="gpt-4o")
        assert config.workspace_root == tmp_path.resolve()
        assert config.model == "gpt-4o"
        assert config.max_tool_rounds == 8

    def test_agent_config_invalid_path(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist"
        with pytest.raises(ValidationError):
            AgentConfig(workspace_root=non_existent)

    def test_read_file_input_validation(self) -> None:
        with pytest.raises(ValidationError):
            ReadFileInput(path="")

    def test_list_files_input_validation(self) -> None:
        inp = ListFilesInput(path="src", max_depth=3)
        assert inp.path == "src"
        assert inp.max_depth == 3

        with pytest.raises(ValidationError):
            ListFilesInput(max_depth=0)
        with pytest.raises(ValidationError):
            ListFilesInput(max_depth=6)

    def test_write_file_input_validation(self) -> None:
        with pytest.raises(ValidationError):
            WriteFileInput(path="", content="hello")

    def test_edit_file_input_validation(self) -> None:
        with pytest.raises(ValidationError):
            EditFileInput(path="a.py", target_content="", replacement_content="b")

    def test_tool_result_structure(self) -> None:
        res = ToolResult(ok=True, content="test output")
        assert res.ok is True
        assert res.content == "test output"
        assert res.error is None


class TestPathResolution:
    """Test safe path resolution within workspace."""

    def test_resolve_normal_relative_path(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("content", encoding="utf-8")

        resolved, err = resolve_relative_path(tmp_path, "hello.txt")
        assert err is None
        assert resolved == f.resolve()

    def test_reject_absolute_path(self, tmp_path: Path) -> None:
        resolved, err = resolve_relative_path(tmp_path, "/etc/passwd")
        assert resolved is None
        assert err is not None
        assert "非法绝对路径" in err
        assert "/etc/passwd" in err

    def test_reject_parent_traversal(self, tmp_path: Path) -> None:
        resolved, err = resolve_relative_path(tmp_path, "../outside.txt")
        assert resolved is None
        assert err is not None
        assert "越界" in err

    def test_reject_external_symlink(self, tmp_path: Path) -> None:
        outside_dir = tmp_path.parent / "outside_dir"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret", encoding="utf-8")

        link = tmp_path / "symlink_file.txt"
        try:
            link.symlink_to(outside_file)
        except OSError:
            pytest.skip("Symlinks not supported on this platform/filesystem")

        resolved, err = resolve_relative_path(tmp_path, "symlink_file.txt")
        assert resolved is None
        assert err is not None
        assert "越界" in err


class TestReadFileTool:
    """Test read_file tool behavior."""

    def test_read_valid_utf8_file(self, tmp_path: Path) -> None:
        sample = tmp_path / "sample.py"
        sample.write_text("print('hello world')", encoding="utf-8")

        result = read_file(ReadFileInput(path="sample.py"), workspace_root=tmp_path)
        assert result.ok is True
        assert result.content == "print('hello world')"
        assert result.error is None
        assert result.metadata["path"] == "sample.py"
        assert result.metadata["truncated"] is False

    def test_read_non_existent_file(self, tmp_path: Path) -> None:
        result = read_file(ReadFileInput(path="missing.txt"), workspace_root=tmp_path)
        assert result.ok is False
        assert "不存在" in (result.error or "")

    def test_read_directory_as_file(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        result = read_file(ReadFileInput(path="subdir"), workspace_root=tmp_path)
        assert result.ok is False
        assert "不是普通文件" in (result.error or "")

    def test_read_binary_file(self, tmp_path: Path) -> None:
        bin_file = tmp_path / "binary.bin"
        bin_file.write_bytes(b"\x80\x81\xfe\xff")

        result = read_file(ReadFileInput(path="binary.bin"), workspace_root=tmp_path)
        assert result.ok is False
        assert "不是 UTF-8" in (result.error or "")

    def test_read_oversized_file(self, tmp_path: Path) -> None:
        big_file = tmp_path / "big.txt"
        big_file.write_text("A" * 1500, encoding="utf-8")

        result = read_file(
            ReadFileInput(path="big.txt"),
            workspace_root=tmp_path,
            max_file_bytes=1000,
        )
        assert result.ok is False
        assert "体积过大" in (result.error or "")


class TestListFilesTool:
    """Test list_files tool behavior."""

    def test_list_normal_directory(self, tmp_path: Path) -> None:
        (tmp_path / "file1.py").write_text("a", encoding="utf-8")
        (tmp_path / "file2.md").write_text("b", encoding="utf-8")
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "subfile.txt").write_text("c", encoding="utf-8")

        result = list_files(ListFilesInput(path=".", max_depth=2), workspace_root=tmp_path)
        assert result.ok is True
        assert "- pkg/" in result.content
        assert "- pkg/subfile.txt" in result.content
        assert "- file1.py" in result.content
        assert "- file2.md" in result.content
        assert result.metadata["total_entries"] == 4

    def test_list_filters_ignored_directories(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref", encoding="utf-8")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "pyvenv.cfg").write_text("home", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "app.py").write_text("code", encoding="utf-8")

        result = list_files(ListFilesInput(path="."), workspace_root=tmp_path)
        assert result.ok is True
        assert ".git" not in result.content
        assert ".venv" not in result.content
        assert "__pycache__" not in result.content
        assert "- app.py" in result.content

    def test_list_respects_max_depth(self, tmp_path: Path) -> None:
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "level1" / "level2" / "deep.txt").write_text("deep", encoding="utf-8")

        result_depth1 = list_files(
            ListFilesInput(path=".", max_depth=1),
            workspace_root=tmp_path,
        )
        assert "- level1/" in result_depth1.content
        assert "level2" not in result_depth1.content

        result_depth2 = list_files(
            ListFilesInput(path=".", max_depth=2),
            workspace_root=tmp_path,
        )
        assert "- level1/" in result_depth2.content
        assert "- level1/level2/" in result_depth2.content
        assert "deep.txt" not in result_depth2.content

    def test_list_non_existent_directory(self, tmp_path: Path) -> None:
        result = list_files(ListFilesInput(path="non_existent"), workspace_root=tmp_path)
        assert result.ok is False
        assert "不存在" in (result.error or "")

    def test_list_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        result = list_files(ListFilesInput(path="empty_dir"), workspace_root=tmp_path)
        assert result.ok is True
        assert "(目录为空)" in result.content


class TestWriteFileTool:
    """Test write_file tool behavior."""

    def test_write_new_file_and_nested_parent(self, tmp_path: Path) -> None:
        inp = WriteFileInput(path="sub/nested/new.py", content="print('created')")
        result = write_file(inp, workspace_root=tmp_path)
        assert result.ok is True
        assert "成功写入" in result.content
        target = tmp_path / "sub" / "nested" / "new.py"
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "print('created')"

    def test_write_file_outside_workspace_rejected(self, tmp_path: Path) -> None:
        inp = WriteFileInput(path="../outside.py", content="evil")
        result = write_file(inp, workspace_root=tmp_path)
        assert result.ok is False
        assert "越界" in (result.error or "")


class TestEditFileTool:
    """Test edit_file tool behavior."""

    def test_edit_existing_file_single_match(self, tmp_path: Path) -> None:
        file_path = tmp_path / "calc.py"
        file_path.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        inp = EditFileInput(
            path="calc.py",
            target_content="    return a - b",
            replacement_content="    return a + b",
        )
        result = edit_file(inp, workspace_root=tmp_path)
        assert result.ok is True
        assert "成功修改" in result.content
        assert file_path.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"

    def test_edit_file_target_not_found(self, tmp_path: Path) -> None:
        file_path = tmp_path / "demo.py"
        file_path.write_text("x = 1", encoding="utf-8")

        inp = EditFileInput(
            path="demo.py",
            target_content="y = 2",
            replacement_content="y = 3",
        )
        result = edit_file(inp, workspace_root=tmp_path)
        assert result.ok is False
        assert "未找到目标代码片段" in (result.error or "")

    def test_edit_file_ambiguous_matches_rejected(self, tmp_path: Path) -> None:
        file_path = tmp_path / "dup.py"
        file_path.write_text("val = 1\nval = 1\n", encoding="utf-8")

        inp = EditFileInput(
            path="dup.py",
            target_content="val = 1",
            replacement_content="val = 2",
        )
        result = edit_file(inp, workspace_root=tmp_path)
        assert result.ok is False
        assert "匹配不唯一" in (result.error or "")


class TestTruncateHelper:
    """Test truncate_text helper."""

    def test_truncate_short_text(self) -> None:
        text, truncated = truncate_text("hello", max_chars=100)
        assert text == "hello"
        assert truncated is False

    def test_truncate_long_text(self) -> None:
        long_text = "start" + ("x" * 500) + "end"
        text, truncated = truncate_text(long_text, max_chars=200)
        assert truncated is True
        assert "已省略" in text
        assert text.startswith("start")
        assert text.endswith("end")


class TestSearchCode:
    """Test search_code filesystem tool."""

    def test_search_code_plain_text(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text(
            "def hello_world():\n    return 'hello'\n", encoding="utf-8"
        )
        (src_dir / "utils.py").write_text("HELLO_CONST = 100\n", encoding="utf-8")

        inp = SearchCodeInput(pattern="hello")
        res = search_code(inp, workspace_root=tmp_path)
        assert res.ok is True
        assert "main.py:1: def hello_world():" in res.content
        assert "utils.py:1: HELLO_CONST = 100" in res.content
        assert res.metadata["total_matches"] == 3

    def test_search_code_case_sensitive(self, tmp_path: Path) -> None:
        file_path = tmp_path / "demo.py"
        file_path.write_text("FooBar\nfoobar\nFOOBAR\n", encoding="utf-8")

        inp = SearchCodeInput(pattern="FooBar", case_sensitive=True)
        res = search_code(inp, workspace_root=tmp_path)
        assert res.ok is True
        assert res.metadata["total_matches"] == 1
        assert "demo.py:1: FooBar" in res.content

    def test_search_code_regex(self, tmp_path: Path) -> None:
        file_path = tmp_path / "router.py"
        file_path.write_text(
            "@app.get('/api/v1/users')\n@app.post('/api/v2/auth')\n", encoding="utf-8"
        )

        inp = SearchCodeInput(pattern=r"/api/v\d+/\w+", is_regex=True)
        res = search_code(inp, workspace_root=tmp_path)
        assert res.ok is True
        assert res.metadata["total_matches"] == 2
        assert "router.py:1:" in res.content
        assert "router.py:2:" in res.content

    def test_search_code_ignores_special_dirs(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("secret_keyword = 1", encoding="utf-8")

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "app.py").write_text("secret_keyword = 2", encoding="utf-8")

        inp = SearchCodeInput(pattern="secret_keyword")
        res = search_code(inp, workspace_root=tmp_path)
        assert res.ok is True
        assert res.metadata["total_matches"] == 1
        assert "src/app.py:1:" in res.content
        assert ".git" not in res.content

    def test_search_code_invalid_regex(self, tmp_path: Path) -> None:
        inp = SearchCodeInput(pattern="[invalid(regex", is_regex=True)
        res = search_code(inp, workspace_root=tmp_path)
        assert res.ok is False
        assert "正则表达式格式错误" in (res.error or "")


class TestGetRepoMapSandbox:
    """get_repo_map must stay inside the workspace via resolve_relative_path."""

    def test_reject_parent_traversal(self, tmp_path: Path) -> None:
        result = get_repo_map(GetRepoMapInput(path="../"), workspace_root=tmp_path)
        assert result.ok is False
        assert "越界" in (result.error or "")

    def test_reject_absolute_path(self, tmp_path: Path) -> None:
        result = get_repo_map(GetRepoMapInput(path=str(tmp_path)), workspace_root=tmp_path)
        assert result.ok is False
        assert "非法绝对路径" in (result.error or "")
