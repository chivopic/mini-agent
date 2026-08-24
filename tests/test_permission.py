"""Unit tests for permission evaluation order and always-memory."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mini_agent.agent import Agent
from mini_agent.llm import FunctionCall, LLMClient, LLMResponse
from mini_agent.models import AgentConfig, PermissionClass
from mini_agent.permission import (
    Decision,
    DefaultPermissionService,
    PermissionRequest,
    Reply,
    is_sensitive_write_path,
    pattern_for_shell,
)


def _req(
    *,
    cls: PermissionClass,
    tool: str,
    resource: str,
    pattern: str,
) -> PermissionRequest:
    return PermissionRequest(cls=cls, tool=tool, resource=resource, pattern=pattern)


class FakeLLMClient(LLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)

    def create_response(
        self,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        if not self.responses:
            return LLMResponse(text="[FakeLLM: No more responses configured]")
        return self.responses.pop(0)


class AskListener:
    def __init__(self, reply: Reply = Reply.ALWAYS) -> None:
        self.asks: list[PermissionRequest] = []
        self.reply = reply

    def on_permission_ask(self, req: PermissionRequest) -> Reply:
        self.asks.append(req)
        return self.reply


class TestSensitiveWritePath:
    def test_backend_env_is_sensitive(self) -> None:
        assert is_sensitive_write_path("backend/.env") is True

    def test_env_local_and_git_and_keys(self) -> None:
        assert is_sensitive_write_path(".env.local") is True
        assert is_sensitive_write_path(".git/hooks/pre-commit") is True
        assert is_sensitive_write_path("certs/site.pem") is True
        assert is_sensitive_write_path("id_rsa.key") is True
        assert is_sensitive_write_path("pkg/.mini-agent.toml") is True

    def test_src_python_is_not_sensitive(self) -> None:
        assert is_sensitive_write_path("src/a.py") is False

    def test_case_sensitive_env(self) -> None:
        assert is_sensitive_write_path("backend/.ENV") is False


class TestPatternForShell:
    def test_uv_run_pytest_prefix(self) -> None:
        assert pattern_for_shell("uv run pytest tests/test_agent.py -q") == "uv run pytest*"

    def test_git_commit_prefix(self) -> None:
        assert pattern_for_shell('git commit -m "x"') == "git commit*"

    def test_git_add_u_is_exact(self) -> None:
        assert pattern_for_shell("git add -u") == "git add -u"
        assert pattern_for_shell("git add -u") != "git add*"

    def test_python_script_is_exact(self) -> None:
        assert pattern_for_shell("python script.py") == "python script.py"

    def test_interpreter_dash_c_cannot_always(self) -> None:
        assert pattern_for_shell('python -c "print(1)"') is None
        assert pattern_for_shell('python3 -c "print(1)"') is None
        assert pattern_for_shell("bash -c 'echo hi'") is None
        assert pattern_for_shell("sh -c echo") is None
        assert pattern_for_shell("zsh -c echo") is None
        assert pattern_for_shell("uv run python -c 'print(1)'") is None

    def test_uv_run_python_script_prefix_does_not_cover_dash_c(self) -> None:
        assert pattern_for_shell("uv run python script.py") == "uv run python*"
        assert pattern_for_shell("uv run python -c 'print(1)'") is None

    def test_npx_prefix(self) -> None:
        assert pattern_for_shell("npx eslint .") == "npx eslint*"

    def test_unclosed_quotes_none(self) -> None:
        assert pattern_for_shell('echo "oops') is None


class TestDefaultPermissionService:
    def test_backend_env_ask_even_if_edit_allow(self) -> None:
        svc = DefaultPermissionService()
        req = _req(
            cls=PermissionClass.EDIT,
            tool="write_file",
            resource="backend/.env",
            pattern="edit:backend/.env",
        )
        assert svc.check(req) == Decision.ASK

    def test_src_py_allow_when_edit_allow(self) -> None:
        svc = DefaultPermissionService()
        req = _req(
            cls=PermissionClass.EDIT,
            tool="write_file",
            resource="src/a.py",
            pattern="edit:src/a.py",
        )
        assert svc.check(req) == Decision.ALLOW

    def test_git_add_dot_deny_cannot_always(self) -> None:
        svc = DefaultPermissionService()
        for command in ("git add .", "git add ./", "git add ./.", 'git add "./"', "git -C . add ."):
            req = _req(
                cls=PermissionClass.SHELL,
                tool="run_shell",
                resource=command,
                pattern=command,
            )
            assert svc.check(req) == Decision.DENY, command
            svc.remember(req, Reply.ALWAYS)
            assert svc.check(req) == Decision.DENY, command

    def test_git_add_all_deny(self) -> None:
        svc = DefaultPermissionService()
        for command in ("git add -A", "git add --all"):
            req = _req(
                cls=PermissionClass.SHELL,
                tool="run_shell",
                resource=command,
                pattern=command,
            )
            assert svc.check(req) == Decision.DENY

    def test_python_dash_c_cannot_always(self) -> None:
        svc = DefaultPermissionService()
        command = 'python -c "print(1)"'
        req = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource=command,
            pattern=pattern_for_shell(command) or "",
        )
        assert req.pattern == ""
        assert svc.check(req) == Decision.ASK
        svc.remember(req, Reply.ALWAYS)
        assert svc.check(req) == Decision.ASK

    def test_uv_run_python_dash_c_not_auto_allow_and_not_covered_by_always(self) -> None:
        svc = DefaultPermissionService()
        script = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="uv run python script.py",
            pattern=pattern_for_shell("uv run python script.py") or "",
        )
        assert script.pattern == "uv run python*"
        assert svc.check(script) == Decision.ALLOW
        svc.remember(script, Reply.ALWAYS)
        dash_c = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="uv run python -c 'print(1)'",
            pattern=pattern_for_shell("uv run python -c 'print(1)'") or "",
        )
        assert dash_c.pattern == ""
        assert svc.check(dash_c) == Decision.ASK
        svc.remember(dash_c, Reply.ALWAYS)
        assert svc.check(dash_c) == Decision.ASK

    def test_always_then_same_pattern_not_asked(self) -> None:
        svc = DefaultPermissionService()
        req = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="git add -u",
            pattern="git add -u",
        )
        assert svc.check(req) == Decision.ASK
        svc.remember(req, Reply.ALWAYS)
        assert svc.check(req) == Decision.ALLOW
        again = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="git add -u",
            pattern="git add -u",
        )
        assert svc.check(again) == Decision.ALLOW

    def test_git_add_u_pattern_exact_does_not_cover_dot(self) -> None:
        svc = DefaultPermissionService()
        add_u = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="git add -u",
            pattern=pattern_for_shell("git add -u") or "",
        )
        assert add_u.pattern == "git add -u"
        svc.remember(add_u, Reply.ALWAYS)
        assert svc.check(add_u) == Decision.ALLOW
        add_file = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="git add src/a.py",
            pattern=pattern_for_shell("git add src/a.py") or "",
        )
        assert add_file.pattern == "git add src/a.py"
        assert svc.check(add_file) == Decision.ASK

    def test_allowlisted_shell_auto_allow(self) -> None:
        svc = DefaultPermissionService()
        req = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="git status",
            pattern="git status*",
        )
        assert svc.check(req) == Decision.ALLOW

    def test_evaluation_order_blocklist_before_memory_and_allowlist(self) -> None:
        svc = DefaultPermissionService()
        blocked = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="git add .",
            pattern="git add .",
        )
        allowlisted = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="pwd",
            pattern="pwd",
        )
        needs_ask = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="python script.py",
            pattern="python script.py",
        )
        assert svc.check(blocked) == Decision.DENY
        assert svc.check(allowlisted) == Decision.ALLOW
        assert svc.check(needs_ask) == Decision.ASK
        svc.remember(needs_ask, Reply.ALWAYS)
        assert svc.check(needs_ask) == Decision.ALLOW
        # Blocklist still wins over a poisoned always entry.
        svc.remember(blocked, Reply.ALWAYS)
        assert svc.check(blocked) == Decision.DENY

    def test_auto_allow_ask_still_denies_blocklist(self) -> None:
        svc = DefaultPermissionService(auto_allow_ask=True)
        blocked = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="git add .",
            pattern="git add .",
        )
        ask = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="python script.py",
            pattern="python script.py",
        )
        assert svc.check(blocked) == Decision.DENY
        assert svc.check(ask) == Decision.ALLOW

    def test_snapshot_roundtrip(self) -> None:
        svc = DefaultPermissionService()
        req = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="uv run pytest tests/test_agent.py -q",
            pattern="uv run pytest*",
        )
        svc.remember(req, Reply.ALWAYS)
        snap = svc.snapshot()
        assert snap == [{"cls": "shell", "pattern": "uv run pytest*", "effect": "allow"}]
        restored = DefaultPermissionService(memory=snap)
        assert restored.check(req) == Decision.ALLOW

    def test_restore_rejects_malformed_nested_data(self) -> None:
        svc = DefaultPermissionService()
        with pytest.raises(TypeError):
            svc.restore("oops")  # type: ignore[arg-type]
        with pytest.raises((ValueError, Exception)):
            svc.restore([{"cls": "shell"}])
        with pytest.raises((ValueError, Exception)):
            svc.restore([{"cls": "shell", "pattern": "pwd", "effect": "ask"}])

    def test_restore_replaces_rather_than_merges(self) -> None:
        svc = DefaultPermissionService()
        old = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="git add -u",
            pattern="git add -u",
        )
        svc.remember(old, Reply.ALWAYS)
        assert svc.check(old) == Decision.ALLOW
        svc.restore([{"cls": "shell", "pattern": "uv run pytest*", "effect": "allow"}])
        assert svc.check(old) == Decision.ASK
        pytest_req = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="uv run pytest tests/test_agent.py -q",
            pattern="uv run pytest*",
        )
        assert svc.check(pytest_req) == Decision.ALLOW
        svc.restore([])
        assert svc.snapshot() == []
        assert svc.check(pytest_req) == Decision.ALLOW  # allowlisted
        assert svc.check(old) == Decision.ASK

    def test_restore_malformed_keeps_existing_memory(self) -> None:
        svc = DefaultPermissionService()
        old = _req(
            cls=PermissionClass.SHELL,
            tool="run_shell",
            resource="git add -u",
            pattern="git add -u",
        )
        svc.remember(old, Reply.ALWAYS)
        with pytest.raises((ValueError, Exception)):
            svc.restore([{"cls": "shell"}])
        assert svc.check(old) == Decision.ALLOW


class TestAgentPermissionIntegration:
    def test_sensitive_write_rejected_without_listener(self, tmp_path: Path) -> None:
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="write_file",
                            call_id="call_env",
                            arguments='{"path": "backend/.env", "content": "SECRET=1"}',
                        )
                    ]
                ),
                LLMResponse(text="已拒绝写入敏感文件。"),
            ]
        )
        agent = Agent(config=AgentConfig(workspace_root=tmp_path), llm_client=fake_llm)
        agent.step("写入 env")
        assert not (tmp_path / "backend" / ".env").exists()

    def test_always_skips_second_ask_for_git_add_u(self, tmp_path: Path) -> None:
        listener = AskListener(Reply.ALWAYS)
        fake_llm = FakeLLMClient(
            [
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="run_shell",
                            call_id="call_1",
                            arguments='{"command": "git add -u"}',
                        )
                    ]
                ),
                LLMResponse(text="第一次已处理。"),
                LLMResponse(
                    function_calls=[
                        FunctionCall(
                            name="run_shell",
                            call_id="call_2",
                            arguments='{"command": "git add -u"}',
                        )
                    ]
                ),
                LLMResponse(text="第二次已处理。"),
            ]
        )
        agent = Agent(
            config=AgentConfig(workspace_root=tmp_path),
            llm_client=fake_llm,
            listener=listener,  # type: ignore[arg-type]
        )
        agent.step("暂存一次")
        agent.step("再暂存一次")
        assert len(listener.asks) == 1
        assert listener.asks[0].pattern == "git add -u"
