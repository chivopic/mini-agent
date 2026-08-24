"""Tests for TOML config merge order and fail-closed loading."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mini_agent.cli import app, run_cli
from mini_agent.config import (
    ConfigError,
    LimitsConfig,
    defaults,
    load_app_config,
)
from mini_agent.llm import LLMClient, LLMResponse
from mini_agent.models import AgentConfig
from mini_agent.permission import Decision

runner = CliRunner()


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_limits_default_max_output_chars_is_12000() -> None:
    limits = LimitsConfig()
    assert limits.max_output_chars == 12_000
    assert limits.max_tool_rounds == 32
    assert defaults().permission.shell is Decision.ASK


def test_agent_config_defaults_without_app_config(tmp_path: Path) -> None:
    config = AgentConfig(workspace_root=tmp_path)
    assert config.max_tool_rounds == 8
    assert config.max_output_chars == 12_000
    assert config.model == "gpt-4o-mini"


def test_toml_changes_max_tool_rounds(tmp_path: Path) -> None:
    _write(tmp_path / ".mini-agent.toml", "[limits]\nmax_tool_rounds = 7\n")
    cfg, _warnings = load_app_config(tmp_path)
    assert cfg.limits.max_tool_rounds == 7
    assert cfg.to_agent_config(tmp_path).max_tool_rounds == 7


def test_env_overrides_toml_model(tmp_path: Path, monkeypatch: object) -> None:
    _write(
        tmp_path / ".mini-agent.toml",
        '[provider]\nmodel = "from-toml"\n',
    )
    monkeypatch.setenv("MINI_AGENT_MODEL", "from-env")  # type: ignore[attr-defined]
    cfg, _warnings = load_app_config(tmp_path)
    assert cfg.provider.model == "from-env"


def test_cli_overrides_env(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("MINI_AGENT_MODEL", "from-env")  # type: ignore[attr-defined]
    cfg, _warnings = load_app_config(tmp_path, cli_model="from-cli")
    assert cfg.provider.model == "from-cli"


def test_api_key_in_toml_ignored(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # type: ignore[attr-defined]
    _write(
        tmp_path / ".mini-agent.toml",
        "[provider]\n"
        'api_key = "sk-from-toml"\n'
        'openai_api_key = "also-secret"\n'
        'OPENAI_API_KEY = "sk-upper"\n'
        'model = "from-toml"\n',
    )
    cfg, warnings = load_app_config(tmp_path)
    assert cfg.provider.model == "from-toml"
    assert os.environ.get("OPENAI_API_KEY") not in {
        "sk-from-toml",
        "also-secret",
        "sk-upper",
    }
    joined = " ".join(warnings).lower()
    assert "api_key" in joined or "openai_api_key" in joined


def test_explicit_base_url_overrides_preset(tmp_path: Path) -> None:
    _write(
        tmp_path / ".mini-agent.toml",
        '[provider]\nname = "deepseek"\nbase_url = "https://example.com/v1"\n',
    )
    cfg, _warnings = load_app_config(tmp_path)
    assert cfg.provider.model == "deepseek-v4"
    assert cfg.provider.base_url == "https://example.com/v1"


def test_preset_name_fills_model_and_base_url(tmp_path: Path) -> None:
    _write(tmp_path / ".mini-agent.toml", '[provider]\nname = "deepseek"\n')
    cfg, _warnings = load_app_config(tmp_path)
    assert cfg.provider.model == "deepseek-v4"
    assert cfg.provider.base_url == "https://api.deepseek.com"


def test_explicit_model_overrides_preset_default(tmp_path: Path) -> None:
    _write(
        tmp_path / ".mini-agent.toml",
        '[provider]\nname = "deepseek"\nmodel = "custom-model"\n',
    )
    cfg, _warnings = load_app_config(tmp_path)
    assert cfg.provider.model == "custom-model"
    assert cfg.provider.base_url == "https://api.deepseek.com"


def test_missing_openai_api_key_still_non_zero_for_default_client(tmp_path: Path) -> None:
    _write(
        tmp_path / ".mini-agent.toml",
        '[provider]\napi_key = "sk-from-toml"\nmodel = "from-toml"\n',
    )
    with patch.dict("os.environ", {}, clear=True):
        result = runner.invoke(app, ["--workspace", str(tmp_path)])
    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.stdout


def test_dotenv_does_not_override_existing_env(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("MINI_AGENT_MODEL", "already-set")  # type: ignore[attr-defined]
    monkeypatch.setenv("OPENAI_API_KEY", "already-key")  # type: ignore[attr-defined]
    _write(
        tmp_path / ".env",
        "MINI_AGENT_MODEL=from-dotenv\nOPENAI_API_KEY=from-dotenv\n",
    )
    cfg, _warnings = load_app_config(tmp_path)
    assert cfg.provider.model == "already-set"
    assert os.environ["MINI_AGENT_MODEL"] == "already-set"
    assert os.environ["OPENAI_API_KEY"] == "already-key"


def test_dotenv_fills_missing_env(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.delenv("MINI_AGENT_MODEL", raising=False)  # type: ignore[attr-defined]
    _write(tmp_path / ".env", "MINI_AGENT_MODEL=from-dotenv\n")
    cfg, _warnings = load_app_config(tmp_path)
    assert cfg.provider.model == "from-dotenv"


def test_config_flag_overrides_user_config_path(tmp_path: Path) -> None:
    default_user = tmp_path / ".user-mini-agent-config.toml"
    _write(default_user, "[limits]\nmax_tool_rounds = 11\n")
    custom = tmp_path / "custom.toml"
    _write(custom, "[limits]\nmax_tool_rounds = 22\n")
    cfg, _warnings = load_app_config(
        tmp_path,
        user_config_path=custom,
        user_config_required=True,
    )
    assert cfg.limits.max_tool_rounds == 22


def test_project_shell_allow_warns(tmp_path: Path) -> None:
    _write(tmp_path / ".mini-agent.toml", '[permission]\nshell = "allow"\n')
    cfg, warnings = load_app_config(tmp_path)
    assert cfg.permission.shell is Decision.ALLOW
    assert any("shell" in warning and "allow" in warning for warning in warnings)


def test_malformed_nested_limits_fail_closed(tmp_path: Path) -> None:
    _write(tmp_path / ".mini-agent.toml", "[limits]\nmax_tool_rounds = { nested = 1 }\n")
    with pytest.raises(ConfigError):
        load_app_config(tmp_path)


def test_invalid_toml_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path / ".mini-agent.toml", "this is not = [valid")
    with pytest.raises(ConfigError):
        load_app_config(tmp_path)


def test_unknown_provider_name_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path / ".mini-agent.toml", '[provider]\nname = "not-a-real-preset"\n')
    with pytest.raises(ConfigError, match="未知服务商预设"):
        load_app_config(tmp_path)


def test_missing_required_config_path_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-config.toml"
    with pytest.raises(ConfigError):
        load_app_config(tmp_path, user_config_path=missing, user_config_required=True)


def test_cli_wired_agent_config_uses_toml_limits(tmp_path: Path) -> None:
    _write(tmp_path / ".mini-agent.toml", "[limits]\nmax_tool_rounds = 9\n")
    captured: dict[str, AgentConfig] = {}

    def factory(config: AgentConfig, client: object, listener: object) -> object:
        captured["config"] = config

        class _Agent:
            def step(self, prompt: str) -> str:
                return "ok"

        return _Agent()

    class DummyLLM(LLMClient):
        def create_response(
            self,
            history: list[dict[str, object]],
            tools: list[dict[str, object]],
            model: str = "gpt-4o-mini",
            on_token: object = None,
        ) -> LLMResponse:
            return LLMResponse(text="ok")

    run_cli(
        workspace=tmp_path,
        prompt="hi",
        llm_client=DummyLLM(),
        agent_factory=factory,  # type: ignore[arg-type]
    )
    assert captured["config"].max_tool_rounds == 9
