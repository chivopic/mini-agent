"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest

_PROVIDER_ENV_KEYS = (
    "MINI_AGENT_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
)


@pytest.fixture(autouse=True)
def isolate_app_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep user TOML and provider env from leaking into tests."""
    user_config = tmp_path / ".user-mini-agent-config.toml"
    monkeypatch.setattr(
        "mini_agent.config.default_user_config_path",
        lambda: user_config,
    )
    for key in _PROVIDER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return user_config
