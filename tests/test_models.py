"""Tests for configuration-level model normalization."""

from pathlib import Path

from mini_agent.models import AgentConfig


def test_legacy_deepseek_default_alias_maps_to_flash(tmp_path: Path) -> None:
    config = AgentConfig(workspace_root=tmp_path, model="deepseek-v4")
    assert config.model == "deepseek-v4-flash"


def test_legacy_reasoner_alias_maps_to_pro(tmp_path: Path) -> None:
    config = AgentConfig(workspace_root=tmp_path, model="deepseek-v4-reasoner")
    assert config.model == "deepseek-v4-pro"


def test_non_deepseek_model_is_unchanged(tmp_path: Path) -> None:
    config = AgentConfig(workspace_root=tmp_path, model="gpt-4o-mini")
    assert config.model == "gpt-4o-mini"
