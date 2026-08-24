"""Unit tests for provider presets."""

import pytest

from mini_agent.providers import (
    expand_provider_preset,
    get_provider_preset,
    list_provider_presets,
)


def test_list_provider_presets() -> None:
    presets = list_provider_presets()
    assert len(presets) >= 6
    names = [p.name for p in presets]
    assert "deepseek" in names
    assert "deepseek-flash" in names
    assert "deepseek-r1" in names
    assert "openai" in names
    assert "ollama" in names


def test_get_provider_preset_deepseek_v4() -> None:
    ds = get_provider_preset("deepseek")
    assert ds is not None
    assert ds.default_model == "deepseek-v4"
    assert "api.deepseek.com" in ds.base_url

    flash = get_provider_preset("deepseek-flash")
    assert flash is not None
    assert flash.default_model == "deepseek-v4-flash"

    r1 = get_provider_preset("deepseek-r1")
    assert r1 is not None
    assert r1.default_model == "deepseek-v4-reasoner"


def test_get_unknown_provider_preset() -> None:
    unknown = get_provider_preset("non_existent_provider")
    assert unknown is None


def test_expand_provider_preset_explicit_base_url_overrides() -> None:
    model, base_url = expand_provider_preset(
        "deepseek",
        base_url="https://example.com/v1",
    )
    assert model == "deepseek-v4"
    assert base_url == "https://example.com/v1"


def test_expand_provider_preset_explicit_model_overrides() -> None:
    model, base_url = expand_provider_preset("openai", model="gpt-4o")
    assert model == "gpt-4o"
    assert base_url == "https://api.openai.com/v1"


def test_expand_unknown_provider_preset_raises() -> None:
    with pytest.raises(ValueError, match="未知服务商预设"):
        expand_provider_preset("not-a-real-preset")
