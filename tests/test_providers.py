"""Unit tests for provider presets."""

from mini_agent.providers import (
    get_provider_preset,
    list_provider_presets,
)


def test_list_provider_presets() -> None:
    presets = list_provider_presets()
    assert len(presets) >= 6
    names = [p.name for p in presets]
    assert "deepseek" in names
    assert "deepseek-flash" in names
    assert "deepseek-pro" in names
    assert "openai" in names
    assert "ollama" in names


def test_get_provider_preset_deepseek_v4() -> None:
    ds = get_provider_preset("deepseek")
    assert ds is not None
    assert ds.default_model == "deepseek-v4-flash"
    assert "api.deepseek.com" in ds.base_url

    flash = get_provider_preset("deepseek-flash")
    assert flash is not None
    assert flash.default_model == "deepseek-v4-flash"

    pro = get_provider_preset("deepseek-pro")
    assert pro is not None
    assert pro.default_model == "deepseek-v4-pro"

    # Retired model labels remain accepted as input aliases but map to current API names.
    assert get_provider_preset("deepseek-chat") == ds
    assert get_provider_preset("deepseek-r1") == pro


def test_get_unknown_provider_preset() -> None:
    unknown = get_provider_preset("non_existent_provider")
    assert unknown is None
