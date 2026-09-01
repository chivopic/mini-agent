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


def test_legacy_deepseek_aliases_map_to_supported_models() -> None:
    assert get_provider_preset("deepseek-chat").default_model == "deepseek-v4-flash"  # type: ignore[union-attr]
    assert get_provider_preset("deepseek-reasoner").default_model == "deepseek-v4-pro"  # type: ignore[union-attr]
    assert get_provider_preset("deepseek-v4").default_model == "deepseek-v4-flash"  # type: ignore[union-attr]
    assert get_provider_preset("v4-pro").default_model == "deepseek-v4-pro"  # type: ignore[union-attr]


def test_get_unknown_provider_preset() -> None:
    unknown = get_provider_preset("non_existent_provider")
    assert unknown is None
