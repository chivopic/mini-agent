"""Unit tests for token usage and dynamic cost calculations."""

from pathlib import Path

from mini_agent.cost import (
    UsageStats,
    calculate_cost_cny,
    estimate_tokens_from_text,
    format_cost_cny,
    get_model_pricing,
    load_pricing_table,
    set_custom_pricing,
)


def test_usage_stats_add() -> None:
    u1 = UsageStats(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    u2 = UsageStats(prompt_tokens=200, completion_tokens=80, total_tokens=280)
    u3 = u1.add(u2)

    assert u3.prompt_tokens == 300
    assert u3.completion_tokens == 130
    assert u3.total_tokens == 430


def test_estimate_tokens_from_text() -> None:
    assert estimate_tokens_from_text("") == 0
    assert estimate_tokens_from_text("hello world") >= 1
    long_text = "def calculate_sum(a: int, b: int) -> int:\n    return a + b\n"
    tokens = estimate_tokens_from_text(long_text)
    assert tokens > 10


def test_deepseek_v4_series_pricing() -> None:
    # Cache-miss input plus output, converted to CNY at the documented approximate rate.
    cost_flash = calculate_cost_cny(1_000_000, 1_000_000, "deepseek-v4-flash")
    assert cost_flash == 3.03

    cost_pro = calculate_cost_cny(1_000_000, 1_000_000, "deepseek-v4-pro")
    assert cost_pro == 9.39

    # Historical session aliases retain a meaningful estimate.
    assert calculate_cost_cny(1_000_000, 1_000_000, "deepseek-v4") == cost_flash


def test_calculate_cost_ollama_free() -> None:
    cost = calculate_cost_cny(500_000, 500_000, "ollama/qwen2.5-coder")
    assert cost == 0.0


def test_format_cost_cny() -> None:
    assert format_cost_cny(0.0) == "免费 (¥0.00)"
    assert "¥" in format_cost_cny(0.0024)
    assert "¥" in format_cost_cny(1.50)


def test_set_custom_pricing_and_load(tmp_path: Path) -> None:
    custom_file = tmp_path / "custom_pricing.json"

    # Set new model pricing
    set_custom_pricing("my-custom-model", 0.25, 0.75, pricing_file=custom_file)
    table = load_pricing_table(pricing_file=custom_file)

    assert "my-custom-model" in table
    assert table["my-custom-model"] == (0.25, 0.75)

    # Calculate cost using custom pricing file
    cost = calculate_cost_cny(1_000_000, 1_000_000, "my-custom-model", pricing_file=custom_file)
    assert cost == 1.0


def test_env_var_pricing_override(monkeypatch: object) -> None:
    monkeypatch.setenv("MINI_AGENT_PRICING", "special-model:0.1,0.2;other-model:5.0,10.0")  # type: ignore[attr-defined]
    p_in, p_out = get_model_pricing("special-model")
    assert p_in == 0.1
    assert p_out == 0.2
