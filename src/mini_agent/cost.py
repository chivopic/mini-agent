"""Dynamic token usage tracking and cost estimation engine for mini-agent."""

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


class UsageStats(BaseModel):
    """Token usage counters."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def add(self, other: "UsageStats") -> "UsageStats":
        """Add another usage stats to this one."""
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


# Built-in pricing per 1M tokens in CNY (RMB).
# DeepSeek uses time-of-day and cache-dependent pricing. Because this estimator does not know
# whether an input token was a cache hit, the built-ins intentionally use the current PEAK,
# CACHE-MISS input rate plus the PEAK output rate as a conservative upper-bound estimate.
# Users can override any model through ~/.mini-agent/pricing.json or MINI_AGENT_PRICING.
BUILTIN_MODEL_PRICING_CNY: dict[str, tuple[float, float]] = {
    # DeepSeek V4 Series (pricing effective 2026-08-16)
    "deepseek-v4-flash": (3.0, 9.0),
    "deepseek-v4-pro": (9.0, 27.0),
    # OpenAI (USD converted to approx CNY @ 7.2; user-overridable estimates)
    "gpt-4o-mini": (1.08, 4.32),
    "gpt-4o": (18.0, 72.0),
    "o1-mini": (21.6, 86.4),
    "o1": (108.0, 432.0),
    # Qwen (Aliyun Bailian; user-overridable estimates)
    "qwen-turbo": (0.3, 0.6),
    "qwen-plus": (0.8, 2.0),
    "qwen-max": (20.0, 60.0),
    # Moonshot
    "moonshot-v1-8k": (12.0, 12.0),
    # GLM
    "glm-4-flash": (0.0, 0.0),
    "glm-4": (10.0, 10.0),
    # Ollama / Local
    "ollama": (0.0, 0.0),
    "local": (0.0, 0.0),
}


def get_default_pricing_file() -> Path:
    """Return path to custom pricing JSON file in ~/.mini-agent/pricing.json."""
    custom_path = os.environ.get("MINI_AGENT_PRICING_FILE")
    if custom_path:
        path = Path(custom_path).resolve()
    else:
        path = Path.home() / ".mini-agent" / "pricing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_pricing_table(pricing_file: Path | None = None) -> dict[str, tuple[float, float]]:
    """Load combined pricing table from built-ins, local custom JSON, and env vars."""
    table = dict(BUILTIN_MODEL_PRICING_CNY)

    # 1. Load from local custom JSON file if exists
    target_file = pricing_file or get_default_pricing_file()
    if target_file.is_file():
        try:
            with open(target_file, encoding="utf-8") as f:
                custom_data = json.load(f)
            if isinstance(custom_data, dict):
                for k, v in custom_data.items():
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        table[k.strip().lower()] = (float(v[0]), float(v[1]))
        except Exception:
            pass

    # 2. Load from environment variable MINI_AGENT_PRICING
    # e.g. "model1:1.0,2.0;model2:0.5,1.0"
    env_pricing = os.environ.get("MINI_AGENT_PRICING", "").strip()
    if env_pricing:
        for item in env_pricing.split(";"):
            if ":" in item:
                m_name, prices = item.split(":", 1)
                if "," in prices:
                    p_in, p_out = prices.split(",", 1)
                    try:
                        table[m_name.strip().lower()] = (float(p_in.strip()), float(p_out.strip()))
                    except ValueError:
                        continue

    return table


def set_custom_pricing(
    model: str,
    input_cny_per_m: float,
    output_cny_per_m: float,
    pricing_file: Path | None = None,
) -> None:
    """Save custom pricing for a model to local pricing.json."""
    target_file = pricing_file or get_default_pricing_file()
    current_data: dict[str, list[float]] = {}

    if target_file.is_file():
        try:
            with open(target_file, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                current_data = loaded
        except Exception:
            current_data = {}

    clean_model = model.strip().lower()
    current_data[clean_model] = [float(input_cny_per_m), float(output_cny_per_m)]

    temp_file = target_file.with_suffix(".tmp")
    with open(temp_file, mode="w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2, ensure_ascii=False)
    temp_file.replace(target_file)


def get_model_pricing(model: str, pricing_file: Path | None = None) -> tuple[float, float]:
    """Get the active (input, output) price per 1M tokens for a model."""
    clean_model = model.strip().lower()
    if "ollama" in clean_model or "local" in clean_model:
        return (0.0, 0.0)

    table = load_pricing_table(pricing_file)

    # 1. Exact match
    if clean_model in table:
        return table[clean_model]

    # 2. Substring match
    for m_key, m_price in table.items():
        if m_key in clean_model:
            return m_price

    # 3. Default fallback pricing: 1.0 CNY input, 2.0 CNY output
    return (1.0, 2.0)


def estimate_tokens_from_text(text: str) -> int:
    """Fallback token estimation: ~1 token per 3.5 characters for mixed text."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


def calculate_cost_cny(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    pricing_file: Path | None = None,
) -> float:
    """Calculate estimated cost in CNY (RMB) for a given token usage and model."""
    input_price_per_m, output_price_per_m = get_model_pricing(model, pricing_file)
    cost = (prompt_tokens / 1_000_000 * input_price_per_m) + (
        completion_tokens / 1_000_000 * output_price_per_m
    )
    return round(cost, 6)


def format_cost_cny(cost: float) -> str:
    """Format cost nicely for terminal display."""
    if cost == 0.0:
        return "免费 (¥0.00)"
    if cost < 0.0001:
        return f"¥{cost:.6f}"
    if cost < 0.01:
        return f"¥{cost:.4f}"
    return f"¥{cost:.2f}"
