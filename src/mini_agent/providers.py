"""Provider presets for mainstream LLM services (DeepSeek V4, OpenAI, Ollama, Qwen, etc.)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPreset:
    """Configuration preset for an LLM provider."""

    name: str
    display_name: str
    default_model: str
    base_url: str
    description: str
    env_key: str = "OPENAI_API_KEY"


PREDEFINED_PROVIDERS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        name="deepseek",
        display_name="DeepSeek V4 (官方标准版)",
        default_model="deepseek-v4",
        base_url="https://api.deepseek.com",
        description="深度求索全新 V4 系列旗舰模型，编程与综合推理能力巅峰",
    ),
    "deepseek-flash": ProviderPreset(
        name="deepseek-flash",
        display_name="DeepSeek V4 Flash (极速版)",
        default_model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        description="极速低延迟轻量模型，价格极具性价比，适合代码检索与快速审查",
    ),
    "deepseek-r1": ProviderPreset(
        name="deepseek-r1",
        display_name="DeepSeek V4 Reasoner (深度思考)",
        default_model="deepseek-v4-reasoner",
        base_url="https://api.deepseek.com",
        description="深度强化学习推理模型，长链路架构设计与复杂 Bug 溯源首选",
    ),
    "deepseek-v3": ProviderPreset(
        name="deepseek-v3",
        display_name="DeepSeek V3 (经典版)",
        default_model="deepseek-chat",
        base_url="https://api.deepseek.com",
        description="DeepSeek-V3 稳定对话版本",
    ),
    "openai": ProviderPreset(
        name="openai",
        display_name="OpenAI (官方)",
        default_model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        description="OpenAI 官方接口，支持 GPT-4o 及 GPT-4o-mini",
    ),
    "ollama": ProviderPreset(
        name="ollama",
        display_name="Ollama (本地离线)",
        default_model="qwen2.5-coder:latest",
        base_url="http://localhost:11434/v1",
        description="本地离线开源大模型，完全免费、零网络开销、数据不出本地",
    ),
    "qwen": ProviderPreset(
        name="qwen",
        display_name="通义千问 (阿里云百炼)",
        default_model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="阿里云 DashScope 兼容接口，支持 Qwen-Plus / Qwen-Turbo",
    ),
    "siliconflow": ProviderPreset(
        name="siliconflow",
        display_name="硅基流动 (SiliconFlow)",
        default_model="deepseek-ai/DeepSeek-V3",
        base_url="https://api.siliconflow.cn/v1",
        description="国内高并发模型托管平台，支持 DeepSeek 系列等",
    ),
    "moonshot": ProviderPreset(
        name="moonshot",
        display_name="Moonshot (Kimi)",
        default_model="moonshot-v1-8k",
        base_url="https://api.moonshot.cn/v1",
        description="月之暗面 Kimi 兼容接口",
    ),
    "zhipu": ProviderPreset(
        name="zhipu",
        display_name="智谱 AI (GLM)",
        default_model="glm-4-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        description="智谱清言大模型开放平台",
    ),
}


def expand_provider_preset(
    name: str | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> tuple[str | None, str | None]:
    """Expand a named preset. Explicit model / base_url override preset defaults."""
    if not name or not str(name).strip():
        return model, base_url
    preset = get_provider_preset(str(name))
    if preset is None:
        raise ValueError(f"未知服务商预设: '{name}'")
    resolved_model = model if model else preset.default_model
    resolved_base_url = base_url if base_url else preset.base_url
    return resolved_model, resolved_base_url


def get_provider_preset(name: str) -> ProviderPreset | None:
    """Lookup a provider preset by name (case-insensitive with aliases)."""
    clean_name = name.strip().lower()
    if clean_name in ("deepseek-reasoner", "reasoner"):
        clean_name = "deepseek-r1"
    if clean_name in ("deepseek-chat", "v3"):
        clean_name = "deepseek-v3"
    if clean_name in ("flash", "v4-flash"):
        clean_name = "deepseek-flash"
    if clean_name in ("v4", "deepseek-v4"):
        clean_name = "deepseek"

    return PREDEFINED_PROVIDERS.get(clean_name)


def list_provider_presets() -> list[ProviderPreset]:
    """Return all predefined provider presets."""
    return list(PREDEFINED_PROVIDERS.values())
