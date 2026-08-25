"""Provider presets for mainstream OpenAI-compatible LLM services."""

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
        display_name="DeepSeek V4 Flash (官方默认)",
        default_model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        description="低延迟、长上下文，支持思考与非思考模式",
    ),
    "deepseek-flash": ProviderPreset(
        name="deepseek-flash",
        display_name="DeepSeek V4 Flash (极速版)",
        default_model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        description="极速低延迟轻量模型，价格极具性价比，适合代码检索与快速审查",
    ),
    "deepseek-pro": ProviderPreset(
        name="deepseek-pro",
        display_name="DeepSeek V4 Pro",
        default_model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        description="旗舰模型，适合复杂架构设计、编码与长链路推理",
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


def get_provider_preset(name: str) -> ProviderPreset | None:
    """Lookup a provider preset by name (case-insensitive with aliases)."""
    clean_name = name.strip().lower()
    if clean_name in (
        "deepseek-r1",
        "deepseek-reasoner",
        "reasoner",
        "pro",
        "v4-pro",
        "deepseek-v4-pro",
    ):
        clean_name = "deepseek-pro"
    if clean_name in ("deepseek-chat", "deepseek-v3", "v3"):
        clean_name = "deepseek"
    if clean_name in ("flash", "v4-flash"):
        clean_name = "deepseek-flash"
    if clean_name in ("v4", "deepseek-v4"):
        clean_name = "deepseek"

    return PREDEFINED_PROVIDERS.get(clean_name)


def list_provider_presets() -> list[ProviderPreset]:
    """Return all predefined provider presets."""
    return list(PREDEFINED_PROVIDERS.values())
