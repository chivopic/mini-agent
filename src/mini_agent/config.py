"""User/project TOML config. Merge order (high wins): CLI > env > project > user > defaults."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mini_agent.models import AgentConfig, PermissionClass
from mini_agent.permission import Decision
from mini_agent.providers import expand_provider_preset

_SECRET_KEY_NAMES = frozenset({"api_key", "openai_api_key"})
_PERMISSION_RANK = {Decision.DENY: 0, Decision.ASK: 1, Decision.ALLOW: 2}
_PERMISSION_KEYS = ("read", "edit", "shell", "git")
DEFAULT_MODEL = "gpt-4o-mini"
PROJECT_CONFIG_NAME = ".mini-agent.toml"


class ConfigError(Exception):
    """Invalid or unreadable config. Fail closed; never fall back to defaults."""


class LimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tool_rounds: int = Field(default=32, ge=1, le=200)
    shell_timeout_seconds: int = Field(default=30, ge=1)
    max_output_chars: int = Field(default=12_000, ge=1024)
    context_window_tokens: int = Field(default=128_000, ge=4000)
    compaction_buffer_tokens: int = Field(default=8_000, ge=1000)
    keep_recent_tokens: int = Field(default=24_000, ge=2000)
    max_parallel_readonly: int = Field(default=4, ge=1, le=8)
    auto_summarize: bool = True


class PermissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read: Decision = Decision.ALLOW
    edit: Decision = Decision.ALLOW
    shell: Decision = Decision.ASK
    git: Decision = Decision.ASK


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    model: str | None = None
    base_url: str | None = None

    @field_validator("name", "model", "base_url", mode="before")
    @classmethod
    def _strip_empty(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    permission: PermissionConfig = Field(default_factory=PermissionConfig)

    def to_agent_config(self, workspace_root: Path) -> AgentConfig:
        return AgentConfig(
            workspace_root=workspace_root,
            model=self.provider.model or DEFAULT_MODEL,
            max_tool_rounds=self.limits.max_tool_rounds,
            shell_timeout_seconds=self.limits.shell_timeout_seconds,
            max_output_chars=self.limits.max_output_chars,
        )

    def permission_class_defaults(self) -> dict[PermissionClass, Decision]:
        return {
            PermissionClass.READ: self.permission.read,
            PermissionClass.EDIT: self.permission.edit,
            PermissionClass.SHELL: self.permission.shell,
            PermissionClass.GIT: self.permission.git,
        }


def defaults() -> AppConfig:
    return AppConfig()


def default_user_config_path() -> Path:
    return Path.home() / ".mini-agent" / "config.toml"


def load_dotenv(workspace_root: Path | None = None) -> None:
    """Load workspace `.env` into os.environ without overriding existing keys."""
    path = (workspace_root / ".env") if workspace_root else Path(".env")
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, val = stripped.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            pass


def load_app_config(
    workspace: Path,
    *,
    cli_model: str | None = None,
    cli_base_url: str | None = None,
    user_config_path: Path | None = None,
    user_config_required: bool = False,
    apply_dotenv: bool = True,
) -> tuple[AppConfig, list[str]]:
    """Load AppConfig.

    Merge order (high wins): CLI flags > process env (after dotenv) >
    project ``<workspace>/.mini-agent.toml`` > user TOML > code defaults.
    ``name`` expands a provider preset; explicit ``base_url`` / ``model`` win.
    A layer that sets ``name`` drops inherited ``model`` / ``base_url`` unless
    that same layer (or a still-higher one) set them.
    """
    warnings: list[str] = []
    if apply_dotenv:
        load_dotenv(workspace)

    user_path = user_config_path if user_config_path is not None else default_user_config_path()
    user_data, user_warnings = _read_toml(user_path, required=user_config_required)
    warnings.extend(user_warnings)

    project_path = workspace / PROJECT_CONFIG_NAME
    project_data, project_warnings = _read_toml(project_path, required=False)
    warnings.extend(project_warnings)
    warnings.extend(_project_permission_warnings(user_data, project_data))

    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, user_data)
    merged = _deep_merge(merged, project_data)
    merged = _deep_merge(merged, _env_overlay())
    merged = _deep_merge(merged, _cli_overlay(cli_model, cli_base_url))

    try:
        cfg = AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"配置无效: {exc}") from exc

    cfg = cfg.model_copy(update={"provider": _finalize_provider(cfg.provider)})
    return cfg, warnings


def _finalize_provider(provider: ProviderConfig) -> ProviderConfig:
    try:
        model, base_url = expand_provider_preset(
            provider.name,
            model=provider.model,
            base_url=provider.base_url,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if not model:
        if base_url and "deepseek" in base_url.lower():
            model = "deepseek-v4"
        else:
            model = DEFAULT_MODEL
    return provider.model_copy(update={"model": model, "base_url": base_url})


def _env_overlay() -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    model = os.environ.get("MINI_AGENT_MODEL", "").strip()
    base_url = (
        os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or ""
    ).strip()
    provider: dict[str, str] = {}
    if model:
        provider["model"] = model
    if base_url:
        provider["base_url"] = base_url
    if provider:
        overlay["provider"] = provider
    return overlay


def _cli_overlay(cli_model: str | None, cli_base_url: str | None) -> dict[str, Any]:
    provider: dict[str, str] = {}
    if cli_model and cli_model.strip():
        provider["model"] = cli_model.strip()
    if cli_base_url and cli_base_url.strip():
        provider["base_url"] = cli_base_url.strip()
    return {"provider": provider} if provider else {}


def _read_toml(path: Path, *, required: bool) -> tuple[dict[str, Any], list[str]]:
    if path.is_file():
        return _load_toml_file(path)
    if path.exists():
        raise ConfigError(f"配置路径不是文件: {path}")
    if required:
        raise ConfigError(f"配置文件不存在: {path}")
    return {}, []


def _load_toml_file(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"无法解析配置文件 {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件 {path} 顶层必须是表")
    warnings: list[str] = []
    cleaned = _strip_secret_keys(raw, warnings, source=str(path))
    if not isinstance(cleaned, dict):
        raise ConfigError(f"配置文件 {path} 顶层必须是表")
    return cleaned, warnings


def _strip_secret_keys(value: object, warnings: list[str], *, source: str) -> object:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        stripped: list[str] = []
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in _SECRET_KEY_NAMES:
                stripped.append(key)
                continue
            cleaned[key] = _strip_secret_keys(nested, warnings, source=source)
        if stripped:
            warnings.append(
                f"已忽略 {source} 中的密钥字段 {', '.join(stripped)}；"
                "请使用环境变量或工作区 .env 提供 OPENAI_API_KEY。"
            )
        return cleaned
    if isinstance(value, list):
        return [_strip_secret_keys(item, warnings, source=source) for item in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if key == "provider" and isinstance(value, dict):
            # Setting name is a preset switch, not a field-wise overlay.
            merged[key] = _merge_provider(existing if isinstance(existing, dict) else {}, value)
        elif isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _merge_provider(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    if "name" in overlay:
        merged.pop("model", None)
        merged.pop("base_url", None)
    for key, value in overlay.items():
        merged[key] = value
    return merged


def _project_permission_warnings(
    user_data: dict[str, Any],
    project_data: dict[str, Any],
) -> list[str]:
    project_perm = project_data.get("permission")
    if not isinstance(project_perm, dict):
        return []
    parent: dict[str, Decision] = {
        "read": Decision.ALLOW,
        "edit": Decision.ALLOW,
        "shell": Decision.ASK,
        "git": Decision.ASK,
    }
    user_perm = user_data.get("permission")
    if isinstance(user_perm, dict):
        for key in _PERMISSION_KEYS:
            parsed = _parse_decision(user_perm.get(key))
            if parsed is not None:
                parent[key] = parsed
    warnings: list[str] = []
    for key in _PERMISSION_KEYS:
        new = _parse_decision(project_perm.get(key))
        if new is None:
            continue
        old = parent[key]
        if _PERMISSION_RANK[new] > _PERMISSION_RANK[old]:
            warnings.append(
                f"警告: 项目 .mini-agent.toml 将 {key} 权限放宽为 {new.value}（无签名校验）。"
            )
    return warnings


def _parse_decision(value: object) -> Decision | None:
    if isinstance(value, Decision):
        return value
    if isinstance(value, str):
        try:
            return Decision(value)
        except ValueError:
            return None
    return None
