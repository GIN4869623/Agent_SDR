"""Shared local/DeepSeek client construction for the server and evaluations."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI

from src.config import Settings, settings


def llm_client_options(config: Settings = settings) -> dict[str, str]:
    config.validate_llm_config()
    return {"base_url": config.llm_base_url, "api_key": config.llm_api_key}


def create_async_llm_client(config: Settings = settings, **options: Any) -> AsyncOpenAI:
    return AsyncOpenAI(**(llm_client_options(config) | options))


def create_llm_client(config: Settings = settings, **options: Any) -> OpenAI:
    return OpenAI(**(llm_client_options(config) | options))


def provider_config(provider: str | None = None, config: Settings = settings) -> Settings:
    selected = provider if provider is not None else config.llm_provider
    if selected not in {"local", "deepseek"}:
        raise ValueError("不支持的模型来源")
    return config.model_copy(update={"llm_provider": selected})


def provider_catalog(config: Settings = settings) -> dict[str, dict[str, Any]]:
    """Public configuration: model IDs and readiness, never credentials."""
    catalog = {}
    for provider, label in (("local", "本地模型"), ("deepseek", "DeepSeek 云端")):
        selected = provider_config(provider, config)
        message = ""
        try:
            selected.validate_llm_config()
        except ValueError as exc:
            message = f"{exc}。请补全 .env 后重启服务。"
        catalog[provider] = {
            "label": label, "models": selected.model_map,
            "configured": not message, "message": message,
        }
    return catalog
