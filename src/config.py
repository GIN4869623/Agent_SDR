"""Unified settings — reads .env, exports a single `settings` instance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file(BASE_DIR / ".env")


class Settings(BaseModel):
    app_name: str = "Agent SDR Control Hub"
    llm_provider: Literal["local", "deepseek"] = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "local").strip().lower(),
        validate_default=True,
    )
    ollama_base_url: str = Field(
        default=os.getenv("OLLAMA_BASE_URL", "http://10.224.184.16:11434/v1")
    )
    api_key: str = Field(default=os.getenv("API_KEY", "ollama"))
    usrp_ip: str = Field(default=os.getenv("USRP_IP", "192.168.40.2"))

    # ── Models ──
    model_think: str = Field(default=os.getenv("MODEL_THINK", "qwen3.5:122b"))
    model_fast: str = Field(default=os.getenv("MODEL_FAST", "qwen3.5:35b"))
    deepseek_base_url: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    )
    deepseek_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("DEEPSEEK_API_KEY", "").strip())
    )
    deepseek_model_think: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL_THINK", "deepseek-flash").strip()
    )
    deepseek_model_fast: str = Field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL_FAST", "deepseek-flash").strip()
    )
    temperature_think: float = Field(
        default=float(os.getenv("TEMPERATURE_THINK", "0.35"))
    )
    temperature_fast: float = Field(
        default=float(os.getenv("TEMPERATURE_FAST", "0.15"))
    )

    # ── Agent ──
    max_tool_turns: int = 4
    session_history_limit: int = 12
    general_history_messages: int = Field(
        default=int(os.getenv("GENERAL_HISTORY_MESSAGES", "8"))
    )
    fast_history_messages: int = Field(
        default=int(os.getenv("FAST_HISTORY_MESSAGES", "4"))
    )

    # ── Agent work log ──
    agent_trace_enabled: bool = Field(
        default=os.getenv("AGENT_TRACE_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    agent_trace_path: Path = Field(
        default=Path(
            os.getenv("AGENT_TRACE_PATH", str(BASE_DIR / "logs" / "agent_runs.jsonl"))
        )
    )
    agent_trace_max_records: int = Field(
        default=int(os.getenv("AGENT_TRACE_MAX_RECORDS", "1000")), ge=1
    )

    # ── Knowledge ──
    bailian_api_key: str = Field(default=os.getenv("BAILIAN_API_KEY", ""))
    bailian_app_id: str = Field(default=os.getenv("BAILIAN_APP_ID", ""))
    bailian_embedding_model: str = Field(
        default=os.getenv("BAILIAN_EMBEDDING_MODEL", "text-embedding-v2")
    )
    # ── Console event reporting ──
    console_event_url: str = Field(
        default=os.getenv("CONSOLE_EVENT_URL", "http://127.0.0.1:3000")
    )
    console_event_token: str = Field(
        default=os.getenv("CONSOLE_EVENT_TOKEN", "dev-local-token-change-in-production")
    )

    knowledge_dir: Path = Field(default=BASE_DIR / "data")
    static_dir: Path = Field(default=BASE_DIR / "static")

    @property
    def model_map(self) -> dict[str, str]:
        if self.llm_provider == "deepseek":
            return {"think": self.deepseek_model_think, "fast": self.deepseek_model_fast}
        return {"think": self.model_think, "fast": self.model_fast}

    @property
    def llm_base_url(self) -> str:
        return self.deepseek_base_url if self.llm_provider == "deepseek" else self.ollama_base_url

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key.get_secret_value()
        return self.api_key

    def validate_llm_config(self) -> None:
        """Fail before opening a client rather than sending invalid credentials."""
        if self.llm_provider == "deepseek":
            for name, value in {
                "DEEPSEEK_API_KEY": self.llm_api_key,
                "DEEPSEEK_BASE_URL": self.deepseek_base_url,
                "DEEPSEEK_MODEL_THINK": self.deepseek_model_think,
                "DEEPSEEK_MODEL_FAST": self.deepseek_model_fast,
            }.items():
                if not value.strip():
                    raise ValueError(f"LLM_PROVIDER=deepseek 时必须配置 {name}")

    @property
    def temperature_map(self) -> dict[str, float]:
        return {"think": self.temperature_think, "fast": self.temperature_fast}


settings = Settings()
