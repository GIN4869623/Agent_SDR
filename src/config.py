"""Unified settings — reads .env, exports a single `settings` instance."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

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
    ollama_base_url: str = Field(
        default=os.getenv("OLLAMA_BASE_URL", "http://10.224.184.16:11434/v1")
    )
    api_key: str = Field(default=os.getenv("API_KEY", "ollama"))
    usrp_ip: str = Field(default=os.getenv("USRP_IP", "192.168.40.2"))

    # ── Models ──
    model_think: str = Field(default=os.getenv("MODEL_THINK", "qwen3.5:122b"))
    model_fast: str = Field(default=os.getenv("MODEL_FAST", "qwen3.5:35b"))
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

    # ── Knowledge ──
    bailian_api_key: str = Field(default=os.getenv("BAILIAN_API_KEY", ""))
    bailian_app_id: str = Field(default=os.getenv("BAILIAN_APP_ID", ""))
    bailian_embedding_model: str = Field(
        default=os.getenv("BAILIAN_EMBEDDING_MODEL", "text-embedding-v2")
    )
    knowledge_dir: Path = Field(default=BASE_DIR / "data")
    static_dir: Path = Field(default=BASE_DIR / "static")

    @property
    def model_map(self) -> dict[str, str]:
        return {"think": self.model_think, "fast": self.model_fast}

    @property
    def temperature_map(self) -> dict[str, float]:
        return {"think": self.temperature_think, "fast": self.temperature_fast}


settings = Settings()
