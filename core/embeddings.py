"""Embedding provider — cloud (DashScope/百炼) text embedding.

文档向量和查询向量必须使用同一个模型，这里集中管理这一选择，
默认使用百炼 `text-embedding-v2`（DashScope SDK）。

当 API key 未配置或调用失败时抛出 :class:`EmbeddingError`，由调用方
（core/rag.py）决定降级策略（当前会回退到关键词匹配）。
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# 百炼 text-embedding-v2 单次调用允许的文本条数上限，超出的分批处理。
DASH_SCOPE_EMBEDDING_BATCH_SIZE = 10


class EmbeddingError(RuntimeError):
    """Raised when cloud embedding is unavailable."""


def _looks_configured(api_key: str) -> bool:
    """判断 API key 是否为真实值（过滤常见占位符）。"""
    key = (api_key or "").strip()
    if not key:
        return False
    return key.lower() not in {"sk-xxxxxxxx", "xxxxxxxx", "your-api-key", "none"}


class DashScopeEmbedder:
    """DashScope text embedding 客户端，保持输入顺序。"""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = (api_key or "").strip()
        self.available = _looks_configured(self.api_key)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document chunks, preserving input order."""
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.available:
            raise EmbeddingError("DashScope API key 未配置（BAILIAN_API_KEY）")

        import dashscope
        from dashscope import TextEmbedding

        dashscope.api_key = self.api_key

        out: list[list[float]] = []
        for start in range(0, len(texts), DASH_SCOPE_EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + DASH_SCOPE_EMBEDDING_BATCH_SIZE]
            resp = TextEmbedding.call(model=self.model, input=batch)
            if resp.status_code != 200:
                raise EmbeddingError(
                    f"embedding 调用失败：{getattr(resp, 'code', '?')} "
                    f"{getattr(resp, 'message', '')}"
                )
            embeddings = (resp.output or {}).get("embeddings", [])
            ordered = sorted(embeddings, key=lambda e: e.get("text_index", 0))
            batch_vecs = [e["embedding"] for e in ordered]
            if len(batch_vecs) != len(batch):
                raise EmbeddingError("embedding 返回条数与输入不一致")
            out.extend(batch_vecs)
        return out
