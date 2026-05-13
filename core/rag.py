from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests


@dataclass
class KnowledgeHit:
    source: str
    snippet: str
    score: float


RETRIEVAL_PROMPT = """从知识库检索与以下查询相关的原文片段。

查询：{query}

严格只输出如下 JSON，禁止输出任何其他文字：

{{"chunks": [{{"source": "文档名称", "content": "原文片段"}}]}}"""


class LocalKnowledgeBase:
    """Retrieve document chunks from Bailian knowledge base via Application API."""

    def __init__(
        self,
        *,
        app_id: str = "",
        api_key: str = "",
        endpoint: str = "https://dashscope.aliyuncs.com/api/v1/apps",
        timeout: float = 30.0,
    ) -> None:
        self.app_id = app_id
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self._available = bool(app_id and api_key)

    @property
    def document_count(self) -> int:
        return 1 if self._available else 0

    def query(self, prompt: str) -> str:
        if not self._available:
            return ""

        retrieval_text = RETRIEVAL_PROMPT.format(query=prompt)

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.endpoint}/{self.app_id}/completion",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"input": {"prompt": retrieval_text}},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    output = data.get("output") or {}
                    text = output.get("text", "")
                    if text:
                        return self._extract_chunks(text, prompt)
                    return ""
            except Exception:
                pass
            if attempt < 2:
                time.sleep(1.5)
        return ""

    def _extract_chunks(self, raw: str, query: str) -> str:
        # 尝试解析 JSON
        try:
            parsed = json.loads(raw)
            chunks = parsed.get("chunks", [])
            if chunks:
                lines = ["以下是知识库检索到的相关文档片段："]
                for i, c in enumerate(chunks, 1):
                    src = c.get("source", "未知来源")
                    content = c.get("content", "")
                    lines.append(f"{i}. [{src}] {content}")
                return "\n".join(lines)
        except (json.JSONDecodeError, TypeError):
            pass

        # 尝试提取 JSON 代码块
        import re

        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            try:
                parsed = json.loads(match.group(1))
                chunks = parsed.get("chunks", [])
                if chunks:
                    lines = ["以下是知识库检索到的相关文档片段："]
                    for i, c in enumerate(chunks, 1):
                        src = c.get("source", "未知来源")
                        content = c.get("content", "")
                        lines.append(f"{i}. [{src}] {content}")
                    return "\n".join(lines)
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: 直接返回原始文本（比完整回答开销小）
        if len(raw) > 100:
            return f"知识库检索结果：\n{raw}"

        return ""

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeHit]:  # noqa: ARG002
        text = self.query(query)
        if text:
            return [
                KnowledgeHit(
                    source="bailian_knowledge_base",
                    snippet=text,
                    score=1.0,
                )
            ]
        return []
