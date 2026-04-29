from __future__ import annotations

from pydantic import Field

from core.rag import LocalKnowledgeBase
from tools import ToolInput, ToolSpec


class SearchKnowledgeRequest(ToolInput):
    query: str = Field(description="要检索的理论问题、术语或设备说明。")
    top_k: int = Field(default=3, ge=1, le=5, description="返回结果数量。")


def build_knowledge_tools(knowledge_base: LocalKnowledgeBase) -> list[ToolSpec]:
    def handler(payload: SearchKnowledgeRequest) -> dict:
        hits = knowledge_base.search(payload.query, top_k=payload.top_k)
        return {
            "status": "success",
            "message": f"已返回 {len(hits)} 条知识检索结果。",
            "results": [
                {
                    "source": hit.source,
                    "snippet": hit.snippet,
                    "score": hit.score,
                }
                for hit in hits
            ],
        }

    return [
        ToolSpec(
            name="search_sdr_knowledge",
            description="查询本地 SDR 理论、实验说明和项目资料。",
            schema_model=SearchKnowledgeRequest,
            handler=handler,
            category="knowledge",
        )
    ]
