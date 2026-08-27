from __future__ import annotations

from pydantic import Field

from core.rag import LocalKnowledgeBase
from core.web_search import search_web
from tools import ToolInput, ToolSpec


class SearchKnowledgeRequest(ToolInput):
    query: str = Field(min_length=1, max_length=512, description="要检索的理论问题、术语或设备说明。")
    top_k: int = Field(default=3, ge=1, le=10, description="返回结果数量。")


class WebSearchRequest(ToolInput):
    query: str = Field(
        min_length=1,
        max_length=256,
        description="用于联网搜索的精简专业关键词，不要传入完整口语问句。",
    )
    max_results: int = Field(default=5, ge=1, le=10, description="返回结果数量。")


def build_knowledge_tools(knowledge_base: LocalKnowledgeBase) -> list[ToolSpec]:
    def search_local(request: SearchKnowledgeRequest) -> dict:
        hits = knowledge_base.search(request.query, top_k=request.top_k)
        message = f"已返回 {len(hits)} 条知识检索结果。"
        if not hits:
            message += " 知识库未命中，可调用 web_search 联网搜索补充。"
        return {
            "status": "success",
            "message": message,
            "results": [
                {"source": hit.source, "snippet": hit.snippet, "score": hit.score}
                for hit in hits
            ],
        }

    def search_online(request: WebSearchRequest) -> dict:
        results = search_web(request.query, max_results=request.max_results)
        if not results:
            return {
                "status": "empty",
                "message": f"未搜索到与「{request.query}」相关的网络内容。",
            }
        return {
            "status": "success",
            "message": f"已返回 {len(results)} 条网络搜索结果。",
            "results": [
                {"title": item.title, "url": item.url, "snippet": item.snippet}
                for item in results
            ],
        }

    return [
        ToolSpec(
            name="search_sdr_knowledge",
            description="从本地知识库检索 SDR 理论、设备规格、实验文档和项目资料。",
            schema_model=SearchKnowledgeRequest,
            handler=search_local,
            category="knowledge",
            risk="read_only",
            retry_safe=True,
        ),
        ToolSpec(
            name="web_search",
            description=(
                "联网搜索 SDR 相关知识。仅当本地知识库没有足够信息时使用，"
                "query 应使用精简专业关键词。"
            ),
            schema_model=WebSearchRequest,
            handler=search_online,
            category="knowledge",
            risk="read_only",
            retry_safe=True,
        ),
    ]
