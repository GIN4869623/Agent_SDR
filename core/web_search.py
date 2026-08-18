"""Web search fallback — pluggable backends, default Bing RSS (no key).

当本地知识库检索不到答案时，Agent Loop 调用 ``web_search`` 工具，
通过这里的后端上网获取补充信息。

默认后端：Bing RSS 搜索（``cn.bing.com/search?format=rss``），零 key、
国内可直连、返回结构化的 title/link/description。备用后端：
  - Wikipedia（部分网络环境可直连）
  - 商业搜索 API（博查 / Tavily 等，填 key 后接入）
"""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_AGENT_UA = "AgentSDR/1.0 (knowledge fallback)"


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


# 常见口语化前缀/后缀，清洗掉以免污染搜索关键词（例如"什么是 OFDM"→"OFDM"）。
_QUERY_FILLERS = (
    "什么是", "是什么", "什么叫", "是什么意思", "什么意思", "的含义", "的意思",
    "介绍一下", "请解释", "请说明", "帮我", "请问", "叫做", "叫作", "如何", "怎么样",
)


def _clean_query(query: str) -> str:
    """Strip conversational fillers from a search query."""
    q = (query or "").strip()
    for filler in _QUERY_FILLERS:
        q = q.replace(filler, " ")
    q = re.sub(r"[，。！？、,;:!?\s]+", " ", q).strip()
    return q or (query or "").strip()


def _search_bing(query: str, max_results: int) -> list[WebSearchResult]:
    """Search via Bing's RSS endpoint (no key required)."""
    resp = requests.get(
        "https://cn.bing.com/search",
        params={"q": query, "format": "rss", "count": max_results},
        timeout=10,
        headers={"User-Agent": _BROWSER_UA},
    )
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    results: list[WebSearchResult] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        snippet = _strip_html(item.findtext("description") or "")
        if title and link:
            results.append(WebSearchResult(title=title, url=link, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


def _search_wikipedia(
    query: str,
    max_results: int,
    lang: str,
) -> list[WebSearchResult]:
    endpoint = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
        "utf8": 1,
    }
    resp = requests.get(
        endpoint,
        params=params,
        timeout=10,
        headers={"User-Agent": _AGENT_UA},
    )
    resp.raise_for_status()
    data = resp.json()

    results: list[WebSearchResult] = []
    for item in (data.get("query") or {}).get("search", []):
        title = item.get("title", "")
        if not title:
            continue
        snippet = _strip_html(item.get("snippet", ""))
        url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        results.append(WebSearchResult(title=title, url=url, snippet=snippet))
    return results


def search_web(
    query: str,
    max_results: int = 5,
    backend: str = "auto",
) -> list[WebSearchResult]:
    """Search the web, returning title/url/snippet results.

    ``backend`` 为预留参数：默认（``auto``/``bing``）走 Bing RSS，失败时
    回退 Wikipedia。任何网络异常都被捕获并返回空列表，保证不因网络问题
    让 Agent 崩溃。
    """
    query = _clean_query(query)

    # 主后端：Bing RSS（国内可直连、零 key）
    if backend in ("auto", "bing"):
        try:
            results = _search_bing(query, max_results)
            if results:
                return results
        except Exception:
            logger.warning("Bing 搜索失败，回退 Wikipedia", exc_info=True)

    # 备用后端：Wikipedia（部分网络环境可直连）
    if backend in ("auto", "bing", "wikipedia"):
        for lang in ("zh", "en"):
            try:
                results = _search_wikipedia(query, max_results, lang)
                if results:
                    return results
            except Exception:
                logger.warning("Wikipedia(%s) 搜索失败", lang, exc_info=True)

    return []
