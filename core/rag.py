"""Local knowledge base — FAISS vector search + cloud (DashScope) embedding.

Architecture:
    data/ 下的 .md/.txt 是唯一资料源（single source of truth）
    分块 → 云端 embedding（百炼 text-embedding-v2）→ 写入本地 FAISS 索引
    （data/.index）持久化
    查询时：query 过同一个 embedding 模型 → FAISS 余弦检索 → 本地返回

对外接口保持不变（``search`` / ``query`` / ``document_count``），因此
src/mcp/__init__.py 的 MCP 工具与 resource 无需修改。

存储后端说明：优先选用 FAISS（cpu），因为本机 chromadb 1.5.x 的 rust
SQLite 后端在部分环境无法初始化；FAISS 无外部数据库依赖，对当前小规模
资料完全够用。若日后数据量增大，可平滑替换为 ChromaDB 等持久化向量库。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.embeddings import DashScopeEmbedder, EmbeddingError

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeHit:
    source: str
    snippet: str
    score: float


# ═══════════════════════════════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════════════════════════════

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _segment_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading_line, body) segments by ATX headings."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    segments: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        segments.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append((m.group(0), text[start:end]))
    return segments


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split a long text into windows of max_chars with the given overlap."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_text(
    text: str,
    source: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> list[dict]:
    """Return list of {text, source} chunks for indexing."""
    chunks: list[dict] = []
    for heading, body in _segment_by_headings(text):
        prefix = (heading + "\n\n") if heading else ""
        for frag in _split_long(body, max_chars, overlap):
            if not frag.strip():
                continue
            chunks.append({"text": prefix + frag, "source": source})
    return chunks


def _bigrams(s: str) -> set[str]:
    """Char bigrams of a string (whitespace removed) for keyword fallback."""
    s = re.sub(r"\s+", "", s or "")
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge base
# ═══════════════════════════════════════════════════════════════════════════


class LocalKnowledgeBase:
    """Local vector knowledge base backed by FAISS + numpy persistence.

    ``app_id`` 保留仅为兼容旧调用，本实现不再使用百炼托管 RAG 应用。
    """

    def __init__(
        self,
        *,
        app_id: str = "",  # noqa: ARG002 — kept for backward compatibility
        api_key: str = "",
        embedding_model: str = "text-embedding-v2",
        knowledge_dir: Path | None = None,
        index_dir: Path | None = None,
        chunk_chars: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self._api_key = api_key
        self._embedding_model = embedding_model
        self._knowledge_dir = Path(knowledge_dir) if knowledge_dir else Path("data")
        self._index_dir = Path(index_dir) if index_dir else self._knowledge_dir / ".index"
        self._chunk_chars = chunk_chars
        self._chunk_overlap = chunk_overlap

        self._embedder = DashScopeEmbedder(model=embedding_model, api_key=api_key)
        # 每条记录: {"source", "text", "chunk_index", "vector"}（vector 未归一化）
        self._records: list[dict] = []
        self._faiss_index = None
        self._build_lock = threading.Lock()
        self._built = False
        self._load_index()

    # ── Index persistence ───────────────────────────────────────────────

    def _load_index(self) -> None:
        vec_path = self._index_dir / "vectors.npy"
        rec_path = self._index_dir / "records.json"
        try:
            if vec_path.exists() and rec_path.exists():
                vecs = np.load(str(vec_path)).astype("float32")
                meta = json.loads(rec_path.read_text(encoding="utf-8"))
                if len(vecs) == len(meta) and len(meta) > 0:
                    self._records = [
                        {
                            "source": m["source"],
                            "text": m["text"],
                            "chunk_index": m["chunk_index"],
                            "vector": vecs[i].tolist(),
                        }
                        for i, m in enumerate(meta)
                    ]
                    self._rebuild_faiss()
        except Exception:
            logger.warning("本地索引加载失败，将按空索引处理", exc_info=True)
            self._records = []
            self._faiss_index = None

    def _save_index(self) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        if not self._records:
            return
        vecs = np.asarray([r["vector"] for r in self._records], dtype="float32")
        np.save(str(self._index_dir / "vectors.npy"), vecs)
        meta = [
            {"source": r["source"], "text": r["text"], "chunk_index": r["chunk_index"]}
            for r in self._records
        ]
        (self._index_dir / "records.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rebuild_faiss(self) -> None:
        if not self._records:
            self._faiss_index = None
            return
        import faiss

        vecs = np.asarray([r["vector"] for r in self._records], dtype="float32")
        faiss.normalize_L2(vecs)
        self._faiss_index = faiss.IndexFlatIP(vecs.shape[1])
        self._faiss_index.add(vecs)

    # ── Public surface (unchanged) ──────────────────────────────────────

    @property
    def document_count(self) -> int:
        return len(self._records)

    def query(self, prompt: str) -> str:
        """Return formatted retrieval text (used by the ``sdr://knowledge`` resource)."""
        hits = self.search(prompt, top_k=3)
        if not hits:
            return ""
        lines = ["以下是知识库检索到的相关文档片段："]
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. [{h.source}] {h.snippet}")
        return "\n".join(lines)

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeHit]:
        """Search the local vector index; falls back to keyword matching."""
        # 首次查询触发一次增量索引同步（空索引则全量构建）。
        if not self._built:
            try:
                self.ensure_index()
            except Exception:
                logger.warning("索引构建失败，回退关键词匹配", exc_info=True)
            finally:
                self._built = True

        if not self._records or self._faiss_index is None:
            return self._keyword_search(query, top_k)

        try:
            vec = np.asarray(self._embedder.embed_query(query), dtype="float32").reshape(1, -1)
        except EmbeddingError as exc:
            logger.warning("query embedding 失败，回退关键词匹配: %s", exc)
            return self._keyword_search(query, top_k)

        import faiss

        faiss.normalize_L2(vec)
        k = min(top_k, len(self._records))
        scores, idxs = self._faiss_index.search(vec, k)

        hits: list[KnowledgeHit] = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            rec = self._records[i]
            score = max(0.0, min(1.0, float(s)))
            hits.append(
                KnowledgeHit(source=rec["source"], snippet=rec["text"], score=round(score, 4))
            )
        return hits

    # ── Indexing ────────────────────────────────────────────────────────

    def ensure_index(self, force: bool = False) -> int:
        """Build/refresh the index. Returns the number of chunks processed."""
        with self._build_lock:
            manifest = self._load_manifest()
            current_map: dict[str, str] = {}
            for f in self._iter_source_files():
                current_map[str(f.relative_to(self._knowledge_dir))] = self._file_hash(f)

            # Remove chunks whose source file was deleted.
            for src in list(manifest.keys()):
                if src not in current_map:
                    self._remove_source_records(src)

            processed = 0
            for f in self._iter_source_files():
                src = str(f.relative_to(self._knowledge_dir))
                digest = current_map[src]
                if not force and manifest.get(src) == digest:
                    continue
                chunks = chunk_text(
                    f.read_text(encoding="utf-8"),
                    source=src,
                    max_chars=self._chunk_chars,
                    overlap=self._chunk_overlap,
                )
                self._index_source(src, chunks)
                manifest[src] = digest
                processed += len(chunks)

            self._save_manifest(manifest)
            self._save_index()
            self._rebuild_faiss()
            self._built = True
            return processed

    def _index_source(self, src: str, chunks: list[dict]) -> None:
        self._remove_source_records(src)
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        vectors = self._embedder.embed_documents(texts)
        for i, (c, vec) in enumerate(zip(chunks, vectors)):
            self._records.append(
                {"source": src, "text": c["text"], "chunk_index": i, "vector": vec}
            )

    def _remove_source_records(self, src: str) -> None:
        self._records = [r for r in self._records if r["source"] != src]

    # ── File discovery / manifest ───────────────────────────────────────

    def _iter_source_files(self) -> list[Path]:
        if not self._knowledge_dir.exists():
            return []
        files: list[Path] = []
        for pattern in ("*.md", "*.txt"):
            for p in self._knowledge_dir.rglob(pattern):
                if ".index" in p.parts or p.name == "README.md":
                    continue
                files.append(p)
        return sorted(files)

    def _manifest_path(self) -> Path:
        return self._index_dir / "_manifest.json"

    def _load_manifest(self) -> dict[str, str]:
        try:
            return json.loads(self._manifest_path().read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        self._manifest_path().parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    # ── Keyword fallback ────────────────────────────────────────────────

    def _keyword_search(self, query: str, top_k: int) -> list[KnowledgeHit]:
        files = self._iter_source_files()
        if not files:
            return []
        q_bg = _bigrams(query)
        if not q_bg:
            return []
        scored: list[tuple[float, str, str]] = []
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            for frag in _split_long(text, self._chunk_chars, self._chunk_overlap):
                if not frag.strip():
                    continue
                c_bg = _bigrams(frag)
                if not c_bg:
                    continue
                score = len(q_bg & c_bg) / len(q_bg)
                if score > 0:
                    scored.append((score, str(f.relative_to(self._knowledge_dir)), frag))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            KnowledgeHit(source=src, snippet=snippet[: self._chunk_chars], score=round(s, 4))
            for s, src, snippet in scored[:top_k]
        ]
