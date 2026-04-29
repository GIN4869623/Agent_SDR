from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_.-]+")


@dataclass
class KnowledgeHit:
    source: str
    snippet: str
    score: float


class LocalKnowledgeBase:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.documents = self._load_documents()

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def _load_documents(self) -> list[dict]:
        docs: list[dict] = []
        if not self.root.exists():
            return docs

        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "content": content.strip(),
                    "tokens": self._tokenize(content),
                }
            )
        return docs

    def _tokenize(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for raw in TOKEN_PATTERN.findall(text.lower()):
            if any("\u4e00" <= ch <= "\u9fff" for ch in raw):
                tokens.add(raw)
                if len(raw) > 1:
                    tokens.update(raw[idx : idx + 2] for idx in range(len(raw) - 1))
            else:
                tokens.add(raw)
        return tokens

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeHit]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        hits: list[KnowledgeHit] = []
        for doc in self.documents:
            overlap = len(query_tokens & doc["tokens"])
            if overlap == 0 and query.lower() not in doc["content"].lower():
                continue

            score = float(overlap)
            if query.lower() in doc["content"].lower():
                score += 2.0

            hits.append(
                KnowledgeHit(
                    source=doc["path"],
                    snippet=self._build_snippet(doc["content"], query),
                    score=score,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def _build_snippet(self, content: str, query: str) -> str:
        compact = " ".join(content.split())
        lowered = compact.lower()
        idx = lowered.find(query.lower())
        if idx == -1:
            idx = 0
        start = max(0, idx - 60)
        end = min(len(compact), idx + 180)
        return compact[start:end]
