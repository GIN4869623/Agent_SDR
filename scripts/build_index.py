"""Offline build/refresh of the local SDR knowledge index.

Usage (run from the project root):
    python scripts/build_index.py            # incremental (new/changed files only)
    python scripts/build_index.py --force    # full rebuild

Index is written to data/.index (FAISS); embedding uses 百炼
text-embedding-v2 via DashScope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag import LocalKnowledgeBase  # noqa: E402
from src.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local SDR knowledge index.")
    parser.add_argument("--force", action="store_true", help="Rebuild all chunks.")
    args = parser.parse_args()

    kb = LocalKnowledgeBase(
        api_key=settings.bailian_api_key,
        embedding_model=settings.bailian_embedding_model,
        knowledge_dir=settings.knowledge_dir,
        index_dir=settings.knowledge_dir / ".index",
    )
    processed = kb.ensure_index(force=args.force)
    print(f"索引构建完成：本次处理 {processed} 个 chunk，当前共 {kb.document_count} 个 chunk。")


if __name__ == "__main__":
    main()
