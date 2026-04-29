from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "data"


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", value.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "document"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_chunks(page_texts: list[str], chunk_pages: int) -> list[tuple[int, int, str]]:
    chunks: list[tuple[int, int, str]] = []
    for start in range(0, len(page_texts), chunk_pages):
        end = min(start + chunk_pages, len(page_texts))
        body_parts: list[str] = []
        for page_index in range(start, end):
            page_no = page_index + 1
            page_text = page_texts[page_index]
            if not page_text:
                continue
            body_parts.append(f"## Page {page_no}\n\n{page_text}")
        if not body_parts:
            continue
        chunks.append((start + 1, end, "\n\n".join(body_parts)))
    return chunks


def export_pdf_to_markdown(
    pdf_path: Path,
    knowledge_dir: Path,
    doc_name: str | None,
    chunk_pages: int,
) -> tuple[Path, int]:
    pdf_path = pdf_path.resolve()
    knowledge_dir = knowledge_dir.resolve()
    title = doc_name or pdf_path.stem
    output_dir = knowledge_dir / slugify(title)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    page_texts = [clean_text(page.extract_text() or "") for page in reader.pages]
    chunks = build_chunks(page_texts, chunk_pages=chunk_pages)

    if not chunks:
        raise RuntimeError("PDF 未提取到可用文本，可能是扫描版，需要先做 OCR。")

    for index, (start_page, end_page, content) in enumerate(chunks, start=1):
        filename = f"chunk_{index:03d}_p{start_page}-p{end_page}.md"
        chunk_path = output_dir / filename
        markdown = (
            f"# {title}\n\n"
            f"- Source PDF: `{pdf_path}`\n"
            f"- Page Range: {start_page}-{end_page}\n\n"
            f"{content}\n"
        )
        chunk_path.write_text(markdown, encoding="utf-8")

    return output_dir, len(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a PDF into the local SDR knowledge base.")
    parser.add_argument("pdf", type=Path, help="Path to the source PDF file.")
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=DEFAULT_KNOWLEDGE_DIR,
        help="Knowledge base directory. Defaults to the project's data/ folder.",
    )
    parser.add_argument(
        "--doc-name",
        type=str,
        default=None,
        help="Optional display name for the imported document.",
    )
    parser.add_argument(
        "--chunk-pages",
        type=int,
        default=6,
        help="Number of PDF pages per markdown chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_pages < 1:
        raise ValueError("--chunk-pages must be >= 1")

    output_dir, chunk_count = export_pdf_to_markdown(
        pdf_path=args.pdf,
        knowledge_dir=args.knowledge_dir,
        doc_name=args.doc_name,
        chunk_pages=args.chunk_pages,
    )
    print(f"Imported into: {output_dir}")
    print(f"Chunks written: {chunk_count}")


if __name__ == "__main__":
    main()
