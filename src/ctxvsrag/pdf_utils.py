"""PDF loading: per-page text extraction, plus a rough token-count estimate
(no access to the local model's actual tokenizer, so this is chars/3 — good
enough to size --num-ctx, not exact)."""

from pypdf import PdfReader


def extract_pages(path: str) -> list[str]:
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)
