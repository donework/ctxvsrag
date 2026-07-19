"""Chunk per-page text into overlapping word-windows for retrieval."""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    page: int


def chunk_pages(pages: list[str], chunk_words: int = 300, overlap_words: int = 50) -> list[Chunk]:
    chunks = []
    for page_num, page_text in enumerate(pages, start=1):
        words = page_text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = start + chunk_words
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append(Chunk(text=chunk_text, page=page_num))
            if end >= len(words):
                break
            start = end - overlap_words
    return chunks
