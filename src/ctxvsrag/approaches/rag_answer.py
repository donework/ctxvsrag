"""Approach B: retrieve relevant chunks via embedding similarity, then answer
using only those chunks as context."""

import time
from dataclasses import dataclass

from ..backends.base import ChatBackend, ChatResult
from ..rag_index import EmbeddingIndex

SYSTEM_PROMPT = (
    "You are a precise analysis assistant. Answer questions strictly based "
    "on the provided excerpts. If the excerpts don't contain the answer, "
    "say so explicitly."
)


@dataclass
class RagResult:
    approach: str
    question: str
    answer: str
    retrieval_s: float
    retrieved_pages: list[int]
    result: ChatResult


def answer_rag(
    backend: ChatBackend,
    model: str,
    index: EmbeddingIndex,
    question: str,
    num_ctx: int,
    k: int = 5,
) -> RagResult:
    retrieval_start = time.perf_counter()
    chunks = index.retrieve(question, k=k)
    retrieval_time = time.perf_counter() - retrieval_start

    context = "\n\n".join(f"[Page {c.page}]\n{c.text}" for c in chunks)

    result = backend.chat(
        model=model,
        system=SYSTEM_PROMPT,
        user=f"Excerpts from the document:\n\n{context}\n\nQuestion: {question}",
        num_ctx=num_ctx,
    )

    return RagResult(
        approach="rag",
        question=question,
        answer=result.text,
        retrieval_s=retrieval_time,
        retrieved_pages=[c.page for c in chunks],
        result=result,
    )
