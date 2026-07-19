"""Approach A: feed the whole extracted PDF text as context on every request.

No document/PDF content-block support on any of these backends' chat APIs -
the extracted plain text goes straight into the prompt, so the model's
context window must be large enough to hold the whole document or it loses
earlier content (silently truncated on Ollama via --num-ctx; a hard
server-side limit on OpenAI-compatible backends like vLLM).
"""

from dataclasses import dataclass

from ..backends.base import ChatBackend, ChatResult

SYSTEM_PROMPT = (
    "You are a precise analysis assistant. Answer questions strictly based "
    "on the attached document."
)


@dataclass
class ApiResult:
    approach: str
    question: str
    answer: str
    result: ChatResult


def answer_full_context(
    backend: ChatBackend,
    model: str,
    document_text: str,
    question: str,
    num_ctx: int,
) -> ApiResult:
    result = backend.chat(
        model=model,
        system=SYSTEM_PROMPT,
        user=f"Document:\n\n{document_text}\n\nQuestion: {question}",
        num_ctx=num_ctx,
    )
    return ApiResult(
        approach="full_context",
        question=question,
        answer=result.text,
        result=result,
    )
