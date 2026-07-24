"""Shared interface all backends implement.

A ChatBackend answers questions; an EmbedBackend turns text into vectors for
RAG retrieval. A single provider (Ollama) can implement both. Others (a vLLM
chat server, for instance) typically only implement chat, since vLLM serves
one model per instance - embeddings then come from a separately configured
backend, which is why the two protocols are kept independent rather than
bundled into one "provider" interface.
"""

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    total_duration_s: float
    # None on backends that only report wall-clock time (i.e. every
    # OpenAI-compatible API) - Ollama's native API is the only one that
    # splits out prompt-processing vs. generation duration per call.
    prompt_eval_s: Optional[float] = None
    eval_s: Optional[float] = None
    # Wall-clock time from request start to the first streamed token. Dominated
    # by prompt processing (prefill) time, so this is the number that actually
    # shows a user-facing effect of "more input tokens" (full-context) vs.
    # "fewer input tokens" (RAG) - unlike prompt_eval_s, it's measurable on
    # every backend since it only requires streaming, not native duration
    # reporting. None if the backend call produced no output at all.
    time_to_first_token_s: Optional[float] = None

    @property
    def precise_timing(self) -> bool:
        return self.eval_s is not None

    @property
    def tokens_per_s(self) -> float:
        """Generation speed. Exact when eval_s is available; otherwise an
        approximation from total wall-clock time (includes network/queueing
        overhead, so it understates true generation speed)."""
        duration = self.eval_s if self.eval_s is not None else self.total_duration_s
        return self.output_tokens / duration if duration > 0 else 0.0


class ChatBackend(Protocol):
    def chat(
        self,
        model: str,
        system: str,
        user: str,
        num_ctx: Optional[int] = None,
        json_schema: Optional[dict] = None,
    ) -> ChatResult:
        """json_schema, if given, requests JSON output. Enforcement strictness
        varies by backend - see each implementation's docstring."""
        ...


class EmbedBackend(Protocol):
    def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...
