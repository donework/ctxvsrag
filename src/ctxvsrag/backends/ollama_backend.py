"""Ollama backend - native /api/chat and /api/embed. This is the only backend
with precise per-call prompt-processing vs. generation timing, since Ollama
reports both durations natively; OpenAI-compatible APIs only report totals."""

import time
from typing import Optional

import ollama

from .base import ChatResult


class OllamaBackend:
    def __init__(self, host: Optional[str] = None):
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        num_ctx: Optional[int] = None,
        json_schema: Optional[dict] = None,
    ) -> ChatResult:
        start = time.perf_counter()
        response = self.client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=json_schema,
            options={"num_ctx": num_ctx} if num_ctx else None,
        )
        total_duration = time.perf_counter() - start

        return ChatResult(
            text=response.message.content or "",
            prompt_tokens=response.get("prompt_eval_count") or 0,
            output_tokens=response.get("eval_count") or 0,
            total_duration_s=total_duration,
            prompt_eval_s=(response.get("prompt_eval_duration") or 0) / 1e9,
            eval_s=(response.get("eval_duration") or 0) / 1e9,
        )

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(model=model, input=texts)
        return list(response.embeddings)
