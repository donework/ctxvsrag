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
        stream = self.client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=json_schema,
            options={"num_ctx": num_ctx} if num_ctx else None,
            stream=True,
        )

        # Streamed rather than a single call so we can time the first token -
        # the final chunk (done=True) still carries the same aggregate counts/
        # durations a non-streamed call would, so nothing else changes.
        time_to_first_token = None
        text_parts = []
        final_chunk = None
        for chunk in stream:
            if time_to_first_token is None and chunk.message.content:
                time_to_first_token = time.perf_counter() - start
            text_parts.append(chunk.message.content or "")
            final_chunk = chunk
        total_duration = time.perf_counter() - start

        final_chunk = final_chunk or {}
        return ChatResult(
            text="".join(text_parts),
            prompt_tokens=final_chunk.get("prompt_eval_count") or 0,
            output_tokens=final_chunk.get("eval_count") or 0,
            total_duration_s=total_duration,
            prompt_eval_s=(final_chunk.get("prompt_eval_duration") or 0) / 1e9,
            eval_s=(final_chunk.get("eval_duration") or 0) / 1e9,
            time_to_first_token_s=time_to_first_token,
        )

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(model=model, input=texts)
        return list(response.embeddings)
