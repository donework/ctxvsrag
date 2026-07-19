"""Generic OpenAI-compatible backend - works with vLLM, LiteLLM (SDK proxy or
server), Open WebUI, and Ollama's own /v1 endpoint.

Two things are deliberately weaker here than the native Ollama backend:

1. Timing: these APIs report only total prompt/completion token counts, no
   separate prompt-processing vs. generation duration. `ChatResult` reflects
   that (prompt_eval_s/eval_s stay None; tokens_per_s falls back to an
   estimate from wall-clock time).

2. Structured output: strict JSON-schema enforcement (vLLM's guided decoding,
   OpenAI's `json_schema` response format) isn't reliably available across
   all four targets, and servers fail differently when it's missing rather
   than something we can cleanly detect and fall back from. So this backend
   always requests the widely-supported loose `json_object` mode instead -
   valid JSON is guaranteed, but not the exact schema. The caller (judge.py)
   already spells out the required shape in the prompt text and validates/
   retries on its side.

3. Context window: there's no per-request equivalent of Ollama's `num_ctx` -
   it's a server-side setting (e.g. vLLM's `--max-model-len`). `num_ctx` is
   accepted for interface compatibility but ignored.
"""

import time
from typing import Optional

from openai import OpenAI

from .base import ChatResult


class OpenAICompatBackend:
    def __init__(self, base_url: str, api_key: str = "not-needed"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        num_ctx: Optional[int] = None,
        json_schema: Optional[dict] = None,
    ) -> ChatResult:
        kwargs = {}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        total_duration = time.perf_counter() - start

        usage = response.usage
        return ChatResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_duration_s=total_duration,
        )

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
