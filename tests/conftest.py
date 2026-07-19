"""Shared test fixtures: fake backends so tests run fast, offline, and deterministically."""

import pytest

from ctxvsrag.backends.base import ChatResult


class FakeChatBackend:
    """Records every call; returns responses from a queue, in order.

    Assign a list to .responses before calling .chat() - either ChatResult
    instances to return, or Exception instances to raise.
    """

    def __init__(self):
        self.responses: list = []
        self.calls: list[dict] = []

    def chat(self, model, system, user, num_ctx=None, json_schema=None):
        self.calls.append(
            {"model": model, "system": system, "user": user, "num_ctx": num_ctx, "json_schema": json_schema}
        )
        if not self.responses:
            raise AssertionError("FakeChatBackend.chat() called but no more responses queued")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeEmbedBackend:
    """Returns vectors from a dict keyed by exact text - lets tests control
    retrieval order precisely instead of depending on a real embedding model."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls: list[dict] = []

    def embed(self, model, texts):
        self.calls.append({"model": model, "texts": list(texts)})
        return [self.vectors[t] for t in texts]


def default_chat_result(text: str = "fake answer", **overrides) -> ChatResult:
    fields = dict(text=text, prompt_tokens=100, output_tokens=20, total_duration_s=1.0)
    fields.update(overrides)
    return ChatResult(**fields)


@pytest.fixture
def fake_chat_backend() -> FakeChatBackend:
    return FakeChatBackend()
