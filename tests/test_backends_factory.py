"""Tests for the make_chat_backend/make_embed_backend factory functions.

These only construct client objects (no network I/O happens at construction
time for either the ollama or openai SDK clients), so no live server needed.
"""

import pytest

from ctxvsrag.backends import (
    OllamaBackend,
    OpenAICompatBackend,
    make_chat_backend,
    make_embed_backend,
)


def test_make_chat_backend_ollama():
    backend = make_chat_backend("ollama", None, "unused")
    assert isinstance(backend, OllamaBackend)


def test_make_chat_backend_openai_requires_host():
    with pytest.raises(SystemExit):
        make_chat_backend("openai", None, "unused")


def test_make_chat_backend_openai_with_host():
    backend = make_chat_backend("openai", "http://localhost:8000/v1", "key")
    assert isinstance(backend, OpenAICompatBackend)


def test_make_chat_backend_unknown_kind_raises():
    with pytest.raises(SystemExit):
        make_chat_backend("bogus", "http://x", "key")


def test_make_embed_backend_ollama():
    backend = make_embed_backend("ollama", None, "unused")
    assert isinstance(backend, OllamaBackend)


def test_make_embed_backend_openai_requires_host():
    with pytest.raises(SystemExit):
        make_embed_backend("openai", None, "unused")
