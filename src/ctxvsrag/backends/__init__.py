"""Backend abstraction so the benchmark can target Ollama, vLLM, LiteLLM,
Open WebUI, or any other OpenAI-compatible server."""

from .base import ChatBackend, ChatResult, EmbedBackend
from .ollama_backend import OllamaBackend
from .openai_compat_backend import OpenAICompatBackend

__all__ = ["ChatBackend", "ChatResult", "EmbedBackend", "OllamaBackend", "OpenAICompatBackend"]


def make_chat_backend(kind: str, host: str | None, api_key: str) -> ChatBackend:
    if kind == "ollama":
        return OllamaBackend(host=host)
    if kind == "openai":
        if not host:
            raise SystemExit(
                "--chat-host is required with --chat-backend=openai "
                "(e.g. http://localhost:8000/v1 for vLLM)"
            )
        return OpenAICompatBackend(base_url=host, api_key=api_key)
    raise SystemExit(f"Unknown chat backend type: {kind}")


def make_embed_backend(kind: str, host: str | None, api_key: str) -> EmbedBackend:
    if kind == "ollama":
        return OllamaBackend(host=host)
    if kind == "openai":
        if not host:
            raise SystemExit(
                "--embed-host is required with --embed-backend=openai"
            )
        return OpenAICompatBackend(base_url=host, api_key=api_key)
    raise SystemExit(f"Unknown embed backend type: {kind}")
