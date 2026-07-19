from ctxvsrag.chunking import Chunk
from ctxvsrag.rag_index import EmbeddingIndex, default_prefixes_for

from conftest import FakeEmbedBackend


def test_retrieve_ranks_by_cosine_similarity():
    chunks = [Chunk(text="about cats", page=1), Chunk(text="about dogs", page=2), Chunk(text="about finance", page=3)]
    vectors = {
        "about cats": [1.0, 0.0],
        "about dogs": [0.9, 0.1],
        "about finance": [0.0, 1.0],
        "tell me about pets": [1.0, 0.0],
    }
    backend = FakeEmbedBackend(vectors)
    index = EmbeddingIndex(backend, chunks, embed_model="fake")

    results = index.retrieve("tell me about pets", k=2)

    assert [c.page for c in results] == [1, 2]  # cats (identical direction) first, dogs (close) second


def test_retrieve_respects_k():
    chunks = [Chunk(text=f"chunk {i}", page=i) for i in range(5)]
    vectors = {f"chunk {i}": [float(i + 1), 1.0] for i in range(5)}
    vectors["query"] = [3.0, 1.0]
    backend = FakeEmbedBackend(vectors)
    index = EmbeddingIndex(backend, chunks, embed_model="fake")

    results = index.retrieve("query", k=2)

    assert len(results) == 2


def test_embed_called_once_per_chunk_at_index_build_time():
    chunks = [Chunk(text="a", page=1), Chunk(text="b", page=2)]
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    backend = FakeEmbedBackend(vectors)

    EmbeddingIndex(backend, chunks, embed_model="fake")

    assert len(backend.calls) == 1  # one batched call for all chunks
    assert backend.calls[0]["texts"] == ["a", "b"]


def test_retrieve_embeds_query_separately():
    chunks = [Chunk(text="a", page=1)]
    vectors = {"a": [1.0, 0.0], "my query": [1.0, 0.0]}
    backend = FakeEmbedBackend(vectors)
    index = EmbeddingIndex(backend, chunks, embed_model="fake")

    index.retrieve("my query", k=1)

    assert backend.calls[-1]["texts"] == ["my query"]


def test_default_prefixes_are_empty_when_not_set():
    chunks = [Chunk(text="a", page=1)]
    vectors = {"a": [1.0, 0.0]}
    backend = FakeEmbedBackend(vectors)

    EmbeddingIndex(backend, chunks, embed_model="fake")

    assert backend.calls[0]["texts"] == ["a"]  # no prefix prepended


def test_document_prefix_is_prepended_at_index_build_time():
    chunks = [Chunk(text="a", page=1), Chunk(text="b", page=2)]
    vectors = {"DOC: a": [1.0, 0.0], "DOC: b": [0.0, 1.0]}
    backend = FakeEmbedBackend(vectors)

    EmbeddingIndex(backend, chunks, embed_model="fake", document_prefix="DOC: ")

    assert backend.calls[0]["texts"] == ["DOC: a", "DOC: b"]


def test_query_prefix_is_prepended_at_retrieval_time():
    chunks = [Chunk(text="a", page=1)]
    vectors = {"a": [1.0, 0.0], "Q: my query": [1.0, 0.0]}
    backend = FakeEmbedBackend(vectors)
    index = EmbeddingIndex(backend, chunks, embed_model="fake", query_prefix="Q: ")

    index.retrieve("my query", k=1)

    assert backend.calls[-1]["texts"] == ["Q: my query"]


def test_default_prefixes_for_nomic_embed_text():
    doc_prefix, query_prefix = default_prefixes_for("nomic-embed-text")
    assert doc_prefix == "search_document: "
    assert query_prefix == "search_query: "


def test_default_prefixes_for_nomic_embed_text_with_tag_suffix():
    # Ollama model tags often carry a suffix, e.g. "nomic-embed-text:latest"
    doc_prefix, query_prefix = default_prefixes_for("nomic-embed-text:latest")
    assert doc_prefix == "search_document: "
    assert query_prefix == "search_query: "


def test_default_prefixes_for_unknown_model_are_empty():
    doc_prefix, query_prefix = default_prefixes_for("some-other-embed-model")
    assert doc_prefix == ""
    assert query_prefix == ""
