"""Embedding-based retrieval index (dense/semantic similarity, cosine distance)
over document chunks, backed by any EmbedBackend (Ollama or an OpenAI-
compatible embeddings endpoint).

Some embedding models require a task-instruction prefix on the input text for
correct retrieval - notably Nomic's embedding models, whose model card states
the prefix "must" be included (trained on "search_document: "/"search_query: "
-prefixed text; without it, embeddings are still produced but retrieval
quality degrades). Ollama does not add this automatically - its template for
nomic-embed-text is just the raw prompt - so it has to happen here. Other
embedding model families use different (or no) prefix conventions, so this
is opt-in per model via default_prefixes_for(), not applied unconditionally.
"""

import numpy as np

from .backends.base import EmbedBackend
from .chunking import Chunk

# embed_model prefix -> (document_prefix, query_prefix). Matched against the
# start of --embed-model, since Ollama tags often carry a suffix (e.g. ":latest").
KNOWN_PREFIX_CONVENTIONS: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


def default_prefixes_for(embed_model: str) -> tuple[str, str]:
    for name, prefixes in KNOWN_PREFIX_CONVENTIONS.items():
        if embed_model.startswith(name):
            return prefixes
    return ("", "")


class EmbeddingIndex:
    def __init__(
        self,
        backend: EmbedBackend,
        chunks: list[Chunk],
        embed_model: str,
        document_prefix: str = "",
        query_prefix: str = "",
    ):
        self.backend = backend
        self.chunks = chunks
        self.embed_model = embed_model
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

        embeddings = backend.embed(embed_model, [document_prefix + c.text for c in chunks])
        matrix = np.array(embeddings, dtype=np.float32)
        self.matrix_normed = _normalize_rows(matrix)

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        embeddings = self.backend.embed(self.embed_model, [self.query_prefix + query])
        query_vec = _normalize_rows(np.array(embeddings, dtype=np.float32))[0]

        scores = self.matrix_normed @ query_vec
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_indices]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return matrix / norms
