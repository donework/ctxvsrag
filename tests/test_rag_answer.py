from conftest import FakeEmbedBackend, default_chat_result
from ctxvsrag.approaches.rag_answer import answer_rag
from ctxvsrag.chunking import Chunk
from ctxvsrag.rag_index import EmbeddingIndex


def test_retrieves_relevant_chunks_and_answers(fake_chat_backend):
    chunks = [Chunk(text="relevant info", page=1), Chunk(text="unrelated info", page=2)]
    vectors = {"relevant info": [1.0, 0.0], "unrelated info": [0.0, 1.0], "question": [1.0, 0.0]}
    index = EmbeddingIndex(FakeEmbedBackend(vectors), chunks, embed_model="fake")

    fake_chat_backend.responses = [default_chat_result(text="The answer.")]

    result = answer_rag(fake_chat_backend, "fake-model", index, "question", num_ctx=8192, k=1)

    assert result.approach == "rag"
    assert result.answer == "The answer."
    assert result.retrieved_pages == [1]
    call = fake_chat_backend.calls[0]
    assert "relevant info" in call["user"]
    assert "unrelated info" not in call["user"]


def test_retrieval_time_is_measured(fake_chat_backend):
    chunks = [Chunk(text="a", page=1)]
    vectors = {"a": [1.0, 0.0], "q": [1.0, 0.0]}
    index = EmbeddingIndex(FakeEmbedBackend(vectors), chunks, embed_model="fake")
    fake_chat_backend.responses = [default_chat_result()]

    result = answer_rag(fake_chat_backend, "fake-model", index, "q", num_ctx=8192, k=1)

    assert result.retrieval_s >= 0
