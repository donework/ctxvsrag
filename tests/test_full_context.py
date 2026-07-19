from conftest import default_chat_result
from ctxvsrag.approaches.full_context import answer_full_context


def test_returns_answer_and_result(fake_chat_backend):
    fake_chat_backend.responses = [default_chat_result(text="The answer is 42.", total_duration_s=1.5)]

    result = answer_full_context(fake_chat_backend, "fake-model", "document text", "What is the answer?", num_ctx=8192)

    assert result.approach == "full_context"
    assert result.answer == "The answer is 42."
    assert result.result.total_duration_s == 1.5


def test_passes_num_ctx_document_and_question_to_backend(fake_chat_backend):
    fake_chat_backend.responses = [default_chat_result()]

    answer_full_context(fake_chat_backend, "fake-model", "MY DOCUMENT CONTENT", "my question", num_ctx=1234)

    call = fake_chat_backend.calls[0]
    assert call["num_ctx"] == 1234
    assert call["model"] == "fake-model"
    assert "MY DOCUMENT CONTENT" in call["user"]
    assert "my question" in call["user"]
    assert call["json_schema"] is None
