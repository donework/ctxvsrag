import json

import pytest

from conftest import default_chat_result
from ctxvsrag.backends.base import ChatResult
from ctxvsrag.judge import Judge, JudgeParseError


def _score_response(**overrides) -> ChatResult:
    data = {
        "accuracy_a": 8, "accuracy_b": 6,
        "completeness_a": 9, "completeness_b": 7,
        "clarity_a": 8, "clarity_b": 7,
        "preferred": "A",
        "reasoning": "A is better.",
    }
    data.update(overrides)
    return default_chat_result(text=json.dumps(data))


# random.random is monkeypatched to a constant below, so both of a question's
# JUDGE_RUNS=2 runs land on the same swap decision - queuing the identical
# response for both runs means the averaged result equals a single run's,
# which keeps these assertions simple while still exercising the averaging path.


def test_no_swap_maps_a_to_full_context(fake_chat_backend, monkeypatch):
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)  # >= 0.5 -> no swap
    fake_chat_backend.responses = [_score_response(), _score_response()]

    judge = Judge(fake_chat_backend, "fake-model", "document text", min_num_ctx=4096)
    result = judge.judge_pair("Q?", "FC answer", "RAG answer", fc_output_tokens=10, rag_output_tokens=10)

    assert result.full_context_scores == {"accuracy": 8, "completeness": 9, "clarity": 8}
    assert result.rag_scores == {"accuracy": 6, "completeness": 7, "clarity": 7}
    assert result.preferred == "full_context"


def test_swap_unswaps_scores_back_correctly(fake_chat_backend, monkeypatch):
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.1)  # < 0.5 -> swap (A=rag, B=full_context)
    fake_chat_backend.responses = [_score_response(), _score_response()]

    judge = Judge(fake_chat_backend, "fake-model", "document text", min_num_ctx=4096)
    result = judge.judge_pair("Q?", "FC answer", "RAG answer", fc_output_tokens=10, rag_output_tokens=10)

    assert result.full_context_scores == {"accuracy": 6, "completeness": 7, "clarity": 7}
    assert result.rag_scores == {"accuracy": 8, "completeness": 9, "clarity": 8}
    assert result.preferred == "rag"


def test_scores_are_averaged_across_the_two_runs(fake_chat_backend, monkeypatch):
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)  # no swap on either run
    fake_chat_backend.responses = [_score_response(accuracy_a=8), _score_response(accuracy_a=4)]

    judge = Judge(fake_chat_backend, "fake-model", "doc", min_num_ctx=4096)
    result = judge.judge_pair("Q?", "FC", "RAG")

    assert result.full_context_scores["accuracy"] == 6  # mean(8, 4)


def test_preferred_derived_from_averaged_scores_not_either_runs_label(fake_chat_backend, monkeypatch):
    # Both runs individually say "A" (= full_context, no swap) is preferred, but
    # their actual score totals average out to a RAG win - preferred must follow
    # the averaged numbers, not either run's own preferred label.
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)
    run1 = _score_response(accuracy_a=6, completeness_a=6, clarity_a=6, accuracy_b=9, completeness_b=9, clarity_b=9, preferred="A")
    run2 = _score_response(accuracy_a=6, completeness_a=6, clarity_a=6, accuracy_b=9, completeness_b=9, clarity_b=9, preferred="A")
    fake_chat_backend.responses = [run1, run2]

    judge = Judge(fake_chat_backend, "fake-model", "doc", min_num_ctx=4096)
    result = judge.judge_pair("Q?", "FC", "RAG")

    assert result.preferred == "rag"


def test_retries_once_on_missing_fields_then_succeeds(fake_chat_backend, monkeypatch):
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)
    incomplete = default_chat_result(text=json.dumps({"accuracy_a": 5}))
    fake_chat_backend.responses = [incomplete, _score_response(), _score_response()]

    judge = Judge(fake_chat_backend, "fake-model", "doc", min_num_ctx=4096)
    result = judge.judge_pair("Q?", "FC", "RAG")

    assert result.preferred == "full_context"
    assert len(fake_chat_backend.calls) == 3  # run 1: retry + success, run 2: success


def test_raises_after_two_failed_attempts(fake_chat_backend, monkeypatch):
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)
    empty = default_chat_result(text="")
    fake_chat_backend.responses = [empty, empty]

    judge = Judge(fake_chat_backend, "fake-model", "doc", min_num_ctx=4096)
    with pytest.raises(JudgeParseError):
        judge.judge_pair("Q?", "FC", "RAG")


def test_rejects_out_of_range_score_and_retries(fake_chat_backend, monkeypatch):
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)
    out_of_range = _score_response(accuracy_a=0)  # schema says 1-10; simulates a backend not enforcing it
    fake_chat_backend.responses = [out_of_range, _score_response(), _score_response()]

    judge = Judge(fake_chat_backend, "fake-model", "doc", min_num_ctx=4096)
    result = judge.judge_pair("Q?", "FC", "RAG")

    assert result.full_context_scores["accuracy"] == 8  # from the retry's valid response, both runs agree
    assert len(fake_chat_backend.calls) == 3


def test_rejects_boolean_masquerading_as_score(fake_chat_backend, monkeypatch):
    # In Python, bool is a subclass of int, so True/False would silently pass
    # an `isinstance(x, int)` check without an explicit bool guard.
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)
    bad = _score_response(accuracy_a=True)
    fake_chat_backend.responses = [bad, _score_response(), _score_response()]

    judge = Judge(fake_chat_backend, "fake-model", "doc", min_num_ctx=4096)
    judge.judge_pair("Q?", "FC", "RAG")

    assert len(fake_chat_backend.calls) == 3  # first response was rejected


def test_rejects_unexpected_preferred_value(fake_chat_backend, monkeypatch):
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)
    bad = _score_response(preferred="C")
    fake_chat_backend.responses = [bad, _score_response(), _score_response()]

    judge = Judge(fake_chat_backend, "fake-model", "doc", min_num_ctx=4096)
    judge.judge_pair("Q?", "FC", "RAG")

    assert len(fake_chat_backend.calls) == 3


def test_context_grows_when_answers_are_long(fake_chat_backend, monkeypatch):
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)
    fake_chat_backend.responses = [_score_response(), _score_response()]

    judge = Judge(fake_chat_backend, "fake-model", "short doc", min_num_ctx=100)
    judge.judge_pair("Q?", "FC", "RAG", fc_output_tokens=5000, rag_output_tokens=5000)

    assert judge.current_num_ctx > 100
    assert fake_chat_backend.calls[0]["num_ctx"] == judge.current_num_ctx


def test_context_never_shrinks_back_down(fake_chat_backend, monkeypatch):
    monkeypatch.setattr("ctxvsrag.judge.random.random", lambda: 0.9)
    fake_chat_backend.responses = [_score_response(), _score_response(), _score_response(), _score_response()]

    judge = Judge(fake_chat_backend, "fake-model", "short doc", min_num_ctx=100)
    judge.judge_pair("Q1", "long fc answer", "long rag answer", fc_output_tokens=5000, rag_output_tokens=5000)
    grown_ctx = judge.current_num_ctx

    judge.judge_pair("Q2", "short", "short", fc_output_tokens=1, rag_output_tokens=1)

    assert judge.current_num_ctx == grown_ctx
