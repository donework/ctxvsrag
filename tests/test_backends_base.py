from ctxvsrag.backends.base import ChatResult


def test_precise_timing_true_when_eval_s_present():
    r = ChatResult(text="x", prompt_tokens=1, output_tokens=1, total_duration_s=1.0, eval_s=0.5)
    assert r.precise_timing is True


def test_precise_timing_false_when_eval_s_missing():
    r = ChatResult(text="x", prompt_tokens=1, output_tokens=1, total_duration_s=1.0)
    assert r.precise_timing is False


def test_tokens_per_s_uses_eval_s_when_available():
    r = ChatResult(text="x", prompt_tokens=1, output_tokens=100, total_duration_s=10.0, eval_s=5.0)
    assert r.tokens_per_s == 20.0


def test_tokens_per_s_falls_back_to_total_duration_when_eval_s_missing():
    r = ChatResult(text="x", prompt_tokens=1, output_tokens=100, total_duration_s=10.0)
    assert r.tokens_per_s == 10.0


def test_tokens_per_s_is_zero_for_zero_duration():
    r = ChatResult(text="x", prompt_tokens=1, output_tokens=100, total_duration_s=0.0)
    assert r.tokens_per_s == 0.0
