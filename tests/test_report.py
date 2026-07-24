from pypdf import PdfReader

from ctxvsrag.approaches.full_context import ApiResult
from ctxvsrag.approaches.rag_answer import RagResult
from ctxvsrag.backends.base import ChatResult
from ctxvsrag.judge import JudgeResult
from ctxvsrag.report import save_pdf_report


def _chat_result(**overrides) -> ChatResult:
    fields = dict(text="answer", prompt_tokens=100, output_tokens=20, total_duration_s=1.0)
    fields.update(overrides)
    return ChatResult(**fields)


def _fc_result(question: str, answer: str = "FC answer") -> ApiResult:
    return ApiResult(approach="full_context", question=question, answer=answer, result=_chat_result(text=answer))


def _rag_result(question: str, answer: str = "RAG answer") -> RagResult:
    return RagResult(
        approach="rag", question=question, answer=answer,
        retrieval_s=0.05, retrieved_pages=[1, 2], result=_chat_result(text=answer),
    )


def _judge_result(question: str, preferred: str = "full_context") -> JudgeResult:
    return JudgeResult(
        question=question,
        full_context_scores={"accuracy": 9, "completeness": 8, "clarity": 9},
        rag_scores={"accuracy": 7, "completeness": 6, "clarity": 7},
        preferred=preferred,
        reasoning="Full-context covered more detail.",
    )


def _pdf_text(path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def test_creates_a_valid_pdf(tmp_path):
    path = tmp_path / "report.pdf"
    save_pdf_report(str(path), [_fc_result("Q1?")], [_rag_result("Q1?")], [_judge_result("Q1?")], ["line one"])

    reader = PdfReader(str(path))
    assert len(reader.pages) >= 1


def test_pdf_contains_question_and_both_answers(tmp_path):
    path = tmp_path / "report.pdf"
    save_pdf_report(
        str(path), [_fc_result("What is the revenue?", "FC answer text")],
        [_rag_result("What is the revenue?", "RAG answer text")],
        [_judge_result("What is the revenue?")], ["line one"],
    )

    text = _pdf_text(path)
    assert "What is the revenue?" in text
    assert "FC answer text" in text
    assert "RAG answer text" in text


def test_pdf_contains_judge_reasoning_and_preferred(tmp_path):
    path = tmp_path / "report.pdf"
    save_pdf_report(str(path), [_fc_result("Q1?")], [_rag_result("Q1?")], [_judge_result("Q1?")], ["line one"])

    text = _pdf_text(path)
    assert "Full-context covered more detail." in text
    assert "Won" in text
    assert "Lost" in text


def test_missing_judge_result_shows_na(tmp_path):
    path = tmp_path / "report.pdf"
    # No JudgeResult for this question - simulates a JudgeParseError that got skipped.
    save_pdf_report(str(path), [_fc_result("Q1?")], [_rag_result("Q1?")], [], ["line one"])

    text = _pdf_text(path)
    assert "N/A (judge failed)" in text


def test_summary_lines_appear_in_pdf(tmp_path):
    path = tmp_path / "report.pdf"
    lines = ["Full-context (7 questions):", "  Total latency: avg=1.23s"]
    save_pdf_report(str(path), [_fc_result("Q1?")], [_rag_result("Q1?")], [_judge_result("Q1?")], lines)

    text = _pdf_text(path)
    assert "Full-context (7 questions):" in text
    assert "Total latency: avg=1.23s" in text


def test_multiple_questions_all_appear(tmp_path):
    path = tmp_path / "report.pdf"
    fc = [_fc_result("Q1?"), _fc_result("Q2?")]
    rag = [_rag_result("Q1?"), _rag_result("Q2?")]
    jr = [_judge_result("Q1?"), _judge_result("Q2?", preferred="rag")]
    save_pdf_report(str(path), fc, rag, jr, ["line one"])

    text = _pdf_text(path)
    assert "Q1?" in text
    assert "Q2?" in text


def test_answer_text_with_special_characters_does_not_break_generation(tmp_path):
    path = tmp_path / "report.pdf"
    tricky = "Revenue < 5% & rising > expectations \"quoted\""
    save_pdf_report(
        str(path), [_fc_result("Q1?", tricky)], [_rag_result("Q1?")], [_judge_result("Q1?")], ["line one"],
    )

    text = _pdf_text(path)
    assert "Revenue" in text and "rising" in text and "expectations" in text


def test_markdown_bold_and_bullets_are_rendered_not_literal(tmp_path):
    path = tmp_path / "report.pdf"
    markdown_answer = "**Revenue** grew:\n- Segment A: 10%\n- Segment B: 5%"
    save_pdf_report(
        str(path), [_fc_result("Q1?", markdown_answer)], [_rag_result("Q1?")], [_judge_result("Q1?")], ["line one"],
    )

    text = _pdf_text(path)
    assert "Revenue" in text
    assert "**" not in text
    assert "- Segment A: 10%" in text


def test_footer_shows_generated_with_ctxvsrag(tmp_path):
    path = tmp_path / "report.pdf"
    save_pdf_report(str(path), [_fc_result("Q1?")], [_rag_result("Q1?")], [_judge_result("Q1?")], ["line one"])

    text = _pdf_text(path)
    assert "Generated with ctxvsrag" in text
    assert "pypi.org/project/ctxvsrag" in text
