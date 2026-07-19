"""LLM-as-judge: blind, order-randomized comparison of full-context vs. RAG answers
against the source document text.

The document text passed in is expected to carry the same "[Page N]" labels
RAG's own context does (see cli.py, which builds this from the page list
rather than the plain joined text used for full-context answers) - otherwise
the judge has no way to verify a page citation and may treat any page
reference as unsupported, even a correct one.

JSON enforcement strength depends on the backend: Ollama constrains decoding
to the exact schema natively; OpenAI-compatible backends only guarantee valid
JSON (not the exact shape). Even schema-constrained decoding isn't a full
guarantee though - grammar-based enforcement (what Ollama uses under the
hood) reliably restricts structure and type, but numeric range constraints
(minimum/maximum) aren't always honored, especially when the model is under
context pressure. So a response can be syntactically valid JSON with the
right fields and still contain garbage values (e.g. a score of 0 or 100
instead of 1-10). _chat_and_parse validates ranges explicitly rather than
trusting the schema alone, and retries once if validation fails.
"""

import json
import random
from dataclasses import dataclass

from .backends.base import ChatBackend
from .pdf_utils import estimate_tokens

SCORE_FIELDS = ("accuracy_a", "accuracy_b", "completeness_a", "completeness_b", "clarity_a", "clarity_b")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy_a": {"type": "integer", "minimum": 1, "maximum": 10},
        "accuracy_b": {"type": "integer", "minimum": 1, "maximum": 10},
        "completeness_a": {"type": "integer", "minimum": 1, "maximum": 10},
        "completeness_b": {"type": "integer", "minimum": 1, "maximum": 10},
        "clarity_a": {"type": "integer", "minimum": 1, "maximum": 10},
        "clarity_b": {"type": "integer", "minimum": 1, "maximum": 10},
        "preferred": {"type": "string", "enum": ["A", "B", "tie"]},
        "reasoning": {"type": "string"},
    },
    "required": [
        "accuracy_a",
        "accuracy_b",
        "completeness_a",
        "completeness_b",
        "clarity_a",
        "clarity_b",
        "preferred",
        "reasoning",
    ],
}

REQUIRED_FIELDS = JUDGE_SCHEMA["required"]

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator. You receive the full text of a "
    "document - marked with [Page N] labels - a question, and two answers "
    "(A and B). Rate each answer on a scale of 1-10 for accuracy (no "
    "fabrications, correct relative to the document), completeness, and "
    "clarity. You don't know which system produced which answer - judge the "
    "content only. An answer may cite a page number as its source; this is "
    "normal and not suspicious by itself - check it against the [Page N] "
    "labels like any other factual claim. A citation matching the labeled "
    "source is correct and should not be penalized; a citation naming the "
    "wrong page is a real accuracy issue like any other incorrect claim. "
    "Respond only with a JSON object with exactly these fields: accuracy_a, "
    "accuracy_b, completeness_a, completeness_b, clarity_a, clarity_b (each "
    'an integer 1-10), preferred ("A", "B", or "tie"), reasoning (brief '
    "free text)."
)


class JudgeParseError(RuntimeError):
    """Raised when the judge model didn't return valid, complete JSON after retrying."""


@dataclass
class JudgeResult:
    question: str
    full_context_scores: dict
    rag_scores: dict
    preferred: str  # "full_context" | "rag" | "tie"
    reasoning: str


class Judge:
    def __init__(self, backend: ChatBackend, model: str, document_text: str, min_num_ctx: int):
        self.backend = backend
        self.model = model
        self.document_text = document_text
        self.document_tokens = estimate_tokens(document_text)
        # Grows across the run if a question's answers need more room than the
        # current size; never shrinks back down. A fixed buffer can't work here -
        # answer length varies a lot per question ("list every X" vs. "who is Y"),
        # and reused prompts get expensive at bigger sizes, so we only pay for what
        # this run's questions actually need, and Ollama only reloads the model
        # (KV cache resize) when the size genuinely has to grow, not every call.
        self.current_num_ctx = min_num_ctx

    def judge_pair(
        self,
        question: str,
        full_context_answer: str,
        rag_answer: str,
        fc_output_tokens: int | None = None,
        rag_output_tokens: int | None = None,
    ) -> JudgeResult:
        # Prefer the actually-measured output token counts from generating the
        # answers (exact) over re-estimating from character count (approximate).
        fc_tokens = fc_output_tokens if fc_output_tokens is not None else estimate_tokens(full_context_answer)
        rag_tokens = rag_output_tokens if rag_output_tokens is not None else estimate_tokens(rag_answer)
        needed = self.document_tokens + fc_tokens + rag_tokens + estimate_tokens(question) + 1500
        if needed > self.current_num_ctx:
            old_num_ctx = self.current_num_ctx
            self.current_num_ctx = 2 ** (needed - 1).bit_length()  # round up to reduce distinct sizes -> fewer reloads
            print(f"    (judge context grown {old_num_ctx} -> {self.current_num_ctx} tokens for this question's answer lengths)")

        swap = random.random() < 0.5
        answer_a, answer_b = (rag_answer, full_context_answer) if swap else (full_context_answer, rag_answer)

        prompt = (
            f"Reference document (full text):\n\n{self.document_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer A:\n{answer_a}\n\n"
            f"Answer B:\n{answer_b}\n\n"
            "Rate both answers as described in your system instructions."
        )

        result = self._chat_and_parse(prompt)

        if swap:
            fc = {"accuracy": result["accuracy_b"], "completeness": result["completeness_b"], "clarity": result["clarity_b"]}
            rag = {"accuracy": result["accuracy_a"], "completeness": result["completeness_a"], "clarity": result["clarity_a"]}
            preferred = {"A": "rag", "B": "full_context", "tie": "tie"}[result["preferred"]]
        else:
            fc = {"accuracy": result["accuracy_a"], "completeness": result["completeness_a"], "clarity": result["clarity_a"]}
            rag = {"accuracy": result["accuracy_b"], "completeness": result["completeness_b"], "clarity": result["clarity_b"]}
            preferred = {"A": "full_context", "B": "rag", "tie": "tie"}[result["preferred"]]

        return JudgeResult(
            question=question,
            full_context_scores=fc,
            rag_scores=rag,
            preferred=preferred,
            reasoning=result["reasoning"],
        )

    def _chat_and_parse(self, prompt: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(2):
            chat_result = self.backend.chat(
                model=self.model,
                system=JUDGE_SYSTEM,
                user=prompt,
                num_ctx=self.current_num_ctx,
                json_schema=JUDGE_SCHEMA,
            )
            try:
                parsed = json.loads(chat_result.text)
                missing = [f for f in REQUIRED_FIELDS if f not in parsed]
                if missing:
                    raise ValueError(f"missing fields: {missing}")
                out_of_range = [
                    (f, parsed[f]) for f in SCORE_FIELDS
                    if not isinstance(parsed[f], int) or isinstance(parsed[f], bool) or not (1 <= parsed[f] <= 10)
                ]
                if out_of_range:
                    raise ValueError(f"score(s) outside expected 1-10 integer range: {out_of_range}")
                if parsed["preferred"] not in ("A", "B", "tie"):
                    raise ValueError(f"unexpected 'preferred' value: {parsed['preferred']!r}")
                return parsed
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                last_error = e
        raise JudgeParseError(f"Judge didn't return valid JSON after 2 attempts: {last_error}")
