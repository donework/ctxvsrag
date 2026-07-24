"""Benchmark: full-context-in-prompt vs. RAG for answering questions over a document.

Supports Ollama natively (with precise prompt/generation timing) as well as
any OpenAI-compatible server (vLLM, LiteLLM, Open WebUI, Ollama's own /v1
endpoint) - wall-clock timing only there, see backends/openai_compat_backend.py.

Usage:
    ctxvsrag --pdf doc.pdf --model qwen3.5:9b --num-ctx 32768

    # against a vLLM server (embeddings still via local Ollama)
    ctxvsrag --pdf doc.pdf \\
        --chat-backend openai --chat-host http://localhost:8000/v1 --model my-model \\
        --embed-backend ollama --embed-model nomic-embed-text
"""

import argparse
import json
import statistics
import sys
from dataclasses import asdict

from .backends import make_chat_backend, make_embed_backend
from .approaches.full_context import answer_full_context
from .approaches.rag_answer import answer_rag
from .chunking import chunk_pages
from .judge import Judge, JudgeParseError
from .pdf_utils import estimate_tokens, extract_pages
from .rag_index import EmbeddingIndex, default_prefixes_for
from .report import save_pdf_report

DEFAULT_QUESTIONS = [
    "Summarize the document's key points in 5 bullet points.",
    "What are the central conclusions or recommendations in the document?",
    "Which numbers, data, or facts are mentioned or emphasized most often in the document?",
    "Does the document mention any risks, open questions, or limitations? Which ones?",
    "Summarize the document in a single paragraph understandable to someone with no prior context.",
]

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_NUM_CTX = 8192


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-Context vs. RAG Benchmark")
    parser.add_argument("--pdf", required=True, help="Path to the PDF document")

    parser.add_argument("--chat-backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--chat-host", help="Server URL. Required for --chat-backend=openai (e.g. http://localhost:8000/v1 for vLLM); optional for --chat-backend=ollama to point at a remote Ollama instance instead of localhost:11434")
    parser.add_argument("--chat-api-key", default="not-needed", help="Only relevant for --chat-backend=openai")

    parser.add_argument("--embed-backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--embed-host", help="Server URL. Required for --embed-backend=openai; optional for --embed-backend=ollama to point at a remote Ollama instance instead of localhost:11434")
    parser.add_argument("--embed-api-key", default="not-needed")

    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Chat model (default: {DEFAULT_MODEL})")
    parser.add_argument("--judge-model", help="Default: same model as --model")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--embed-document-prefix", help="Prefix prepended to each chunk before embedding. Default: auto-detected from --embed-model (e.g. Nomic's 'search_document: ')")
    parser.add_argument("--embed-query-prefix", help="Prefix prepended to the question before embedding. Default: auto-detected from --embed-model (e.g. Nomic's 'search_query: ')")

    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Only effective with --chat-backend=ollama")
    parser.add_argument("--judge-num-ctx", type=int, help="Starting/minimum context for the judge (default: --num-ctx + 4096). Grows automatically per question based on measured answer length - this just sets the floor")
    parser.add_argument("--k", type=int, default=5, help="Number of RAG chunks retrieved per question")
    parser.add_argument("--chunk-words", type=int, default=300)
    parser.add_argument("--questions", help="Optional text file, one question per line")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--report-output", default="report.pdf", help="PDF report (summary, charts, per-question detail) for sharing with non-technical colleagues. Set to an empty string to skip")
    parser.add_argument("--no-judge", action="store_true", help="Skip the quality comparison")
    parser.add_argument("--excerpt-length", type=int, default=200, help="Characters of each answer to print per question (0 to disable)")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    questions = DEFAULT_QUESTIONS
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]

    print(f"Loading PDF: {args.pdf}")
    pages = extract_pages(args.pdf)
    full_text = "\n\n".join(pages)
    print(f"{len(pages)} pages, {len(full_text)} characters extracted.")

    # The judge gets a page-annotated copy (same "[Page N]" labels RAG's context
    # uses) instead of the plain full_text used for full-context answers - so it
    # can actually verify page citations a RAG answer makes, rather than having
    # no way to check them and treating any page reference as unverifiable.
    judge_document_text = "\n\n".join(f"[Page {i}]\n{page}" for i, page in enumerate(pages, start=1))

    if len(full_text) < len(pages) * 200:
        print(
            "WARNING: Very little extracted text per page - the PDF might be "
            "scanned/image-based. RAG retrieval will barely work in that case."
        )

    needed_tokens = estimate_tokens(full_text) + 1000  # + buffer for system prompt, question, answer
    print(f"Estimated token need for full-context: ~{needed_tokens} (rough estimate, chars/3)")

    # Starting floor for the judge's context - Judge grows this per question based on
    # the actually measured answer lengths (see judge.py), since a fixed buffer can't
    # account for how much longer some questions' answers are than others'.
    judge_num_ctx = args.judge_num_ctx or (args.num_ctx + 4096)

    if args.chat_backend == "ollama":
        if needed_tokens > args.num_ctx:
            print(
                f"WARNING: --num-ctx={args.num_ctx} is probably too small for the whole document. "
                f"Ollama will then truncate the context. Try --num-ctx {2 ** (needed_tokens - 1).bit_length()} or higher."
            )
    else:
        print(
            "Note: --num-ctx/--judge-num-ctx only affect --chat-backend=ollama. On OpenAI-compatible "
            "servers the context window is configured server-side (e.g. vLLM's "
            "--max-model-len) - make sure it covers at least the estimates above."
        )

    chunks = chunk_pages(pages, chunk_words=args.chunk_words)
    print(f"{len(chunks)} chunks created for RAG.")

    chat_backend = make_chat_backend(args.chat_backend, args.chat_host, args.chat_api_key)
    embed_backend = make_embed_backend(args.embed_backend, args.embed_host, args.embed_api_key)

    default_doc_prefix, default_query_prefix = default_prefixes_for(args.embed_model)
    document_prefix = args.embed_document_prefix if args.embed_document_prefix is not None else default_doc_prefix
    query_prefix = args.embed_query_prefix if args.embed_query_prefix is not None else default_query_prefix
    if document_prefix or query_prefix:
        print(f"Using embedding prefixes for '{args.embed_model}': document={document_prefix!r} query={query_prefix!r}")

    print(f"Embedding {len(chunks)} chunks with '{args.embed_model}' ({args.embed_backend})...")
    index = EmbeddingIndex(
        embed_backend, chunks, embed_model=args.embed_model,
        document_prefix=document_prefix, query_prefix=query_prefix,
    )

    judge = None if args.no_judge else Judge(chat_backend, args.judge_model or args.model, judge_document_text, judge_num_ctx)

    full_context_results = []
    rag_results = []
    judge_results = []

    for i, question in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] {question}")

        fc = answer_full_context(chat_backend, args.model, full_text, question, num_ctx=args.num_ctx)
        print(f"  Full-context: {_format_result(fc.result)}")
        if args.excerpt_length > 0:
            print(f"    → {_excerpt(fc.answer, args.excerpt_length)}")
        full_context_results.append(fc)

        rag = answer_rag(chat_backend, args.model, index, question, num_ctx=args.num_ctx, k=args.k)
        total_rag_s = rag.result.total_duration_s + rag.retrieval_s
        print(f"  RAG:          {_format_result(rag.result, extra=f'retrieval {rag.retrieval_s * 1000:.0f}ms')} | total={total_rag_s:.2f}s")
        if args.excerpt_length > 0:
            print(f"    → {_excerpt(rag.answer, args.excerpt_length)}")
        rag_results.append(rag)

        if judge is not None:
            try:
                jr = judge.judge_pair(
                    question, fc.answer, rag.answer,
                    fc_output_tokens=fc.result.output_tokens,
                    rag_output_tokens=rag.result.output_tokens,
                )
                judge_results.append(jr)
                print(f"  Judge: preferred={jr.preferred} | FC={jr.full_context_scores} | RAG={jr.rag_scores}")
            except JudgeParseError as e:
                print(f"  Judge: FAILED ({e}) - skipping quality comparison for this question. Consider raising --judge-num-ctx.")

    summary_lines = build_summary_lines(full_context_results, rag_results, judge_results)
    print()
    for line in summary_lines:
        print(line)

    save_results(args.output, full_context_results, rag_results, judge_results)

    if args.report_output:
        save_pdf_report(args.report_output, full_context_results, rag_results, judge_results, summary_lines)
        print(f"Report saved: {args.report_output}")

    return 0


def _excerpt(text: str, length: int) -> str:
    flat = " ".join(text.split())  # collapse newlines/whitespace for single-line display
    if len(flat) <= length:
        return flat
    return flat[:length].rstrip() + "..."


def _format_result(result, extra: str = "") -> str:
    parts = [f"{result.total_duration_s:.2f}s"]
    if extra:
        parts.append(f"({extra})")
    if result.time_to_first_token_s is not None:
        parts.append(f"first token={result.time_to_first_token_s:.2f}s")
    if result.precise_timing:
        parts.append(f"prompt={result.prompt_tokens}tok ({result.prompt_eval_s:.2f}s)")
        parts.append(f"out={result.output_tokens}tok ({result.tokens_per_s:.1f} tok/s)")
    else:
        parts.append(f"prompt={result.prompt_tokens}tok out={result.output_tokens}tok (~{result.tokens_per_s:.1f} tok/s estimated)")
    return " | ".join(parts)


def build_summary_lines(fc_results, rag_results, judge_results) -> list[str]:
    """The narrative summary, as a list of lines - used both for the CLI
    printout and as the summary section of the PDF report, so
    there's exactly one place that composes this text."""
    lines = ["SUMMARY"]

    fc_latencies = [r.result.total_duration_s for r in fc_results]
    rag_latencies = [r.result.total_duration_s + r.retrieval_s for r in rag_results]
    fc_tok_s = [r.result.tokens_per_s for r in fc_results if r.result.tokens_per_s > 0]
    rag_tok_s = [r.result.tokens_per_s for r in rag_results if r.result.tokens_per_s > 0]
    precise = fc_results[0].result.precise_timing if fc_results else False

    # Wall-clock time until the first token, incl. retrieval for RAG (that
    # happens before the model call, so it's part of what the user waits
    # through) - this is the number that most directly shows whether feeding
    # fewer input tokens (RAG) actually gets a user their first token sooner
    # than stuffing the whole document in (full-context).
    fc_ttfts = [r.result.time_to_first_token_s for r in fc_results if r.result.time_to_first_token_s is not None]
    rag_ttfts = [
        r.result.time_to_first_token_s + r.retrieval_s
        for r in rag_results if r.result.time_to_first_token_s is not None
    ]

    lines.append("")
    lines.append(f"Full-context ({len(fc_results)} questions):")
    lines.append(f"  Total latency: avg={statistics.mean(fc_latencies):.2f}s min={min(fc_latencies):.2f}s max={max(fc_latencies):.2f}s")
    if fc_ttfts:
        lines.append(f"  Time to first token: avg={statistics.mean(fc_ttfts):.2f}s min={min(fc_ttfts):.2f}s max={max(fc_ttfts):.2f}s")
    if precise:
        fc_prompt_eval = [r.result.prompt_eval_s for r in fc_results]
        lines.append(f"  Prompt processing: avg={statistics.mean(fc_prompt_eval):.2f}s (dominates with long context)")
    lines.append(f"  Generation speed: avg={statistics.mean(fc_tok_s):.1f} tok/s" + ("" if precise else " (estimated)") if fc_tok_s else "  Generation speed: n/a")

    lines.append("")
    lines.append(f"RAG ({len(rag_results)} questions):")
    lines.append(f"  Total latency: avg={statistics.mean(rag_latencies):.2f}s min={min(rag_latencies):.2f}s max={max(rag_latencies):.2f}s")
    if rag_ttfts:
        lines.append(f"  Time to first token: avg={statistics.mean(rag_ttfts):.2f}s min={min(rag_ttfts):.2f}s max={max(rag_ttfts):.2f}s (incl. retrieval)")
    if precise:
        rag_prompt_eval = [r.result.prompt_eval_s for r in rag_results]
        lines.append(f"  Prompt processing: avg={statistics.mean(rag_prompt_eval):.2f}s (only the retrieved chunks)")
    lines.append(f"  Generation speed: avg={statistics.mean(rag_tok_s):.1f} tok/s" + ("" if precise else " (estimated)") if rag_tok_s else "  Generation speed: n/a")

    if fc_ttfts and rag_ttfts:
        lines.append("")
        ratio = statistics.mean(fc_ttfts) / statistics.mean(rag_ttfts) if statistics.mean(rag_ttfts) > 0 else None
        lines.append(f"Time-to-first-token ratio (Full-context / RAG): {ratio:.2f}x" if ratio else "")

    if judge_results:
        prefs = [j.preferred for j in judge_results]
        lines.append("")
        lines.append(f"Quality (LLM judge, {len(judge_results)} comparisons):")
        lines.append(
            f"  Full-context preferred: {prefs.count('full_context')}, "
            f"RAG preferred: {prefs.count('rag')}, Tie: {prefs.count('tie')}"
        )
        fc_acc = statistics.mean(j.full_context_scores["accuracy"] for j in judge_results)
        rag_acc = statistics.mean(j.rag_scores["accuracy"] for j in judge_results)
        fc_comp = statistics.mean(j.full_context_scores["completeness"] for j in judge_results)
        rag_comp = statistics.mean(j.rag_scores["completeness"] for j in judge_results)
        fc_clar = statistics.mean(j.full_context_scores["clarity"] for j in judge_results)
        rag_clar = statistics.mean(j.rag_scores["clarity"] for j in judge_results)
        lines.append(f"  Avg. accuracy:     Full-context={fc_acc:.1f}  RAG={rag_acc:.1f}")
        lines.append(f"  Avg. completeness: Full-context={fc_comp:.1f}  RAG={rag_comp:.1f}")
        lines.append(f"  Avg. clarity:      Full-context={fc_clar:.1f}  RAG={rag_clar:.1f}")

        # Derived from the measured numbers above, not a judge criterion: the judge never
        # sees timing, this is purely (avg. quality score) / (avg. latency), computed after
        # the fact so you can weigh the quality/speed trade-off yourself.
        fc_quality = (fc_acc + fc_comp + fc_clar) / 3
        rag_quality = (rag_acc + rag_comp + rag_clar) / 3
        fc_efficiency = fc_quality / statistics.mean(fc_latencies)
        rag_efficiency = rag_quality / statistics.mean(rag_latencies)
        lines.append("")
        lines.append("Efficiency (avg. quality score ÷ avg. latency — informational only, not part of the judge):")
        lines.append(f"  Full-context: {fc_quality:.1f}/10 in {statistics.mean(fc_latencies):.1f}s → {fc_efficiency:.3f} quality/s")
        lines.append(f"  RAG:          {rag_quality:.1f}/10 in {statistics.mean(rag_latencies):.1f}s → {rag_efficiency:.3f} quality/s")
        if fc_efficiency > 0 and rag_efficiency > 0:
            better, ratio = ("RAG", rag_efficiency / fc_efficiency) if rag_efficiency > fc_efficiency else ("Full-context", fc_efficiency / rag_efficiency)
            lines.append(f"  → {better} delivers ~{ratio:.1f}x more quality per second")

    lines.append("")
    lines.append(f"Speed ratio (Full-context / RAG): {statistics.mean(fc_latencies) / statistics.mean(rag_latencies):.2f}x")
    return lines


def save_results(path, fc_results, rag_results, judge_results):
    data = {
        "full_context": [asdict(r) for r in fc_results],
        "rag": [asdict(r) for r in rag_results],
        "judge": [asdict(r) for r in judge_results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {path}")


if __name__ == "__main__":
    sys.exit(main())
