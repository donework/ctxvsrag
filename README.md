# ctxvsrag

Compares two ways of answering questions over a long PDF with a local/self-hosted LLM:

- **Full-context** — dump the entire extracted PDF text into the prompt on every request.
- **RAG** — chunk the PDF, embed the chunks, retrieve the top-k most similar chunks per question via cosine similarity, and only send those.

For each question it measures latency, generation speed, and (via an LLM-as-judge pass) answer quality — so you can see the actual speed/quality trade-off on your own hardware and documents, not just take it on faith.

## Supported backends

| Backend | Chat | Embeddings | Notes |
|---|---|---|---|
| **Ollama** | ✅ native | ✅ native | The reference backend — reports precise prompt-processing vs. generation timing per call, which nothing else here does |
| **vLLM** | ✅ OpenAI-compatible | ✅ OpenAI-compatible (if serving an embedding model) | Usually one model per server instance — you'll likely point chat and embeddings at two different vLLM instances/ports |
| **LiteLLM** | ✅ OpenAI-compatible | ✅ OpenAI-compatible | Works both as the LiteLLM proxy server and via the SDK's own OpenAI-compatible surface |
| **Open WebUI** | ✅ OpenAI-compatible | usually not — it's a UI, not an embedding provider | Point `--embed-backend` elsewhere (e.g. straight at Ollama) if Open WebUI doesn't expose one |

Chat and embeddings are configured **independently** (`--chat-*` / `--embed-*` flags) precisely because that split — different server, different model, sometimes different backend entirely — is normal once you're not on Ollama alone.

## Requirements

- Python 3.10+
- At least one running backend from the table above.
  - **Ollama** (simplest default): `ollama serve`, then pull a chat model and an embedding model:
    ```bash
    ollama pull qwen3.5:9b
    ollama pull nomic-embed-text
    ```
    Prefer a chat model with a large context window if you plan to test full-context on long documents — check with `ollama show <model>`.
  - **vLLM / LiteLLM / Open WebUI**: have the server(s) running and know their base URL(s) (e.g. `http://localhost:8000/v1`) and model name(s).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs the package in editable mode plus dev dependencies (pytest), and puts a `ctxvsrag` command on your PATH. Drop `[dev]` for a plain runtime install (`pip install -e .`).

## Quick start

`examples/example.pdf` is a synthetic ~21-page annual report checked into this repo, with a
matching `examples/example_questions.txt` — enough to see the mechanics and a real (if modest)
full-context/RAG gap before pointing the tool at your own, much larger document. Everything in it
is a deliberate placeholder — company is "ACME Corporation", people are John Doe / Jane Doe /
Richard Roe, every figure follows an obviously-sequential pattern (12,345 / 23,456 / ...) —
specifically so nobody mistakes it for a real company's real report. The disclaimer is also
printed on the PDF's own title page.

```bash
ctxvsrag --pdf examples/example.pdf --questions examples/example_questions.txt
```

Fits comfortably under the default `--num-ctx 8192`, so it runs out of the box with no tuning.
It's still 5–10x smaller than the 100-page case the trade-off is really about — treat it as a
smoke test, not as the answer to "which one is better."

### The 100-page example

`examples/example_large.pdf` (with `examples/example_large_questions.txt`) is the same fictional ACME Corporation,
same placeholder conventions, but at the scale the full-context/RAG trade-off actually matters:
100 pages, ~76,000 characters, ~25,000 estimated tokens — structurally varied rather than padded
(12 monthly updates, 6 segments × 3 pages each, 10 offices × 2 pages each, 12 customer case
studies, 8 risk factors × 2 pages each, plus governance/compensation/environmental detail
sections), so RAG retrieval has genuinely different chunks to discriminate between and
full-context has real "lost in the middle" pressure to contend with.

This one does **not** fit under the default context window — size `--num-ctx` accordingly (the
tool will estimate and warn if you don't):

```bash
ctxvsrag --pdf examples/example_large.pdf --questions examples/example_large_questions.txt --num-ctx 32768
```

The bundled questions include a few designed to stress-test each approach differently: one asks
to list *every* risk factor with its mitigation (scattered across 16 pages — a completeness test
for both approaches), one asks about a single specific month (a precision-retrieval test for
RAG), and one asks for detail on one specific office by name (tests whether RAG's top-k actually
surfaces the right chunk among ten near-identical office reports).

## Usage

### Ollama only (default)

```bash
ctxvsrag --pdf your_document.pdf --model qwen3.5:9b --num-ctx 32768
```

### vLLM for chat, Ollama for embeddings

```bash
ctxvsrag --pdf your_document.pdf \
  --chat-backend openai --chat-host http://localhost:8000/v1 --model my-model \
  --embed-backend ollama --embed-model nomic-embed-text
```

### Both through LiteLLM / Open WebUI

```bash
ctxvsrag --pdf your_document.pdf \
  --chat-backend openai --chat-host http://localhost:4000 --chat-api-key sk-litellm-... --model gpt-4o-mini \
  --embed-backend openai --embed-host http://localhost:4000 --embed-api-key sk-litellm-... --embed-model text-embedding-3-small
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--pdf` | *(required)* | Path to the PDF to benchmark |
| `--chat-backend` | `ollama` | `ollama` or `openai` (any OpenAI-compatible server) |
| `--chat-host` | `localhost:11434` | Required for `--chat-backend=openai` (e.g. `http://localhost:8000/v1`); also works with `ollama` to point at a remote Ollama instance |
| `--chat-api-key` | `not-needed` | Only relevant for `--chat-backend=openai` |
| `--embed-backend` | `ollama` | `ollama` or `openai` |
| `--embed-host` | `localhost:11434` | Required for `--embed-backend=openai`; also works with `ollama` to point at a remote Ollama instance |
| `--embed-api-key` | `not-needed` | Only relevant for `--embed-backend=openai` |
| `--model` | `qwen3.5:9b` | Chat model used for both approaches |
| `--judge-model` | same as `--model` | Model used to score answer quality (uses `--chat-backend`) |
| `--embed-model` | `nomic-embed-text` | Embedding model used for RAG retrieval (uses `--embed-backend`) |
| `--embed-document-prefix` | auto-detected | Prefix prepended to each chunk before embedding — see below |
| `--embed-query-prefix` | auto-detected | Prefix prepended to the question before embedding — see below |
| `--num-ctx` | `8192` | Context window in tokens. **Only applies to `--chat-backend=ollama`** — see below |
| `--judge-num-ctx` | `--num-ctx + 4096` | Starting/minimum context for the judge — grows automatically per question, see below |
| `--k` | `5` | Number of chunks retrieved per question in the RAG path |
| `--chunk-words` | `300` | Words per RAG chunk (with overlap) |
| `--questions` | *(built-in defaults)* | Text file, one question per line |
| `--output` | `results.json` | Where to write raw results |
| `--xlsx-output` | `report.xlsx` | Two-sheet Excel report for non-technical colleagues — see below. Empty string to skip |
| `--no-judge` | off | Skip the quality comparison (faster/cheaper iteration) |
| `--excerpt-length` | `200` | Characters of each answer printed per question (`0` to disable) |

### Sizing the context window

Ollama does **not** automatically use a model's max supported context — it defaults to a small window and silently truncates whatever doesn't fit, controlled per-request via `--num-ctx`. The script estimates the token count needed for your PDF (`chars / 3`, a rough heuristic) and warns if `--num-ctx` is too small.

On OpenAI-compatible backends there's no per-request equivalent — the context window is a **server-side setting** (e.g. vLLM's `--max-model-len` at launch). `--num-ctx` is ignored there; the script just prints the same token estimate so you can check it against your server's configured limit yourself.

A 100-page PDF typically needs somewhere around 30,000–50,000 tokens of context either way — that's real RAM/VRAM, not just a config flag.

**The judge needs more room than a single answer** — its prompt holds the document plus *both* the full-context and RAG answers, and answer length varies a lot per question ("list every X" produces much longer answers than "who is Y"). `--judge-num-ctx` only sets a starting floor; the judge measures each answer's actual token count and grows its own context automatically when a question needs more, printing `(judge context grown X -> Y tokens ...)` when it does. It never shrinks back down within a run, to avoid repeated Ollama model reloads.

### Embedding prefixes

Some embedding models are trained to expect a task-instruction prefix on the input text - notably [Nomic's models](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), whose model card states the prefix *must* be included for correct retrieval: `"search_document: "` on chunks at index time, `"search_query: "` on the question at retrieval time. Ollama doesn't add this automatically (its template for `nomic-embed-text` is just the raw prompt), and it's a per-model convention - other embedding families use different prefixes or none at all - so it's not applied unconditionally.

`default_prefixes_for()` in `rag_index.py` auto-detects known conventions from `--embed-model` (currently just Nomic's). When it matches, you'll see `Using embedding prefixes for '<model>': document=... query=...` printed at startup. For a different embedding model with its own convention (e.g. E5's `"query: "`/`"passage: "`, BGE's longer instruction strings), set `--embed-document-prefix`/`--embed-query-prefix` explicitly; pass empty strings to force no prefix even for a model in the known-conventions table.

## What gets measured

**On Ollama**, timing is split into:
- **`prompt_eval`** — time spent processing the input context. This is where full-context pays its tax on every request (the whole document, every time) versus RAG's much smaller retrieved-chunk prompt.
- **`eval`** — time spent generating the answer, reported as tokens/sec.

Ollama also appears to reuse the KV cache automatically when consecutive requests to the same loaded model share a prefix — you may see `prompt_eval` time drop sharply after the first full-context question. This narrows RAG's speed advantage for repeated questions on the same document.

**On OpenAI-compatible backends**, only total wall-clock latency is available — these APIs don't report a prompt/generation split, so results print with an `(estimated)` flag and `tokens_per_s` is derived from total time (includes network overhead, so it understates true generation speed). This is a real capability gap, not a rounding difference — keep it in mind when comparing Ollama numbers against vLLM/LiteLLM numbers directly.

RAG's `retrieval_s` includes one embedding API call per question plus the in-memory cosine-similarity search.

Quality is scored by a second LLM call (the "judge") that sees the full document text, the question, and both answers with the labels **A/B randomized** to avoid position bias. It rates each answer 1–10 on accuracy, completeness, and clarity, and states a preference. Scores outside 1–10 (or answers that aren't valid JSON) are rejected and retried once, regardless of backend — schema-constrained decoding reduces how often that happens but isn't a hard guarantee on any backend.

The summary also prints a derived **efficiency** figure (avg. quality score ÷ avg. latency) for each approach — purely computed from the numbers above after the fact, never seen by the judge itself, so the quality/speed trade-off stays visible instead of being silently baked into one blended score.

## Sharing results

Every run writes three outputs: the CLI printout, `--output` (raw JSON, everything, for further scripting/analysis), and `--xlsx-output` (`report.xlsx` by default) — a two-sheet Excel file meant to be handed to a colleague without needing to explain the tool first:

- **"Results" sheet** — one row per question, answers and the judge's verdict/reasoning in the first columns, technical metrics (tokens, prompt-processing time, retrieved pages) further right. Ignore whatever columns don't mean anything to you.
- **"Summary" sheet** — the exact same narrative text printed to the CLI, one line per row, so the overall story (which approach won, by how much, on speed vs. quality) is readable without opening the data sheet at all.

Set `--xlsx-output ""` to skip generating it.

## Development

```bash
pytest
```

Tests use fake `ChatBackend`/`EmbedBackend` implementations (see `tests/conftest.py`) instead of a live server, so the suite runs in well under a second with no Ollama/vLLM/etc. required. It covers chunking, retrieval ranking, the judge's validation/retry/context-growth logic, the xlsx report's structure, and CLI argument parsing. It does **not** cover the end-to-end `main()` orchestration loop (needs a live backend) — that's exercised manually against `examples/example.pdf` before releases.

## Project layout

```
pyproject.toml
examples/                         example.pdf + example_large.pdf, each with a matching *_questions.txt
src/ctxvsrag/
├── cli.py                        entry point (the `ctxvsrag` command)
├── pdf_utils.py                  PDF text extraction, rough token estimation
├── chunking.py                   splits page text into overlapping word-based chunks
├── rag_index.py                  embedding index + cosine-similarity retrieval, backend-agnostic
├── judge.py                      LLM-as-judge quality comparison
├── report.py                     two-sheet .xlsx report for sharing with non-technical colleagues
├── approaches/
│   ├── full_context.py           Approach A: whole document in every prompt
│   └── rag_answer.py             Approach B: retrieve chunks, then answer
└── backends/
    ├── base.py                   ChatBackend/EmbedBackend protocols, shared ChatResult type
    ├── ollama_backend.py         native Ollama backend (precise timing)
    └── openai_compat_backend.py  generic OpenAI-compatible backend (vLLM, LiteLLM, Open WebUI, ...)
tests/                            pytest suite, fake backends, no live server needed
```

## Known limitations

- **No cost tracking.** Everything runs against self-hosted infrastructure, so there's nothing to bill — the metrics focus on latency and throughput instead.
- **Timing precision isn't uniform across backends.** Only Ollama reports a prompt/generation time split; comparing an Ollama run against a vLLM run means comparing precise numbers to wall-clock estimates. Fine within one backend, be careful across backends.
- **Retrieval quality depends entirely on the embedding model.** A different embedding model may retrieve noticeably better on domain-specific documents (legal, medical, code) — worth A/B-ing via `--embed-model` before drawing conclusions about "RAG quality" in general.
- **Judge re-sends the full document on every question**, and on non-Ollama backends there's no cache to reuse it across calls — this is the most expensive part of a run in wall-clock terms. Use `--no-judge` for quick iteration.
- Scanned/image-only PDFs won't extract usable text via `pypdf` — the script warns if extraction looks suspiciously sparse, but doesn't OCR.
