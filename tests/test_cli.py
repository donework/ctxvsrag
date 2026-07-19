import pytest

from ctxvsrag.cli import DEFAULT_MODEL, DEFAULT_NUM_CTX, _excerpt, build_arg_parser


def test_excerpt_short_text_unchanged():
    assert _excerpt("short text", 200) == "short text"


def test_excerpt_truncates_long_text_with_ellipsis():
    text = "a" * 300
    result = _excerpt(text, 200)
    assert result == "a" * 200 + "..."


def test_excerpt_collapses_whitespace_and_newlines():
    text = "line one\n\nline   two\nline three"
    assert _excerpt(text, 200) == "line one line two line three"


def test_excerpt_empty_text():
    assert _excerpt("", 200) == ""


def test_arg_parser_requires_pdf():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(["--pdf", "doc.pdf"])

    assert args.model == DEFAULT_MODEL
    assert args.num_ctx == DEFAULT_NUM_CTX
    assert args.chat_backend == "ollama"
    assert args.embed_backend == "ollama"
    assert args.no_judge is False
    assert args.excerpt_length == 200
    assert args.embed_document_prefix is None  # unset -> cli.py auto-detects from --embed-model
    assert args.embed_query_prefix is None


def test_arg_parser_embed_prefix_override():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--pdf", "doc.pdf",
        "--embed-document-prefix", "passage: ",
        "--embed-query-prefix", "query: ",
    ])

    assert args.embed_document_prefix == "passage: "
    assert args.embed_query_prefix == "query: "


def test_arg_parser_openai_backend_flags():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--pdf", "doc.pdf",
        "--chat-backend", "openai",
        "--chat-host", "http://localhost:8000/v1",
        "--model", "my-model",
    ])

    assert args.chat_backend == "openai"
    assert args.chat_host == "http://localhost:8000/v1"
    assert args.model == "my-model"


def test_arg_parser_rejects_unknown_backend():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--pdf", "doc.pdf", "--chat-backend", "bogus"])
