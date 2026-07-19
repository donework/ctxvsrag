from pathlib import Path

from ctxvsrag.pdf_utils import estimate_tokens, extract_pages

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_PDF = EXAMPLES_DIR / "example.pdf"
EXAMPLE_LARGE_PDF = EXAMPLES_DIR / "example_large.pdf"


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("abc" * 100) > estimate_tokens("abc")


def test_estimate_tokens_minimum_is_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("ab") == 1


def test_estimate_tokens_is_roughly_chars_over_three():
    assert estimate_tokens("x" * 300) == 100


def test_extract_pages_reads_the_example_pdf():
    pages = extract_pages(str(EXAMPLE_PDF))

    assert len(pages) > 15  # example.pdf is ~21 pages
    assert "ACME" in "".join(pages)


def test_extract_pages_reads_the_large_example_pdf():
    pages = extract_pages(str(EXAMPLE_LARGE_PDF))

    assert len(pages) == 100
    assert "ACME" in "".join(pages)
