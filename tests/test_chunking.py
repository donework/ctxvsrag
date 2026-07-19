from ctxvsrag.chunking import Chunk, chunk_pages


def test_basic_chunking():
    pages = ["one two three four five six", "seven eight nine ten"]
    chunks = chunk_pages(pages, chunk_words=3, overlap_words=1)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].page == 1
    assert chunks[0].text == "one two three"


def test_overlap_between_consecutive_chunks():
    pages = ["a b c d e f g h"]
    chunks = chunk_pages(pages, chunk_words=4, overlap_words=2)

    assert chunks[0].text == "a b c d"
    assert chunks[1].text == "c d e f"
    assert chunks[2].text == "e f g h"


def test_empty_and_whitespace_only_pages_are_skipped():
    pages = ["some content here", "   ", ""]
    chunks = chunk_pages(pages, chunk_words=10)

    assert all(c.page == 1 for c in chunks)
    assert len(chunks) == 1


def test_page_numbers_are_1_indexed_and_tracked_per_page():
    pages = ["page one text", "page two text"]
    chunks = chunk_pages(pages, chunk_words=10)

    assert {c.page for c in chunks} == {1, 2}


def test_short_page_produces_single_chunk():
    pages = ["just a few words"]
    chunks = chunk_pages(pages, chunk_words=300, overlap_words=50)

    assert len(chunks) == 1
    assert chunks[0].text == "just a few words"
