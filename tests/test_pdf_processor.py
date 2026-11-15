import io

import pytest

from pdf_processor import PDFProcessor
from scripts.make_sample_pdf import build_pdf


@pytest.fixture()
def processor():
    return PDFProcessor()


@pytest.fixture()
def sample_pdf():
    return io.BytesIO(build_pdf())


def test_extracts_text_from_pdf(processor, sample_pdf):
    text = processor.extract_text_from_pdf(sample_pdf)
    assert "Retrieval-augmented generation" in text
    assert "38 percent" in text


def test_clean_text_normalises_whitespace(processor):
    raw = "Hello   world\t.\n\n\n\nSecond   paragraph.\x00"
    assert processor.clean_text(raw) == "Hello world .\n\nSecond paragraph."


def test_count_tokens_is_positive(processor):
    assert processor.count_tokens("a short sentence") > 0


def test_chunk_text_respects_token_budget(processor):
    text = " ".join(f"Sentence number {i} about summarisation." for i in range(400))
    chunks = processor.chunk_text(text, max_tokens=200)
    assert len(chunks) > 1
    assert all(processor.count_tokens(c) <= 200 for c in chunks)
    assert " ".join(chunks).count("Sentence number") == 400


def test_chunk_text_hard_splits_oversized_sentence(processor):
    giant = "word " * 2000
    chunks = processor.chunk_text(giant, max_tokens=100)
    assert len(chunks) > 1
    assert all(processor.count_tokens(c) <= 100 for c in chunks)


def test_chunk_text_rejects_empty_text(processor):
    with pytest.raises(ValueError, match="No extractable text"):
        processor.chunk_text("   \n  ")


def test_metadata(processor, sample_pdf):
    meta = processor.get_pdf_metadata(sample_pdf)
    assert meta["num_pages"] == 1
    assert meta["title"] == "Unknown"
