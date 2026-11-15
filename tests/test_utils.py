import pandas as pd

from utils import create_summary_dataframe, estimate_processing_time, export_summary_to_text

SUMMARY_DATA = {
    "summary_type": "bullet_points",
    "total_chunks": 2,
    "combined_summary": "Final combined text.",
    "individual_summaries": [
        {"chunk_number": 1, "summary": "First summary.", "ok": True,
         "original_length": 100, "summary_length": 25},
        {"chunk_number": 2, "summary": "Error processing chunk: boom", "ok": False,
         "original_length": 50, "summary_length": 0},
    ],
}


def test_dataframe_has_one_row_per_chunk_and_ratio():
    df = create_summary_dataframe(SUMMARY_DATA)

    assert isinstance(df, pd.DataFrame)
    assert list(df["Chunk"]) == [1, 2]
    assert df.loc[0, "Compression Ratio"] == 0.25
    assert df.loc[1, "Compression Ratio"] == 0
    assert list(df["Status"]) == ["Success", "Error"]


def test_text_export_contains_header_summary_and_failed_marker():
    report = export_summary_to_text(SUMMARY_DATA, "paper.pdf")

    assert "File: paper.pdf" in report
    assert "Summary Type: Bullet Points" in report
    assert "Final combined text." in report
    assert "Chunk 2 (FAILED):" in report


def test_estimate_processing_time_scales():
    assert estimate_processing_time(4).startswith("~")
    assert "min" not in estimate_processing_time(4)
    assert "min" in estimate_processing_time(200)
