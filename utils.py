import os
import time

import pandas as pd
import streamlit as st

PLACEHOLDER_KEY = "your_gemini_api_key_here"


def format_summary_display(summary_data: dict):
    """Render the summary statistics, final summary and per-chunk summaries."""
    st.subheader("📊 Summary Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", summary_data['total_chunks'])
    col2.metric("Successful Summaries", sum(s['ok'] for s in summary_data['individual_summaries']))
    col3.metric("Summary Type", summary_data['summary_type'].replace('_', ' ').title())

    st.subheader("📝 Final Summary")
    st.write(summary_data['combined_summary'])

    with st.expander("View Individual Chunk Summaries"):
        for s in summary_data['individual_summaries']:
            st.write(f"**Chunk {s['chunk_number']}**" + ("" if s['ok'] else " ⚠ failed"))
            st.write(s['summary'])
            st.write(f"*Original length: {s['original_length']} chars, "
                     f"Summary length: {s['summary_length']} chars*")
            st.divider()


def export_summary_to_text(summary_data: dict, filename: str = "") -> str:
    """Build the plain-text report; filename (if given) appears in the header."""
    lines = ["PDF Summary Report", "==================", ""]
    if filename:
        lines.append(f"File: {filename}")
    lines += [
        f"Summary Type: {summary_data['summary_type'].replace('_', ' ').title()}",
        f"Total Chunks Processed: {summary_data['total_chunks']}",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FINAL SUMMARY",
        "=============",
        "",
        summary_data['combined_summary'],
        "",
        "INDIVIDUAL CHUNK SUMMARIES",
        "==========================",
        "",
    ]
    for s in summary_data['individual_summaries']:
        lines += [f"Chunk {s['chunk_number']}{'' if s['ok'] else ' (FAILED)'}:", s['summary'], ""]
    return "\n".join(lines)


def create_summary_dataframe(summary_data: dict) -> pd.DataFrame:
    """One row per chunk with lengths, compression ratio and status."""
    return pd.DataFrame([{
        'Chunk': s['chunk_number'],
        'Original Length': s['original_length'],
        'Summary Length': s['summary_length'],
        'Compression Ratio': round(s['summary_length'] / s['original_length'], 2) if s['original_length'] else 0,
        'Status': 'Success' if s['ok'] else 'Error',
    } for s in summary_data['individual_summaries']])


def estimate_processing_time(num_chunks: int) -> str:
    """Rough estimate: ~3s per chunk with 4 running concurrently, plus 5s to combine."""
    seconds = int(num_chunks * 3 / 4 + 5)
    return f"~{seconds} seconds" if seconds < 60 else f"~{seconds // 60} min {seconds % 60}s"


def validate_api_key() -> bool:
    """Check GEMINI_API_KEY is set; show a quiet sidebar caption when it is."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == PLACEHOLDER_KEY:
        st.error("❌ GEMINI_API_KEY not found! Copy `.env.example` to `.env`, add your key, and restart.")
        st.info("Get a free key at https://aistudio.google.com/apikey")
        return False
    if len(api_key) < 20:
        st.error("❌ GEMINI_API_KEY seems incomplete. Get a new one at https://aistudio.google.com/apikey")
        return False
    st.sidebar.caption("🔑 Gemini API key loaded")
    return True
