"""End-to-end smoke test: real PDF -> chunks -> real Gemini call.

Costs a couple of API calls. Run from the project root:

    .venv\\Scripts\\python scripts\\smoke_test.py [path/to/file.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from pdf_processor import PDFProcessor  # noqa: E402
from summarizer import PDFSummarizer  # noqa: E402

load_dotenv(ROOT / ".env")


def main() -> int:
    pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "sample/sample.pdf")
    if not pdf_path.exists():
        print(f"missing input: {pdf_path} (run scripts/make_sample_pdf.py first)")
        return 1

    processor = PDFProcessor()
    with pdf_path.open("rb") as fh:
        raw = processor.extract_text_from_pdf(fh)
    text = processor.clean_text(raw)
    chunks = processor.chunk_text(text, max_tokens=4000)

    print(f"file      : {pdf_path}")
    print(f"chars     : {len(text)}")
    print(f"tokens    : {processor.count_tokens(text)}")
    print(f"chunks    : {len(chunks)}")

    summarizer = PDFSummarizer()
    print(f"model     : {summarizer.model}")

    result = summarizer.summarize_chunks(chunks, "brief")
    ok = sum(s["ok"] for s in result["individual_summaries"])
    print(f"chunk ok  : {ok}/{result['total_chunks']}")
    print("\n--- final summary ---")
    print(result["combined_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
