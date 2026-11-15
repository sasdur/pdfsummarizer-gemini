"""Generate a small text PDF without third-party dependencies.

Used by the test-suite and for the local smoke test::

    python scripts/make_sample_pdf.py sample/sample.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

NL = chr(10)
BACKSLASH = chr(92)

LINES = [
    "Quarterly Research Brief: Retrieval-Augmented Generation in Practice",
    "",
    "1. Background",
    "Retrieval-augmented generation (RAG) pairs a language model with an external",
    "knowledge base. The model retrieves relevant passages at query time instead of",
    "relying only on parameters learned during training.",
    "",
    "2. Findings",
    "Teams that adopted RAG reported a 38 percent drop in unsupported answers.",
    "The largest gains came from documents that change frequently, such as policies",
    "and product manuals. Freshness mattered more than model size.",
    "",
    "3. Recommendations",
    "Start with a narrow corpus, measure answer groundedness, and add citations.",
    "Review chunking strategy quarterly. Scanned documents still need OCR before",
    "they can be indexed reliably.",
    "",
    "4. Conclusion",
    "RAG is not a replacement for evaluation. It is a way to keep answers tied to",
    "sources that a reviewer can inspect.",
]


def _escape(text: str) -> str:
    return (
        text.replace(BACKSLASH, BACKSLASH * 2)
        .replace("(", BACKSLASH + "(")
        .replace(")", BACKSLASH + ")")
    )


def _content_stream() -> str:
    parts = ["BT", "/F1 12 Tf", "16 TL", "50 760 Td"]
    for line in LINES:
        if line:
            parts.append("(" + _escape(line) + ") Tj T*")
        else:
            parts.append("T*")
    parts.append("ET")
    return NL.join(parts)


def build_pdf() -> bytes:
    """Assemble a minimal one-page PDF with the sample text."""
    stream = _content_stream().encode("latin-1")
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>" + NL.encode()
        + b"stream" + NL.encode() + stream + NL.encode() + b"endstream",
    ]

    out = bytearray(("%PDF-1.4" + NL).encode())
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        raw = body.encode("latin-1") if isinstance(body, str) else body
        out += f"{index} 0 obj{NL}".encode() + raw + f"{NL}endobj{NL}".encode()

    xref_at = len(out)
    out += f"xref{NL}0 {len(objects) + 1}{NL}".encode()
    out += b"0000000000 65535 f " + NL.encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n {NL}".encode()
    out += (
        f"trailer{NL}<< /Size {len(objects) + 1} /Root 1 0 R >>{NL}"
        f"startxref{NL}{xref_at}{NL}%%EOF{NL}"
    ).encode()
    return bytes(out)


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "sample/sample.pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_pdf())
    print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
