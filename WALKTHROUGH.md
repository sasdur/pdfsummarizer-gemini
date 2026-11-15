# Walkthrough: how a PDF becomes a summary

This document follows one upload through the whole system. Read it top to bottom to
understand every module, the design decisions behind it, and where to change things.

---

## 0. At a glance

| Layer | File | Responsibility |
|---|---|---|
| UI | `app.py` | Sidebar settings, upload, progress, rendering, downloads |
| Text | `pdf_processor.py` | Extract → clean → count tokens → chunk |
| Prompts | `prompts.py` | System persona + four `ChatPromptTemplate`s |
| LLM | `summarizer.py` | Builds LCEL chains, runs map-reduce, analysis, quotes |
| Glue | `utils.py` | Metrics, text/CSV export, API-key validation |

The app is stateless between uploads and stateful across Streamlit reruns via
`st.session_state` (results, processor, file id).

---

## 1. Startup

1. `load_dotenv()` reads `.env` and exposes `GEMINI_API_KEY` / `GEMINI_MODEL`.
2. `validate_api_key()` in `utils.py` blocks the page when the key is missing, is the
   `.env.example` placeholder, or is too short — with a link to AI Studio.
3. `get_summarizer(model)` is wrapped in `@st.cache_resource`, so the Gemini client is
   created **once per model** and reused across reruns and browser sessions. Streamlit
   reruns the script on every interaction; without this, you would rebuild the client
   (and its HTTP session) on every click.
4. `PDFProcessor()` is cheap, so it simply lives in session state.

---

## 2. Upload and metadata

`st.file_uploader` returns a `UploadedFile`, which is a file-like object with
`.seek(0)`, so it can be handed straight to `pypdf`.

`PDFProcessor.get_pdf_metadata()` reads `/Title`, `/Author` and `/Subject` and the page
count. Missing fields return `"Unknown"` rather than raising: many PDFs simply have no
document info dictionary.

A `file_id` guard clears old results whenever the uploaded file changes:

```python
file_id = uploaded_file.file_id if uploaded_file else None
if st.session_state.get('file_id') != file_id:
    st.session_state.pop('results', None)
    st.session_state.file_id = file_id
```

Without it, uploading a second PDF would briefly show the first PDF's summary.

---

## 3. Extraction

```python
reader = PdfReader(pdf_file)
pages = [page.extract_text() or "" for page in reader.pages]
return "\n\n".join(pages)
```

Notes:

- Each page is wrapped in its own `try/except`; a corrupt page is logged and skipped
  instead of failing the document.
- `or ""` normalises `None` for image-only pages.
- A document-level failure is re-raised as `RuntimeError("Error reading PDF: ...")` so
  the UI can show a clean message.
- No OCR. If the result is empty, the app tells the user the PDF is probably scanned.

---

## 4. Cleaning

`clean_text()` applies four regex passes, in order:

| Step | Regex | Effect |
|---|---|---|
| 1 | `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]` | Drop control characters |
| 2 | `[ \t]+` → `" "` | Collapse runs of spaces/tabs |
| 3 | ` ?\n ?` → `"\n"` | Strip spaces hugging a newline |
| 4 | `\n{3,}` → `"\n\n"` | At most one blank line between paragraphs |

Punctuation, capitalisation and paragraph boundaries are deliberately preserved —
the summary prompt relies on sentence structure.

---

## 5. Chunking (the important part)

Gemini accepts large contexts, but per-chunk summarization keeps quality high and lets
chunks run in parallel. Chunking is done in **tokens**, not characters, using
`tiktoken.get_encoding("cl100k_base")` as a cheap approximation of the Gemini tokenizer.

Algorithm (`chunk_text`):

1. Split on sentence boundaries: `re.split(r"(?<=[.!?])\s+", text)`.
2. Tokenize each sentence. If a single sentence exceeds `max_tokens` (rare, e.g. a
   pasted table with no punctuation), hard-split it by decoding token windows.
3. Greedily pack sentences into a chunk until the next sentence would overflow
   `max_tokens`; then start a new chunk. The `+1` per sentence approximates the
   joining space.

Properties:

- Chunks never exceed the budget (verified by tests).
- No text is lost: concatenating chunks reproduces every sentence.
- No overlap is used, which keeps the reduce step free of duplicated claims.

`max_tokens` comes from the sidebar slider (4000–10000, default 8000). Smaller chunks
mean more calls (slower, more tokens overall) but tighter, more focused summaries.

Empty input raises `ValueError("No extractable text — the PDF may be scanned images")`.

---

## 6. Map step: per-chunk summaries

`PDFSummarizer.summarize_chunks()`:

```python
self.chunk_chain = CHUNK_SUMMARY | self.llm | parser
results = self.chunk_chain.batch(inputs,
                                 config={"max_concurrency": 4},
                                 return_exceptions=True)
```

- LCEL (`|`) composes prompt → model → parser into one `Runnable`.
- `parser` is `RunnableLambda(lambda m: str(m.text))`. Gemini 3 returns content as a
  list of blocks; `m.text` joins them, and `str()` turns the `TextAccessor` into a real
  string so downstream code never sees a LangChain object.
- `batch(..., return_exceptions=True)` runs up to 4 chunks concurrently and gives back
  exception objects instead of aborting the whole batch.
- Each chunk becomes a record:

```python
{"chunk_number": 1, "summary": "...", "ok": True,
 "original_length": 4210, "summary_length": 512}
```

A failed chunk keeps `ok: False`, `summary_length: 0`, and a readable error string.
The UI labels it `⚠ failed`, and the CSV export marks it `Error`.

---

## 7. Reduce step: combine

`combine_summaries()`:

- 0 valid summaries → `"No valid summaries were generated."`
- exactly 1 → return it as the final summary (no wasted API call)
- otherwise → render `COMBINE` with numbered sections and ask the model to merge them
  without repetition.

If the combine call itself fails, the code degrades gracefully to
`"Combined Summary:\n\n<numbered sections>"` instead of raising. Partial results beat
an error screen.

---

## 8. Optional extras

**Document analysis** (`analyze_document_structure`): sends the first 3000 characters
through `ANALYSIS` and asks for document type, themes, sections, audience, purpose. The
truncation keeps the call cheap; structure is visible in the opening pages. Returns
`{"analysis": str, "status": "success" | "error"}`.

**Key quotes** (`extract_key_quotes`): sends the first 5000 characters through `QUOTES`,
then post-processes each line:

- strips leading numbering/bullets (`1.`, `-`, `•`)
- strips surrounding straight or curly quotes
- drops empty lines, caps at 10

The prompt forbids adding numbering in the first place; the cleanup is belt-and-braces.

---

## 9. Rendering and export

`format_summary_display()` renders three metrics (chunks, successful summaries,
summary type), the final summary, and an expander with every chunk's summary plus its
original/summary character counts.

`export_summary_to_text()` builds a plain-text report with a header (file, type, chunk
count, timestamp), the final summary, then per-chunk sections, marking failures.

`create_summary_dataframe()` builds one row per chunk: lengths, compression ratio
(`summary/original`, 0 when original is empty), and status. The UI shows it in a
dataframe plus average compression and success rate. The CSV download is the same table.

Downloads use `st.download_button`, which triggers a rerun — results are read from
`st.session_state['results']`, so nothing is recomputed (and no tokens are spent).

---

## 10. Failure modes and what the user sees

| Failure | Handling |
|---|---|
| No `.env` / placeholder key | Full-page error with AI Studio link; `st.stop()` |
| Corrupt PDF | `RuntimeError` → `st.error("Error extracting text: ...")` |
| Scanned PDF | `ValueError` → "No extractable text — the PDF may be scanned images" |
| One chunk fails (safety block, 500, timeout) | Chunk marked failed, others continue |
| All chunks fail | "No valid summaries were generated." |
| Combine call fails | Fallback to concatenated sections |
| Analysis fails | Warning banner, summary still shown |
| Quotes fail | Warning banner only |
| 429 rate limit | Visible error; retry, smaller file, or different model |

---

## 11. Testing strategy

`tests/` runs offline:

- `test_pdf_processor.py` builds a real PDF in memory with a dependency-free generator
  (`scripts/make_sample_pdf.py`) and checks extraction, cleaning, token counting,
  budget enforcement, hard-splitting, empty input and metadata.
- `test_summarizer.py` injects a fake chat model into `PDFSummarizer` (the `llm=`
  parameter exists for exactly this reason) and asserts prompt rendering, map/reduce
  call counts, per-chunk failure isolation, single-chunk shortcut, combine fallback,
  analysis shapes and quote cleanup.
- `test_utils.py` checks dataframe ratios, text export content and time estimates.

Nothing in the suite needs a key, so it can run in CI.

---

## 12. Where to extend

| Goal | Change |
|---|---|
| New summary style | add to `STYLES` in `prompts.py`; add the name in `app.py` |
| Different model default | `GEMINI_MODEL` in `.env` |
| Bigger/smaller chunks | slider range in `app.py` |
| More concurrency | `max_concurrency` in `summarize_chunks` |
| Chat with the PDF | store chunks in session state, add a retrieval step, reuse the client |
| OCR for scanned PDFs | add `pytesseract`/`pdf2image` in `pdf_processor.extract_text_from_pdf` |
| Cost tracking | wrap `self.llm` with a callback handler that accumulates `usage_metadata` |
