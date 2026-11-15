from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from prompts import CHUNK_SUMMARY, COMBINE, STYLES
from summarizer import PDFSummarizer

QUOTES_REPLY = "\n".join(f'{i}. "Quote number {i}"' for i in range(1, 13))


class FakeGemini:
    """Deterministic stand-in for ChatGoogleGenerativeAI.

    Replies are chosen from the rendered prompt, not from a queue, so the tests do
    not depend on the order in which concurrent batch calls finish.
    """

    def __init__(self, fail_on=(), map_reply="chunk summary", combine_reply="FINAL"):
        self.fail_on = tuple(fail_on)
        self.map_reply = map_reply
        self.combine_reply = combine_reply
        self.prompts: list[str] = []

    def as_runnable(self):
        def respond(prompt_value):
            text = prompt_value.to_string()
            self.prompts.append(text)

            if any(pattern in text for pattern in self.fail_on):
                raise RuntimeError("simulated API failure")
            if "<summaries>" in text:
                return AIMessage(content=self.combine_reply)
            if "Analyze the structure" in text:
                return AIMessage(content="Document type: report")
            if "Extract 5-10 key quotes" in text:
                return AIMessage(content=QUOTES_REPLY)
            return AIMessage(content=self.map_reply)

        return RunnableLambda(respond)


def make_summarizer(**kwargs):
    fake = FakeGemini(**kwargs)
    return PDFSummarizer(model="fake-model", llm=fake.as_runnable()), fake


def test_summarize_text_renders_style_into_prompt():
    summarizer, fake = make_summarizer(map_reply="A short summary.")
    assert summarizer.summarize_text("Some text.", "brief") == "A short summary."
    assert STYLES["brief"] in fake.prompts[0]
    assert "Some text." in fake.prompts[0]


def test_unknown_style_falls_back_to_detailed():
    summarizer, fake = make_summarizer()
    summarizer.summarize_text("text", "not-a-style")
    assert STYLES["detailed"] in fake.prompts[0]


def test_map_reduce_produces_per_chunk_stats_and_combined_summary():
    summarizer, fake = make_summarizer()

    result = summarizer.summarize_chunks(["chunk one text", "chunk two text"], "detailed")

    assert result["total_chunks"] == 2
    assert result["summary_type"] == "detailed"
    assert [s["ok"] for s in result["individual_summaries"]] == [True, True]
    assert result["individual_summaries"][0]["original_length"] == len("chunk one text")
    assert result["combined_summary"] == "FINAL"
    # map, map, reduce
    assert len(fake.prompts) == 3
    assert "Section 1" in fake.prompts[-1]


def test_single_chunk_skips_reduce_step():
    summarizer, fake = make_summarizer(map_reply="only summary")
    result = summarizer.summarize_chunks(["single chunk"], "brief")
    assert result["combined_summary"] == "only summary"
    assert len(fake.prompts) == 1


def test_failed_chunk_is_isolated_and_reported():
    summarizer, fake = make_summarizer(fail_on=("EXPLODE",))
    result = summarizer.summarize_chunks(["fine text", "EXPLODE here"], "detailed")

    first, second = result["individual_summaries"]
    assert first["ok"] is True
    assert second["ok"] is False
    assert second["summary_length"] == 0
    assert "Error processing chunk" in second["summary"]
    # only one valid summary left, so the reduce shortcut returns it directly
    assert result["combined_summary"] == "chunk summary"
    assert len(fake.prompts) == 2


def test_all_chunks_failing_returns_message():
    summarizer, _ = make_summarizer(fail_on=("EXPLODE",))
    result = summarizer.summarize_chunks(["EXPLODE", "EXPLODE"], "detailed")
    assert result["combined_summary"] == "No valid summaries were generated."


def test_combine_failure_falls_back_to_joined_sections():
    summarizer, _ = make_summarizer(fail_on=("<summaries>",))
    result = summarizer.summarize_chunks(["a", "b"], "brief")
    assert result["combined_summary"].startswith("Combined Summary:")
    assert "Section 1: chunk summary" in result["combined_summary"]


def test_analysis_success_and_failure_shapes():
    summarizer, _ = make_summarizer()
    out = summarizer.analyze_document_structure("text")
    assert out == {"analysis": "Document type: report", "status": "success"}

    failing, _ = make_summarizer(fail_on=("Analyze the structure",))
    out = failing.analyze_document_structure("text")
    assert out["status"] == "error"
    assert "Error analyzing document" in out["analysis"]


def test_analysis_is_truncated_to_first_3000_chars():
    summarizer, fake = make_summarizer()
    summarizer.analyze_document_structure("A" * 5000 + "TAIL-MARKER")
    assert "TAIL-MARKER" not in fake.prompts[0]


def test_quotes_are_cleaned_and_capped():
    summarizer, _ = make_summarizer()
    quotes = summarizer.extract_key_quotes("text")
    assert len(quotes) == 10
    assert quotes[0] == "Quote number 1"
    assert not quotes[0].startswith("1.")


def test_prompt_templates_render_without_missing_variables():
    style = STYLES["executive"]
    assert "body" in CHUNK_SUMMARY.format(style=style, text="body")
    assert style in CHUNK_SUMMARY.format(style=style, text="body")
    assert "Section 1" in COMBINE.format(style=style, summaries="Section 1: x")
