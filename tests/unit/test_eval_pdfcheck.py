"""D68's validity check (D81) — everything about it that does not need arXiv.

The check's credibility rests on three things a test can hold down without the
network: that the two builds differ ONLY in where their text came from, that the
shared normaliser folds exactly the differences D68 names and nothing more, and
that a query judged on one side alone never reaches the delta. The fetch itself is
tested against a stub, including the case that would otherwise be reported as a
catastrophic parsing regression: arXiv serving a holding page instead of a PDF.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from scholiast.application.ports.parsing import ParsedDocument
from scholiast.domain.chunking import BlockKind, ParsedBlock, ParsedSection
from scholiast.domain.paper import SectionKind
from scholiast.eval.dret import DRet, Grade, Judgment, Query, Split, build_dret
from scholiast.eval.pdfcheck import (
    USER_AGENT,
    comparable_queries,
    diagnose_mapping,
    fetch_and_parse,
    fetch_pdf,
    judged_query_ids,
    normalise_for_pdf_match,
    paper_from_document,
    queries_by_id,
    sample_papers,
)
from scholiast.eval.qasper import build_paper, normalise_for_match


def _row(paper_id: str, *, question: str = "what was measured?") -> dict[str, Any]:
    return {
        "id": paper_id,
        "title": f"Paper {paper_id}",
        "abstract": "An abstract sentence.",
        "full_text": {
            "section_name": ["Results"],
            "paragraphs": [["The system reached 91.2 accuracy on the benchmark."]],
        },
        "qas": {
            "question": [question],
            "question_id": [f"{paper_id}-q1"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "evidence": ["The system reached 91.2 accuracy on the benchmark."],
                        }
                    ]
                }
            ],
        },
    }


def _document(paragraph: str, *, title: str = "Results") -> ParsedDocument:
    block = ParsedBlock(
        kind=BlockKind.PARAGRAPH,
        text=paragraph,
        page=1,
        char_start=0,
        char_end=len(paragraph),
    )
    return ParsedDocument(
        sections=(
            ParsedSection(
                kind=SectionKind.RESULTS,
                path=title,
                blocks=(block,),
                text=paragraph,
                raw_title=title,
            ),
        ),
        title="A Paper",
        abstract=None,
        page_count=1,
        parser_version="grobid/0.8.2.1-crf",
        stats={},
    )


def _dataset(*, queries: list[Query], judgments: list[Judgment]) -> DRet:
    return DRet(queries=queries, judgments=judgments, chunks={}, papers={})


class TestTheSharedNormaliser:
    """It must fold exactly what D68 names, and leave the rest alone."""

    def test_a_hyphen_at_a_line_break_is_joined(self) -> None:
        # PDF line-wrapping. Without this the evidence string never matches and the
        # judgment is silently lost.
        assert normalise_for_pdf_match("inter-\nnational") == "international"

    def test_a_real_hyphen_survives(self) -> None:
        # The failure mode on the other side: over-folding would merge distinct
        # terms and match evidence to the wrong paragraph.
        assert normalise_for_pdf_match("state-of-the-art") == "state-of-the-art"

    def test_ligatures_are_folded(self) -> None:
        assert normalise_for_pdf_match("The ﬁrst ﬂaw") == "the first flaw"

    def test_whitespace_is_collapsed_and_case_ignored(self) -> None:
        assert normalise_for_pdf_match("  Two   Spaces\n here ") == "two spaces here"

    def test_it_is_strictly_more_permissive_than_the_published_matcher(self) -> None:
        # The reason it exists (D81). If these two ever agreed on GROBID-shaped
        # text, the PDF side would not need its own matcher at all.
        grobid_text = "The ﬁrst inter-\nnational evaluation"
        qasper_text = "The first international evaluation"
        assert normalise_for_pdf_match(grobid_text) == normalise_for_pdf_match(qasper_text)
        assert normalise_for_match(grobid_text) != normalise_for_match(qasper_text)


class TestTheSample:
    def test_it_is_deterministic_for_a_seed(self) -> None:
        # The sample is part of the result: a reader has to be able to reproduce it.
        dataset = _dataset(
            queries=[Query(f"q{i}", "text", f"paper-{i}", Split.EVAL) for i in range(20)],
            judgments=[],
        )
        first = sample_papers(dataset, count=5, seed=7)
        assert first == sample_papers(dataset, count=5, seed=7)
        assert first != sample_papers(dataset, count=5, seed=8)
        assert first == sorted(first)

    def test_only_eval_split_papers_are_eligible(self) -> None:
        # Sampling a train paper would measure the parser on a paper no published
        # number is measured on.
        dataset = _dataset(
            queries=[
                Query("q1", "t", "eval-paper", Split.EVAL),
                Query("q2", "t", "train-paper", Split.TRAIN),
            ],
            judgments=[],
        )
        assert sample_papers(dataset, count=5) == ["eval-paper"]

    def test_a_dataset_with_no_eval_papers_is_refused(self) -> None:
        dataset = _dataset(queries=[Query("q", "t", "p", Split.TRAIN)], judgments=[])
        with pytest.raises(ValueError, match="no eval-split papers"):
            sample_papers(dataset)


class TestBuildingAPaperFromGrobid:
    def test_the_index_is_keyed_by_normalised_block_text(self) -> None:
        # This index is what the evidence lookup reads; if its keys are not
        # normalised the same way the lookup is, nothing matches.
        paper = paper_from_document(
            _row("1234.5678"),
            _document("The ﬁrst result."),
            normaliser=normalise_for_pdf_match,
        )
        # The KEY is normalised; the OFFSETS index the stored text, where the
        # ligature is a single character (16, not 17). Normalising before recording
        # offsets is the precise failure `ParsedBlock`'s round trip exists to catch.
        assert paper.paragraph_index["the first result."] == ("Results", 0, 16)
        assert paper.section_headings == frozenset({"results"})

    def test_the_labels_still_come_from_qasper(self) -> None:
        # The check varies where the TEXT comes from, never the labels.
        paper = paper_from_document(
            _row("1234.5678"), _document("Body text."), normaliser=normalise_for_pdf_match
        )
        assert paper.paper_id == "1234.5678"
        assert paper.title == "Paper 1234.5678"


class TestTheBuilderInjection:
    def test_the_defaults_reproduce_the_published_build(self) -> None:
        # The injection points exist for this check alone. If their defaults ever
        # stop being the published behaviour, every number phases 2-6 reported
        # moves and nothing says so.
        rows = [_row("1111.1111"), _row("2222.2222")]
        default = build_dret(rows)
        explicit = build_dret(rows, paper_factory=build_paper, normaliser=normalise_for_match)
        assert sorted(default.chunks) == sorted(explicit.chunks)
        assert [(j.query_id, j.chunk_id, j.grade) for j in default.judgments] == [
            (j.query_id, j.chunk_id, j.grade) for j in explicit.judgments
        ]
        assert default.evidence.as_dict() == explicit.evidence.as_dict()

    def test_a_pdf_side_build_uses_the_same_judgment_machinery(self) -> None:
        # The whole design: the PDF side is `build_dret` with a different paper
        # factory, so grading, splitting and CORE selection cannot drift.
        rows = [_row("1111.1111")]
        evidence = "The system reached 91.2 accuracy on the benchmark."

        def from_pdf(row: dict[str, Any], *, normaliser: Any) -> Any:
            # Same sentence, PDF-shaped: a ligature and a wrapped hyphen.
            return paper_from_document(
                row, _document(evidence.replace("reached", "re-\nached")), normaliser=normaliser
            )

        side = build_dret(rows, paper_factory=from_pdf, normaliser=normalise_for_pdf_match)
        assert side.judgments, "the wrapped hyphen should still match under the shared matcher"
        assert side.evidence.total == 1


class TestWhatCanBeCompared:
    def test_only_queries_judged_on_both_sides_enter_the_delta(self) -> None:
        # D81's basis: a query judged on one side only measures the matcher.
        json_side = _dataset(
            queries=[],
            judgments=[
                Judgment("shared", "c1", Grade.DIRECTLY_ANSWERS),
                Judgment("json-only", "c2", Grade.DIRECTLY_ANSWERS),
            ],
        )
        pdf_side = _dataset(
            queries=[],
            judgments=[
                Judgment("shared", "p1", Grade.DIRECTLY_ANSWERS),
                Judgment("pdf-only", "p2", Grade.DIRECTLY_ANSWERS),
            ],
        )
        comparable = comparable_queries(json_side, pdf_side)
        assert comparable.shared == ("shared",)
        assert comparable.json_only == ("json-only",)
        assert comparable.pdf_only == ("pdf-only",)
        assert comparable.report == {
            "scored_on_both": 1,
            "judged_only_on_json": 1,
            "judged_only_on_pdf": 1,
        }

    def test_judged_query_ids_reads_the_judgments_not_the_queries(self) -> None:
        # A query can exist in a build with no judgments; scoring it would divide
        # by an empty relevant set.
        dataset = _dataset(
            queries=[Query("unjudged", "t", "p", Split.EVAL)],
            judgments=[Judgment("judged", "c", Grade.DIRECTLY_ANSWERS)],
        )
        assert judged_query_ids(dataset) == {"judged"}

    def test_each_side_is_scored_with_its_own_query_objects_in_order(self) -> None:
        # Chunk ids differ between builds, so a side must be scored against its own
        # judgments; the fixed order is what lets the paired bootstrap line them up.
        dataset = _dataset(
            queries=[Query(qid, "t", "p", Split.EVAL) for qid in ("b", "a", "c")],
            judgments=[],
        )
        assert [q.id for q in queries_by_id(dataset, ["a", "b"])] == ["a", "b"]

    def test_a_query_absent_from_a_build_is_refused(self) -> None:
        dataset = _dataset(queries=[Query("a", "t", "p", Split.EVAL)], judgments=[])
        with pytest.raises(ValueError, match="absent from this build"):
            queries_by_id(dataset, ["a", "missing"])


class TestDiagnosingAFailedMapping:
    """A bare unresolved count is uninterpretable; the causes are the finding."""

    def _diagnose(self, paragraph: str) -> dict[str, int]:
        row = _row("1234.5678")
        paper = paper_from_document(row, _document(paragraph), normaliser=normalise_for_pdf_match)
        return diagnose_mapping([row], {"1234.5678": paper}).as_dict()

    def test_an_exact_paragraph_is_counted_as_resolved(self) -> None:
        counts = self._diagnose("The system reached 91.2 accuracy on the benchmark.")
        assert counts["resolved_whole_block"] == 1

    def test_text_split_across_paragraphs_is_named_as_a_boundary_problem(self) -> None:
        # The corpus HAS the text; the matcher wants it as one block. A
        # boundary-insensitive matcher would recover these, so they are not a
        # parsing loss and must not be counted as one.
        counts = self._diagnose("Before. The system reached 91.2 accuracy on the benchmark. After.")
        assert counts["unresolved_boundaries_differ"] == 1
        assert counts["unresolved_absent_from_parse"] == 0

    def test_a_sentence_that_diverges_after_its_opening_is_named_as_such(self) -> None:
        # The real-world case, and the largest bucket in the actual run: QASPER's
        # text comes from arXiv LaTeX while GROBID reads the typeset PDF, so a
        # citation appears in different forms mid-sentence. The opening matches and
        # the string as a whole never will — no normaliser bridges content.
        evidence = (
            "The multilingual model uses word alignments between sentences in a "
            "parallel corpus to exploit role correspondence."
        )
        typeset = (
            "The multilingual model uses word alignments between sentences in a "
            "parallel corpus (Lang et al., 2011) to exploit role correspondence."
        )
        row = _row("1234.5678")
        row["qas"]["answers"][0]["answer"][0]["evidence"] = [evidence]
        paper = paper_from_document(row, _document(typeset), normaliser=normalise_for_pdf_match)

        counts = diagnose_mapping([row], {"1234.5678": paper}).as_dict()
        assert counts["unresolved_tail_differs"] == 1
        assert counts["unresolved_absent_from_parse"] == 0, (
            "the paragraph is present; calling this a parsing loss would blame the wrong component"
        )

    def test_missing_text_is_named_as_missing(self) -> None:
        counts = self._diagnose("An unrelated sentence about something else entirely.")
        assert counts["unresolved_absent_from_parse"] == 1

    def test_float_references_are_counted_apart_from_failures(self) -> None:
        # ~9% of QASPER evidence points at tables and figures. Counting those as
        # parsing failures would inflate the number this check reports.
        row = _row("1234.5678")
        row["qas"]["answers"][0]["answer"][0]["evidence"] = ["FLOAT SELECTED: Table 2 shows..."]
        paper = paper_from_document(row, _document("Body."), normaliser=normalise_for_pdf_match)
        counts = diagnose_mapping([row], {"1234.5678": paper}).as_dict()
        assert counts["float_reference"] == 1
        assert counts["unresolved_absent_from_parse"] == 0


class StubResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class StubClient:
    def __init__(self, content: bytes = b"%PDF-1.7 body") -> None:
        self.content = content
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def get(self, url: str, **kwargs: Any) -> StubResponse:
        self.calls.append((url, kwargs.get("headers", {})))
        return StubResponse(self.content)


class StubParser:
    version = "grobid/0.8.2.1-crf"

    def __init__(self) -> None:
        self.parsed: list[str] = []

    async def parse(self, pdf: bytes, *, filename: str = "") -> ParsedDocument:
        self.parsed.append(filename)
        return _document("Parsed body.")


class TestFetching:
    async def test_a_cached_paper_is_not_refetched(self, tmp_path: Path) -> None:
        (tmp_path / "1234.5678.pdf").write_bytes(b"%PDF cached")
        client = StubClient()
        assert await fetch_pdf("1234.5678", cache_dir=tmp_path, client=client) == b"%PDF cached"
        assert client.calls == [], "a cached paper must cost arXiv nothing"

    async def test_a_fetched_paper_is_cached_and_identifies_itself(self, tmp_path: Path) -> None:
        client = StubClient()
        content = await fetch_pdf("1234.5678", cache_dir=tmp_path, client=client)
        assert content == b"%PDF-1.7 body"
        assert (tmp_path / "1234.5678.pdf").read_bytes() == content
        url, headers = client.calls[0]
        assert url.endswith("1234.5678")
        assert headers["User-Agent"] == USER_AGENT

    async def test_a_holding_page_is_refused_rather_than_parsed(self, tmp_path: Path) -> None:
        # arXiv serves HTML for withdrawn papers. Handing that to GROBID produces
        # an empty parse, which would read as a catastrophic chunking regression
        # rather than as the fetch failure it is.
        client = StubClient(b"<!DOCTYPE html><title>arXiv</title>")
        with pytest.raises(ValueError, match="not a PDF"):
            await fetch_pdf("9999.9999", cache_dir=tmp_path, client=client)
        assert not (tmp_path / "9999.9999.pdf").exists(), "a non-PDF must not be cached"

    async def test_fetches_are_spaced_but_cached_papers_are_not(self, tmp_path: Path) -> None:
        # Ten PDFs is a small favour to ask of arXiv and a rude one to ask at once.
        (tmp_path / "cached.pdf").write_bytes(b"%PDF cached")
        slept: list[float] = []

        async def record(seconds: float) -> None:
            slept.append(seconds)

        parser = StubParser()
        original = asyncio.sleep
        try:
            asyncio.sleep = record  # type: ignore[assignment]
            await fetch_and_parse(
                ["first", "cached", "third"],
                cache_dir=tmp_path,
                client=StubClient(),
                parser=parser,
                delay_seconds=2.5,
            )
        finally:
            asyncio.sleep = original  # type: ignore[assignment]

        # No delay before the first request, none for the cached paper, one before
        # the third — which is uncached and not first.
        assert slept == [2.5]
        assert parser.parsed == ["first.pdf", "cached.pdf", "third.pdf"]
