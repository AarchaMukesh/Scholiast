"""Chunker tests — the three rules from doc 6 §6.3, asserted rather than assumed.

Each test names the failure it prevents. A test called `test_chunking_works` tells
a future reader nothing about why the behaviour matters.
"""

from __future__ import annotations

import itertools

import pytest

from scholiast.domain.chunking import (
    CHUNKER_VERSION,
    BlockKind,
    ChunkingConfig,
    ParsedBlock,
    ParsedSection,
    StructureAwareChunker,
    approximate_tokens,
)
from scholiast.domain.errors import InvariantViolation
from scholiast.domain.paper import ChunkRole, SectionKind

PAPER = "paper-1"


def block(
    text: str, *, kind: BlockKind = BlockKind.PARAGRAPH, page: int = 1, start: int = 0
) -> ParsedBlock:
    return ParsedBlock(
        kind=kind, text=text, page=page, char_start=start, char_end=start + len(text)
    )


def prose(n_sentences: int, *, word: str = "retrieval") -> str:
    return " ".join(
        f"Sentence {i} discusses {word} in detail and adds enough words to matter."
        for i in range(n_sentences)
    )


def section(kind: SectionKind, text: str, *, path: str | None = None) -> ParsedSection:
    return ParsedSection(kind=kind, path=path or kind.value.title(), blocks=(block(text),))


class TestSectionBoundaries:
    """Rule 1: a chunk must never span two sections."""

    def test_chunks_never_span_sections(self) -> None:
        # Prevents: a chunk straddling Related Work into Method, which loses the
        # signal that the first half describes someone else's contribution.
        sections = [
            ParsedSection(
                SectionKind.RELATED_WORK,
                "Related Work",
                (block("Prior work established baselines.", start=0),),
            ),
            ParsedSection(
                SectionKind.METHOD, "Method", (block("We propose a new approach.", start=100),)
            ),
        ]
        result = StructureAwareChunker().chunk(PAPER, sections)

        for chunk in result.chunks:
            text = chunk.text
            assert not ("Prior work" in text and "We propose" in text), (
                "a chunk merged content from two different sections"
            )

    def test_each_chunk_carries_its_own_section_kind(self) -> None:
        sections = [
            section(SectionKind.RELATED_WORK, prose(3)),
            section(SectionKind.RESULTS, prose(3)),
        ]
        result = StructureAwareChunker().chunk(PAPER, sections)

        kinds = {c.provenance.section_kind for c in result.chunks}
        assert kinds == {SectionKind.RELATED_WORK, SectionKind.RESULTS}

    def test_non_body_sections_are_skipped_by_default(self) -> None:
        sections = [
            section(SectionKind.METHOD, prose(3)),
            section(SectionKind.REFERENCES, prose(20)),
            section(SectionKind.APPENDIX, prose(20)),
        ]
        result = StructureAwareChunker().chunk(PAPER, sections)

        assert result.sections_processed == 1
        assert result.sections_skipped == 2

    def test_non_body_sections_included_when_configured(self) -> None:
        sections = [section(SectionKind.APPENDIX, prose(3))]
        chunker = StructureAwareChunker(ChunkingConfig(skip_non_body=False))
        assert chunker.chunk(PAPER, sections).sections_processed == 1


class TestParentChildWindows:
    """Rule 2: small-to-big — children are searched, parents are returned."""

    def test_children_are_smaller_than_their_parent(self) -> None:
        result = StructureAwareChunker().chunk(PAPER, [section(SectionKind.METHOD, prose(60))])
        assert result.parents, "expected at least one parent window"
        assert result.children, "expected child chunks"

        by_parent: dict[str, list[int]] = {}
        for child in result.children:
            assert child.parent_id is not None
            by_parent.setdefault(child.parent_id, []).append(child.token_count)

        for parent in result.parents:
            for child_tokens in by_parent.get(parent.id, []):
                assert child_tokens <= parent.token_count

    def test_every_child_references_an_existing_parent(self) -> None:
        # Prevents: orphaned children, which would make parent expansion fail
        # silently at query time and quietly shrink generation context.
        result = StructureAwareChunker().chunk(PAPER, [section(SectionKind.METHOD, prose(80))])
        parent_ids = {p.id for p in result.parents}
        for child in result.children:
            assert child.parent_id in parent_ids

    def test_parents_are_not_embedded(self) -> None:
        # Embedding parents would roughly double index size for no retrieval
        # gain — they exist to be returned, not matched.
        result = StructureAwareChunker().chunk(PAPER, [section(SectionKind.METHOD, prose(60))])
        assert all(c.role is not ChunkRole.PARENT for c in result.embeddable)

    def test_child_chunks_overlap(self) -> None:
        # Prevents: a claim split across a boundary appearing in neither chunk.
        result = StructureAwareChunker(
            ChunkingConfig(
                child_target_tokens=60,
                parent_target_tokens=600,
                overlap_ratio=0.3,
                min_child_tokens=10,
            )
        ).chunk(PAPER, [section(SectionKind.METHOD, prose(40))])

        children = result.children
        assert len(children) >= 2, "need multiple children to test overlap"

        overlaps = 0
        for first, second in itertools.pairwise(children):
            tail_words = set(first.text.split()[-12:])
            head_words = set(second.text.split()[:12])
            if tail_words & head_words:
                overlaps += 1
        assert overlaps > 0, "no sibling overlap found"

    def test_no_orphaned_short_fragment(self) -> None:
        # Prevents: a 20-token trailing chunk with too little context to retrieve.
        cfg = ChunkingConfig(child_target_tokens=100, parent_target_tokens=900, min_child_tokens=40)
        result = StructureAwareChunker(cfg).chunk(PAPER, [section(SectionKind.METHOD, prose(25))])
        children = result.children
        if len(children) > 1:
            assert all(c.token_count >= cfg.min_child_tokens for c in children[:-1])


class TestTablesAreAtomic:
    """Rule 3: tables hold the facts users ask for, and must survive intact."""

    def test_table_becomes_its_own_chunk(self) -> None:
        table = "| Model | nDCG@10 |\n| BM25 | 0.61 |\n| Hybrid | 0.74 |"
        sections = [
            ParsedSection(
                SectionKind.RESULTS,
                "Results",
                (
                    block("We report results below.", start=0),
                    block(table, kind=BlockKind.TABLE, start=100),
                ),
            )
        ]
        result = StructureAwareChunker().chunk(PAPER, sections)

        tables = [c for c in result.chunks if c.role is ChunkRole.TABLE]
        assert len(tables) == 1
        assert tables[0].text == table, "table text was altered"
        assert result.tables_preserved == 1

    def test_table_is_not_merged_into_prose_children(self) -> None:
        # Prevents: a table fragment inside a prose chunk, where a split mid-row
        # produces confidently wrong numbers.
        table = "| A | B |\n| 1 | 2 |"
        sections = [
            ParsedSection(
                SectionKind.RESULTS,
                "Results",
                (block(prose(5), start=0), block(table, kind=BlockKind.TABLE, start=1000)),
            )
        ]
        result = StructureAwareChunker().chunk(PAPER, sections)

        for child in result.children:
            assert "| 1 | 2 |" not in child.text

    def test_table_larger_than_a_window_still_survives_whole(self) -> None:
        big_table = "\n".join(f"| row {i} | value {i} |" for i in range(400))
        sections = [
            ParsedSection(
                SectionKind.RESULTS,
                "Results",
                (block(big_table, kind=BlockKind.TABLE, start=0),),
            )
        ]
        result = StructureAwareChunker().chunk(PAPER, sections)

        tables = [c for c in result.chunks if c.role is ChunkRole.TABLE]
        assert len(tables) == 1
        assert tables[0].text == big_table


class TestContextualHeaders:
    """ML-3: the cheapest quality win — and it must actually mark attribution."""

    def test_header_names_paper_and_section(self) -> None:
        result = StructureAwareChunker().chunk(PAPER, [section(SectionKind.METHOD, prose(5))])
        header = result.children[0].contextual_header("Attention Is All You Need", 2017)
        assert "Attention Is All You Need" in header
        assert "2017" in header
        assert "Method" in header

    def test_related_work_chunks_are_marked_as_other_work(self) -> None:
        # This is the defence against the misattribution failure in doc 2 §2.3:
        # without the marker, a baseline's result reads as this paper's finding.
        result = StructureAwareChunker().chunk(PAPER, [section(SectionKind.RELATED_WORK, prose(5))])
        header = result.children[0].contextual_header("Some Paper", 2024)
        assert "other work" in header.lower()

    def test_method_chunks_are_not_marked_as_other_work(self) -> None:
        result = StructureAwareChunker().chunk(PAPER, [section(SectionKind.METHOD, prose(5))])
        header = result.children[0].contextual_header("Some Paper", 2024)
        assert "other work" not in header.lower()

    def test_header_omits_year_when_unknown(self) -> None:
        result = StructureAwareChunker().chunk(PAPER, [section(SectionKind.METHOD, prose(5))])
        header = result.children[0].contextual_header("Untitled Preprint", None)
        assert "None" not in header


def multipage_section(pages: tuple[int, ...] = (4, 5, 6)) -> tuple[ParsedSection, str]:
    """A section spanning several pages, plus the canonical text its offsets index."""
    blocks: list[ParsedBlock] = []
    cursor = 0
    for page in pages:
        text = f"Sentence about retrieval quality on page {page} with enough words. " * 20
        blocks.append(block(text, page=page, start=cursor))
        cursor += len(text)
    document = "".join(b.text for b in blocks)
    return ParsedSection(SectionKind.METHOD, "Method", tuple(blocks)), document


class TestChildProvenanceIsItsOwn:
    """A citation must point at the chunk that was cited, not at its neighbourhood."""

    def test_child_span_is_narrower_than_its_parent_window(self) -> None:
        # Prevents: every child in a window inheriting the window's span, so
        # clicking a citation highlights ~1000 tokens of which ~300 are the
        # actual evidence. FR-I5 asks for the span, not the vicinity.
        sec, _ = multipage_section()
        result = StructureAwareChunker().chunk(PAPER, [sec])
        parent = result.parents[0]
        children = result.children
        assert len(children) > 1, "need several children for this to mean anything"
        for child in children:
            assert child.provenance.length < parent.provenance.length

    def test_children_do_not_all_share_one_span(self) -> None:
        # Prevents the specific regression: identical provenance on every child,
        # which looks correct in isolation and is wrong for all but one of them.
        sec, _ = multipage_section()
        children = StructureAwareChunker().chunk(PAPER, [sec]).children
        spans = {(c.provenance.char_start, c.provenance.char_end) for c in children}
        assert len(spans) == len(children)

    def test_child_page_is_where_the_chunk_starts(self) -> None:
        # Prevents: sending the reader to the first page of the window when the
        # cited text is three pages further in.
        sec, _ = multipage_section()
        children = StructureAwareChunker().chunk(PAPER, [sec]).children
        assert {c.provenance.page for c in children} != {4}, (
            "every child reported the window's first page — pages are not being tracked"
        )

    def test_child_text_is_recoverable_from_its_span(self) -> None:
        # FR-I5's round trip: given a chunk, locate the exact source text. The
        # contract is containment, not equality — joining sentences normalises
        # inter-sentence whitespace — but every sentence must lie inside.
        sec, document = multipage_section()
        for child in StructureAwareChunker().chunk(PAPER, [sec]).children:
            source = document[child.provenance.char_start : child.provenance.char_end]
            for sentence in child.text.split(". "):
                assert sentence.strip(". ") in source

    def test_overlapping_children_have_overlapping_spans(self) -> None:
        # The 15% overlap exists so a claim straddling a boundary survives. If
        # the spans did not overlap too, the offsets would be describing a
        # different chunking than the one that was embedded.
        sec, _ = multipage_section()
        children = StructureAwareChunker().chunk(PAPER, [sec]).children
        pairs = list(itertools.pairwise(children))
        assert any(b.provenance.char_start < a.provenance.char_end for a, b in pairs)


class TestBlockOffsetsAreTrustworthy:
    def test_block_offsets_must_match_its_own_text(self) -> None:
        # Prevents a parser recording offsets against the raw PDF text while
        # handing over normalised text: chunk spans are derived by adding
        # within-block positions to char_start, so the two must describe the
        # same string. A mismatch would shift every citation silently.
        with pytest.raises(InvariantViolation, match="canonical parsed text"):
            ParsedBlock(
                kind=BlockKind.PARAGRAPH,
                text="short text",
                page=1,
                char_start=0,
                char_end=9999,
            )


class TestProvenanceAndVersioning:
    def test_every_chunk_carries_resolvable_provenance(self) -> None:
        # FR-I5: without this a citation is a gesture, not a reference.
        result = StructureAwareChunker().chunk(PAPER, [section(SectionKind.METHOD, prose(30))])
        for chunk in result.chunks:
            p = chunk.provenance
            assert p.paper_id == PAPER
            assert p.page >= 1
            assert p.char_end > p.char_start
            assert p.section_path

    def test_every_chunk_is_version_stamped(self) -> None:
        # NFR-M5: this is what makes "was it the reranker or the re-chunk?"
        # answerable.
        result = StructureAwareChunker().chunk(PAPER, [section(SectionKind.METHOD, prose(10))])
        assert result.chunks
        assert all(c.chunker_version == CHUNKER_VERSION for c in result.chunks)

    def test_chunk_ids_are_unique(self) -> None:
        result = StructureAwareChunker().chunk(
            PAPER,
            [
                section(SectionKind.METHOD, prose(40)),
                section(SectionKind.RESULTS, prose(40), path="Results"),
            ],
        )
        ids = [c.id for c in result.chunks]
        assert len(ids) == len(set(ids))

    def test_chunking_is_deterministic(self) -> None:
        # FR-E4: eval reproducibility depends on identical input giving identical
        # chunks.
        sections = [section(SectionKind.METHOD, prose(40))]
        a = StructureAwareChunker().chunk(PAPER, sections)
        b = StructureAwareChunker().chunk(PAPER, sections)
        assert [c.id for c in a.chunks] == [c.id for c in b.chunks]
        assert [c.text for c in a.chunks] == [c.text for c in b.chunks]


class TestConfigValidation:
    def test_parent_must_exceed_child(self) -> None:
        with pytest.raises(InvariantViolation, match="small-to-big"):
            ChunkingConfig(child_target_tokens=500, parent_target_tokens=300)

    @pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
    def test_overlap_ratio_must_be_a_fraction(self, ratio: float) -> None:
        with pytest.raises(InvariantViolation, match="overlap_ratio"):
            ChunkingConfig(overlap_ratio=ratio)

    def test_min_child_must_be_below_target(self) -> None:
        with pytest.raises(InvariantViolation, match="min_child_tokens"):
            ChunkingConfig(child_target_tokens=100, min_child_tokens=100)


class TestEdgeCases:
    def test_empty_document_produces_no_chunks(self) -> None:
        assert StructureAwareChunker().chunk(PAPER, []).chunks == []

    def test_empty_sections_are_ignored(self) -> None:
        sections = [ParsedSection(SectionKind.METHOD, "Method", ())]
        assert StructureAwareChunker().chunk(PAPER, sections).chunks == []

    def test_single_short_paragraph_yields_one_child(self) -> None:
        sections = [section(SectionKind.METHOD, "A short method description.")]
        result = StructureAwareChunker().chunk(PAPER, sections)
        assert len(result.children) == 1
        assert len(result.parents) == 1

    def test_block_with_no_sentence_punctuation_is_not_lost(self) -> None:
        sections = [section(SectionKind.METHOD, "no punctuation here at all")]
        result = StructureAwareChunker().chunk(PAPER, sections)
        assert any("no punctuation" in c.text for c in result.children)

    def test_injected_token_counter_is_used(self) -> None:
        calls: list[str] = []

        def counting(text: str) -> int:
            calls.append(text)
            return approximate_tokens(text)

        StructureAwareChunker(token_counter=counting).chunk(
            PAPER, [section(SectionKind.METHOD, prose(10))]
        )
        assert calls, "injected token counter was never called"
