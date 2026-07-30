"""Paper, section normalisation, and the privacy carve-out.

Two behaviours here are load-bearing beyond ordinary data modelling:

`normalise_section` is what makes attribution possible — if "Related Work" fails
to normalise, chunks from it lose their marker and a baseline's result reads as
this paper's finding.

`citation_impact_per_year` is the ranking detail from FR-D4. Raw citation counts
are confounded by age; without normalising, 2015 papers permanently bury strong
2025 work.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scholiast.domain.errors import InvariantViolation
from scholiast.domain.paper import (
    Chunk,
    ChunkRole,
    Paper,
    PaperVisibility,
    Provenance,
    SectionKind,
    normalise_section,
)


def paper(**kw: object) -> Paper:
    defaults: dict[str, object] = {
        "id": "p-1",
        "content_hash": "sha256:abc",
        "visibility": PaperVisibility.PUBLIC,
        "title": "Attention Is All You Need",
        "year": 2017,
    }
    return Paper(**(defaults | kw))  # type: ignore[arg-type]


class TestSectionNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Abstract", SectionKind.ABSTRACT),
            ("1 Introduction", SectionKind.INTRODUCTION),
            ("1. Introduction", SectionKind.INTRODUCTION),
            ("II. Introduction", SectionKind.INTRODUCTION),
            ("2 Related Work", SectionKind.RELATED_WORK),
            ("Prior Work", SectionKind.RELATED_WORK),
            ("Background", SectionKind.RELATED_WORK),
            ("3 Method", SectionKind.METHOD),
            ("Methodology", SectionKind.METHOD),
            ("Our Approach", SectionKind.METHOD),
            ("Proposed Method", SectionKind.METHOD),
            ("4 Experiments", SectionKind.EXPERIMENTS),
            ("Experimental Setup", SectionKind.EXPERIMENTS),
            ("Implementation Details", SectionKind.EXPERIMENTS),
            ("5 Results", SectionKind.RESULTS),
            ("Main Results", SectionKind.RESULTS),
            ("Ablation Study", SectionKind.DISCUSSION),
            ("Limitations", SectionKind.LIMITATIONS),
            ("Broader Impact", SectionKind.LIMITATIONS),
            ("6 Conclusion", SectionKind.CONCLUSION),
            ("Future Work", SectionKind.CONCLUSION),
            ("References", SectionKind.REFERENCES),
            ("A Appendix", SectionKind.APPENDIX),
            ("Supplementary Material", SectionKind.APPENDIX),
        ],
    )
    def test_real_world_headings_normalise(self, raw: str, expected: SectionKind) -> None:
        assert normalise_section(raw) == expected

    def test_longest_phrase_wins(self) -> None:
        # "related work" must beat a bare "work" match, and "experimental setup"
        # must beat "setup" — ordering in the pattern table is load-bearing.
        assert normalise_section("Related Work") is SectionKind.RELATED_WORK
        assert normalise_section("Experimental Setup") is SectionKind.EXPERIMENTS

    def test_unknown_heading_falls_back_rather_than_guessing(self) -> None:
        # OTHER is retrievable but carries no section prior — the safe default.
        assert normalise_section("Acknowledgements") is SectionKind.OTHER
        assert normalise_section("Zorblax Protocol") is SectionKind.OTHER

    def test_normalisation_is_case_insensitive(self) -> None:
        assert normalise_section("RELATED WORK") is SectionKind.RELATED_WORK
        assert normalise_section("related work") is SectionKind.RELATED_WORK


class TestAttributionSignals:
    def test_related_work_and_references_describe_other_work(self) -> None:
        # The single most important property in this module: it is what stops a
        # cited baseline's result being attributed to the paper being read.
        assert SectionKind.RELATED_WORK.describes_other_work
        assert SectionKind.REFERENCES.describes_other_work

    @pytest.mark.parametrize(
        "kind",
        [
            SectionKind.METHOD,
            SectionKind.RESULTS,
            SectionKind.EXPERIMENTS,
            SectionKind.CONCLUSION,
            SectionKind.ABSTRACT,
        ],
    )
    def test_own_contribution_sections_are_not_marked(self, kind: SectionKind) -> None:
        assert not kind.describes_other_work

    def test_back_matter_is_not_body(self) -> None:
        assert not SectionKind.REFERENCES.is_body
        assert not SectionKind.APPENDIX.is_body
        assert SectionKind.METHOD.is_body


class TestPrivacyCarveOut:
    def test_private_paper_requires_an_owner(self) -> None:
        # Doc 4 §4.4.2: an unpublished manuscript must never become a globally
        # shared registry entry. Encoded as an invariant, mirrored by partial
        # unique indexes in the schema so one layer's bug cannot break it alone.
        with pytest.raises(InvariantViolation, match="private papers must have an owner"):
            paper(visibility=PaperVisibility.PRIVATE, owner_id=None)

    def test_public_paper_must_not_be_owner_scoped(self) -> None:
        with pytest.raises(InvariantViolation, match="must not be owner-scoped"):
            paper(visibility=PaperVisibility.PUBLIC, owner_id="user-1")

    def test_only_public_papers_are_globally_shared(self) -> None:
        assert paper().is_globally_shared
        assert not paper(visibility=PaperVisibility.PRIVATE, owner_id="user-1").is_globally_shared

    def test_blank_title_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation, match="title"):
            paper(title="   ")


class TestAgeNormalisedCitationImpact:
    def test_older_paper_with_more_citations_can_lose_to_a_recent_one(self) -> None:
        # FR-D4: this is the whole reason the normalisation exists.
        current_year = datetime.now(tz=UTC).year
        old = paper(year=current_year - 10, citation_count=1000)
        recent = paper(year=current_year - 1, citation_count=400)

        assert old.citation_count > recent.citation_count  # type: ignore[operator]
        assert recent.citation_impact_per_year > old.citation_impact_per_year  # type: ignore[operator]

    def test_current_year_paper_does_not_divide_by_zero(self) -> None:
        p = paper(year=datetime.now(tz=UTC).year, citation_count=10)
        impact = p.citation_impact_per_year
        assert impact is not None
        assert impact == pytest.approx(20.0)  # 10 citations / 0.5 years

    @pytest.mark.parametrize(
        ("year", "count"),
        [(None, 100), (2020, None), (None, None)],
    )
    def test_missing_signal_returns_none_rather_than_zero(
        self, year: int | None, count: int | None
    ) -> None:
        # Defaulting an unknown signal to zero would be a silent ranking penalty
        # for papers with incomplete metadata; the ranker must handle absence.
        assert paper(year=year, citation_count=count).citation_impact_per_year is None


class TestChunkInvariants:
    def _provenance(self) -> Provenance:
        return Provenance(
            paper_id="p-1",
            section_path="Method",
            section_kind=SectionKind.METHOD,
            page=3,
            char_start=100,
            char_end=400,
        )

    def test_blank_chunk_text_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation, match="blank"):
            Chunk(
                id="c1",
                text="   ",
                role=ChunkRole.PARENT,
                ordinal=0,
                provenance=self._provenance(),
                chunker_version="v1",
                token_count=1,
            )

    def test_child_without_parent_is_rejected(self) -> None:
        # An orphaned child would make parent expansion fail silently at query
        # time, quietly shrinking generation context.
        with pytest.raises(InvariantViolation, match="parent window"):
            Chunk(
                id="c1",
                text="text",
                role=ChunkRole.CHILD,
                ordinal=0,
                provenance=self._provenance(),
                chunker_version="v1",
                token_count=1,
            )

    def test_embedding_stamp_is_immutable_and_returns_a_copy(self) -> None:
        original = Chunk(
            id="c1",
            text="text",
            role=ChunkRole.PARENT,
            ordinal=0,
            provenance=self._provenance(),
            chunker_version="v1",
            token_count=1,
        )
        assert not original.is_indexed
        stamped = original.with_embedding_version("bge-base/1.5")
        assert stamped.is_indexed
        assert stamped.embedder_version == "bge-base/1.5"
        assert not original.is_indexed, "original chunk was mutated"

    def test_provenance_rejects_inverted_offsets(self) -> None:
        with pytest.raises(InvariantViolation, match="char_end"):
            Provenance(
                paper_id="p",
                section_path="M",
                section_kind=SectionKind.METHOD,
                page=1,
                char_start=400,
                char_end=100,
            )

    def test_provenance_requires_one_indexed_pages(self) -> None:
        with pytest.raises(InvariantViolation, match="1-indexed"):
            Provenance(
                paper_id="p",
                section_path="M",
                section_kind=SectionKind.METHOD,
                page=0,
                char_start=0,
                char_end=10,
            )
