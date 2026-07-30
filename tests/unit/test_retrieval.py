"""Retrieval logic tests: RRF, evidence assembly, citation resolution, abstention.

The most valuable test here is `test_fusion_is_immune_to_score_scale`: it encodes
*why* RRF was chosen over weighted score fusion (doc 6 §6.6). If someone later
"optimises" fusion to use raw scores, that test fails and explains itself.
"""

from __future__ import annotations

import pytest

from scholiast.domain.errors import InvariantViolation
from scholiast.domain.retrieval import (
    RRF_K,
    CitationStatus,
    EvidenceSet,
    Query,
    QueryIntent,
    RetrievalArm,
    ScoredChunk,
    reciprocal_rank_fusion,
)


def ranked(
    *ids: str, arm: RetrievalArm = RetrievalArm.DENSE, paper: str = "p-1", base: float = 1.0
) -> list[ScoredChunk]:
    """A ranked list, best-first, with descending synthetic scores."""
    return [
        ScoredChunk(chunk_id=cid, paper_id=paper, score=base - i * 0.01, arm=arm, rank=i + 1)
        for i, cid in enumerate(ids)
    ]


class TestReciprocalRankFusion:
    def test_agreed_top_result_wins(self) -> None:
        fused = reciprocal_rank_fusion(
            [
                ranked("a", "b", "c"),
                ranked("a", "c", "b", arm=RetrievalArm.LEXICAL),
            ]
        )
        assert fused[0].chunk_id == "a"

    def test_fusion_is_immune_to_score_scale(self) -> None:
        # THE reason RRF was chosen (ML-7). BM25 scores are unbounded while cosine
        # sits in ~0.6-0.9; a score-based fuser would let the lexical arm dominate
        # purely through magnitude. Rank-based fusion must not care.
        dense = ranked("a", "b", base=0.9)
        lexical_small = ranked("b", "a", arm=RetrievalArm.LEXICAL, base=0.8)
        lexical_huge = [
            ScoredChunk(c.chunk_id, c.paper_id, c.score * 10_000, c.arm, c.rank)
            for c in lexical_small
        ]

        assert [c.chunk_id for c in reciprocal_rank_fusion([dense, lexical_small])] == [
            c.chunk_id for c in reciprocal_rank_fusion([dense, lexical_huge])
        ]

    def test_chunk_in_both_lists_beats_a_chunk_in_one(self) -> None:
        # Corroboration across arms is the signal hybrid retrieval exists to use.
        fused = reciprocal_rank_fusion(
            [
                ranked("both", "dense-only"),
                ranked("both", "lexical-only", arm=RetrievalArm.LEXICAL),
            ]
        )
        assert fused[0].chunk_id == "both"

    def test_scores_match_the_rrf_formula(self) -> None:
        fused = reciprocal_rank_fusion([ranked("a", "b")])
        assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))
        assert fused[1].score == pytest.approx(1.0 / (RRF_K + 2))

    def test_output_is_marked_as_fused(self) -> None:
        # Arm provenance is what lets the eval harness attribute a win to an arm
        # rather than to "hybrid" as an opaque whole.
        fused = reciprocal_rank_fusion([ranked("a")])
        assert all(c.arm is RetrievalArm.FUSED for c in fused)
        assert fused[0].rank == 1

    def test_ties_break_deterministically(self) -> None:
        # FR-E4: identical input must produce identical output across runs.
        lists = [ranked("z"), ranked("a", arm=RetrievalArm.LEXICAL)]
        first = [c.chunk_id for c in reciprocal_rank_fusion(lists)]
        for _ in range(5):
            assert [c.chunk_id for c in reciprocal_rank_fusion(lists)] == first

    def test_limit_truncates_after_fusing(self) -> None:
        fused = reciprocal_rank_fusion([ranked("a", "b", "c", "d")], limit=2)
        assert len(fused) == 2
        assert fused[0].chunk_id == "a"

    def test_empty_input_yields_empty_output(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(InvariantViolation, match="k must be positive"):
            reciprocal_rank_fusion([ranked("a")], k=0)


class TestEvidenceAssembly:
    def test_labels_are_sequential_and_stable(self) -> None:
        ev = EvidenceSet.assemble(ranked("a", "b", "c"), limit=10)
        assert ev.labels == {"a": "C1", "b": "C2", "c": "C3"}

    def test_per_paper_cap_prevents_one_paper_dominating(self) -> None:
        # The QA-path echo of FR-C2. Without the cap, a verbose paper turns a
        # multi-paper workspace into a single-paper answer.
        verbose = ranked(*[f"v{i}" for i in range(10)], paper="verbose")
        other = ranked("o1", "o2", paper="other")
        ev = EvidenceSet.assemble([*verbose, *other], limit=6, per_paper_share=0.5)

        per_paper = {p: sum(1 for c in ev.chunks if c.paper_id == p) for p in ev.paper_ids}
        assert per_paper["verbose"] <= 3
        assert "other" in per_paper, "the quieter paper was crowded out entirely"

    def test_cap_is_not_applied_when_only_one_paper_is_in_scope(self) -> None:
        # Applying it would discard evidence for no reason.
        single = ranked(*[f"c{i}" for i in range(8)], paper="only")
        ev = EvidenceSet.assemble(single, limit=6, per_paper_share=0.5)
        assert len(ev.chunks) == 6

    def test_rerank_order_is_preserved_among_survivors(self) -> None:
        ev = EvidenceSet.assemble(ranked("a", "b", "c"), limit=3)
        assert [c.chunk_id for c in ev.chunks] == ["a", "b", "c"]

    def test_limit_must_be_positive(self) -> None:
        with pytest.raises(InvariantViolation, match="limit must be positive"):
            EvidenceSet.assemble(ranked("a"), limit=0)

    def test_disclosure_fields_are_carried(self) -> None:
        # Every degraded or partial answer must say so (NFR-R4, NFR-U4).
        ev = EvidenceSet.assemble(
            ranked("a"),
            limit=5,
            excluded_paper_ids=["still-ingesting"],
            degraded=["reranker_unavailable"],
        )
        assert ev.excluded_paper_ids == ("still-ingesting",)
        assert ev.degraded == ("reranker_unavailable",)

    def test_paper_ids_preserve_first_appearance_order(self) -> None:
        ev = EvidenceSet.assemble(
            [*ranked("b1", paper="second"), *ranked("a1", paper="first")], limit=5
        )
        assert ev.paper_ids == ("second", "first")


class TestCitationResolution:
    def test_label_resolves_to_its_chunk(self) -> None:
        ev = EvidenceSet.assemble(ranked("a", "b"), limit=5)
        assert ev.resolve("C1") == "a"
        assert ev.resolve("C2") == "b"

    def test_unknown_label_does_not_resolve(self) -> None:
        # Layer 2 of citation integrity: a fabricated label is caught
        # deterministically and for free (doc 6 §6.11).
        ev = EvidenceSet.assemble(ranked("a"), limit=5)
        assert ev.resolve("C99") is None

    def test_citation_status_gates_trust(self) -> None:
        from scholiast.domain.retrieval import Citation

        verified = Citation("C1", "a", "claim", CitationStatus.VERIFIED, 0.95)
        unverified = Citation("C1", "a", "claim", CitationStatus.UNVERIFIED, 0.3)
        fabricated = Citation("C9", "?", "claim", CitationStatus.FABRICATED)

        assert verified.is_trustworthy
        assert not unverified.is_trustworthy
        assert not fabricated.is_trustworthy


class TestAbstention:
    def test_empty_evidence_always_abstains(self) -> None:
        # FR-Q4: answering from parametric knowledge here is a correctness
        # failure, not a degraded experience.
        assert EvidenceSet().should_abstain(threshold=0.0)

    def test_weak_evidence_abstains(self) -> None:
        weak = [ScoredChunk("a", "p-1", 0.05, RetrievalArm.RERANKED)]
        assert EvidenceSet.assemble(weak, limit=5).should_abstain(threshold=0.4)

    def test_strong_evidence_answers(self) -> None:
        strong = [ScoredChunk("a", "p-1", 0.91, RetrievalArm.RERANKED)]
        assert not EvidenceSet.assemble(strong, limit=5).should_abstain(threshold=0.4)

    def test_threshold_uses_the_top_score_not_the_mean(self) -> None:
        # One strong piece of evidence is enough to answer; averaging would let a
        # long tail of weak chunks suppress a good answer.
        mixed = [
            ScoredChunk("strong", "p-1", 0.90, RetrievalArm.RERANKED),
            *[ScoredChunk(f"weak{i}", "p-1", 0.01, RetrievalArm.RERANKED) for i in range(9)],
        ]
        assert not EvidenceSet.assemble(mixed, limit=10).should_abstain(threshold=0.5)


class TestQueryValidation:
    def test_blank_query_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation, match="query text"):
            Query(text="  ", workspace_id="ws-1", paper_ids=("p-1",))

    def test_unscoped_query_is_rejected(self) -> None:
        # Guards the failure mode where an empty scope becomes a global search
        # across tenants.
        with pytest.raises(InvariantViolation, match="at least one paper"):
            Query(text="what is RAG?", workspace_id="ws-1", paper_ids=())

    def test_rewrite_is_detected(self) -> None:
        rewritten = Query(
            text="What ablations does the Lost-in-the-Middle paper report?",
            workspace_id="ws-1",
            paper_ids=("p-1",),
            original_text="what about their ablations?",
        )
        assert rewritten.was_rewritten

    def test_unrewritten_query_reports_no_rewrite(self) -> None:
        plain = Query(text="what is RAG?", workspace_id="ws-1", paper_ids=("p-1",))
        assert not plain.was_rewritten
        assert plain.intent is QueryIntent.LOOKUP
