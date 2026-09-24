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
    Citation,
    CitationStatus,
    EvidenceSet,
    Query,
    QueryIntent,
    RetrievalArm,
    ScoredChunk,
    audit_citations,
    reciprocal_rank_fusion,
    strip_labels,
    weighted_score_fusion,
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


class TestWeightedScoreFusion:
    """RRF's ablation arm (doc 6 §6.6, D76) — the one that keeps the scores.

    Doc 6 rejects raw score fusion because cosine and BM25 scales are not
    comparable, and keeps weighted fusion as an ablation only because
    normalisation can remove the mismatch. These tests are about whether it
    actually does: the class exists to be *compared* with RRF, so the properties
    RRF is chosen for are the ones worth pinning here.
    """

    def test_normalisation_makes_it_immune_to_score_scale(self) -> None:
        # The counterpart of test_fusion_is_immune_to_score_scale. Raw weighted
        # fusion would hand the whole ranking to whichever arm has the larger
        # numbers; per-query min-max is exactly what buys back scale immunity, so
        # if someone drops the normalisation this fails and says why.
        dense = ranked("a", "b", base=0.9)
        lexical_small = ranked("b", "a", arm=RetrievalArm.LEXICAL, base=0.8)
        lexical_huge = [
            ScoredChunk(c.chunk_id, c.paper_id, c.score * 10_000, c.arm, c.rank)
            for c in lexical_small
        ]
        weights = (0.5, 0.5)

        assert [
            c.chunk_id for c in weighted_score_fusion([dense, lexical_small], weights=weights)
        ] == [c.chunk_id for c in weighted_score_fusion([dense, lexical_huge], weights=weights)]

    def test_a_full_weight_reproduces_that_arm_alone(self) -> None:
        # The degenerate ends of the grid are in the sweep on purpose (D76): a fit
        # landing on 0.0 or 1.0 is the finding that fusion added nothing.
        dense = ranked("a", "b", "c")
        lexical = ranked("c", "b", "a", arm=RetrievalArm.LEXICAL)
        dense_only = weighted_score_fusion([dense, lexical], weights=(1.0, 0.0))
        lexical_only = weighted_score_fusion([dense, lexical], weights=(0.0, 1.0))

        assert [c.chunk_id for c in dense_only] == ["a", "b", "c"]
        assert [c.chunk_id for c in lexical_only] == ["c", "b", "a"]

    def test_the_weight_actually_shifts_the_ranking(self) -> None:
        # Without this, every weight on the grid could be scoring the same order
        # and the fit would be meaningless.
        dense = ranked("d-top", "shared")
        lexical = ranked("l-top", "shared", arm=RetrievalArm.LEXICAL)
        leans_dense = weighted_score_fusion([dense, lexical], weights=(0.9, 0.1))
        leans_lexical = weighted_score_fusion([dense, lexical], weights=(0.1, 0.9))

        assert leans_dense[0].chunk_id == "d-top"
        assert leans_lexical[0].chunk_id == "l-top"

    def test_a_chunk_missing_from_one_arm_scores_zero_there_rather_than_vanishing(self) -> None:
        # The union, not the intersection: single-arm hits are the majority, and
        # dropping them would make fusion a filter instead of a combiner.
        fused = weighted_score_fusion(
            [ranked("dense-only"), ranked("lexical-only", arm=RetrievalArm.LEXICAL)],
            weights=(0.5, 0.5),
        )
        assert sorted(c.chunk_id for c in fused) == ["dense-only", "lexical-only"]

    def test_a_single_hit_takes_its_arms_full_weight(self) -> None:
        # min and max coincide, so the normalisation is undefined and the choice
        # is visible here: 1.0, not 0.0. At 0.0 an arm returning one strong
        # candidate would contribute nothing at all.
        fused = weighted_score_fusion(
            [ranked("only"), ranked("other", arm=RetrievalArm.LEXICAL)], weights=(0.7, 0.3)
        )
        by_id = {c.chunk_id: c.score for c in fused}
        assert by_id["only"] == pytest.approx(0.7)
        assert by_id["other"] == pytest.approx(0.3)

    def test_identical_scores_within_an_arm_do_not_collapse_to_zero(self) -> None:
        flat = [
            ScoredChunk(chunk_id=cid, paper_id="p-1", score=0.5, arm=RetrievalArm.DENSE, rank=i)
            for i, cid in enumerate(("a", "b"), start=1)
        ]
        fused = weighted_score_fusion([flat], weights=(1.0,))
        assert [c.score for c in fused] == [pytest.approx(1.0), pytest.approx(1.0)]

    def test_output_is_marked_as_fused_and_ranked(self) -> None:
        fused = weighted_score_fusion([ranked("a", "b")], weights=(1.0,))
        assert all(c.arm is RetrievalArm.FUSED for c in fused)
        assert [c.rank for c in fused] == [1, 2]

    def test_ties_break_deterministically(self) -> None:
        # FR-E4 again, and more likely here than under RRF: normalised scores
        # collide whenever two arms return the same-shaped list.
        lists = [ranked("z"), ranked("a", arm=RetrievalArm.LEXICAL)]
        first = [c.chunk_id for c in weighted_score_fusion(lists, weights=(0.5, 0.5))]
        assert first == ["a", "z"]
        for _ in range(5):
            assert [c.chunk_id for c in weighted_score_fusion(lists, weights=(0.5, 0.5))] == first

    def test_limit_truncates_after_fusing(self) -> None:
        fused = weighted_score_fusion([ranked("a", "b", "c", "d")], weights=(1.0,), limit=2)
        assert [c.chunk_id for c in fused] == ["a", "b"]

    def test_one_weight_per_ranking_is_required(self) -> None:
        with pytest.raises(InvariantViolation, match="one weight per ranking"):
            weighted_score_fusion([ranked("a"), ranked("b")], weights=(1.0,))

    def test_negative_weights_are_refused(self) -> None:
        # A negative weight would rank a chunk higher for being irrelevant to one
        # arm, which is not a configuration anyone means to run.
        with pytest.raises(InvariantViolation, match="non-negative"):
            weighted_score_fusion([ranked("a")], weights=(-0.5,))

    def test_silencing_every_arm_is_refused(self) -> None:
        # All-zero weights rank the whole union by chunk_id, which looks like a
        # result rather than a misconfiguration.
        with pytest.raises(InvariantViolation, match="at least one fusion weight"):
            weighted_score_fusion([ranked("a"), ranked("b")], weights=(0.0, 0.0))


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
        verified = Citation("C1", "a", "claim", CitationStatus.VERIFIED, 0.95)
        unverified = Citation("C1", "a", "claim", CitationStatus.UNVERIFIED, 0.3)
        fabricated = Citation("C9", None, "claim", CitationStatus.FABRICATED)

        assert verified.is_trustworthy
        assert not unverified.is_trustworthy
        assert not fabricated.is_trustworthy

    def test_fabricated_citation_cannot_carry_a_chunk_id(self) -> None:
        # Prevents a sentinel chunk id ("?", "", "unknown") on a fabricated
        # citation, which a caller that checks `chunk_id is not None` instead of
        # `status` would treat as a real reference.
        with pytest.raises(InvariantViolation, match="fabricated"):
            Citation("C9", "chunk-1", "claim", CitationStatus.FABRICATED)

    def test_resolved_citation_must_name_its_chunk(self) -> None:
        with pytest.raises(InvariantViolation, match="fabricated"):
            Citation("C1", None, "claim", CitationStatus.UNVERIFIED)


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


class TestCitationAudit:
    """Layer 2: every emitted label is checked, never trusted (FR-Q2)."""

    def evidence(self) -> EvidenceSet:
        return EvidenceSet.assemble(ranked("chunk-a", "chunk-b"), limit=5)

    def test_fabricated_label_is_caught(self) -> None:
        # The M1 acceptance criterion in one test: "every emitted citation ID
        # exists in the retrieved context (validated, not trusted)". A model
        # citing [C7] over a two-chunk context is the crudest failure there is,
        # and catching it costs nothing.
        audit = audit_citations("Transformers scale well [C7].", self.evidence())
        assert not audit.is_clean
        assert audit.fabricated_labels == ("C7",)
        assert audit.fabricated[0].chunk_id is None

    def test_resolved_label_is_not_promoted_to_verified(self) -> None:
        # Prevents laundering a syntactic check as a semantic one. Passing layer
        # 2 proves the label exists, not that the chunk supports the claim —
        # that is layer 3's job, and marking these VERIFIED would mean the
        # groundedness number counts unchecked claims.
        audit = audit_citations("Attention is quadratic [C1].", self.evidence())
        assert audit.citations[0].status is CitationStatus.UNVERIFIED
        assert not audit.citations[0].is_trustworthy

    def test_comma_grouped_labels_are_not_dropped(self) -> None:
        # Prevents the silent failure: a model emits [C1, C2], a strict regex
        # matches neither, and two real citations vanish from the answer with no
        # error anywhere. Losing a citation is worse than accepting sloppy
        # syntax, because nothing surfaces it.
        audit = audit_citations("Both report gains [C1, C2].", self.evidence())
        assert [c.label for c in audit.citations] == ["C1", "C2"]
        assert audit.is_clean

    def test_one_label_cited_for_two_claims_yields_two_citations(self) -> None:
        # Layer 3 judges (evidence, claim) pairs, so the same chunk supporting
        # two different claims has to be two entailment checks, not one.
        answer = "It uses masked LM [C1]. It trains on BooksCorpus [C1]."
        audit = audit_citations(answer, self.evidence())
        assert len(audit.citations) == 2
        assert {c.claim_text for c in audit.citations} == {
            "It uses masked LM [C1].",
            "It trains on BooksCorpus [C1].",
        }

    def test_claim_text_is_the_sentence_that_cited_it(self) -> None:
        answer = "Retrieval is hard. BM25 beats dense on rare entities [C2]. That matters."
        audit = audit_citations(answer, self.evidence())
        assert audit.citations[0].claim_text == "BM25 beats dense on rare entities [C2]."

    def test_uncited_answer_is_clean_but_that_is_not_groundedness(self) -> None:
        # `is_clean` means "no fabricated labels", NOT "adequately cited". An
        # answer with no citations at all passes layer 2 trivially — it satisfies
        # FR-Q2 (citation *validity*) while saying nothing about FR-Q11 (citation
        # *coverage*), which is measured in M5 (D66). Conflating them would let
        # an uncited answer report as verified.
        audit = audit_citations("Transformers are popular.", self.evidence())
        assert audit.is_clean
        assert audit.citations == ()

    def test_fabricated_labels_are_stripped_without_leaving_artifacts(self) -> None:
        # Showing [C7] invites a click that cannot resolve. Stripping it must not
        # leave " ." behind, which reads to a user as a rendering bug rather than
        # as a removed citation.
        answer = "Masked LM works [C1]. MMLU is 86.4 [C7]."
        audit = audit_citations(answer, self.evidence())
        assert strip_labels(answer, audit.fabricated_labels) == (
            "Masked LM works [C1]. MMLU is 86.4."
        )

    def test_stripping_keeps_the_valid_half_of_a_mixed_marker(self) -> None:
        # A marker holding one real and one fabricated label must lose only the
        # fabricated one — dropping the whole marker would discard a citation
        # that resolved perfectly well.
        answer = "Both agree [C1, C9]."
        audit = audit_citations(answer, self.evidence())
        assert strip_labels(answer, audit.fabricated_labels) == "Both agree [C1]."

    def test_stripping_nothing_leaves_the_answer_alone(self) -> None:
        answer = "Masked LM works [C1]."
        assert strip_labels(answer, ()) == answer
