"""Production's hybrid retrieval and the guards around it, against fakes (D167, D171, D172).

Phase A ships the code with today's behaviour as the default, so two kinds of test
live here: that dense mode is exactly what it was, and that hybrid mode - once it
is switched on - reads both arms over the same tenant-derived scope, refuses a
collection with holes in it, and records what it did. Each test names the failure
it prevents.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import pytest

from scholiast.application.ports.vector import VectorRecord
from scholiast.application.use_cases.answer_question import (
    DEFAULT_CANDIDATES,
    HYBRID_DEPTH,
    NO_EVIDENCE_ANSWER,
    AnswerQuestionUseCase,
    EvidenceReady,
)
from scholiast.application.use_cases.ingest_paper import IngestPaperUseCase
from scholiast.application.use_cases.reconcile import ReconcileUseCase
from scholiast.application.use_cases.verify_hybrid_index import (
    HybridIndexIncomplete,
    VerifyHybridIndexUseCase,
)
from scholiast.domain.errors import InvariantViolation
from scholiast.domain.lexical import LEXICAL_VERSION, chunk_weights, query_weights, tokenize
from scholiast.domain.retrieval import (
    EvidenceSet,
    RetrievalArm,
    RetrievalMode,
    ScoredChunk,
    reciprocal_rank_fusion,
)
from scholiast.infrastructure.db.repositories import _evidence_from_json, _evidence_to_json
from scholiast.infrastructure.vector.mirrored import MirroredVectorIndex
from tests.unit import test_ingestion as ingestion
from tests.unit import test_rebuild_index as rebuild
from tests.unit.test_answering import (
    OTHER_PAPER,
    PAPER,
    FakeChunks,
    FakeConversations,
    FakeEmbedder,
    FakePapers,
    FakeProvider,
    FakeWorkspaces,
    chunk,
    collect,
    final,
    workspace,
)

# --- fakes ----------------------------------------------------------------


class RecordingIndex:
    """A dense arm that applies the filter it is given and records every call."""

    def __init__(self, hits: Sequence[ScoredChunk]) -> None:
        self._hits = list(hits)
        self.calls: list[tuple[tuple[str, ...], int]] = []

    async def search(
        self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.calls.append((tuple(paper_ids), limit))
        return [h for h in self._hits if h.paper_id in set(paper_ids)][:limit]


class RecordingLexical:
    """A lexical arm with the adapter's contract: an empty query matches nothing."""

    def __init__(self, hits: Sequence[ScoredChunk]) -> None:
        self._hits = list(hits)
        self.calls: list[tuple[dict[int, float], tuple[str, ...], int]] = []

    async def search_lexical(
        self, weights: dict[int, float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.calls.append((dict(weights), tuple(paper_ids), limit))
        if not weights:
            return []
        return [h for h in self._hits if h.paper_id in set(paper_ids)][:limit]


class RefusingLexical:
    async def search_lexical(self, *args: object, **kwargs: object) -> list[ScoredChunk]:
        raise AssertionError("dense mode searched the lexical arm")


def dense(chunk_id: str, rank: int, paper_id: str = PAPER) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        score=1.0 - rank / 100,
        arm=RetrievalArm.DENSE,
        rank=rank,
    )


def lexical(chunk_id: str, rank: int, paper_id: str = PAPER) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        score=10.0 - rank,
        arm=RetrievalArm.LEXICAL,
        rank=rank,
    )


def answer_use_case(
    *,
    dense_hits: Sequence[ScoredChunk],
    lexical_arm: Any,
    mode: RetrievalMode,
    ids: Sequence[str] = ("a", "b", "c", "d"),
    queryable: tuple[str, ...] = (PAPER,),
) -> tuple[AnswerQuestionUseCase, RecordingIndex, FakeConversations]:
    # Each chunk row belongs to the paper its hit names, as the store and the
    # database agree in production.
    paper_of = {h.chunk_id: h.paper_id for h in dense_hits}
    index = RecordingIndex(dense_hits)
    conversations = FakeConversations()
    use_case = AnswerQuestionUseCase(
        workspaces=FakeWorkspaces(workspace()),
        papers=FakePapers(queryable),  # type: ignore[arg-type]
        chunks=FakeChunks(  # type: ignore[arg-type]
            [chunk(i, f"passage {i}", paper_id=paper_of.get(i, PAPER)) for i in ids]
        ),
        conversations=conversations,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        index=index,  # type: ignore[arg-type]
        providers=[FakeProvider("ollama", "An answer [C1].")],
        mode=mode,
        lexical=lexical_arm,
    )
    return use_case, index, conversations


def labels(events: Sequence[object]) -> list[str]:
    ready = next(e for e in events if isinstance(e, EvidenceReady))
    return [item.chunk_id for item in ready.items]


# --- dense mode is today's path ----------------------------------------------


class TestDenseModeIsTodaysPath:
    def test_the_default_mode_is_dense(self) -> None:
        # Phase A must change nothing until someone switches it: a hybrid default
        # here would switch every deployment that merely upgraded.
        default = inspect.signature(AnswerQuestionUseCase).parameters["mode"].default
        assert default is RetrievalMode.DENSE

    async def test_dense_mode_never_touches_the_lexical_arm(self) -> None:
        use_case, index, _ = answer_use_case(
            dense_hits=[dense("a", 1)], lexical_arm=RefusingLexical(), mode=RetrievalMode.DENSE
        )
        await collect(use_case)
        assert index.calls == [((PAPER,), DEFAULT_CANDIDATES)]

    async def test_dense_mode_keeps_its_candidates_and_their_order(self) -> None:
        use_case, _, _ = answer_use_case(
            dense_hits=[dense("b", 1), dense("a", 2)], lexical_arm=None, mode=RetrievalMode.DENSE
        )
        assert labels(await collect(use_case)) == ["b", "a"]


# --- hybrid mode -----------------------------------------------------------------


class TestHybridModeFusesBothArms:
    async def test_both_arms_search_the_same_derived_scope_at_the_measured_depth(self) -> None:
        # §5.8 layer 3: the lexical arm must take the tenant-derived scope too. A
        # lexical filter of its own would be a second route across tenants.
        arm = RecordingLexical([lexical("a", 1)])
        use_case, index, _ = answer_use_case(
            dense_hits=[dense("a", 1)],
            lexical_arm=arm,
            mode=RetrievalMode.HYBRID,
            queryable=(PAPER, OTHER_PAPER),
        )
        await collect(use_case)
        assert index.calls == [((PAPER, OTHER_PAPER), HYBRID_DEPTH)]
        assert [(scope, limit) for _, scope, limit in arm.calls] == [
            ((PAPER, OTHER_PAPER), HYBRID_DEPTH)
        ]
        assert HYBRID_DEPTH == 50  # what M2 and D168's gate measured (D172)

    async def test_the_evidence_is_the_rrf_order_of_both_lists(self) -> None:
        dense_hits = [dense("a", 1), dense("b", 2), dense("c", 3)]
        lexical_hits = [lexical("c", 1), lexical("d", 2)]
        use_case, _, _ = answer_use_case(
            dense_hits=dense_hits,
            lexical_arm=RecordingLexical(lexical_hits),
            mode=RetrievalMode.HYBRID,
        )
        events = await collect(use_case)
        # `c` is in both lists, so it outranks everything either arm ranked alone.
        assert labels(events)[0] == "c"
        expected = [s.chunk_id for s in reciprocal_rank_fusion([dense_hits, lexical_hits])]
        assert labels(events) == expected

    async def test_a_chunk_only_the_lexical_arm_found_reaches_the_evidence(self) -> None:
        # The reason for the lexical arm: an identifier the dense arm ranks nowhere.
        use_case, _, _ = answer_use_case(
            dense_hits=[dense("a", 1)],
            lexical_arm=RecordingLexical([lexical("d", 1)]),
            mode=RetrievalMode.HYBRID,
        )
        assert "d" in labels(await collect(use_case))

    async def test_the_lexical_arm_is_queried_with_the_questions_terms(self) -> None:
        arm = RecordingLexical([])
        use_case, _, _ = answer_use_case(
            dense_hits=[dense("a", 1)], lexical_arm=arm, mode=RetrievalMode.HYBRID
        )
        question = "Does LLaMA-2-70B beat nDCG@10?"
        await collect(use_case, question)
        assert arm.calls[0][0] == query_weights(tokenize(question))

    async def test_a_question_with_no_lexical_terms_ranks_the_dense_list_alone(self) -> None:
        arm = RecordingLexical([lexical("d", 1)])
        use_case, _, _ = answer_use_case(
            dense_hits=[dense("b", 1), dense("a", 2)], lexical_arm=arm, mode=RetrievalMode.HYBRID
        )
        events = await collect(use_case, "?! ...")
        assert arm.calls[0][0] == {}
        assert labels(events) == ["b", "a"]

    async def test_an_empty_scope_reaches_neither_arm(self) -> None:
        use_case, index, _ = answer_use_case(
            dense_hits=[dense("a", 1)],
            lexical_arm=RefusingLexical(),
            mode=RetrievalMode.HYBRID,
            queryable=(),
        )
        events = await collect(use_case)
        assert index.calls == []
        assert final(events).answer == NO_EVIDENCE_ANSWER.strip()

    async def test_the_stored_turn_says_its_evidence_was_fused(self) -> None:
        # A fused score is not a cosine; a stored turn that hid the arm would be
        # misread by anyone comparing scores across turns.
        use_case, _, conversations = answer_use_case(
            dense_hits=[dense("a", 1), dense("b", 2)],
            lexical_arm=RecordingLexical([lexical("b", 1)]),
            mode=RetrievalMode.HYBRID,
        )
        await collect(use_case)
        evidence = conversations.turns[-1].evidence
        assert evidence and all(c.arm is RetrievalArm.FUSED for c in evidence)
        assert [c.rank for c in evidence] == list(range(1, len(evidence) + 1))

    async def test_the_fused_list_runs_deep_enough_for_the_per_paper_cap_to_fill(self) -> None:
        # Assembly caps one paper at half the 8 slots. When one paper fills the
        # top ranks, the slots it may not take come from further down the fused
        # list - so a list cut to 8 would leave the evidence half empty.
        mine = [dense(f"m{i}", i) for i in range(1, 11)]
        theirs = [dense(f"t{i}", 10 + i, paper_id=OTHER_PAPER) for i in range(1, 5)]
        use_case, _, _ = answer_use_case(
            dense_hits=mine + theirs,
            lexical_arm=RecordingLexical([]),
            mode=RetrievalMode.HYBRID,
            ids=[c.chunk_id for c in mine + theirs],
            queryable=(PAPER, OTHER_PAPER),
        )
        chosen = labels(await collect(use_case))
        assert len(chosen) == 8
        assert sum(c.startswith("t") for c in chosen) == 4

    def test_hybrid_without_a_lexical_index_is_refused_at_construction(self) -> None:
        # Otherwise it would answer dense-only while every record called it hybrid.
        with pytest.raises(ValueError, match="lexical index"):
            answer_use_case(dense_hits=[], lexical_arm=None, mode=RetrievalMode.HYBRID)


# --- τ -------------------------------------------------------------------------


class TestAFusedScoreIsNotTau:
    def test_abstention_refuses_fused_evidence(self) -> None:
        # D167: RRF's score measures how far the arms agreed, so a threshold on it
        # would abstain on disagreement, not on weak evidence.
        fused = reciprocal_rank_fusion([[dense("a", 1)], [lexical("a", 1)]])
        evidence = EvidenceSet.assemble(fused, limit=8)
        with pytest.raises(InvariantViolation, match="fused"):
            evidence.should_abstain(0.5)

    def test_abstention_still_reads_a_dense_score(self) -> None:
        evidence = EvidenceSet.assemble([dense("a", 1)], limit=8)
        assert evidence.should_abstain(0.999) is True
        assert evidence.should_abstain(0.5) is False


# --- the completeness check -----------------------------------------------------


class EmbeddedRows:
    def __init__(self, ids: set[str]) -> None:
        self.ids = ids

    async def embedded_chunk_ids(self) -> set[str]:
        return set(self.ids)

    async def count_embedded(self) -> int:
        return len(self.ids)


class HybridPoints:
    """The hybrid collection's points, by lexical version."""

    def __init__(self, by_version: dict[str, set[str]]) -> None:
        self.by_version = by_version

    async def lexical_chunk_ids(self, *, lexical_version: str) -> set[str]:
        return set(self.by_version.get(lexical_version, set()))

    async def count_lexical(self, *, lexical_version: str) -> int:
        return len(self.by_version.get(lexical_version, set()))


def verifier(rows: set[str], points: dict[str, set[str]]) -> VerifyHybridIndexUseCase:
    return VerifyHybridIndexUseCase(chunks=EmbeddedRows(rows), lexical=HybridPoints(points))  # type: ignore[arg-type]


class TestAHybridReadWaitsForACompleteCollection:
    async def test_a_complete_collection_passes(self) -> None:
        coverage = await verifier({"a", "b"}, {LEXICAL_VERSION: {"a", "b"}}).require_complete()
        assert coverage.complete and coverage.present == coverage.expected == 2

    async def test_an_empty_collection_is_refused(self) -> None:
        # The failure D172 was written against: a collection created and never filled.
        with pytest.raises(HybridIndexIncomplete, match="scholiast-reindex --hybrid"):
            await verifier({"a", "b"}, {}).require_complete()

    async def test_one_missing_chunk_is_refused_by_name(self) -> None:
        with pytest.raises(HybridIndexIncomplete, match="missing b") as raised:
            await verifier({"a", "b"}, {LEXICAL_VERSION: {"a"}}).require_complete()
        assert raised.value.coverage.missing == ("b",)

    async def test_points_with_an_earlier_lexical_version_do_not_count(self) -> None:
        # After a weighting change, an old point is not a current one: a hybrid
        # read must wait until the reindex has rewritten every point.
        coverage = await verifier({"a"}, {"bm25-identifier/1.0.0": {"a"}}).execute()
        assert coverage.missing == ("a",)

    async def test_points_no_embedded_chunk_accounts_for_are_reported_not_failed(self) -> None:
        coverage = await verifier({"a"}, {LEXICAL_VERSION: {"a", "orphan"}}).execute()
        assert coverage.complete and coverage.unaccounted == 1

    async def test_the_cheap_check_finds_a_writer_that_does_not_mirror(self) -> None:
        assert await verifier({"a", "b"}, {LEXICAL_VERSION: {"a"}}).falls_short() is True
        assert await verifier({"a"}, {LEXICAL_VERSION: {"a", "b"}}).falls_short() is False
        assert await verifier({"a"}, {LEXICAL_VERSION: {"a"}}).falls_short() is False

    async def test_an_orphan_can_hide_a_hole_from_the_count_but_not_from_the_id_check(
        self,
    ) -> None:
        # Why startup checks ids: two counts can balance while a chunk is missing.
        check = verifier({"a", "b"}, {LEXICAL_VERSION: {"a", "orphan"}})
        assert await check.falls_short() is False
        assert (await check.execute()).missing == ("b",)


# --- the sweep -------------------------------------------------------------------


class _Nothing:
    async def list_deleting(self) -> list[str]:
        return []

    async def orphaned_paper_ids(self) -> list[str]:
        return []

    async def all_paper_ids(self) -> tuple[str, ...]:
        return ()

    async def delete_orphans(self, live: Sequence[str]) -> int:
        return 0

    async def list_keys(self, prefix: str) -> list[str]:
        return []


class TestTheSweepReportsAHybridShortfall:
    async def test_missing_hybrid_points_are_counted(self) -> None:
        nothing = _Nothing()
        sweep = ReconcileUseCase(
            workspaces=nothing,  # type: ignore[arg-type]
            papers=nothing,  # type: ignore[arg-type]
            index=nothing,  # type: ignore[arg-type]
            store=nothing,  # type: ignore[arg-type]
            chunks=EmbeddedRows({"a", "b", "c"}),  # type: ignore[arg-type]
            hybrid=HybridPoints({LEXICAL_VERSION: {"a"}}),  # type: ignore[arg-type]
        )
        assert (await sweep.execute()).hybrid_missing == 2

    async def test_without_a_hybrid_collection_there_is_nothing_to_report(self) -> None:
        nothing = _Nothing()
        sweep = ReconcileUseCase(
            workspaces=nothing,  # type: ignore[arg-type]
            papers=nothing,  # type: ignore[arg-type]
            index=nothing,  # type: ignore[arg-type]
            store=nothing,  # type: ignore[arg-type]
        )
        assert (await sweep.execute()).hybrid_missing == 0


# --- what gets indexed ----------------------------------------------------------


def _ingester(
    journal: ingestion.Journal, index: Any
) -> tuple[IngestPaperUseCase, ingestion.FakeChunks]:
    subject = ingestion.stored_paper()
    papers = ingestion.FakePapers(journal)
    papers.by_id[subject.id] = subject
    chunks = ingestion.FakeChunks(journal)
    store = ingestion.FakeStore(journal)
    store.objects[ingestion.source_key(subject.id)] = ingestion.PDF
    use_case = IngestPaperUseCase(
        papers=papers,
        sections=ingestion.FakeSections(journal),
        chunks=chunks,
        store=store,
        parser=ingestion.FakeParser(ingestion.document()),
        embedder=ingestion.FakeEmbedder(journal),
        index=index,
        queue=ingestion.FakeQueue(journal),
        uow=ingestion.FakeUnitOfWork(journal),
    )
    return use_case, chunks


class TestEveryIndexedRecordCarriesItsLexicalWeights:
    async def test_ingestion_writes_production_weights_and_their_version(self) -> None:
        journal = ingestion.Journal()
        use_case, _, chunks, index, _, _ = ingestion.make_ingester(
            journal, parser=ingestion.FakeParser(ingestion.document())
        )
        await ingestion.ingest(use_case, ingestion.stored_paper().id)
        text = {c.id: c.text for c in chunks.written}
        assert index.records
        for record in index.records:
            assert record.lexical == chunk_weights(text[record.chunk_id])
            assert record.lexical_version == LEXICAL_VERSION

    async def test_the_rebuild_writes_the_same_weights_ingestion_wrote(self) -> None:
        # A rebuild that weighted differently would change the index it claims to
        # reproduce (NFR-R6, D101) - here, every lexical score.
        use_case, _, index = rebuild.build()
        await use_case.execute()
        source = rebuild.a_chunk()
        (written,) = index.upserted
        assert written.lexical == chunk_weights(source.text)
        assert written.lexical_version == LEXICAL_VERSION

    def test_weights_depend_on_the_text_alone(self) -> None:
        # No corpus, batch or order goes in, so none can change what comes out:
        # the property a running average would have broken.
        texts = ["LLaMA-2-70B on SQuAD v1.1", "BM25 is a ranking function", "x"]
        forwards = {t: chunk_weights(t) for t in texts}
        backwards = {t: chunk_weights(t) for t in reversed(texts)}
        assert forwards == backwards


class _Collection:
    """A store stand-in with a name, as the mirror wrapper requires."""

    def __init__(self, journal: ingestion.Journal, name: str, *, fails: bool = False) -> None:
        self.collection = name
        self.points: dict[str, VectorRecord] = {}
        self._journal = journal
        self.fails = fails

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        self._journal.record(f"{self.collection}.upsert")
        if self.fails:
            raise ConnectionError(f"{self.collection} refused the write")
        self.points.update({r.chunk_id: r for r in records})
        return len(records)


class TestAFailedMirroredWriteIsRedoneNotRecorded:
    async def test_a_failed_mirror_write_leaves_the_rows_unstamped(self) -> None:
        # D53 across two collections: a row stamped while the mirror lacks its
        # point would be a chunk the completeness check reports and no job redoes.
        journal = ingestion.Journal()
        primary = _Collection(journal, "dense")
        mirror = _Collection(journal, "hybrid", fails=True)
        use_case, chunks = _ingester(journal, MirroredVectorIndex(primary, mirror))  # type: ignore[arg-type]
        outcome = await ingestion.ingest(use_case, ingestion.stored_paper().id)
        assert not outcome.succeeded
        # The primary took the write and the mirror refused it; neither is
        # recorded as done, and the job retries as any store failure does.
        assert journal.index_of("dense.upsert") < journal.index_of("hybrid.upsert")
        assert "chunks.stamp" not in journal.events
        assert chunks.stamped == []
        assert outcome.failure is not None and outcome.failure.retryable is True

    async def test_a_retry_converges_on_one_point_per_chunk_in_each_collection(self) -> None:
        journal = ingestion.Journal()
        primary = _Collection(journal, "dense")
        mirror = _Collection(journal, "hybrid", fails=True)
        use_case, _ = _ingester(journal, MirroredVectorIndex(primary, mirror))  # type: ignore[arg-type]
        assert not (await ingestion.ingest(use_case, ingestion.stored_paper().id)).succeeded

        mirror.fails = False
        retry, chunks = _ingester(ingestion.Journal(), MirroredVectorIndex(primary, mirror))  # type: ignore[arg-type]
        assert (await ingestion.ingest(retry, ingestion.stored_paper().id)).succeeded
        assert primary.points and set(primary.points) == set(mirror.points) == set(chunks.stamped)


# --- stored evidence --------------------------------------------------------------


class TestStoredEvidenceKeepsItsArm:
    def test_fused_evidence_round_trips_with_its_arm_and_rank(self) -> None:
        fused = tuple(reciprocal_rank_fusion([[dense("a", 1)], [lexical("b", 1)]]))
        assert _evidence_from_json(_evidence_to_json(fused)) == fused

    def test_a_row_written_before_the_arm_was_stored_reads_back_as_dense(self) -> None:
        # Every turn before D172 was retrieved dense-only, so that is what it was.
        legacy = {"chunks": [{"chunk_id": "a", "paper_id": PAPER, "score": 0.8}]}
        (read,) = _evidence_from_json(legacy)
        assert read.arm is RetrievalArm.DENSE and read.rank is None

    def test_no_evidence_is_stored_as_none(self) -> None:
        assert _evidence_to_json(()) is None
        assert _evidence_from_json(None) == ()


def test_the_lexical_arm_is_wired_by_keyword_not_position() -> None:
    # A positional `mode` slipped in before `providers` would quietly change what
    # every existing caller passes; both new parameters are keyword-only.
    parameters = inspect.signature(AnswerQuestionUseCase).parameters
    assert parameters["mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["lexical"].kind is inspect.Parameter.KEYWORD_ONLY
