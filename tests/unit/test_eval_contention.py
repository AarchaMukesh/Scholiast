"""The contention harness (doc 8 deliverable 6, §5.7, D79).

The numbers this produces are wall-clock and cannot be asserted. What CAN be
asserted is everything that decides whether they mean anything: that the ingestion
arm actually ran while the queries were timed, that a load which died is not
silently reported as a fast condition, and that nothing but timing varied between
conditions. Those are the failures that would turn a latency table into fiction,
and none of them needs a GPU.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.parsing import ParsedDocument
from scholiast.domain.paper import Chunk, ChunkRole, Provenance, SectionKind
from scholiast.domain.retrieval import RetrievalArm, ScoredChunk
from scholiast.eval.contention import (
    Condition,
    ConditionReport,
    IngestionLoad,
    measure_condition,
    verify_contended,
    verify_same_retrieval,
)
from scholiast.eval.dret import DRet, Grade, Judgment, Query, Split
from scholiast.eval.qasper import CorpusPaper

IDS = ("a", "b", "c", "target")


class StubEmbedder:
    """Records what it was asked to embed, and suspends like the real one."""

    version = "stub/1.0.0"

    def __init__(self, *, explode: bool = False) -> None:
        self.roles: list[TextRole] = []
        self.batch_sizes: list[int] = []
        self._explode = explode

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        if self._explode and role is TextRole.PASSAGE:
            raise RuntimeError("CUDA out of memory")
        await asyncio.sleep(0)
        self.roles.append(role)
        if batch_size is not None:
            self.batch_sizes.append(batch_size)
        return [[1.0, 0.0, 0.0] for _ in texts]


class StubIndex:
    def __init__(self) -> None:
        self.searches = 0

    async def search(
        self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.searches += 1
        return [
            ScoredChunk(
                chunk_id=cid, paper_id="paper-1", score=1.0 - i / 10, arm=RetrievalArm.DENSE
            )
            for i, cid in enumerate(IDS[:limit])
        ]


class StubLexical:
    async def search_lexical(
        self, weights: dict[int, float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk_id=cid, paper_id="paper-1", score=1.0 - i / 10, arm=RetrievalArm.LEXICAL
            )
            for i, cid in enumerate(IDS[:limit])
        ]


class StubReranker:
    async def rerank(
        self, query: str, passages: Sequence[str], *, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        await asyncio.sleep(0)
        return [(i, 1.0 - i / 10) for i in range(len(passages))]


def _dataset() -> DRet:
    chunks = {
        cid: Chunk(
            id=cid,
            text=f"text of {cid}",
            role=ChunkRole.CHILD,
            ordinal=i,
            provenance=Provenance(
                paper_id="paper-1",
                section_path="Results",
                section_kind=SectionKind.RESULTS,
                page=1,
                char_start=0,
                char_end=10,
            ),
            chunker_version="structure-aware/1.2.0",
            token_count=3,
            parent_id="paper-1:p0",
        )
        for i, cid in enumerate(IDS)
    }
    paper = CorpusPaper(
        paper_id="paper-1",
        title="A Paper With A Title",
        document=ParsedDocument(sections=()),
        paragraph_index={},
        section_headings=frozenset(),
    )
    return DRet(
        queries=[Query("q1", "what was measured?", "paper-1", Split.EVAL, in_core=True)],
        judgments=[Judgment("q1", "target", Grade.DIRECTLY_ANSWERS)],
        chunks=chunks,
        papers={"paper-1": paper},
    )


def _condition(**kwargs: object) -> Condition:
    defaults: dict[str, object] = {
        "label": "t",
        "ingesting": True,
        "gated": False,
        "batch_size": 2,
    }
    return Condition(**{**defaults, **kwargs})  # type: ignore[arg-type]


class TestTheIngestionLoad:
    async def test_it_keeps_embedding_until_it_is_stopped(self) -> None:
        embedder = StubEmbedder()
        load = IngestionLoad(embedder, ["x", "y", "z"], batch_size=2)
        await load.start()
        for _ in range(20):
            await asyncio.sleep(0)
        await load.stop()
        assert load.batches > 0
        assert load.chunks == load.batches * 2
        assert set(embedder.roles) == {TextRole.PASSAGE}

    async def test_it_cycles_the_corpus_rather_than_running_out(self) -> None:
        # A loop that stopped at the end of the corpus would leave the query arm
        # measuring an idle GPU and reporting it as contention.
        load = IngestionLoad(StubEmbedder(), ["only-one"], batch_size=3)
        await load.start()
        for _ in range(10):
            await asyncio.sleep(0)
        await load.stop()
        assert load.batches >= 2
        assert load.chunks == load.batches * 3

    async def test_a_load_that_dies_is_re_raised_rather_than_going_quiet(self) -> None:
        # The worst available failure: every condition still produces numbers, and
        # they are idle numbers wearing a contention label.
        load = IngestionLoad(StubEmbedder(explode=True), ["x"], batch_size=1)
        await load.start()
        for _ in range(5):
            await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="out of memory"):
            await load.stop()

    async def test_an_empty_corpus_is_refused(self) -> None:
        with pytest.raises(ValueError, match="idle GPU"):
            IngestionLoad(StubEmbedder(), [], batch_size=4)

    async def test_it_embeds_at_the_batch_size_the_condition_asked_for(self) -> None:
        # The batch-16 diagnostic is only a diagnostic if the size reaches the
        # adapter; §5.7's bound is one batch, so the size IS the variable.
        embedder = StubEmbedder()
        load = IngestionLoad(embedder, ["x", "y"], batch_size=16)
        await load.start()
        for _ in range(5):
            await asyncio.sleep(0)
        await load.stop()
        assert set(embedder.batch_sizes) == {16}


class TestTheGuards:
    def test_a_contended_condition_that_ingested_nothing_is_refused(self) -> None:
        report = ConditionReport(condition=_condition(ingesting=True), batches_during=0)
        with pytest.raises(ValueError, match="idle numbers under a contention label"):
            verify_contended(report)

    def test_an_idle_condition_that_ingested_is_refused(self) -> None:
        report = ConditionReport(condition=_condition(ingesting=False), batches_during=4)
        with pytest.raises(ValueError, match="idle reference"):
            verify_contended(report)

    def test_a_healthy_pair_passes(self) -> None:
        verify_contended(ConditionReport(condition=_condition(ingesting=True), batches_during=3))
        verify_contended(ConditionReport(condition=_condition(ingesting=False)))

    def test_conditions_that_retrieved_differently_are_refused(self) -> None:
        # Ranking cannot depend on how the GPU was shared. If it did, the harness
        # changed something it had no business changing, and a latency table would
        # never show it.
        first = ConditionReport(condition=_condition(label="idle"), metrics={"mrr": 0.5})
        second = ConditionReport(condition=_condition(label="gated"), metrics={"mrr": 0.4})
        with pytest.raises(ValueError, match="retrieved differently"):
            verify_same_retrieval([first, second])

    def test_identical_retrieval_passes(self) -> None:
        reports = [
            ConditionReport(condition=_condition(label=name), metrics={"mrr": 0.5})
            for name in ("idle", "ungated", "gated")
        ]
        verify_same_retrieval(reports)


class TestMeasuringACondition:
    async def _measure(self, condition: Condition, repeats: int = 2) -> ConditionReport:
        dataset = _dataset()
        embedder = StubEmbedder()
        return await measure_condition(
            condition,
            dataset,
            dataset.core,
            StubIndex(),  # type: ignore[arg-type]
            StubLexical(),
            embedder,
            StubReranker(),
            paper_ids=["paper-1"],
            load_texts=[c.text for c in dataset.chunks.values()],
            repeats=repeats,
            depth=4,
        )

    async def test_every_pass_over_the_query_set_contributes_samples(self) -> None:
        # p95 over one pass of 53 queries moves too much to compare conditions
        # with; the repeats are what make the percentile readable (D79).
        report = await self._measure(_condition(), repeats=3)
        assert report.queries == 3
        assert len(report.rerank_ms) == 3
        assert len(report.total_ms) == 3

    async def test_a_contended_condition_records_the_ingestion_that_overlapped(self) -> None:
        report = await self._measure(_condition(ingesting=True))
        assert report.batches_during > 0
        assert report.ingestion_chunks_per_second is not None

    async def test_an_idle_condition_runs_no_ingestion_and_reports_no_throughput(self) -> None:
        report = await self._measure(_condition(label="idle", ingesting=False))
        assert report.batches_during == 0
        assert report.ingestion_chunks_per_second is None

    async def test_the_rerank_stage_is_sampled_apart_from_the_whole_stage(self) -> None:
        # NFR-P2 budgets the stage; §5.7 is about the rerank burst inside it. One
        # merged number would answer neither question.
        report = await self._measure(_condition())
        assert report.rerank_ms and report.total_ms
        assert all(
            total >= rerank for total, rerank in zip(report.total_ms, report.rerank_ms, strict=True)
        )

    async def test_the_models_are_warm_before_the_clock_starts(self) -> None:
        # Whichever condition touches the models first pays ~1.5 GB of loading. The
        # percentiles have their own warm-ups, but INGESTION THROUGHPUT divides work
        # by wall-clock, and the first real run deflated the gated arm's throughput
        # by about a fifth this way — the cost side the gate is judged on.
        dataset = _dataset()
        embedder = StubEmbedder()
        reranker = StubReranker()
        await measure_condition(
            _condition(ingesting=False),
            dataset,
            dataset.core,
            StubIndex(),  # type: ignore[arg-type]
            StubLexical(),
            embedder,
            reranker,
            paper_ids=["paper-1"],
            load_texts=["x"],
            repeats=1,
            depth=4,
        )
        # One passage embed before any query work: the warm-up, not the load.
        assert embedder.roles[0] is TextRole.PASSAGE

    async def test_the_gate_count_covers_the_measured_work_and_not_the_warm_up(self) -> None:
        # The gated conditions share one gate, so a block reporting totals would
        # carry the previous condition's acquisitions — and a count taken before
        # the warm-up would include calls that were never measured.
        from scholiast.infrastructure.ml.gpu import Priority

        class StubGate:
            """Counts nothing, so any non-zero delta is the harness's own doing."""

            def __init__(self) -> None:
                self.acquisitions: dict[Priority, int] = {Priority.QUERY: 7, Priority.INGESTION: 3}

        gate = StubGate()
        dataset = _dataset()
        report = await measure_condition(
            _condition(ingesting=False),
            dataset,
            dataset.core,
            StubIndex(),  # type: ignore[arg-type]
            StubLexical(),
            StubEmbedder(),
            StubReranker(),
            paper_ids=["paper-1"],
            load_texts=["x"],
            repeats=1,
            depth=4,
            gate=gate,  # type: ignore[arg-type]
        )
        # The stub gate never counts anything itself, so a correct delta is zero
        # rather than the 7/3 it started at.
        assert report.acquisitions == {"QUERY": 0, "INGESTION": 0}

    async def test_the_retrieval_metrics_come_back_for_the_cross_condition_check(self) -> None:
        # The payload `verify_same_retrieval` compares. Depth 4 here, so the recall
        # cutoff is 4 — D77's rule that a run cannot report a cutoff deeper than
        # the candidate list applies to this report too.
        report = await self._measure(_condition())
        assert set(report.metrics) == {"recall@4", "ndcg@10", "mrr"}

    def test_percentiles_are_empty_rather_than_zero_when_nothing_was_sampled(self) -> None:
        # A zero would read as "instant" in the table; absent reads as absent.
        assert ConditionReport(condition=_condition()).percentiles([]) == {}
        assert ConditionReport(condition=_condition()).percentiles([1.0, 2.0])["p50"] == 2.0
