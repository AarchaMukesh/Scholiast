"""The harness, tested against stand-ins so the logic is separable from the GPU.

The real run needs a 419 MB encoder and a live Qdrant, which NFR-T3 keeps out of
the default test run. What is testable without either is the part most likely to be
wrong quietly: whether the collection name tracks the dataset's identity, whether a
rebuild is correctly skipped, whether metrics are computed over the ranking the
index actually returned, and whether the report keeps CORE and FULL apart the way
D69 requires.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.vector import VectorRecord
from scholiast.domain.paper import SectionKind
from scholiast.domain.retrieval import RetrievalArm, ScoredChunk
from scholiast.eval.cli import EvalReport, _render, _summarise
from scholiast.eval.dret import DRet, Grade, Judgment, Query, Split
from scholiast.eval.harness import build_index, collection_name, run_retrieval


class StubEmbedder:
    """Deterministic and cheap. Never loads a model."""

    def __init__(self, version: str = "stub/1.0.0") -> None:
        self._version = version
        self.passages_embedded = 0
        self.roles: list[TextRole] = []

    @property
    def version(self) -> str:
        return self._version

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        self.roles.append(role)
        if role is TextRole.PASSAGE:
            self.passages_embedded += len(texts)
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]


@dataclass
class StubIndex:
    """Returns a scripted ranking so metric arithmetic is checkable by hand."""

    ranking: list[str] = field(default_factory=list)
    points: int = 0
    upserts: list[int] = field(default_factory=list)
    searches: int = 0
    last_paper_ids: Sequence[str] = ()

    async def ensure_ready(self) -> None:
        return None

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        self.upserts.append(len(records))
        self.points += len(records)
        return len(records)

    async def search(
        self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.searches += 1
        self.last_paper_ids = paper_ids
        return [
            ScoredChunk(chunk_id=cid, paper_id="p", score=1.0 - i / 100, arm=RetrievalArm.DENSE)
            for i, cid in enumerate(self.ranking[:limit])
        ]

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        return 0

    async def count(self, *, paper_ids: Sequence[str] | None = None) -> int:
        return self.points


def _dataset_with(ranking_target: str) -> DRet:
    """A one-query dataset whose single relevant chunk is `ranking_target`."""
    from scholiast.domain.paper import Chunk, ChunkRole, Provenance

    chunk = Chunk(
        id=ranking_target,
        text="some retrievable text",
        role=ChunkRole.CHILD,
        ordinal=0,
        provenance=Provenance(
            paper_id="paper-1",
            section_path="Results",
            section_kind=SectionKind.RESULTS,
            page=1,
            char_start=0,
            char_end=21,
        ),
        chunker_version="structure-aware/1.2.0",
        token_count=5,
        parent_id="paper-1:p0",
    )
    from scholiast.application.ports.parsing import ParsedDocument
    from scholiast.eval.qasper import CorpusPaper

    paper = CorpusPaper(
        paper_id="paper-1",
        title="A Paper With A Title",
        document=ParsedDocument(sections=()),
        paragraph_index={},
        section_headings=frozenset(),
    )
    return DRet(
        queries=[Query("q1", "what was measured?", "paper-1", Split.EVAL, in_core=True)],
        judgments=[Judgment("q1", ranking_target, Grade.DIRECTLY_ANSWERS)],
        chunks={chunk.id: chunk},
        papers={paper.paper_id: paper},
    )


class TestTheCollectionNameCarriesTheDatasetIdentity:
    def test_the_same_versions_give_the_same_name(self) -> None:
        dataset = _dataset_with("c1")
        assert collection_name(dataset, "e/1") == collection_name(dataset, "e/1")

    def test_a_different_embedder_gives_a_different_collection(self) -> None:
        # Otherwise a re-embed silently queries vectors from the previous model
        # and the report describes a corpus that no longer exists.
        dataset = _dataset_with("c1")
        assert collection_name(dataset, "e/1") != collection_name(dataset, "e/2")

    def test_a_different_dataset_version_gives_a_different_collection(self) -> None:
        dataset = _dataset_with("c1")
        before = collection_name(dataset, "e/1")
        # Mutating the module constant is the only way to simulate a version bump
        import scholiast.eval.dret as dret_module

        original = dret_module.DRET_VERSION
        try:
            dret_module.DRET_VERSION = "d-ret/9.9.9"
            assert collection_name(dataset, "e/1") != before
        finally:
            dret_module.DRET_VERSION = original

    def test_the_name_is_a_legal_qdrant_identifier(self) -> None:
        # Embedder versions contain "/" — concatenating them would produce a name
        # with a path separator in it.
        name = collection_name(_dataset_with("c1"), "BAAI/bge-base-en-v1.5/fp16/1.0.0")
        assert "/" not in name
        assert name.startswith("scholiast_eval_")


class TestIndexBuilding:
    async def test_a_complete_collection_is_not_rebuilt(self) -> None:
        dataset = _dataset_with("c1")
        index = StubIndex(points=len(dataset.chunks))
        embedder = StubEmbedder()
        report = await build_index(dataset, index, embedder, collection="x")
        assert not report.rebuilt
        assert embedder.passages_embedded == 0, "a complete index must cost no embedding"

    async def test_an_empty_collection_is_populated(self) -> None:
        dataset = _dataset_with("c1")
        index = StubIndex()
        embedder = StubEmbedder()
        report = await build_index(dataset, index, embedder, collection="x")
        assert report.rebuilt
        assert report.chunks_indexed == len(dataset.chunks)
        assert index.points == len(dataset.chunks)

    async def test_chunks_are_embedded_as_passages_not_queries(self) -> None:
        # An asymmetric encoder given the wrong prefix degrades retrieval silently
        # — no error, just worse numbers, which is the worst possible failure in
        # a measurement harness.
        dataset = _dataset_with("c1")
        embedder = StubEmbedder()
        await build_index(dataset, StubIndex(), embedder, collection="x")
        assert embedder.roles == [TextRole.PASSAGE]

    async def test_the_report_names_the_collection_it_was_given(self) -> None:
        report = await build_index(_dataset_with("c1"), StubIndex(), StubEmbedder(), collection="k")
        assert report.collection == "k"


class TestRunningRetrieval:
    async def test_metrics_reflect_the_ranking_the_index_returned(self) -> None:
        dataset = _dataset_with("target")
        # Relevant chunk sits at position 3 -> RR = 1/3, recall@10 = 1.0.
        index = StubIndex(ranking=["x", "y", "target", "z"])
        run = await run_retrieval(
            dataset,
            dataset.core,
            index,
            StubEmbedder(),
            label="t",
            paper_ids=["paper-1"],
        )
        assert run.metrics.rr == [pytest.approx(1 / 3)]
        assert run.metrics.recall[10] == [1.0]

    async def test_a_miss_scores_zero_rather_than_being_dropped(self) -> None:
        dataset = _dataset_with("target")
        run = await run_retrieval(
            dataset,
            dataset.core,
            StubIndex(ranking=["x", "y"]),
            StubEmbedder(),
            label="t",
            paper_ids=["paper-1"],
        )
        assert run.metrics.rr == [0.0]
        assert run.metrics.recall[10] == [0.0]
        assert run.metrics.n == 1, "a missed query must still count in the mean"

    async def test_the_query_is_embedded_as_a_query(self) -> None:
        dataset = _dataset_with("target")
        embedder = StubEmbedder()
        await run_retrieval(
            dataset,
            dataset.core,
            StubIndex(ranking=["target"]),
            embedder,
            label="t",
            paper_ids=["paper-1"],
        )
        assert set(embedder.roles) == {TextRole.QUERY}

    async def test_an_untimed_warmup_query_precedes_the_measured_ones(self) -> None:
        # FR-E5's latency must be retrieval, not CUDA context creation. Measured
        # without this: CORE's retrieve p99 was 13,595 ms against a p50 of 16 ms.
        # Pinned because the warm-up looks like a redundant extra call and is the
        # kind of thing a later cleanup removes.
        dataset = _dataset_with("target")
        embedder = StubEmbedder()
        index = StubIndex(ranking=["target"])
        run = await run_retrieval(
            dataset, dataset.core, index, embedder, label="t", paper_ids=["paper-1"]
        )
        assert index.searches == len(dataset.core) + 1, "no warm-up search was issued"
        assert len(embedder.roles) == len(dataset.core) + 1
        assert len(run.latencies_ms) == len(dataset.core), "the warm-up must not be timed"

    async def test_the_retrieval_scope_is_passed_through_to_the_index(self) -> None:
        # D70: the whole corpus. If the harness narrowed it, every number would
        # describe an easier benchmark than the one the report claims.
        dataset = _dataset_with("target")
        index = StubIndex(ranking=["target"])
        scope = ["paper-1", "paper-2", "paper-3"]
        await run_retrieval(
            dataset, dataset.core, index, StubEmbedder(), label="t", paper_ids=scope
        )
        assert list(index.last_paper_ids) == scope

    async def test_latency_is_recorded_per_query(self) -> None:
        dataset = _dataset_with("target")
        run = await run_retrieval(
            dataset,
            dataset.core,
            StubIndex(ranking=["target"]),
            StubEmbedder(),
            label="t",
            paper_ids=["paper-1"],
        )
        assert len(run.latencies_ms) == 1
        assert set(run.latency_percentiles()) == {"p50", "p95", "p99"}


class TestTheReportKeepsCoreAndFullApart:
    async def _run(self) -> tuple[dict, dict]:
        dataset = _dataset_with("target")
        index = StubIndex(ranking=["target"])
        run = await run_retrieval(
            dataset, dataset.core, index, StubEmbedder(), label="t", paper_ids=["paper-1"]
        )
        return _summarise(run, graded=True), _summarise(run, graded=False)

    async def test_the_graded_block_reports_ndcg(self) -> None:
        graded, _ = await self._run()
        assert "ndcg@10" in graded["metrics"]

    async def test_the_ungraded_block_omits_ndcg_entirely(self) -> None:
        # D69: FULL's judgments are binary, so its nDCG is not the graded nDCG
        # CORE reports. Omitted from the ARTIFACT, not just the printout — a value
        # that must not be quoted should not be greppable in the JSON.
        _, ungraded = await self._run()
        assert "ndcg@10" not in ungraded["metrics"]
        assert "recall@10" in ungraded["metrics"]

    async def test_every_metric_carries_an_interval(self) -> None:
        graded, _ = await self._run()
        for name, values in graded["metrics"].items():
            assert "ci95" in values, f"{name} was reported without an interval"
            assert len(values["ci95"]) == 2

    async def _report(self, *, with_delta: bool = False) -> EvalReport:
        graded, _ = await self._run()
        arm: dict[str, object] = {
            "model": "BAAI/bge-base-en-v1.5",
            "collection": "scholiast_eval_abc",
            "is_m1_baseline": True,
            "resources": {
                "dimensions": 768,
                "vector_bytes_computed": 768 * 100 * 4,
                "index_build_seconds": 12.5,
                "index_chunks_per_second": 8.0,
                "index_reused": False,
            },
            "blocks": {"D-RET-CORE": {**graded, "note": "primary"}},
            "latency": {
                "D-RET-CORE": {
                    "total": {"p50": 16.0, "p95": 27.0, "p99": 69.0},
                    "embed_p50_ms": 12.0,
                    "search_p50_ms": 4.0,
                }
            },
        }
        if with_delta:
            arm["is_m1_baseline"] = False
            arm["deltas_vs_baseline"] = {
                "D-RET-CORE": {
                    "recall@10": {"delta": -0.02, "ci95": [-0.09, 0.05], "within_noise": True},
                }
            }
        return EvalReport(
            generated_at="2026-09-15T00:00:00+00:00",
            config={
                "dataset_versions": {"dret": "d-ret/1.0.0"},
                "retrieval_scope": "whole corpus, 281 papers (D70)",
                "grade_scale": "binary 2/0 - grade 1 (CONTEXT) unpopulated",
                "comparison": "ML-4 embedding-model comparison",
                "latency_measured": "retrieve only: query embed + dense search, idle system",
            },
            blocks={"bge-base (M1 baseline)": arm},
        )

    async def test_the_rendered_report_states_the_grade_scale(self) -> None:
        # D69 requires every report to say the scale rather than imply 3 points.
        rendered = _render(await self._report())
        assert "binary 2/0" in rendered
        assert "whole corpus" in rendered
        assert "scholiast_eval_abc" in rendered

    async def test_the_baseline_arm_is_labelled_as_the_m1_baseline(self) -> None:
        # The user of this report must never have to infer which arm is the
        # incumbent — a comparison where the control is ambiguous is not one.
        assert "[M1 DENSE BASELINE]" in _render(await self._report())

    async def test_a_delta_within_noise_is_rendered_as_such_not_as_a_win(self) -> None:
        # Doc 9 section 9.4: an interval straddling zero is reported as noise, not
        # as a win for whichever point estimate is higher.
        rendered = _render(await self._report(with_delta=True))
        assert "within noise" in rendered
        assert "SIGNIFICANT" not in rendered

    async def test_the_report_separates_embed_from_search_latency(self) -> None:
        # ML-4's tradeoff turns on the embed cost specifically; one combined
        # number would attribute a smaller encoder's win to "retrieval".
        rendered = _render(await self._report())
        assert "embed p50" in rendered
        assert "search p50" in rendered

    async def test_the_report_states_which_latency_stages_it_measured(self) -> None:
        # A table headed "latency" with no scope invites the reader to assume it
        # covers the pipeline. Rerank and generation do not exist yet, and rows
        # for unbuilt stages would be fabrication (doc 9 §9.8).
        rendered = _render(await self._report())
        assert "retrieve only" in rendered
