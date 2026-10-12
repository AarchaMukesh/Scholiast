"""D167's workspace-scale gate, held to the parameters D168 fixed before it existed.

What can go wrong quietly here is not the retrieval (that is M2's, called
unchanged) but the plumbing around it: a draw that depends on corpus order, sizes
that are not nested, dense and hybrid given different scopes, an average taken
over the wrong axis, a precondition that tolerates a difference, or a rule that
reads the wrong metric. Each class below pins one of those.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from scholiast.application.ports.parsing import ParsedDocument
from scholiast.application.ports.vector import VectorRecord
from scholiast.domain.paper import Chunk, ChunkRole, Provenance, SectionKind
from scholiast.domain.retrieval import RetrievalArm, ScoredChunk
from scholiast.eval import cli
from scholiast.eval.dret import DRet, Grade, Judgment, Query, Split
from scholiast.eval.gate import (
    DECIDING,
    GATE_KEY,
    compare_with_archive,
    decide,
    drawn,
    measure_scoped,
    scope_for,
    whole_corpus,
)
from scholiast.eval.harness import run_hybrid, run_retrieval
from scholiast.eval.qasper import CorpusPaper
from tests.unit.test_eval_harness import StubEmbedder

# --- a corpus small enough to reason about ----------------------------------


def _chunk(chunk_id: str, paper_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=f"text of {chunk_id}",
        role=ChunkRole.CHILD,
        ordinal=0,
        provenance=Provenance(
            paper_id=paper_id,
            section_path="Results",
            section_kind=SectionKind.RESULTS,
            page=1,
            char_start=0,
            char_end=10,
        ),
        chunker_version="structure-aware/1.2.0",
        token_count=4,
        parent_id=f"{paper_id}:p0",
    )


def _corpus(papers: int, queries: Sequence[tuple[str, str, bool]]) -> DRet:
    """`papers` papers p00.., one chunk each (`c-pNN`); each query judged on its own paper's chunk.

    `queries` is (query id, own paper, in CORE). Every query is in the eval split,
    so FULL holds them all and CORE the flagged ones, as D-RET's are.
    """
    ids = [f"p{i:02d}" for i in range(papers)]
    return DRet(
        queries=[
            Query(qid, f"question {qid}", paper, Split.EVAL, in_core=core)
            for qid, paper, core in queries
        ],
        judgments=[
            Judgment(qid, f"c-{paper}", Grade.DIRECTLY_ANSWERS) for qid, paper, _ in queries
        ],
        chunks={f"c-{p}": _chunk(f"c-{p}", p) for p in ids},
        papers={
            p: CorpusPaper(
                paper_id=p,
                title=f"Paper {p}",
                document=ParsedDocument(sections=()),
                paragraph_index={},
                section_headings=frozenset(),
            )
            for p in ids
        },
    )


#: The index operations that create, index or write: `ensure_ready` is among them
#: because the real adapter creates a missing collection and re-asserts its
#: payload index on every call (qdrant.py), which is exactly why the gate must
#: not use it.
WRITES = frozenset({"ensure_ready", "upsert", "upsert_sparse", "delete_by_papers"})


@dataclass
class ScopedIndex:
    """Ranks a fixed global order of chunks, keeping only those in the scope.

    `dense` and `lexical` are the two global orders. Each call's scope is recorded
    per method, so a test can see exactly which papers each arm was given, and
    every operation is logged in `calls`, so a test can see whether the store was
    touched at all and whether anything wrote to it.
    """

    dataset: DRet
    dense: list[str]
    lexical: list[str] = field(default_factory=list)
    dense_scopes: list[tuple[str, ...]] = field(default_factory=list)
    lexical_scopes: list[tuple[str, ...]] = field(default_factory=list)
    points: int | None = None
    exists: bool = True
    upserted: int = 0
    calls: list[str] = field(default_factory=list)

    @property
    def writes(self) -> list[str]:
        return [c for c in self.calls if c in WRITES]

    def _ranked(
        self, order: list[str], paper_ids: Sequence[str], limit: int, arm: RetrievalArm
    ) -> list[ScoredChunk]:
        allowed = set(paper_ids)
        kept = [c for c in order if self.dataset.chunks[c].provenance.paper_id in allowed][:limit]
        return [
            ScoredChunk(
                chunk_id=c,
                paper_id=self.dataset.chunks[c].provenance.paper_id,
                score=1.0 - i / 100,
                arm=arm,
                rank=i + 1,
            )
            for i, c in enumerate(kept)
        ]

    async def ensure_ready(self) -> None:
        self.calls.append("ensure_ready")

    async def count(self, *, paper_ids: Sequence[str] | None = None) -> int:
        self.calls.append("count")
        return len(self.dataset.chunks) if self.points is None else self.points

    async def existing_count(self) -> int | None:
        # The adapter's read-only check: None for a missing collection, which the
        # stand-in never creates either way.
        self.calls.append("existing_count")
        if not self.exists:
            return None
        return len(self.dataset.chunks) if self.points is None else self.points

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        self.calls.append("upsert")
        self.upserted += len(records)
        return len(records)

    async def upsert_sparse(self, records: Sequence[object]) -> int:
        self.calls.append("upsert_sparse")
        self.upserted += len(records)
        return len(records)

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        self.calls.append("delete_by_papers")
        return 0

    async def search(
        self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.calls.append("search")
        self.dense_scopes.append(tuple(paper_ids))
        return self._ranked(self.dense, paper_ids, limit, RetrievalArm.DENSE)

    async def search_lexical(
        self, weights: dict[int, float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.calls.append("search_lexical")
        self.lexical_scopes.append(tuple(paper_ids))
        return self._ranked(self.lexical, paper_ids, limit, RetrievalArm.LEXICAL)


def _papers(n: int) -> list[str]:
    return [f"paper-{i:03d}" for i in range(n)]


# --- the draw ----------------------------------------------------------------


class TestTheScopeIsDrawnAsD168Fixed:
    def test_the_hash_key_is_the_one_d168_recorded(self) -> None:
        # Changing the key redraws every scope; D168 names it, so a test pins it.
        papers = _papers(6)
        own, others = papers[0], papers[1:]
        expected_order = sorted(
            others,
            key=lambda p: hashlib.sha256(f"d167-gate:3:q-7:{p}".encode()).digest(),
        )
        assert GATE_KEY == "d167-gate"
        assert scope_for("q-7", own, papers, size=3, draw=3) == tuple(
            sorted([own, *expected_order[:2]])
        )

    def test_a_scope_is_its_own_paper_and_size_minus_one_others(self) -> None:
        papers = _papers(40)
        scope = scope_for("q", "paper-017", papers, size=10, draw=0)

        assert len(scope) == len(set(scope)) == 10
        assert "paper-017" in scope
        assert set(scope) <= set(papers)

    def test_the_draw_does_not_depend_on_how_the_corpus_is_listed(self) -> None:
        papers = _papers(60)
        shuffled = papers[1::2] + papers[::2]

        for draw in range(5):
            assert (
                scope_for("q", "paper-003", papers, size=10, draw=draw)
                == scope_for("q", "paper-003", reversed(papers), size=10, draw=draw)
                == scope_for("q", "paper-003", shuffled, size=10, draw=draw)
            )

    def test_smaller_scopes_sit_inside_larger_ones(self) -> None:
        papers = _papers(281)
        for q in range(30):
            for draw in range(5):
                five, ten, twenty_five = (
                    set(scope_for(f"q{q}", "paper-000", papers, size=n, draw=draw))
                    for n in (5, 10, 25)
                )
                assert five < ten < twenty_five

    def test_each_draw_is_a_different_scope(self) -> None:
        papers = _papers(281)
        scopes = {scope_for("q", "paper-000", papers, size=10, draw=d) for d in range(5)}
        assert len(scopes) == 5

    def test_two_queries_on_one_paper_get_different_scopes(self) -> None:
        papers = _papers(281)
        assert scope_for("q1", "paper-000", papers, size=10, draw=0) != scope_for(
            "q2", "paper-000", papers, size=10, draw=0
        )

    def test_no_paper_is_favoured_by_where_its_name_sorts(self) -> None:
        # Uniform means every other paper is about equally likely; a draw that
        # leaned on sort order would pile into the first or last names. Expected
        # 2000 x 9 / 280 = 64.3 per paper, binomial sd ~7.9; the bounds are wide
        # because the draw is deterministic, not to forgive a bias.
        papers = _papers(281)
        counts = dict.fromkeys(papers[1:], 0)
        for q in range(2000):
            for p in scope_for(f"query-{q}", "paper-000", papers, size=10, draw=0):
                if p != "paper-000":
                    counts[p] += 1
        assert min(counts.values()) >= 35
        assert max(counts.values()) <= 100
        first, last = sum(list(counts.values())[:28]), sum(list(counts.values())[-28:])
        assert abs(first - last) < 0.25 * (first + last) / 2

    @pytest.mark.parametrize(
        ("own", "size"),
        [("paper-000", 0), ("paper-000", 11), ("paper-999", 5)],
    )
    def test_a_scope_that_cannot_be_drawn_is_refused(self, own: str, size: int) -> None:
        with pytest.raises(ValueError):
            scope_for("q", own, _papers(10), size=size, draw=0)

    def test_the_whole_corpus_scope_is_m2s_for_every_query_and_draw(self) -> None:
        dataset = _corpus(5, [("q", "p02", True)])
        scope = whole_corpus(dataset)

        assert [scope(dataset.queries[0], d) for d in range(3)] == [sorted(dataset.papers)] * 3


# --- measuring on scopes -------------------------------------------------------


def _three_query_corpus() -> tuple[DRet, ScopedIndex, ScopedIndex]:
    # Dense ranks papers in name order and lexical in reverse, so a query's rank
    # in either depends on which papers its draw put in the scope.
    dataset = _corpus(
        12,
        [("qa", "p05", True), ("qb", "p08", True), ("qc", "p10", False)],
    )
    dense_order = ["c-p00", "c-p01", *[f"c-p{i:02d}" for i in range(2, 12)]]
    lexical_order = [*[f"c-p{i:02d}" for i in range(11, 1, -1)], "c-p01", "c-p00"]
    dense_index = ScopedIndex(dataset, dense_order)
    hybrid_index = ScopedIndex(dataset, dense_order, lexical_order)
    return dataset, dense_index, hybrid_index


class TestMeasuringOnScopes:
    async def test_dense_and_hybrid_search_the_identical_scope_at_every_draw(self) -> None:
        dataset, dense_index, hybrid_index = _three_query_corpus()

        await measure_scoped(
            dataset,
            dataset.full,
            dense_index,
            hybrid_index,
            StubEmbedder(),
            label="FULL",
            scope=drawn(dataset, size=4),
            draws=5,
        )

        # Every call searches twice (an untimed warm-up, then the real query) on
        # the same scope, so the three logs must line up call for call.
        assert dense_index.dense_scopes == hybrid_index.dense_scopes == hybrid_index.lexical_scopes
        assert len(dense_index.dense_scopes) == 3 * 5 * 2
        expected = [
            scope_for(q.id, q.paper_id, sorted(dataset.papers), size=4, draw=d)
            for q in dataset.full
            for d in range(5)
            for _ in range(2)
        ]
        assert dense_index.dense_scopes == expected

    async def test_each_arm_reads_the_collection_m2s_arm_read(self) -> None:
        # Dense from the dense-only collection, hybrid from the sparse one: the
        # precondition compares against M2's arms, which read them that way.
        dataset, dense_index, hybrid_index = _three_query_corpus()

        await measure_scoped(
            dataset,
            dataset.full,
            dense_index,
            hybrid_index,
            StubEmbedder(),
            label="FULL",
            scope=drawn(dataset, size=4),
            draws=2,
        )

        assert dense_index.lexical_scopes == []
        assert len(hybrid_index.dense_scopes) == len(dense_index.dense_scopes) > 0

    async def test_a_query_scores_the_mean_over_its_draws(self) -> None:
        dataset, dense_index, hybrid_index = _three_query_corpus()

        runs = await measure_scoped(
            dataset,
            dataset.full,
            dense_index,
            hybrid_index,
            StubEmbedder(),
            label="FULL",
            scope=drawn(dataset, size=3),
            draws=5,
        )

        def dense_rr(query: Query, draw: int) -> float:
            # Rank = 1 + the in-scope papers whose chunk the dense order puts first.
            scope = scope_for(query.id, query.paper_id, sorted(dataset.papers), size=3, draw=draw)
            ahead = dense_index.dense[: dense_index.dense.index(f"c-{query.paper_id}")]
            return 1.0 / (1 + sum(1 for c in ahead if c.removeprefix("c-") in scope))

        expected = [sum(dense_rr(q, d) for d in range(5)) / 5 for q in dataset.full]
        assert runs.dense.metrics.rr == expected
        # The draws genuinely differ here, or the test would not test averaging.
        assert (
            len({dense_rr(dataset.full[0], d) for d in range(5)}) > 1
            or len({dense_rr(dataset.full[1], d) for d in range(5)}) > 1
        )

    async def test_one_draw_over_the_whole_corpus_is_m2s_run_exactly(self) -> None:
        # The precondition in miniature: given M2's scope, the gate's per-query
        # values must be the ones M2's own functions return over the query list.
        dataset, dense_index, hybrid_index = _three_query_corpus()
        everything = sorted(dataset.papers)
        embedder = StubEmbedder()

        runs = await measure_scoped(
            dataset,
            dataset.full,
            dense_index,
            hybrid_index,
            embedder,
            label="FULL",
            scope=whole_corpus(dataset),
            draws=1,
        )
        dense = await run_retrieval(
            dataset,
            dataset.full,
            dense_index,
            embedder,
            label="FULL",
            paper_ids=everything,
            depth=50,
        )
        hybrid = await run_hybrid(
            dataset,
            dataset.full,
            hybrid_index,
            hybrid_index,
            embedder,
            label="FULL",
            paper_ids=everything,
            arms="hybrid",
            depth=50,
        )

        for ours, m2 in ((runs.dense, dense), (runs.hybrid, hybrid)):
            assert ours.metrics.recall == m2.metrics.recall
            assert ours.metrics.ndcg == m2.metrics.ndcg
            assert ours.metrics.rr == m2.metrics.rr
            assert ours.ceiling_recall == m2.ceiling_recall
            assert ours.own_paper_in_top10 == m2.own_paper_in_top10

    async def test_the_spread_is_the_mean_delta_at_each_draw(self) -> None:
        dataset, dense_index, hybrid_index = _three_query_corpus()

        runs = await measure_scoped(
            dataset,
            dataset.full,
            dense_index,
            hybrid_index,
            StubEmbedder(),
            label="FULL",
            scope=drawn(dataset, size=3),
            draws=4,
        )

        assert set(runs.draw_deltas) == {"recall@10", "ndcg@10"}
        assert all(len(values) == 4 for values in runs.draw_deltas.values())
        mean_of_draws = sum(runs.draw_deltas["ndcg@10"]) / 4
        mean_of_queries = sum(runs.hybrid.metrics.ndcg) / 3 - sum(runs.dense.metrics.ndcg) / 3
        assert mean_of_draws == pytest.approx(mean_of_queries)

    async def test_a_query_with_no_draws_is_refused(self) -> None:
        dataset, dense_index, hybrid_index = _three_query_corpus()
        with pytest.raises(ValueError, match="at least one draw"):
            await measure_scoped(
                dataset,
                dataset.full,
                dense_index,
                hybrid_index,
                StubEmbedder(),
                label="FULL",
                scope=drawn(dataset, size=3),
                draws=0,
            )


# --- the precondition --------------------------------------------------------


def _summary(recall: float = 0.25) -> dict[str, Any]:
    return {
        "n": 53,
        "metrics": {"recall@10": {"point": recall, "ci95": [0.1, 0.4]}},
        "diagnostics": {"own_paper_share_of_top10": {"point": 0.2, "ci95": [0.1, 0.3]}},
        "retrieve_latency_ms": {"p50": 12.0},
    }


def _delta(delta: float = 0.08) -> dict[str, Any]:
    return {"recall@10": {"delta": delta, "ci95": [0.004, 0.177], "within_noise": False}}


def _archive(**overrides: Any) -> dict[str, Any]:
    archive = {
        "blocks": {
            "dense": {"blocks": {"CORE": _summary(0.177)}},
            "hybrid": {
                "blocks": {"CORE": _summary(0.266)},
                "deltas_vs_control": {"CORE": _delta()},
            },
        }
    }
    archive["blocks"].update(overrides)
    return archive


def _measured(dense: float = 0.177, hybrid: float = 0.266, delta: float = 0.08) -> dict[str, Any]:
    return {
        "dense": {"CORE": _summary(dense)},
        "hybrid": {"CORE": _summary(hybrid)},
        "deltas": {"CORE": _delta(delta)},
    }


class TestThePreconditionToleratesNothing:
    def test_identical_numbers_reproduce(self) -> None:
        assert compare_with_archive(_measured(), _archive(), dense="dense", hybrid="hybrid") == []

    def test_a_difference_in_the_last_bit_is_a_difference(self) -> None:
        nudged = 0.177 + 2**-55
        assert nudged != 0.177

        found = compare_with_archive(
            _measured(dense=nudged), _archive(), dense="dense", hybrid="hybrid"
        )

        assert found == [f"dense/CORE/metrics/recall@10/point: measured {nudged!r}, archived 0.177"]

    def test_a_different_diagnostic_is_a_difference(self) -> None:
        # The own-paper share is M2's mechanism evidence; a run that moved it has
        # not reproduced M2, whatever its metrics say.
        measured = _measured()
        measured["hybrid"]["CORE"]["diagnostics"]["own_paper_share_of_top10"]["point"] = 0.21
        found = compare_with_archive(measured, _archive(), dense="dense", hybrid="hybrid")
        assert found == [
            "hybrid/CORE/diagnostics/own_paper_share_of_top10/point: measured 0.21, archived 0.2"
        ]

    def test_a_different_delta_is_a_difference(self) -> None:
        found = compare_with_archive(
            _measured(delta=0.081), _archive(), dense="dense", hybrid="hybrid"
        )
        assert len(found) == 1 and "deltas_vs_control/CORE/recall@10/delta" in found[0]

    def test_latency_is_not_compared(self) -> None:
        measured = _measured()
        measured["dense"]["CORE"]["retrieve_latency_ms"] = {"p50": 99.0}
        assert compare_with_archive(measured, _archive(), dense="dense", hybrid="hybrid") == []

    def test_a_missing_block_is_a_difference(self) -> None:
        archive = _archive()
        del archive["blocks"]["dense"]
        found = compare_with_archive(_measured(), archive, dense="dense", hybrid="hybrid")
        assert found == ["dense: not in the archived artifact"]

    def test_a_number_archived_but_not_measured_is_a_difference(self) -> None:
        archive = _archive()
        archive["blocks"]["dense"]["blocks"]["CORE"]["metrics"]["mrr"] = {
            "point": 0.1,
            "ci95": [0, 1],
        }
        found = compare_with_archive(_measured(), archive, dense="dense", hybrid="hybrid")
        assert found == ["dense/CORE/metrics/mrr: archived but not measured"]

    def test_true_is_not_one(self) -> None:
        measured = _measured()
        measured["deltas"]["CORE"]["recall@10"]["within_noise"] = 0
        found = compare_with_archive(measured, _archive(), dense="dense", hybrid="hybrid")
        assert found == [
            "hybrid/deltas_vs_control/CORE/recall@10/within_noise: measured 0, archived False"
        ]

    def test_a_query_set_archived_but_not_measured_is_a_difference(self) -> None:
        # D168 reproduces CORE and FULL; a run that skipped FULL must not pass on
        # CORE alone.
        archive = _archive()
        archive["blocks"]["dense"]["blocks"]["FULL"] = _summary(0.196)
        archive["blocks"]["hybrid"]["blocks"]["FULL"] = _summary(0.229)
        archive["blocks"]["hybrid"]["deltas_vs_control"]["FULL"] = _delta(0.033)

        found = compare_with_archive(_measured(), archive, dense="dense", hybrid="hybrid")

        assert found == [
            "dense/FULL: archived but not measured",
            "hybrid/FULL: archived but not measured",
            "hybrid/deltas_vs_control/FULL: archived but not measured",
        ]

    def test_an_empty_run_does_not_reproduce_an_empty_archive(self) -> None:
        # Comparing nothing with nothing finds no difference; that must not read
        # as the precondition met.
        nothing = {"dense": {}, "hybrid": {}, "deltas": {}}
        empty = {"blocks": {"dense": {"blocks": {}}, "hybrid": {"blocks": {}}}}

        found = compare_with_archive(nothing, empty, dense="dense", hybrid="hybrid")

        assert found == [
            "dense: the whole-corpus run measured nothing to compare",
            "hybrid: the whole-corpus run measured nothing to compare",
            "deltas: the whole-corpus run measured nothing to compare",
        ]


# --- the rule ----------------------------------------------------------------


def _deltas(**highs_and_lows: tuple[float, float]) -> dict[str, dict[str, dict[str, Any]]]:
    """Every deciding metric within noise unless overridden as name=(low, high)."""
    out: dict[str, dict[str, dict[str, Any]]] = {
        "D-RET-CORE": {
            "recall@10": {"ci95": [-0.02, 0.05]},
            "ndcg@10": {"ci95": [-0.01, 0.04]},
            "recall@20": {"ci95": [-0.03, 0.06]},
            "mrr": {"ci95": [-0.03, 0.06]},
        },
        "D-RET-FULL": {
            "recall@10": {"ci95": [-0.01, 0.03]},
            "recall@20": {"ci95": [-0.01, 0.03]},
            "mrr": {"ci95": [-0.01, 0.03]},
        },
    }
    for key, interval in highs_and_lows.items():
        name, metric = key.split("__")
        out[f"D-RET-{name}"][metric.replace("_", "@")] = {"ci95": list(interval)}
    return out


class TestTheRule:
    def test_a_null_is_adopted_and_reported_as_one(self) -> None:
        verdict = decide(True, _deltas())

        assert verdict["outcome"] == "adopt hybrid"
        assert verdict["within_noise"] == [f"{n} {m}" for n, m in DECIDING]
        assert verdict["significantly_worse"] == verdict["significantly_better"] == []

    @pytest.mark.parametrize("key", ["CORE__recall_10", "CORE__ndcg_10", "FULL__recall_10"])
    def test_hybrid_significantly_worse_on_any_deciding_metric_keeps_dense(self, key: str) -> None:
        verdict = decide(True, _deltas(**{key: (-0.09, -0.001)}))

        assert verdict["outcome"] == "stay dense-only"
        assert len(verdict["significantly_worse"]) == 1

    def test_an_interval_touching_zero_is_not_below_it(self) -> None:
        assert decide(True, _deltas(CORE__recall_10=(-0.09, 0.0)))["outcome"] == "adopt hybrid"

    @pytest.mark.parametrize(
        "key", ["CORE__recall_20", "CORE__mrr", "FULL__recall_20", "FULL__mrr"]
    )
    def test_metrics_the_rule_does_not_name_decide_nothing(self, key: str) -> None:
        assert decide(True, _deltas(**{key: (-0.2, -0.1)}))["outcome"] == "adopt hybrid"

    def test_significantly_better_is_reported(self) -> None:
        verdict = decide(True, _deltas(CORE__recall_10=(0.01, 0.09)))
        assert verdict["significantly_better"] == ["D-RET-CORE recall@10"]

    def test_without_the_precondition_there_is_no_verdict_whatever_the_deltas(self) -> None:
        verdict = decide(False, _deltas(CORE__recall_10=(0.05, 0.2)))
        assert verdict["outcome"] == "no verdict"

    def test_a_deciding_metric_that_was_not_measured_is_an_error(self) -> None:
        deltas = _deltas()
        del deltas["D-RET-CORE"]["ndcg@10"]
        with pytest.raises(ValueError, match="ndcg@10"):
            decide(True, deltas)


# --- the run, end to end on stand-ins -------------------------------------------


@pytest.fixture
def gate_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A 30-paper corpus behind the real CLI, with M2's own code producing the archive.

    The archive is written by `_measure` and `_deltas` — the functions that wrote
    the real arms artifact — so a reproduced precondition here means the gate's
    scope plumbing returns what M2's path returns, not what a test author wrote.
    """
    queries = [(f"q{i}", f"p{(i * 7) % 30:02d}", i % 3 == 0) for i in range(12)]
    dataset = _corpus(30, queries)
    order = [f"c-p{i:02d}" for i in range(30)]
    dense_index = ScopedIndex(dataset, order)
    hybrid_index = ScopedIndex(dataset, order, list(reversed(order)))
    embedder = StubEmbedder("BAAI/bge-base-en-v1.5/fp16/1.0.0")

    def build(_dataset: DRet, _spec: Any, *, sparse: bool = False) -> tuple[Any, Any, str]:
        return (
            embedder,
            hybrid_index if sparse else dense_index,
            "eval_hybrid" if sparse else "eval",
        )

    built: list[str] = []

    def build_dret(_rows: Any) -> DRet:
        built.append("dret")
        return dataset

    monkeypatch.setattr(cli, "_build_dependencies", build)
    monkeypatch.setattr(cli, "load_snapshot", lambda: [])
    monkeypatch.setattr(cli, "build_dret", build_dret)
    monkeypatch.setattr(cli, "verify_split", lambda _d: None)
    monkeypatch.setattr(cli, "verify_core", lambda _d: None)
    return {
        "dataset": dataset,
        "dense": dense_index,
        "hybrid": hybrid_index,
        "dir": tmp_path,
        "built": built,
        "monkeypatch": monkeypatch,
    }


def _pin(world: dict[str, Any], path: Path) -> None:
    """Make `path` the pinned reference, as the real one is pinned by its hash."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    world["monkeypatch"].setattr(cli, "GATE_REFERENCE_SHA256", digest)


async def _write_m2_archive(world: dict[str, Any]) -> Path:
    dataset: DRet = world["dataset"]
    base = next(s for s in cli.COMPARED_EMBEDDERS if s.baseline)
    scope = sorted(dataset.papers)
    dense_blocks, dense_runs, dense_config = await cli._measure(
        dataset, cli.Cell("d", base, arm="dense"), scope
    )
    hybrid_blocks, hybrid_runs, hybrid_config = await cli._measure(
        dataset, cli.Cell("h", base, arm="hybrid"), scope
    )
    archive = cli.EvalReport(
        generated_at="2026-09-24T05:35:49+00:00",
        config={},
        blocks={
            # Each block carries its cell's configuration, collection and embedder
            # version included, as the real arms artifact's blocks do.
            cli.GATE_DENSE_BLOCK: {**dense_config, "blocks": dense_blocks},
            cli.GATE_HYBRID_BLOCK: {
                **hybrid_config,
                "blocks": hybrid_blocks,
                "deltas_vs_control": {
                    name: cli._deltas(hybrid_runs[name], dense_runs[name], graded=graded)
                    for name, _a, graded, _n in cli.QUERY_SETS
                },
            },
        },
    )
    path = world["dir"] / "arms.json"
    path.write_text(archive.to_json() + "\n", encoding="utf-8")
    _pin(world, path)
    # Building the reference touched the stand-ins as M2's run did (it prepares
    # the index and attaches lexical weights); the gate is judged from here on.
    for index in (world["dense"], world["hybrid"]):
        index.calls.clear()
        index.dense_scopes.clear()
        index.lexical_scopes.clear()
        index.upserted = 0
    return path


def _edit_archive(world: dict[str, Any], path: Path, edit: Any) -> None:
    """Edit the reference and pin the edited file, so a test reaches the check it
    is about (identity, precondition) instead of the hash refusal."""
    content = json.loads(path.read_text(encoding="utf-8"))
    edit(content["blocks"])
    path.write_text(json.dumps(content), encoding="utf-8")
    _pin(world, path)


class TestTheGateRun:
    async def test_it_reproduces_an_arms_run_made_by_m2s_own_code(
        self, gate_world: dict[str, Any]
    ) -> None:
        archive = await _write_m2_archive(gate_world)

        report = await cli.run_workspace_gate(reproduces=archive)

        assert report.blocks["reproduction"] == {
            "reproduced": True,
            "mismatches": [],
            "compared_against": [cli.GATE_DENSE_BLOCK, cli.GATE_HYBRID_BLOCK],
        }
        assert report.blocks["verdict"]["outcome"] in {"adopt hybrid", "stay dense-only"}
        assert list(report.blocks["sizes"]) == ["N=5", "N=10", "N=25"]
        json.loads(report.to_json())  # the artifact must serialise

    async def test_a_successful_run_creates_rebuilds_and_writes_nothing(
        self, gate_world: dict[str, Any]
    ) -> None:
        # D168: the gate reads the collections M2 measured. Anything that writes,
        # `ensure_ready` included, would make them not those collections.
        archive = await _write_m2_archive(gate_world)

        report = await cli.run_workspace_gate(reproduces=archive)

        assert report.blocks["reproduction"]["reproduced"] is True
        for which in ("dense", "hybrid"):
            index: ScopedIndex = gate_world[which]
            assert index.writes == [], which
            assert index.upserted == 0, which
            assert index.calls.count("existing_count") == 1, which
        assert gate_world["dense"].lexical_scopes == []  # dense never served a lexical query

    async def test_a_failed_precondition_runs_no_scoped_measurement(
        self, gate_world: dict[str, Any]
    ) -> None:
        archive = await _write_m2_archive(gate_world)

        def nudge(blocks: dict[str, Any]) -> None:
            blocks[cli.GATE_DENSE_BLOCK]["blocks"]["D-RET-CORE"]["metrics"]["mrr"]["point"] += 1e-12

        _edit_archive(gate_world, archive, nudge)

        report = await cli.run_workspace_gate(reproduces=archive)

        assert report.blocks["reproduction"]["reproduced"] is False
        assert report.blocks["verdict"]["outcome"] == "no verdict"
        assert report.blocks["sizes"] == {}
        assert report.blocks["scoped_measurement"] == {
            "run": False,
            "reason": "the precondition failed (D168)",
        }
        # Only the whole-corpus precondition searched: no 5-, 10- or 25-paper scope.
        searched = gate_world["dense"].dense_scopes + gate_world["hybrid"].dense_scopes
        assert searched and {len(s) for s in searched} == {30}
        rendered = cli._render_gate(report)
        assert "VERDICT: no verdict" in rendered
        assert "scoped measurement: not run" in rendered
        assert "=== N=" not in rendered

    @pytest.mark.parametrize(
        ("block", "key", "recorded"),
        [
            ("dense", "collection", "scholiast_eval_000000000000"),
            ("hybrid", "collection", "scholiast_eval_000000000000_hybrid"),
            ("dense", "embedder_version", "BAAI/bge-base-en-v1.5/fp32/1.0.0"),
            ("hybrid", "embedder_version", "BAAI/bge-base-en-v1.5/fp32/1.0.0"),
        ],
    )
    async def test_a_collection_the_archive_does_not_record_is_refused_before_any_qdrant_call(
        self, gate_world: dict[str, Any], block: str, key: str, recorded: str
    ) -> None:
        archive = await _write_m2_archive(gate_world)
        label = cli.GATE_DENSE_BLOCK if block == "dense" else cli.GATE_HYBRID_BLOCK
        _edit_archive(gate_world, archive, lambda blocks: blocks[label].__setitem__(key, recorded))

        with pytest.raises(
            ValueError, match="would not read the collections M2's arms read"
        ) as err:
            await cli.run_workspace_gate(reproduces=archive)

        assert recorded in str(err.value)
        assert "precision and device" in str(err.value)
        assert gate_world["dense"].calls == gate_world["hybrid"].calls == []

    @pytest.mark.parametrize("which", ["dense", "hybrid"])
    async def test_a_missing_collection_is_refused_without_being_created(
        self, gate_world: dict[str, Any], which: str
    ) -> None:
        archive = await _write_m2_archive(gate_world)
        gate_world[which].exists = False

        with pytest.raises(ValueError, match="does not exist"):
            await cli.run_workspace_gate(reproduces=archive)

        assert gate_world["dense"].writes == gate_world["hybrid"].writes == []
        assert "search" not in gate_world["dense"].calls + gate_world["hybrid"].calls

    async def test_the_deciding_size_is_ten_and_its_scopes_hold_ten_papers(
        self, gate_world: dict[str, Any]
    ) -> None:
        archive = await _write_m2_archive(gate_world)
        before = len(gate_world["dense"].dense_scopes)

        report = await cli.run_workspace_gate(reproduces=archive)

        sizes = {len(s) for s in gate_world["dense"].dense_scopes[before:]}
        assert sizes == {5, 10, 25, 30}  # the three sizes plus the whole-corpus precondition
        assert "N = 10" in report.config["rule"]
        assert report.config["scope"]["draws_per_query"] == 5

    async def test_the_verdict_is_read_from_the_ten_paper_block_only(
        self, gate_world: dict[str, Any]
    ) -> None:
        archive = await _write_m2_archive(gate_world)

        report = await cli.run_workspace_gate(reproduces=archive)

        sizes = report.blocks["sizes"]
        at = {
            size: {n: b["deltas_hybrid_vs_dense"] for n, b in sizes[size].items()} for size in sizes
        }
        assert report.blocks["verdict"] == decide(True, at["N=10"])
        # The other sizes must read differently here, or this would not tell the
        # deciding block from the others.
        assert decide(True, at["N=5"]) != decide(True, at["N=10"]) != decide(True, at["N=25"])

    async def test_latency_never_reaches_the_artifact(self, gate_world: dict[str, Any]) -> None:
        archive = await _write_m2_archive(gate_world)

        report = await cli.run_workspace_gate(reproduces=archive)

        assert "retrieve_latency_ms" not in report.to_json()

    @pytest.mark.parametrize("which", ["dense", "hybrid"])
    async def test_an_incomplete_collection_is_refused_not_built(
        self, gate_world: dict[str, Any], which: str
    ) -> None:
        archive = await _write_m2_archive(gate_world)
        index: ScopedIndex = gate_world[which]
        index.points = 7

        with pytest.raises(ValueError, match="must not build them"):
            await cli.run_workspace_gate(reproduces=archive)
        assert gate_world["dense"].writes == gate_world["hybrid"].writes == []
        assert gate_world["dense"].upserted == gate_world["hybrid"].upserted == 0

    async def test_a_missing_reference_artifact_is_refused(
        self, gate_world: dict[str, Any]
    ) -> None:
        with pytest.raises(FileNotFoundError, match="precondition"):
            await cli.run_workspace_gate(reproduces=gate_world["dir"] / "absent.json")

    @pytest.mark.parametrize("how", ["modified", "replaced"])
    async def test_a_reference_that_is_not_the_pinned_one_is_refused_before_anything_runs(
        self, gate_world: dict[str, Any], how: str
    ) -> None:
        # F1: the precondition must reproduce the arms run M2 published, so a
        # reference whose bytes differ from the pinned ones is refused before the
        # dataset is built, any collection is read, or anything is measured.
        archive = await _write_m2_archive(gate_world)
        content = json.loads(archive.read_text(encoding="utf-8"))
        if how == "modified":
            # One number moved, the kind of edit nobody would spot by eye.
            content["blocks"][cli.GATE_DENSE_BLOCK]["blocks"]["D-RET-CORE"]["n"] += 1
            archive.write_text(json.dumps(content), encoding="utf-8")
        else:
            # The same JSON written again, as a re-export or restored copy would
            # be: equal when parsed, different bytes, so not the pinned file.
            archive.write_text(json.dumps(content, indent=4), encoding="utf-8")
        assert json.loads(archive.read_text(encoding="utf-8")) == content

        with pytest.raises(ValueError, match="does not match the pinned M2 reference"):
            await cli.run_workspace_gate(reproduces=archive)

        assert gate_world["built"] == []
        assert gate_world["dense"].calls == gate_world["hybrid"].calls == []

    def test_the_pinned_hash_is_the_one_recorded_for_m2s_arms_run(self) -> None:
        # Changing the pin re-points the precondition at a different reference, so
        # it is pinned here as the D168 hash key is.
        assert cli.GATE_REFERENCE_SHA256 == (
            "53accec29c820489d9d7290a56aa8b3dd5e72ce49b161567df57f694cc4d51b8"
        )
        assert cli.GATE_REPRODUCES.name == "arms-20260924T053549+0000.json"

    @pytest.mark.skipif(
        not cli.GATE_REPRODUCES.exists(),
        reason="the M2 reference lives in gitignored var/eval/, present on the dev machine only",
    )
    def test_the_local_m2_reference_matches_its_pin(self) -> None:
        # Where the real file exists, the pin must accept it, or the gate could
        # never run; read-only, no Qdrant, no measurement.
        assert set(cli._load_reference(cli.GATE_REPRODUCES)["blocks"]) >= {
            cli.GATE_DENSE_BLOCK,
            cli.GATE_HYBRID_BLOCK,
        }

    def test_the_command_line_archives_it_as_a_gate_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        report = cli.EvalReport(
            generated_at="2026-10-11T10:00:00+00:00",
            config={
                "dataset_versions": {},
                "grade_scale": "binary",
                "rule": "r",
                "reproduces": "x",
                "scope": {"draw": "d"},
            },
            blocks={
                "reproduction": {"reproduced": True, "mismatches": [], "compared_against": []},
                "sizes": {},
                "verdict": {"outcome": "adopt hybrid", "within_noise": ["D-RET-CORE recall@10"]},
            },
        )

        async def fake_gate() -> cli.EvalReport:
            return report

        monkeypatch.setattr(cli, "run_workspace_gate", fake_gate)

        assert cli.main(["--workspace-gate", "--report-dir", str(tmp_path)]) == 0
        assert [p.name for p in tmp_path.iterdir()] == ["gate-20261011T100000+0000.json"]
