"""The cross-encoder reranker adapter (doc 6 §6.7, D18, D74).

Most of what goes wrong with a reranker adapter is decidable without a model: the
ordering contract, the precision rule, whether the gate is taken and released. Those
run here with no GPU. One test uses the real 1.1 GB model and skips unless it is
already cached AND a CUDA device exists, because NFR-T3 requires the default run to
make zero network calls — a warm HF cache on the dev machine hides that completely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scholiast.application.ports.ml import Precision
from scholiast.infrastructure.ml.gpu import GpuPriorityGate, Priority
from scholiast.infrastructure.ml.reranker import DEFAULT_RERANKER, LocalReranker, rank_scores


class TestTheOrderingContract:
    def test_scores_come_back_best_first_with_original_indices(self) -> None:
        # Indices refer back into the caller's passage list, so the reranker never
        # needs to know what a chunk is (the port's contract).
        assert rank_scores([0.1, 0.9, 0.5]) == [(1, 0.9), (2, 0.5), (0, 0.1)]

    def test_ties_keep_original_index_order_so_runs_reproduce(self) -> None:
        # fp16 makes exact ties more likely, not less, so the tie order is a
        # reproducibility contract (FR-E4). This pins the ORDER, not the mechanism:
        # stable sorting over enumerate already yields it, so deleting the explicit
        # tiebreak is harmless and survives — but a WRONG tiebreak (descending
        # index) is caught here, which is the regression that would actually
        # change results.
        assert rank_scores([0.5, 0.5, 0.5]) == [(0, 0.5), (1, 0.5), (2, 0.5)]
        assert rank_scores([0.2, 0.7, 0.7]) == [(1, 0.7), (2, 0.7), (0, 0.2)]

    def test_top_k_truncates_after_ordering_not_before(self) -> None:
        # Truncating before sorting would keep the first k candidates rather than
        # the best k — silently reranking only a prefix of the shortlist.
        assert rank_scores([0.1, 0.2, 0.9, 0.8], top_k=2) == [(2, 0.9), (3, 0.8)]

    def test_no_top_k_returns_every_candidate_exactly_once(self) -> None:
        # The property the evaluation depends on: reranking reorders a shortlist
        # and never resizes it, so recall@50 is unchanged by construction.
        ranked = rank_scores([0.3, 0.1, 0.4, 0.2])
        assert sorted(index for index, _ in ranked) == [0, 1, 2, 3]

    def test_a_negative_top_k_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            rank_scores([0.1], top_k=-1)


class TestPrecisionIsPartOfTheContract:
    def test_gpu_defaults_to_fp16(self) -> None:
        # D18: fp32 reranking took 743 ms of an 800 ms whole-stage budget.
        assert LocalReranker(device="cuda").precision is Precision.FP16

    def test_cpu_falls_back_to_fp32_and_says_so(self) -> None:
        assert LocalReranker(device="cpu").precision is Precision.FP32

    def test_precision_appears_in_the_version_stamp(self) -> None:
        # fp16 and fp32 scores differ; a report must be able to say which produced
        # its numbers.
        assert "fp16" in LocalReranker(device="cuda").version
        assert "fp32" in LocalReranker(device="cpu").version

    def test_construction_does_not_load_the_model(self) -> None:
        # Wiring happens at import time; a 1.1 GB load there would make every test
        # that merely constructs an adapter pay for the model.
        assert LocalReranker()._model is None


class TestTheGpuGate:
    @staticmethod
    def _fake(reranker: LocalReranker, scores: list[float]) -> list[int]:
        seen: list[int] = []

        def predict(pairs: list[tuple[str, str]]) -> list[float]:
            seen.append(len(pairs))
            return scores[: len(pairs)]

        reranker._predict = predict  # type: ignore[method-assign]
        return seen

    async def test_a_rerank_takes_the_gate_at_query_priority(self) -> None:
        gate = GpuPriorityGate()
        reranker = LocalReranker(gate=gate)
        self._fake(reranker, [0.2, 0.8])
        order = await reranker.rerank("q", ["a", "b"])
        assert order == [(1, 0.8), (0, 0.2)]
        assert gate.acquisitions[Priority.QUERY] == 1
        assert gate.acquisitions[Priority.INGESTION] == 0
        assert not gate.busy

    async def test_a_failing_rerank_still_releases_the_gate(self) -> None:
        # A model error must not leave the GPU held for every later call.
        gate = GpuPriorityGate()
        reranker = LocalReranker(gate=gate)

        def explode(pairs: list[tuple[str, str]]) -> list[float]:
            raise RuntimeError("CUDA out of memory")

        reranker._predict = explode  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="out of memory"):
            await reranker.rerank("q", ["a"])
        assert not gate.busy

    async def test_without_a_gate_it_reranks_ungated(self) -> None:
        reranker = LocalReranker()
        self._fake(reranker, [0.4])
        assert await reranker.rerank("q", ["a"]) == [(0, 0.4)]

    async def test_no_passages_returns_nothing_without_touching_the_model_or_gate(self) -> None:
        gate = GpuPriorityGate()
        reranker = LocalReranker(gate=gate)
        seen = self._fake(reranker, [])
        assert await reranker.rerank("q", []) == []
        assert seen == []
        assert gate.acquisitions[Priority.QUERY] == 0


def _model_cached() -> bool:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    return (cache / f"models--{DEFAULT_RERANKER.replace('/', '--')}").exists()


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # any import failure means "no usable CUDA"
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not (_model_cached() and _has_cuda()), reason="needs cached model + CUDA")
class TestTheRealModel:
    async def test_the_relevant_passage_outranks_an_irrelevant_one(self) -> None:
        # The only test that proves the adapter is wired to a model that actually
        # judges relevance, rather than to one that returns some scores.
        reranker = LocalReranker(gate=GpuPriorityGate())
        passages = [
            "The weather in Lisbon was mild throughout the spring season.",
            "BM25 ranks documents by term frequency saturation and inverse document "
            "frequency, with length normalisation.",
        ]
        order = await reranker.rerank("how does BM25 score documents?", passages)
        assert order[0][0] == 1
        assert sorted(i for i, _ in order) == [0, 1]
        assert reranker.precision is Precision.FP16
