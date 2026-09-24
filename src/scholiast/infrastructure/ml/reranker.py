"""Local cross-encoder reranker (doc 6 §6.7, ML-8, FR-Q3, D18, D74).

A bi-encoder embeds query and passage separately and never sees them together; a
cross-encoder attends over the pair jointly, which is why it judges *relevance*
rather than *similarity* — and why it is only affordable on a shortlist.

**Its quality benefit is a design hypothesis, not a project result.** Doc 6 §6.7,
written before this project measured any retrieval quality, calls
retrieve-then-rerank "the single most reliable quality win in practical RAG". This
adapter exists to test that expectation, not to assume it: FR-Q3 accepts reranking
only if hybrid+rerank beats dense-only on the eval set, and adoption waits on the
phase 5 ablation (D74).

**fp16 on GPU is specified, not preferred (D18).** At fp32 reranking 50 candidates
took 743 ms against an 800 ms whole-stage budget (NFR-P2); at fp16, 224 ms in M0 and
160-165 ms re-measured on real passages in phase 4 (D74). Precision is part of the
version stamp for the same reason as the embedder's: fp16 and fp32 scores differ,
and a report must be able to say which produced its numbers.

**Takes the GPU gate at QUERY priority when one is supplied (D74).** A rerank is a
small burst a user is waiting on; with a gate it goes ahead of queued embedding
batches instead of contending with them. Without one it behaves exactly as an
ungated adapter, which is what every test and the dense-only baseline rely on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from scholiast.application.ports.ml import Precision
from scholiast.infrastructure.ml.gpu import GpuPriorityGate, Priority

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "BAAI/bge-reranker-base"

#: The cross-encoder's context window. A query (~20 tokens) plus the ML-3 contextual
#: header (~20) plus a ~300-token child chunk fits well inside it; parent windows
#: (~1000 tokens) would not, which is one reason reranking scores children.
MAX_LENGTH = 512


class LocalReranker:
    """`Reranker` adapter over sentence-transformers' CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        *,
        device: str = "cuda",
        precision: Precision = Precision.FP16,
        gate: GpuPriorityGate | None = None,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self._model_name = model_name
        self._device = device
        # Same rule as the embedder: fp16 on CPU is slower and partly unsupported,
        # so the CPU path runs fp32 and says so in its version stamp.
        self._precision = precision if device.startswith("cuda") else Precision.FP32
        self._gate = gate
        self._max_length = max_length
        self._model: CrossEncoder | None = None

    @property
    def version(self) -> str:
        return f"{self._model_name}/{self._precision.value}/1.0.0"

    @property
    def precision(self) -> Precision:
        """Exposed so callers and tests can assert D18 rather than trust it."""
        return self._precision

    def _load(self) -> CrossEncoder:
        """Load on first use — constructing an adapter must not cost 1.1 GB."""
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {}
            if self._precision is Precision.FP16:
                kwargs["torch_dtype"] = torch.float16
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
                model_kwargs=kwargs,
            )
        return self._model

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        if self._gate is None:
            scores = await asyncio.to_thread(self._predict, pairs)
        else:
            async with self._gate.hold(Priority.QUERY):
                scores = await asyncio.to_thread(self._predict, pairs)
        return rank_scores(scores, top_k=top_k)

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load()
        # `predict` is typed as an invariant `list` of a very wide union, so a plain
        # `list[tuple[str, str]]` is rejected even though it is exactly what the
        # method accepts. Widened here, at the one call, rather than by weakening
        # this method's own signature.
        inputs: Any = pairs
        raw: Any = model.predict(inputs, show_progress_bar=False, convert_to_numpy=True)
        return [float(score) for score in raw]


def rank_scores(scores: Sequence[float], *, top_k: int | None = None) -> list[tuple[int, float]]:
    """Order `(original_index, score)` best-first, deterministically.

    **Contract:** ties keep the original index order, ascending, so two runs over
    the same candidates agree (FR-E4) — and fp16 makes exact ties more likely, not
    less.

    What guarantees it is `sorted`'s stability over `enumerate(...)`, whose input is
    already in index order. The explicit `pair[0]` in the key is therefore
    redundant today; it is kept to state the contract in the code rather than lean
    on an implementation property, so it still holds if the input is ever built in
    some other order. An earlier docstring claimed the key was what prevented
    nondeterminism; mutation testing showed removing it changes nothing, which is
    what corrected that.

    Kept as a free function so the ordering contract is testable without a model.
    """
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    ordered = sorted(enumerate(scores), key=lambda pair: (-pair[1], pair[0]))
    return ordered if top_k is None else ordered[:top_k]
