"""Local inference ports: embedding, reranking, entailment.

These are the three models that run on our own GPU (D8). The interfaces are
narrow on purpose — swapping bge-base for bge-small, or a fine-tuned reranker for
an off-the-shelf one (M7), must be an adapter change with no use-case edits. That
claim is checked by `import-linter`, not merely asserted.

Two things measured in M0 are encoded here rather than left to adapter authors:

  fp16 (D18)  reranking 50 candidates costs 743 ms at fp32 and 224 ms at fp16.
              Precision is part of the contract because getting it wrong silently
              consumes ~93% of the whole-stage latency budget.

  versions    every adapter reports a version string that is stamped onto stored
              artifacts (NFR-M5), so a quality change can be attributed to a
              specific cause and re-indexing can be targeted.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum, unique
from typing import Protocol, runtime_checkable


@unique
class Precision(StrEnum):
    """Inference precision.

    FP16 is the default for GPU inference (D18). FP32 exists for CPU paths, where
    fp16 is usually slower rather than faster, and for numerical debugging.
    """

    FP16 = "fp16"
    FP32 = "fp32"


@unique
class TextRole(StrEnum):
    """Asymmetric encoders need to know which side they are encoding.

    Getting this wrong degrades retrieval **silently** — no error, just worse
    results — because instruction-prefixed models expect different prefixes for
    queries and passages. It is a contract parameter precisely so it cannot be
    forgotten at a call site.
    """

    QUERY = "query"
    PASSAGE = "passage"


@runtime_checkable
class Embedder(Protocol):
    """Dense encoder for chunks and queries."""

    @property
    def version(self) -> str:
        """Stable identifier stamped onto every embedded chunk (NFR-M5).

        Must change whenever the model, its revision, or the pooling strategy
        changes — otherwise targeted re-indexing silently skips stale vectors.
        """
        ...

    @property
    def dimensions(self) -> int: ...

    async def embed(
        self,
        texts: Sequence[str],
        *,
        role: TextRole,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Encode texts, applying the model's role-appropriate prefix.

        `batch_size` is bounded by the caller for ingestion work: small batches
        (~50-100 ms each) are what let a high-priority rerank preempt a bulk
        embedding job without waiting for a large batch to drain (§5.7).
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder that scores (query, passage) pairs jointly.

    Unlike a bi-encoder it sees both sides together, which is why it judges
    *relevance* rather than *similarity* — and why it is worth running on a
    shortlist despite costing more per pair.
    """

    @property
    def version(self) -> str: ...

    @property
    def precision(self) -> Precision:
        """Exposed so the query path can assert fp16 on GPU (D18)."""
        ...

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Score passages against the query.

        Returns `(original_index, score)` pairs sorted best-first. Indices refer
        back into `passages` so callers keep their own chunk mapping — the
        reranker never needs to know what a chunk is.
        """
        ...


@runtime_checkable
class EntailmentChecker(Protocol):
    """NLI model used for claim-level citation verification (ML-12, D16).

    This is the **primary** groundedness metric, not a fallback: it is
    deterministic, free, reproducible, and — because the same mechanism runs in
    production and in the eval harness — the reported groundedness number is
    computed by the code the product actually uses. That is what defuses the
    "you used an LLM to judge an LLM" objection (RSK-4).
    """

    @property
    def version(self) -> str: ...

    async def entails(
        self,
        premises: Sequence[str],
        claims: Sequence[str],
    ) -> list[float]:
        """Probability that each premise entails its paired claim.

        `premises[i]` is checked against `claims[i]`; the sequences must be the
        same length.
        """
        ...


class GpuBudgetExceeded(RuntimeError):
    """Raised when a batch cannot fit the VRAM budget.

    Handled by halving the batch and retrying (FR-I10) rather than failing the
    job — classified as RESOURCE_EXHAUSTED, which is retryable.

    Sized against ~6.8 GB *usable*, not the 7.96 GB the device reports: the
    Windows desktop compositor holds 1.1-1.3 GB before we allocate anything
    (REV-3). Budgeting against the nameplate figure produces a phantom surplus
    that only surfaces as an OOM under load.
    """

    def __init__(self, requested_gb: float, available_gb: float) -> None:
        super().__init__(
            f"batch needs ~{requested_gb:.2f} GB but only {available_gb:.2f} GB is free"
        )
        self.requested_gb = requested_gb
        self.available_gb = available_gb
