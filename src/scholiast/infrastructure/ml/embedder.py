"""Local dense encoder (ML-4, §6.4, D18).

Three details here are load-bearing, and two of them fail *silently* when wrong —
which is the entire reason they are asserted rather than assumed.

**Instruction prefixes.** BGE is asymmetric: a query must be prefixed, a passage
must not. §6.4 calls getting this wrong "degrades retrieval silently — no error,
just worse results", and it is the classic interview gotcha. The prefix is
applied here, in the adapter, so no call site can forget it, and it is
unit-tested.

**fp16 is specification, not optimisation** (D18, §6.16). Measured M0: reranking
50 candidates costs 743 ms at fp32 versus 224 ms at fp16, which at fp32 consumes
~93% of NFR-P2's whole-stage budget on its own. The adapter sets it explicitly
and reports it, and the reported precision is part of `version`.

**Precision is part of the version stamp.** fp16 and fp32 do not produce
identical vectors, so under NFR-M5 they are different embedders for the purpose
of "re-embed only what changed". CPU runs fp32 — fp16 on CPU is usually slower,
not faster — so the CPU path stamps a different version than the GPU path, on
purpose.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from scholiast.application.ports.ml import GpuBudgetExceeded, Precision, TextRole
from scholiast.infrastructure.ml.gpu import GpuPriorityGate, Priority

if TYPE_CHECKING:  # pragma: no cover - import cost is the point
    from sentence_transformers import SentenceTransformer

#: BGE v1.5's documented retrieval instruction. Queries only — prefixing a
#: passage with this actively hurts, because the model was trained with the
#: instruction on one side of the pair.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

#: Model families that take the BGE-style query instruction. Anything else gets
#: no prefix rather than a guessed one: a wrong instruction is worse than none,
#: and silently so.
_QUERY_PREFIXES: dict[str, str] = {
    "bge": BGE_QUERY_PREFIX,
    "gte": "",
    "e5": "query: ",
}

#: Ingestion batches are bounded (§5.7). 32 is a throughput compromise for M1,
#: where embedding is the only GPU consumer. M2 tightens this to the ~50-100 ms
#: batches the priority semaphore needs, once a reranker exists to contend with.
DEFAULT_BATCH_SIZE = 32


def _prefix_for(model_name: str) -> str:
    lowered = model_name.lower()
    for family, prefix in _QUERY_PREFIXES.items():
        if family in lowered:
            return prefix
    return ""


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether this exception means "the GPU had no room".

    Two ways of asking, because there are two ways torch says it. Modern builds
    raise `torch.cuda.OutOfMemoryError`; older ones, and some allocator paths,
    raise a plain `RuntimeError` whose message carries "out of memory". Matching
    on the message as well as the type is also what makes the halving ladder
    testable without a GPU — CI has none, so a test that could only raise the
    typed error would exercise nothing there.

    Torch is imported lazily and its absence is not an error: this adapter is
    constructed on CPU-only machines, where a CUDA OOM cannot occur and pulling
    in torch to name an impossible exception would be a large import for nothing.
    """
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    try:
        import torch
    except ImportError:  # pragma: no cover - CPU-only environments
        return False
    return isinstance(exc, torch.cuda.OutOfMemoryError)


def _release_cached_blocks() -> None:
    """Return a failed allocation's reservation to the device.

    A caught OOM does not free what torch had already reserved for the attempt,
    so a retry at half the size can fail against memory nothing is using.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - never worth failing an encode over
        return


def _reserved_gb() -> float:
    """What torch had reserved when the allocation failed. 0.0 without CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.memory_reserved()) / 1024**3
    except Exception:  # pragma: no cover - diagnostic only
        return 0.0
    return 0.0


def _free_gb() -> float:
    """What the device reports free. See D24 on why this is NOT a budget.

    It excludes other processes entirely on Windows WDDM, so it is carried as a
    diagnostic on the exception rather than used to decide anything.
    """
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return float(free) / 1024**3
    except Exception:  # pragma: no cover - diagnostic only
        return 0.0
    return 0.0


class LocalEmbedder:
    """`Embedder` adapter over sentence-transformers."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        precision: Precision = Precision.FP16,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gate: GpuPriorityGate | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        # Opt-in (D74). With no gate the adapter keeps its original single `encode`
        # call, byte for byte, so every published phase 2 and 3 number reproduces.
        # With a gate it must loop over batches itself — the only way a waiting
        # rerank can go ahead between two of them.
        self._gate = gate
        # fp16 on CPU is generally slower than fp32 and unsupported for some
        # ops, so the CPU path runs fp32 and says so rather than pretending.
        self._precision = precision if device.startswith("cuda") else Precision.FP32
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._query_prefix = _prefix_for(model_name)

    @property
    def version(self) -> str:
        """Stamped onto every embedded chunk (NFR-M5).

        Includes precision because fp16 and fp32 vectors differ: a targeted
        re-index of "everything embedded at fp32" is only expressible if the two
        carry different stamps.
        """
        return f"{self._model_name}/{self._precision.value}/1.0.0"

    @property
    def precision(self) -> Precision:
        """Exposed so tests can assert D18 rather than trust it."""
        return self._precision

    @property
    def dimensions(self) -> int:
        return int(self._load().get_sentence_embedding_dimension() or 0)

    @property
    def device(self) -> str:
        model = self._load()
        return str(next(model.parameters()).device)

    def _load(self) -> SentenceTransformer:
        """Load on first use, not at construction.

        Constructing an adapter should not cost two seconds and half a gigabyte
        of VRAM — dependency wiring happens at import time, and tests that never
        embed anything should not pay for a model.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._model_name, device=self._device)
            if self._precision is Precision.FP16:
                model = model.half()
            self._model = model
        return self._model

    def _apply_prefix(self, texts: Sequence[str], role: TextRole) -> list[str]:
        if role is TextRole.QUERY and self._query_prefix:
            return [f"{self._query_prefix}{text}" for text in texts]
        return list(texts)

    async def embed(
        self,
        texts: Sequence[str],
        *,
        role: TextRole,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        prepared = self._apply_prefix(texts, role)
        size = batch_size or self._batch_size

        # sentence-transformers is synchronous and releases the GIL inside
        # torch, but a multi-second encode on the event loop would stall every
        # in-flight request. The API and the worker share a process in M1.
        if self._gate is None:
            return await asyncio.to_thread(self._encode, prepared, size)

        # Gated: one GPU acquisition PER BATCH, releasing between batches. Holding
        # the gate for the whole call would make a rerank wait for an entire paper
        # instead of one batch — §5.7's bound is "at most one batch", and it only
        # holds if the gate is released at batch boundaries.
        #
        # Priority follows the ROLE, not the adapter. A query embedding is on the
        # user-facing path exactly as a rerank is; giving it ingestion priority
        # would queue a user's question behind a paper being indexed, which is the
        # precise inversion §5.7 exists to prevent.
        priority = Priority.QUERY if role is TextRole.QUERY else Priority.INGESTION
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), size):
            async with self._gate.hold(priority):
                vectors.extend(
                    await asyncio.to_thread(self._encode, prepared[start : start + size], size)
                )
        return vectors

    def _encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Encode, halving the batch on an out-of-memory error (FR-I10, D98).

        **Reactive, not predictive.** No VRAM is estimated before allocating,
        because on Windows WDDM there is nothing trustworthy to estimate against:
        D24 measured that `nvidia-smi` reports `0 MiB`/`[N/A]` and torch sees only
        its own context, so a budget check would be blind to the desktop
        compositor's 1.1-1.3 GB and to every other process. Letting the
        allocation fail cannot be wrong about what it never measured.

        **Two levels, because two different things go wrong.** A batch that is
        momentarily too large is the caller's sizing problem and is solved here,
        in milliseconds, by trying a smaller one. A *single item* that cannot
        allocate is a capacity problem this process cannot solve by trying
        harder, so it escalates to `GpuBudgetExceeded` → `RESOURCE_EXHAUSTED` →
        a requeue with backoff, which runs it when whatever else held the GPU
        has finished.

        The cache is emptied between attempts because a caught OOM leaves the
        failed allocation's reservation in torch's pool; retrying without it
        would keep failing against memory nothing is using.
        """
        model = self._load()
        attempt = batch_size
        while True:
            try:
                return self._encode_at(model, texts, attempt)
            except Exception as exc:
                if not _is_out_of_memory(exc):
                    raise
                if attempt <= 1:
                    # A single item will not fit. Nothing smaller to try.
                    raise GpuBudgetExceeded(
                        requested_gb=_reserved_gb(), available_gb=_free_gb()
                    ) from exc
                attempt = max(1, attempt // 2)
                _release_cached_blocks()

    @staticmethod
    def _encode_at(
        model: SentenceTransformer, texts: list[str], batch_size: int
    ) -> list[list[float]]:
        vectors: Any = model.encode(
            texts,
            batch_size=batch_size,
            # BGE expects L2-normalised vectors for cosine similarity. Qdrant
            # normalises for COSINE too, but normalising here keeps the vectors
            # directly comparable wherever else they are used.
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]
