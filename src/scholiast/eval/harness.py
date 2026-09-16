"""Build the eval index and measure retrieval over it (FR-E2, FR-E4, D70).

Two responsibilities that belong together because they share one invariant: the
index a run measures must be the index the dataset describes. Splitting them across
modules is how a report ends up computed against a stale collection.

**The collection name carries the dataset's identity** (NFR-M5). It is derived from
the D-RET, builder, chunker and embedder versions, so changing any of them names a
*different* collection rather than querying a stale one. The alternative — one fixed
name plus a remembered rebuild — fails silently and produces numbers for a corpus
that no longer exists, which is the exact class of error version stamps exist to
prevent.

**Retrieval spans the whole corpus** (D70). Every query ranks against all 281
papers; judgments live only in its own paper, so a hit elsewhere is a false
positive. Paper-scoping would make Recall@50 degenerate for 96% of CORE queries and
destroy §9.3's reranking-ceiling diagnostic before phase 4 can use it.

**This is the dense-only baseline** (D56) — the ablation *control*, not the adopted
design. The lexical arm, fusion and the cross-encoder arrive in later phases
precisely so their contribution is measured against this number rather than assumed.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.vector import SparseRecord, VectorIndex, VectorRecord
from scholiast.domain.lexical import (
    average_length,
    document_weights,
    query_weights,
    tokenize,
)
from scholiast.domain.paper import Chunk
from scholiast.domain.retrieval import ScoredChunk, reciprocal_rank_fusion
from scholiast.eval.dret import DRet, Grade, Query
from scholiast.eval.metrics import (
    NDCG_CUTOFF,
    RECALL_CUTOFFS,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class EmbedderLike(Protocol):
    """Just the surface the harness needs, so tests can supply a stand-in.

    Narrower than `application.ports.ml.Embedder` on purpose: the harness never
    calls `dimensions`, and a Protocol that demands more than it uses forces every
    test double to implement dead methods.
    """

    @property
    def version(self) -> str: ...

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]: ...


def collection_name(dataset: DRet, embedder_version: str, *, prefix: str = "scholiast_eval") -> str:
    """A name that changes whenever anything upstream of the vectors changes.

    Hashed rather than concatenated because embedder versions contain `/` and Qdrant
    collection names are path segments. The prefix stays readable so a human
    listing collections can tell eval indexes from the production one at a glance.
    """
    identity = "|".join(
        [*sorted(f"{k}={v}" for k, v in dataset.versions.items()), embedder_version]
    )
    return f"{prefix}_{hashlib.sha256(identity.encode()).hexdigest()[:12]}"


@dataclass(slots=True)
class IndexReport:
    """What building the index did, so a run can say whether it rebuilt or reused."""

    collection: str
    chunks_indexed: int
    already_present: int
    seconds: float

    @property
    def rebuilt(self) -> bool:
        return self.chunks_indexed > 0


async def build_index(
    dataset: DRet,
    index: VectorIndex,
    embedder: EmbedderLike,
    *,
    collection: str,
    batch_size: int = 32,
) -> IndexReport:
    """Embed and index the corpus. Idempotent, and skips the work when complete.

    Chunk ids are deterministic (D46) and the collection name encodes the versions
    that produce them, so a complete collection cannot be stale — which is what
    makes "skip if the count matches" safe rather than a shortcut.

    `collection` is passed rather than read off the adapter: the name is the
    caller's decision (it encodes the dataset identity) and the port does not
    expose it. Reaching for it with `getattr` would silently report "?" the day an
    adapter renamed its private attribute, which is the kind of thing a report
    carries for months.
    """
    started = time.perf_counter()
    await index.ensure_ready()
    present = await index.count()
    if present == len(dataset.chunks):
        return IndexReport(
            collection=collection,
            chunks_indexed=0,
            already_present=present,
            seconds=time.perf_counter() - started,
        )

    ordered: list[Chunk] = [dataset.chunks[cid] for cid in sorted(dataset.chunks)]
    indexed = 0
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        missing = {c.provenance.paper_id for c in batch} - dataset.papers.keys()
        if missing:
            # Raised rather than defaulted to "". The contextual header is ML-3's
            # cheapest quality win and it is what marks a Related Work chunk as
            # describing someone else's contribution; embedding without it degrades
            # retrieval with no error and no symptom except worse numbers.
            raise KeyError(
                f"chunks reference papers absent from the dataset: {sorted(missing)[:3]} — "
                "the corpus and the chunk set disagree, so the contextual header "
                "(ML-3) cannot be built"
            )
        titles = {c.provenance.paper_id: dataset.papers[c.provenance.paper_id].title for c in batch}
        # ML-3's contextual header, the same one ingestion embeds. Omitting it here
        # would measure a different embedding than production produces.
        vectors = await embedder.embed(
            [c.embedding_input(titles[c.provenance.paper_id], None) for c in batch],
            role=TextRole.PASSAGE,
            batch_size=batch_size,
        )
        indexed += await index.upsert(
            [
                VectorRecord(
                    chunk_id=c.id,
                    paper_id=c.provenance.paper_id,
                    vector=tuple(v),
                    section_kind=c.provenance.section_kind,
                    chunker_version=c.chunker_version,
                    embedder_version=embedder.version,
                )
                for c, v in zip(batch, vectors, strict=True)
            ]
        )
    return IndexReport(
        collection=collection,
        chunks_indexed=indexed,
        already_present=present,
        seconds=time.perf_counter() - started,
    )


@dataclass(slots=True)
class MetricSet:
    """Per-query metric values, kept as lists so §9.4's bootstrap can resample them.

    Means alone would make confidence intervals impossible to compute after the
    fact, and doc 9 is explicit that a point estimate without an interval is, at
    this n, frequently noise.
    """

    recall: dict[int, list[float]] = field(default_factory=dict)
    ndcg: list[float] = field(default_factory=list)
    rr: list[float] = field(default_factory=list)

    def means(self) -> dict[str, float]:
        out = {f"recall@{k}": _mean(v) for k, v in sorted(self.recall.items())}
        out[f"ndcg@{NDCG_CUTOFF}"] = _mean(self.ndcg)
        out["mrr"] = _mean(self.rr)
        return out

    @property
    def n(self) -> int:
        return len(self.rr)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(slots=True)
class RunReport:
    """One configuration measured over one query set."""

    label: str
    metrics: MetricSet
    #: Per-query retrieval latency in milliseconds. FR-E5 wants percentiles per
    #: stage; this is the retrieve stage, recorded from the first run rather than
    #: retrofitted onto runs that never had it.
    latencies_ms: list[float] = field(default_factory=list)
    #: The same latency split into its two stages. Kept separate because the
    #: embedding-model comparison (ML-4) turns on exactly this: a smaller encoder
    #: changes the *embed* cost and the vector width, and leaves the ANN search
    #: cost to move for a different reason. One combined number would attribute
    #: both to "retrieval" and make the tradeoff unreadable.
    embed_ms: list[float] = field(default_factory=list)
    search_ms: list[float] = field(default_factory=list)
    #: Share of the top 10 that came from the query's own paper. A **diagnostic**,
    #: not a headline metric: under D70's whole-corpus scope a low Recall@10 has two
    #: very different causes — the right paper was never reached, or it was reached
    #: and the wrong chunk within it was ranked first. This separates them, the same
    #: way §9.3's Recall@50-pre-rerank separates a weak reranker from a weak
    #: candidate set. Measured on the dense-only baseline: 2.28 of 10.
    own_paper_in_top10: list[float] = field(default_factory=list)

    def latency_percentiles(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {}
        ordered = sorted(self.latencies_ms)
        return {
            name: ordered[min(len(ordered) - 1, int(len(ordered) * q))]
            for name, q in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
        }


async def run_retrieval(
    dataset: DRet,
    queries: Sequence[Query],
    index: VectorIndex,
    embedder: EmbedderLike,
    *,
    label: str,
    paper_ids: Sequence[str],
    depth: int = max(RECALL_CUTOFFS),
) -> RunReport:
    """Retrieve for every query and score the ranking. One configuration, one pass.

    `paper_ids` is the retrieval scope and is passed explicitly rather than derived
    here: the port requires it (§5.8 layer 3), and D70's choice of the whole corpus
    is a decision the caller states rather than a default this function invents.
    """
    report = RunReport(label=label, metrics=MetricSet(recall={k: [] for k in RECALL_CUTOFFS}))

    # One untimed query first. Without it the first measurement carries CUDA
    # context creation and the encoder's first forward pass: measured, that put
    # CORE's retrieve p99 at 13,595 ms against a p50 of 16 ms, which would have
    # been published as a retrieval latency (FR-E5) when it is a cold start. The
    # larger FULL set hid it by diluting the single outlier below the percentile —
    # so the defect was visible on n=53 and invisible on n=259, which is the worst
    # way for a measurement bug to behave.
    if queries:
        warm = (await embedder.embed([queries[0].text], role=TextRole.QUERY))[0]
        await index.search(warm, paper_ids=paper_ids, limit=depth)

    for query in queries:
        qrels: dict[str, int] = {
            cid: int(grade)
            for cid, grade in dataset.qrels(query.id).items()
            if grade > Grade.NOT_RELEVANT
        }
        if not qrels:
            raise ValueError(
                f"query {query.id!r} reached scoring with no judgments — D-RET drops "
                "these at build time, so the dataset and the query list disagree"
            )
        started = time.perf_counter()
        vector = (await embedder.embed([query.text], role=TextRole.QUERY))[0]
        embedded = time.perf_counter()
        hits = await index.search(vector, paper_ids=paper_ids, limit=depth)
        finished = time.perf_counter()
        report.embed_ms.append((embedded - started) * 1000.0)
        report.search_ms.append((finished - embedded) * 1000.0)
        report.latencies_ms.append((finished - started) * 1000.0)

        ranking = [hit.chunk_id for hit in hits]
        for k in RECALL_CUTOFFS:
            report.metrics.recall[k].append(recall_at_k(ranking, qrels, k))
        report.metrics.ndcg.append(ndcg_at_k(ranking, qrels, NDCG_CUTOFF))
        report.metrics.rr.append(reciprocal_rank(ranking, qrels))
        # Read off the hits already in hand — no second search, so the diagnostic
        # costs nothing beyond the arithmetic.
        report.own_paper_in_top10.append(
            sum(1 for hit in hits[:10] if hit.paper_id == query.paper_id) / 10.0
        )
    return report


class LexicalLike(Protocol):
    """The lexical surface the harness needs (doc 6 section 6.5, D73)."""

    async def upsert_sparse(self, records: Sequence[SparseRecord]) -> int: ...

    async def search_lexical(
        self, weights: dict[int, float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]: ...


async def build_lexical_index(dataset: DRet, index: LexicalLike) -> int:
    """Attach BM25 term weights to the corpus. Pure arithmetic plus one upsert.

    `avgdl` is computed over the WHOLE corpus before any weight is emitted, not
    per batch: BM25's length normalisation divides by it, so a per-batch average
    would make a chunk's weight depend on which batch it landed in and the same
    corpus indexed at a different batch size would score differently (D73).
    """
    tokens = {cid: tokenize(chunk.text) for cid, chunk in dataset.chunks.items()}
    avgdl = average_length(tokens.values())
    records = [
        SparseRecord(
            chunk_id=cid,
            paper_id=dataset.chunks[cid].provenance.paper_id,
            weights=document_weights(toks, average_length=avgdl),
        )
        for cid, toks in sorted(tokens.items())
        if toks
    ]
    written = 0
    for start in range(0, len(records), 256):
        written += await index.upsert_sparse(records[start : start + 256])
    return written


async def run_hybrid(
    dataset: DRet,
    queries: Sequence[Query],
    index: VectorIndex,
    lexical: LexicalLike,
    embedder: EmbedderLike,
    *,
    label: str,
    paper_ids: Sequence[str],
    arms: str = "hybrid",
    depth: int = max(RECALL_CUTOFFS),
) -> RunReport:
    """One pass over the query set for a lexical-only or RRF-fused configuration.

    `arms` selects the configuration under test: "lexical" runs BM25 alone,
    "hybrid" fuses both by RRF. Dense-only keeps its own function, unchanged, so
    the phase 2 baseline cannot drift because a shared code path was edited.

    **Each arm retrieves `depth` and the fused list is truncated to `depth`**, per
    doc 6 section 6.6's candidate budget. Fusing truncated lists rather than full
    ones is what makes RRF's rank positions mean the same thing in both arms.
    """
    report = RunReport(label=label, metrics=MetricSet(recall={k: [] for k in RECALL_CUTOFFS}))
    if queries:
        warm = tokenize(queries[0].text)
        await lexical.search_lexical(query_weights(warm), paper_ids=paper_ids, limit=depth)
        if arms == "hybrid":
            warm_vector = (await embedder.embed([queries[0].text], role=TextRole.QUERY))[0]
            await index.search(warm_vector, paper_ids=paper_ids, limit=depth)

    for query in queries:
        qrels: dict[str, int] = {
            cid: int(grade)
            for cid, grade in dataset.qrels(query.id).items()
            if grade > Grade.NOT_RELEVANT
        }
        if not qrels:
            raise ValueError(f"query {query.id!r} reached scoring with no judgments")

        # Query PREPARATION first, both arms', so `embed_ms` means the same thing
        # here as in the dense-only runner. The first version of this timed only
        # tokenisation and left the GPU embed inside `search_ms` — which reported
        # `embed p50 0.02ms` for hybrid and quietly attributed the encoder's cost
        # to the store. A mislabelled stage is worse than an unmeasured one.
        started = time.perf_counter()
        sparse = query_weights(tokenize(query.text))
        vector = (
            None
            if arms == "lexical"
            else (await embedder.embed([query.text], role=TextRole.QUERY))[0]
        )
        prepared = time.perf_counter()

        lexical_hits = await lexical.search_lexical(sparse, paper_ids=paper_ids, limit=depth)
        if vector is None:
            hits = lexical_hits
        else:
            dense_hits = await index.search(vector, paper_ids=paper_ids, limit=depth)
            hits = reciprocal_rank_fusion([dense_hits, lexical_hits], limit=depth)
        finished = time.perf_counter()

        report.embed_ms.append((prepared - started) * 1000.0)
        report.search_ms.append((finished - prepared) * 1000.0)
        report.latencies_ms.append((finished - started) * 1000.0)

        ranking = [hit.chunk_id for hit in hits]
        for k in RECALL_CUTOFFS:
            report.metrics.recall[k].append(recall_at_k(ranking, qrels, k))
        report.metrics.ndcg.append(ndcg_at_k(ranking, qrels, NDCG_CUTOFF))
        report.metrics.rr.append(reciprocal_rank(ranking, qrels))
        report.own_paper_in_top10.append(
            sum(1 for hit in hits[:10] if hit.paper_id == query.paper_id) / 10.0
        )
    return report
