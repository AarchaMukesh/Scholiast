"""Relevance for discovery's ranking: dense, with work vectors cached per user (D151, D153).

**What is embedded.** The query as a query, and each work's "title. abstract" as a
passage (D151), with the API's loaded encoder: the same one, at the same precision,
that produced the committed scores the weights were fitted on (D152).

**A cached search embeds only its query** (D153). Each work's vector is looked up
in the vector cache first, per user and keyed by the encoder build and a hash of
the exact text; only the misses are embedded, and they are stored for next time.
Measured before this cache existed: dense ranking took p95 351 ms per search on an
idle GPU, most of NFR-P1's 400 ms for a cached search.

**An encoder failure costs the ranking's quality, never the search** (FR-D6's rule,
applied to ranking). If embedding raises, relevance falls back to lexical BM25
over the same texts, and the response says which signal ranked it. The cache is
isolated the same way: a cache that raises is a cache that missed.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import structlog

from scholiast.application.ports.discovery import VectorCache, VectorKey
from scholiast.application.ports.ml import Embedder, TextRole
from scholiast.domain.dedup import DiscoveredWork
from scholiast.domain.ranking import RelevanceSignal, lexical_relevance, relevance_text

log = structlog.get_logger(__name__)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


@dataclass(frozen=True, slots=True)
class Relevance:
    scores: tuple[float, ...]
    #: Which signal produced them: dense, or lexical when the encoder failed.
    signal: RelevanceSignal
    #: How many work vectors came from the cache rather than the encoder.
    cached_vectors: int = 0


class RelevanceScorer(Protocol):
    """Relevance of each work to the query, in the works' order."""

    async def score(
        self, query: str, works: Sequence[DiscoveredWork], *, scope: str
    ) -> Relevance: ...


class DenseRelevance:
    """Cosine between the query and each work, with work vectors cached per user."""

    def __init__(self, embedder: Embedder, cache: VectorCache | None = None) -> None:
        self._embedder = embedder
        self._cache = cache

    async def score(self, query: str, works: Sequence[DiscoveredWork], *, scope: str) -> Relevance:
        texts = [relevance_text(w) for w in works]
        if not works:
            return Relevance((), RelevanceSignal.DENSE)
        try:
            return await self._dense(query, works, texts, scope)
        except Exception as exc:
            # An encoder that fails (out of memory, a driver error, a bug) must not
            # fail the search; lexical relevance needs nothing but the texts.
            log.error(
                "discovery.dense_relevance_failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return Relevance(tuple(lexical_relevance(query, texts)), RelevanceSignal.LEXICAL)

    async def _dense(
        self, query: str, works: Sequence[DiscoveredWork], texts: list[str], scope: str
    ) -> Relevance:
        keys = [
            VectorKey(
                scope=scope,
                embedder=self._embedder.version,
                record=f"{w.add.source}:{w.add.source_id}",
                text_sha256=text_sha256(text),
            )
            for w, text in zip(works, texts, strict=True)
        ]
        vectors = await self._lookup(keys)
        missing = [i for i, v in enumerate(vectors) if v is None]
        [query_vector] = await self._embedder.embed([query], role=TextRole.QUERY)
        if missing:
            fresh = await self._embedder.embed([texts[i] for i in missing], role=TextRole.PASSAGE)
            for i, vector in zip(missing, fresh, strict=True):
                vectors[i] = vector
            await self._store([(keys[i], vectors[i]) for i in missing])
        scores = tuple(cosine(query_vector, v) for v in vectors if v is not None)
        if len(scores) != len(works):
            raise RuntimeError("the encoder returned fewer vectors than it was given texts")
        return Relevance(scores, RelevanceSignal.DENSE, cached_vectors=len(works) - len(missing))

    async def _lookup(self, keys: list[VectorKey]) -> list[list[float] | None]:
        if self._cache is None:
            return [None] * len(keys)
        try:
            found = await self._cache.get_vectors(keys)
        except Exception:
            # The port forbids raising; a cache with a bug must still only miss.
            return [None] * len(keys)
        if len(found) != len(keys):
            return [None] * len(keys)
        return list(found)

    async def _store(self, entries: list[tuple[VectorKey, list[float] | None]]) -> None:
        if self._cache is None:
            return
        try:
            await self._cache.put_vectors([(k, v) for k, v in entries if v is not None])
        except Exception:
            # A cache that cannot store costs the next search an embedding, nothing more.
            return


class LexicalRelevance:
    """BM25 relevance, for a deployment with no encoder: what dense falls back to."""

    async def score(self, query: str, works: Sequence[DiscoveredWork], *, scope: str) -> Relevance:
        del scope  # lexical relevance has nothing per user to look up
        texts = [relevance_text(w) for w in works]
        return Relevance(tuple(lexical_relevance(query, texts)), RelevanceSignal.LEXICAL)
