"""Dense relevance in the request: what is embedded, what is cached, what survives a failure (D153).

What fails silently here is a cached search that still embeds every work (it would
quietly blow NFR-P1's cached budget), a vector served for text that has since
changed, one user's vectors answering another's search, and an encoder failure that
fails the search instead of only its ranking.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from scholiast.application.ports.discovery import VectorKey
from scholiast.application.ports.ml import TextRole
from scholiast.application.use_cases.rank_works import DenseRelevance, LexicalRelevance
from scholiast.domain.dedup import DiscoveredWork, merge
from scholiast.domain.discovery import PaperCandidate
from scholiast.domain.ranking import RelevanceSignal


def work(n: int, abstract: str | None = None) -> DiscoveredWork:
    return merge(
        [
            PaperCandidate(
                source="arxiv",
                source_id=f"{n}",
                rank=n,
                title=f"paper {n}",
                abstract=abstract,
                adapter_version="t",
            )
        ]
    )


class CountingEmbedder:
    """Unit vectors along one axis per text: deterministic, and every call is counted."""

    version = "fake-encoder/1"
    dimensions = 3

    def __init__(self, *, fails: bool = False) -> None:
        self.calls: list[tuple[TextRole, list[str]]] = []
        self.fails = fails

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        self.calls.append((role, list(texts)))
        if self.fails:
            raise RuntimeError("CUDA out of memory")
        return [[1.0, 0.0, 0.0] if "1" in t else [0.0, 1.0, 0.0] for t in texts]

    def passages_embedded(self) -> int:
        return sum(len(texts) for role, texts in self.calls if role is TextRole.PASSAGE)


class MemoryVectorCache:
    def __init__(self, *, raises: bool = False) -> None:
        self.store: dict[VectorKey, list[float]] = {}
        self.raises = raises

    async def get_vectors(self, keys: Sequence[VectorKey]) -> list[list[float] | None]:
        if self.raises:
            raise RuntimeError("a buggy cache")
        return [self.store.get(k) for k in keys]

    async def put_vectors(self, entries: Sequence[tuple[VectorKey, Sequence[float]]]) -> None:
        if self.raises:
            raise RuntimeError("a buggy cache")
        for key, vector in entries:
            self.store[key] = list(vector)


async def test_a_repeated_search_embeds_only_its_query() -> None:
    embedder, cache = CountingEmbedder(), MemoryVectorCache()
    dense = DenseRelevance(embedder, cache)
    works = [work(1), work(2)]

    first = await dense.score("paper 1", works, scope="u-1")
    embedded_first = embedder.passages_embedded()
    second = await dense.score("paper 1", works, scope="u-1")

    assert embedded_first == 2
    assert embedder.passages_embedded() == 2, "the cached search embedded the works again"
    assert second.cached_vectors == 2
    assert second.scores == first.scores
    assert [role for role, _ in embedder.calls].count(TextRole.QUERY) == 2


async def test_the_query_is_scored_against_each_work_by_cosine() -> None:
    relevance = await DenseRelevance(CountingEmbedder()).score(
        "paper 1", [work(1), work(2)], scope="u-1"
    )

    assert relevance.signal is RelevanceSignal.DENSE
    assert relevance.scores == pytest.approx((1.0, 0.0))


async def test_a_work_whose_text_changed_is_embedded_again() -> None:
    embedder, cache = CountingEmbedder(), MemoryVectorCache()
    dense = DenseRelevance(embedder, cache)

    await dense.score("q", [work(1, abstract="old abstract")], scope="u-1")
    await dense.score("q", [work(1, abstract="new abstract")], scope="u-1")

    assert embedder.passages_embedded() == 2, "a vector of the old text ranked the new one"


async def test_one_users_vectors_never_answer_anothers_search() -> None:
    embedder, cache = CountingEmbedder(), MemoryVectorCache()
    dense = DenseRelevance(embedder, cache)

    await dense.score("q", [work(1)], scope="u-1")
    await dense.score("q", [work(1)], scope="u-2")

    assert embedder.passages_embedded() == 2
    assert {k.scope for k in cache.store} == {"u-1", "u-2"}


async def test_a_vector_from_another_encoder_is_never_used() -> None:
    cache = MemoryVectorCache()
    await DenseRelevance(CountingEmbedder(), cache).score("q", [work(1)], scope="u-1")
    other = CountingEmbedder()
    other.version = "another-encoder/2"

    await DenseRelevance(other, cache).score("q", [work(1)], scope="u-1")

    assert other.passages_embedded() == 1


async def test_an_encoder_failure_costs_the_ranking_not_the_search() -> None:
    relevance = await DenseRelevance(CountingEmbedder(fails=True)).score(
        "paper 1", [work(1), work(2)], scope="u-1"
    )

    assert relevance.signal is RelevanceSignal.LEXICAL
    assert len(relevance.scores) == 2


async def test_a_cache_that_raises_is_a_cache_that_missed() -> None:
    embedder = CountingEmbedder()

    relevance = await DenseRelevance(embedder, MemoryVectorCache(raises=True)).score(
        "paper 1", [work(1), work(2)], scope="u-1"
    )

    assert relevance.signal is RelevanceSignal.DENSE
    assert embedder.passages_embedded() == 2


async def test_no_works_asks_the_encoder_nothing() -> None:
    embedder = CountingEmbedder()

    relevance = await DenseRelevance(embedder).score("q", [], scope="u-1")

    assert relevance.scores == ()
    assert embedder.calls == []


async def test_lexical_relevance_needs_no_encoder() -> None:
    relevance = await LexicalRelevance().score("paper 1", [work(1), work(2)], scope="u-1")

    assert relevance.signal is RelevanceSignal.LEXICAL
    assert relevance.scores[0] > relevance.scores[1]
