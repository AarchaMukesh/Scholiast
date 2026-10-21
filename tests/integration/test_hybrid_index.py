"""The hybrid collection against a real Qdrant (D172, D174).

What a fake cannot establish: that a point written with both vectors is searchable
by both arms through the store's own filter and IDF, that the lexical arm honours
the tenant filter the way the dense arm does, that coverage is read by payload
version, that a rebuild run twice leaves the collection byte-for-byte as it
was, and that a process losing a race to create the collection gets the refusal
it expects. Each test uses its own collection and deletes it.
"""

from __future__ import annotations

import hashlib
import json
import struct
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from scholiast.application.ports.vector import VectorRecord
from scholiast.application.use_cases.rebuild_index import RebuildIndexUseCase
from scholiast.domain.lexical import LEXICAL_VERSION, chunk_weights, query_weights, tokenize
from scholiast.domain.paper import SectionKind
from scholiast.domain.retrieval import RetrievalArm, reciprocal_rank_fusion
from scholiast.infrastructure.vector.qdrant import CollectionNotFound, QdrantVectorIndex
from tests.integration.test_vector_index import QDRANT_URL, needs_qdrant, unit
from tests.unit import test_rebuild_index as rebuild


def record(chunk_id: str, paper_id: str, text: str, vector: list[float]) -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        paper_id=paper_id,
        vector=tuple(vector),
        section_kind=SectionKind.METHOD,
        chunker_version="structure-aware/1.2.0",
        embedder_version="test/fp16/1.0.0",
        lexical=chunk_weights(text),
        lexical_version=LEXICAL_VERSION,
    )


@pytest_asyncio.fixture
async def hybrid() -> AsyncIterator[QdrantVectorIndex]:
    store = QdrantVectorIndex(
        QDRANT_URL, collection=f"test_hybrid_{uuid.uuid4().hex[:10]}", dimensions=8, sparse=True
    )
    await store.ensure_ready()
    yield store
    await store._client.delete_collection(store.collection)
    await store.close()


async def fingerprint(index: QdrantVectorIndex) -> str:
    """Every point's id, payload and both vectors, in id order."""
    digest = hashlib.sha256()
    points, offset = [], None
    while True:
        page, offset = await index._client.scroll(
            index.collection, limit=256, offset=offset, with_payload=True, with_vectors=True
        )
        points.extend(page)
        if offset is None:
            break
    for point in sorted(points, key=lambda p: str(p.id)):
        digest.update(str(point.id).encode())
        digest.update(json.dumps(point.payload, sort_keys=True).encode())
        vectors = point.vector
        assert isinstance(vectors, dict)
        dense = vectors["dense"]
        digest.update(struct.pack(f"<{len(dense)}f", *dense))
        sparse = vectors.get("lexical")
        if sparse is not None:
            digest.update(struct.pack(f"<{len(sparse.indices)}I", *sparse.indices))
            digest.update(struct.pack(f"<{len(sparse.values)}f", *sparse.values))
    return digest.hexdigest()


@pytest.mark.integration
@needs_qdrant
class TestTheHybridCollectionAgainstRealQdrant:
    async def test_an_identifier_the_dense_arm_misses_is_lifted_by_the_lexical_arm(
        self, hybrid: QdrantVectorIndex
    ) -> None:
        # D167's reason for the lexical arm, end to end through the store's IDF.
        # Cosines to the query vector of 1.0, 0.71 and 0.0: the dense arm ranks
        # the identifier's chunk last, and no two chunks tie.
        halfway = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        await hybrid.upsert(
            [
                record("p:c0", "p", "Results for the base model on our benchmark.", unit(0)),
                record("p:c1", "p", "Discussion of training cost and data.", halfway),
                record("p:c2", "p", "LLaMA-2-70B reaches 71.3 on the benchmark.", unit(5)),
            ]
        )
        question = "How does LLaMA-2-70B score?"
        dense = await hybrid.search(unit(0), paper_ids=["p"], limit=50)
        lexical = await hybrid.search_lexical(
            query_weights(tokenize(question)), paper_ids=["p"], limit=50
        )

        assert [c.chunk_id for c in dense] == ["p:c0", "p:c1", "p:c2"]
        assert [c.chunk_id for c in lexical] == ["p:c2"]
        assert lexical[0].arm is RetrievalArm.LEXICAL
        # Last for one arm and first for the other beats first for one alone.
        fused = reciprocal_rank_fusion([dense, lexical], limit=50)
        assert fused[0].chunk_id == "p:c2"

    async def test_the_lexical_arm_cannot_cross_into_another_tenants_paper(
        self, hybrid: QdrantVectorIndex
    ) -> None:
        # §5.8 layer 3 for the lexical arm: identical text, different papers.
        text = "LLaMA-2-70B reaches 71.3 on the benchmark."
        await hybrid.upsert(
            [record("mine:c0", "mine", text, unit(0)), record("theirs:c0", "theirs", text, unit(0))]
        )
        hits = await hybrid.search_lexical(
            query_weights(tokenize("LLaMA-2-70B")), paper_ids=["mine"], limit=10
        )
        assert [h.paper_id for h in hits] == ["mine"]

    async def test_coverage_is_read_by_lexical_version(self, hybrid: QdrantVectorIndex) -> None:
        # After a weighting change, a point written before it is not current.
        current = record("p:c0", "p", "current weights", unit(0))
        stale = VectorRecord(
            chunk_id="p:c1",
            paper_id="p",
            vector=tuple(unit(1)),
            section_kind=SectionKind.METHOD,
            chunker_version="structure-aware/1.2.0",
            embedder_version="test/fp16/1.0.0",
            lexical=chunk_weights("older weights"),
            lexical_version="bm25-identifier/1.0.0",
        )
        await hybrid.upsert([current, stale])

        assert await hybrid.lexical_chunk_ids(lexical_version=LEXICAL_VERSION) == {"p:c0"}
        assert await hybrid.count_lexical(lexical_version=LEXICAL_VERSION) == 1

    async def test_a_rebuild_run_twice_leaves_the_collection_byte_for_byte_the_same(
        self,
    ) -> None:
        # The migration runs `--hybrid` twice; the second must converge, not drift.
        store = QdrantVectorIndex(
            QDRANT_URL, collection=f"test_hybrid_{uuid.uuid4().hex[:10]}", dimensions=4, sparse=True
        )
        try:
            await store.ensure_ready()
            chunks = {rebuild.PAPER_ID: [rebuild.a_chunk(ordinal=0), rebuild.a_chunk(ordinal=1)]}

            async def run() -> str:
                use_case = RebuildIndexUseCase(
                    papers=rebuild.FakePapers({rebuild.PAPER_ID: rebuild.a_paper()}),  # type: ignore[arg-type]
                    chunks=rebuild.FakeChunks(chunks),  # type: ignore[arg-type]
                    embedder=rebuild.FakeEmbedder(),
                    index=store,
                )
                await use_case.execute()
                return await fingerprint(store)

            first = await run()
            assert await store.count() == 2
            assert await store.lexical_chunk_ids(lexical_version=LEXICAL_VERSION) == {
                c.id for c in chunks[rebuild.PAPER_ID]
            }
            assert await run() == first
        finally:
            await store._client.delete_collection(store.collection)
            await store.close()

    async def test_requiring_a_missing_collection_does_not_create_it(self) -> None:
        name = f"test_hybrid_{uuid.uuid4().hex[:10]}"
        store = QdrantVectorIndex(QDRANT_URL, collection=name, dimensions=8, sparse=True)
        try:
            with pytest.raises(CollectionNotFound):
                await store.require_existing()
            assert await store._client.collection_exists(name) is False
        finally:
            await store.close()

    async def test_a_process_that_loses_the_create_race_starts_anyway(self) -> None:
        # D174: two processes starting on an empty corpus can both find the
        # collection missing. The second's create is refused by the real store,
        # and that refusal must read as a race lost, not as a failed start.
        name = f"test_hybrid_{uuid.uuid4().hex[:10]}"
        first = QdrantVectorIndex(QDRANT_URL, collection=name, dimensions=8, sparse=True)
        second = QdrantVectorIndex(QDRANT_URL, collection=name, dimensions=8, sparse=True)
        try:
            await first.ensure_ready()
            client = second._client
            real = client.collection_exists
            asked: list[str] = []

            async def missing_at_first(collection_name: str, **kwargs: object) -> bool:
                # What the second process saw just before the first created it.
                asked.append(collection_name)
                return False if len(asked) == 1 else await real(collection_name, **kwargs)

            client.collection_exists = missing_at_first  # type: ignore[method-assign]
            await second.ensure_ready()
            # Missing, the create refused, then found: the race lost, not a failure.
            assert len(asked) == 2
            await second.require_existing()
        finally:
            await first._client.delete_collection(name)
            await first.close()
            await second.close()
