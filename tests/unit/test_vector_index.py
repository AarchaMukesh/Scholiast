"""Vector index behaviour that is decidable without a running Qdrant.

The point-id mapping (D46) and the mandatory-filter refusal (§5.8 layer 3) are
pure logic, and both are properties that must never quietly change — the first
because it is baked into stored data, the second because it is a tenancy
boundary. `tests/integration/test_vector_index.py` covers the rest against a
real Qdrant.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models

from scholiast.application.ports.vector import (
    EmptyRetrievalScope,
    VectorDimensionMismatch,
    VectorRecord,
)
from scholiast.domain.paper import SectionKind
from scholiast.infrastructure.vector.qdrant import (
    DENSE_VECTOR,
    PAYLOAD_LEXICAL_VERSION,
    POINT_NAMESPACE,
    SPARSE_VECTOR,
    CollectionNotFound,
    CollectionSchemaMismatch,
    QdrantVectorIndex,
    point_id_for,
)


class TestPointIdMapping:
    """D46: deterministic UUID5, because idempotent re-indexing depends on it."""

    def test_the_same_chunk_always_maps_to_the_same_point(self) -> None:
        # The property the whole decision rests on. Re-chunking a paper
        # regenerates identical chunk ids, so a re-index must upsert onto the
        # same points instead of accumulating a second copy of the corpus.
        chunk = "0d9c2f1e-1111-2222-3333-444455556666:p3:c2"
        assert point_id_for(chunk) == point_id_for(chunk)

    def test_different_chunks_map_to_different_points(self) -> None:
        base = "0d9c2f1e-1111-2222-3333-444455556666:p3"
        assert point_id_for(f"{base}:c1") != point_id_for(f"{base}:c2")

    def test_the_mapping_produces_a_valid_uuid(self) -> None:
        # Qdrant accepts only an unsigned integer or a UUID; anything else is
        # rejected at upsert time, per-batch, deep inside ingestion.
        assert uuid.UUID(point_id_for("paper:p0:c0")).version == 5

    def test_the_namespace_is_pinned(self) -> None:
        # A changed namespace relocates every point silently: the next re-index
        # would duplicate the entire corpus rather than replace it, and nothing
        # would error. Pinning it here makes that change impossible to make by
        # accident.
        assert str(POINT_NAMESPACE) == "6f9619ff-8b86-d011-b42d-00c04fc964ff"
        assert point_id_for("paper:p0:c0") == str(uuid.uuid5(POINT_NAMESPACE, "paper:p0:c0"))


class TestTheFilterIsMandatory:
    """§5.8 layer 3, expressed as control flow rather than convention."""

    async def test_searching_with_no_papers_in_scope_raises(self) -> None:
        # The single input that would turn a workspace-scoped ANN query into a
        # global one. Returning an empty list instead would be worse than
        # raising: it looks like "no relevant chunks", so the bug would present
        # as a quality problem rather than as a tenancy one.
        index = QdrantVectorIndex("http://qdrant.invalid:6333")
        with pytest.raises(EmptyRetrievalScope):
            await index.search([0.1] * 768, paper_ids=[], limit=10)

    async def test_deleting_nothing_is_a_no_op_not_a_wildcard(self) -> None:
        # The dangerous asymmetry: an empty filter on delete would match
        # everything. It must never reach Qdrant as an unfiltered selector.
        index = QdrantVectorIndex("http://qdrant.invalid:6333")
        assert await index.delete_by_papers([]) == 0

    async def test_constructing_the_adapter_makes_no_network_call(self) -> None:
        # NFR-T3: the default test run makes zero outbound calls. The Qdrant
        # client performs a server version check in its constructor, so building
        # one eagerly would make merely wiring dependencies a network operation.
        index = QdrantVectorIndex("http://qdrant.invalid:6333")
        assert index._connection is None

    def test_the_filter_targets_the_indexed_payload_key(self) -> None:
        # A typo in a filter key does not error — Qdrant matches nothing, and a
        # retrieval returning nothing is indistinguishable from a workspace with
        # no relevant content.
        built = QdrantVectorIndex._paper_filter(["a", "b"])
        condition = built.must[0]  # type: ignore[index]
        assert condition.key == "paper_id"  # type: ignore[union-attr]
        assert condition.match.any == ["a", "b"]  # type: ignore[union-attr]


def _dense(size: int = 768) -> models.VectorParams:
    return models.VectorParams(size=size, distance=models.Distance.COSINE)


def _lexical(modifier: models.Modifier | None = models.Modifier.IDF) -> models.SparseVectorParams:
    return models.SparseVectorParams(modifier=modifier)


#: What `ensure_ready` builds, as Qdrant reports it back: the dense-only
#: collection's one unnamed vector, and the hybrid collection's named dense
#: vector plus its IDF-weighted lexical sparse vector (D73).
DENSE_ONLY: dict[str, Any] = {"vectors": _dense()}
HYBRID: dict[str, Any] = {
    "vectors": {DENSE_VECTOR: _dense()},
    "sparse_vectors": {SPARSE_VECTOR: _lexical()},
}


class ReadOnlyClient:
    """A Qdrant client stand-in that permits three reads and fails on anything else.

    Any other attribute, `create_collection` and `create_payload_index` among
    them, fails the test the moment it is looked up, so a method that writes or
    creates cannot pass by accident. The collection's configuration is built from
    qdrant-client's own models, so the schema check reads the shapes a real
    server returns.
    """

    def __init__(
        self, *, exists: bool, params: dict[str, Any] | None = None, points: int = 7854
    ) -> None:
        self.reads: list[str] = []
        self._exists = exists
        self._params = DENSE_ONLY if params is None else params
        self._points = points

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"existing_count used {name!r}; only reads are allowed")

    async def collection_exists(self, collection: str) -> bool:
        self.reads.append("collection_exists")
        return self._exists

    async def get_collection(self, collection: str) -> object:
        self.reads.append("get_collection")
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(**self._params)))

    async def count(self, collection: str, **kwargs: object) -> object:
        self.reads.append("count")
        return SimpleNamespace(count=self._points)


def _index_over(client: ReadOnlyClient, *, sparse: bool = False) -> QdrantVectorIndex:
    index = QdrantVectorIndex("http://qdrant.invalid:6333", collection="eval", sparse=sparse)
    index._connection = client  # type: ignore[assignment]
    return index


class TestCheckingACollectionWithoutCreatingIt:
    """`existing_count` is the read-only check the workspace gate relies on (D168).

    `ensure_ready` creates a missing collection and re-asserts its payload index
    on every call, so a caller that must leave the store exactly as it found it
    needs a check that only reads. Every test here also proves nothing was
    created or modified: the stand-in client fails on any method but three reads.
    """

    async def test_a_missing_collection_is_reported_not_created(self) -> None:
        client = ReadOnlyClient(exists=False)

        assert await _index_over(client).existing_count() is None
        assert client.reads == ["collection_exists"]

    async def test_an_existing_dense_collection_is_counted_by_reads_alone(self) -> None:
        client = ReadOnlyClient(exists=True, points=7854)

        assert await _index_over(client).existing_count() == 7854
        assert client.reads == ["collection_exists", "get_collection", "count"]

    async def test_a_complete_hybrid_collection_is_counted_by_reads_alone(self) -> None:
        client = ReadOnlyClient(exists=True, params=HYBRID, points=7854)

        assert await _index_over(client, sparse=True).existing_count() == 7854
        assert client.reads == ["collection_exists", "get_collection", "count"]

    async def test_a_hybrid_collection_missing_its_lexical_vector_is_refused(self) -> None:
        # The lexical arm queries the `lexical` sparse vector; without it the
        # hybrid arm is not the one M2 measured, so nothing is read from it.
        client = ReadOnlyClient(exists=True, params={"vectors": {DENSE_VECTOR: _dense()}})

        with pytest.raises(CollectionSchemaMismatch) as refused:
            await _index_over(client, sparse=True).existing_count()

        assert refused.value.problems == ("no 'lexical' sparse vector",)
        assert client.reads == ["collection_exists", "get_collection"]

    async def test_a_lexical_vector_without_idf_is_refused(self) -> None:
        # D73: IDF comes from Qdrant's modifier. Without it BM25 is term
        # frequency alone, a different lexical arm under the same name.
        params = {**HYBRID, "sparse_vectors": {SPARSE_VECTOR: _lexical(models.Modifier.NONE)}}
        client = ReadOnlyClient(exists=True, params=params)

        with pytest.raises(CollectionSchemaMismatch, match="not IDF-weighted"):
            await _index_over(client, sparse=True).existing_count()
        assert "count" not in client.reads

    async def test_a_hybrid_collection_without_a_named_dense_vector_is_refused(self) -> None:
        # An unnamed vector is the dense-only form; `search` on the hybrid
        # collection asks for the vector named `dense`.
        params = {**HYBRID, "vectors": _dense()}
        client = ReadOnlyClient(exists=True, params=params)

        with pytest.raises(CollectionSchemaMismatch, match="no named 'dense' vector"):
            await _index_over(client, sparse=True).existing_count()
        assert "count" not in client.reads

    async def test_a_dense_only_adapter_refuses_a_named_vector_collection(self) -> None:
        client = ReadOnlyClient(exists=True, params=HYBRID)

        with pytest.raises(CollectionSchemaMismatch, match="no unnamed dense vector"):
            await _index_over(client).existing_count()
        assert "count" not in client.reads

    async def test_an_unreadable_vector_configuration_is_refused_not_passed(self) -> None:
        # The lenient check `ensure_ready` keeps would pass this, having found
        # nothing to compare; a check that finds nothing must not look like one.
        client = ReadOnlyClient(exists=True, params={"vectors": None})

        with pytest.raises(CollectionSchemaMismatch, match="no unnamed dense vector"):
            await _index_over(client).existing_count()
        assert "count" not in client.reads

    @pytest.mark.parametrize("sparse", [False, True])
    async def test_a_dense_vector_whose_size_cannot_be_read_is_refused(self, sparse: bool) -> None:
        # The vector is there but carries no readable size: nothing to compare
        # with the encoder's dimensions, so it is refused rather than assumed.
        sizeless = SimpleNamespace(size=None)
        params = (
            {**HYBRID, "vectors": {DENSE_VECTOR: sizeless}} if sparse else {"vectors": sizeless}
        )
        client = ReadOnlyClient(exists=True, params=params)

        with pytest.raises(CollectionSchemaMismatch, match="size is unreadable"):
            await _index_over(client, sparse=sparse).existing_count()
        assert "count" not in client.reads

    @pytest.mark.parametrize("sparse", [False, True])
    async def test_a_collection_built_for_another_embedder_is_refused_before_counting(
        self, sparse: bool
    ) -> None:
        params = (
            {**HYBRID, "vectors": {DENSE_VECTOR: _dense(384)}}
            if sparse
            else {"vectors": _dense(384)}
        )
        client = ReadOnlyClient(exists=True, params=params)

        with pytest.raises(VectorDimensionMismatch):
            await _index_over(client, sparse=sparse).existing_count()
        assert client.reads == ["collection_exists", "get_collection"]


class TestRequiringACollectionThisProcessMustNotCreate:
    """`require_existing`: how the API and worker treat the hybrid collection (D172).

    Only `scholiast-reindex --hybrid` creates it, filled. The stand-in client fails
    on any method but its reads, so a path that would create the collection or its
    payload index fails here rather than passing quietly.
    """

    async def test_a_missing_collection_stops_the_caller_and_is_not_created(self) -> None:
        client = ReadOnlyClient(exists=False)

        with pytest.raises(CollectionNotFound, match="scholiast-reindex --hybrid"):
            await _index_over(client, sparse=True).require_existing()
        assert client.reads == ["collection_exists"]

    async def test_a_complete_hybrid_schema_passes_on_reads_alone(self) -> None:
        client = ReadOnlyClient(exists=True, params=HYBRID)

        await _index_over(client, sparse=True).require_existing()
        assert client.reads == ["collection_exists", "get_collection"]

    async def test_a_hybrid_collection_without_its_lexical_vector_is_refused(self) -> None:
        client = ReadOnlyClient(exists=True, params={"vectors": {DENSE_VECTOR: _dense()}})

        with pytest.raises(CollectionSchemaMismatch, match="no 'lexical' sparse vector"):
            await _index_over(client, sparse=True).require_existing()


def _record(
    lexical: dict[int, float] | None, version: str | None = "bm25-identifier/1.1.0"
) -> VectorRecord:
    return VectorRecord(
        chunk_id="paper:p0:c0",
        paper_id="paper",
        vector=(0.1, 0.2),
        section_kind=SectionKind.METHOD,
        chunker_version="structure-aware/1.2.0",
        embedder_version="test/fp16/1.0.0",
        lexical=lexical,
        lexical_version=version,
    )


class WritingClient:
    """Records what `upsert` sends, and answers the coverage reads it is asked."""

    def __init__(self, *, pages: list[list[str]] | None = None, counted: int = 0) -> None:
        self.points: list[models.PointStruct] = []
        self.filters: list[object] = []
        self._pages = pages or []
        self._counted = counted

    async def upsert(
        self, collection: str, *, points: list[models.PointStruct], wait: bool
    ) -> None:
        self.points.extend(points)

    async def scroll(self, collection: str, **kwargs: Any) -> tuple[list[object], object]:
        self.filters.append(kwargs["scroll_filter"])
        assert kwargs["with_vectors"] is False, "the completeness check needs ids, not vectors"
        page = kwargs["offset"] or 0
        ids = self._pages[page] if page < len(self._pages) else []
        points = [SimpleNamespace(payload={"chunk_id": chunk_id}) for chunk_id in ids]
        return points, (page + 1 if page + 1 < len(self._pages) else None)

    async def count(self, collection: str, **kwargs: Any) -> object:
        self.filters.append(kwargs["count_filter"])
        return SimpleNamespace(count=self._counted)


def _writing(client: WritingClient, *, sparse: bool) -> QdrantVectorIndex:
    index = QdrantVectorIndex("http://qdrant.invalid:6333", collection="c", sparse=sparse)
    index._connection = client  # type: ignore[assignment]
    return index


class TestAHybridPointIsWrittenWhole:
    """D172: the dense and lexical vectors of one chunk travel in one point."""

    async def test_a_hybrid_point_carries_both_vectors_and_its_lexical_version(self) -> None:
        client = WritingClient()
        await _writing(client, sparse=True).upsert([_record({11: 0.5, 7: 1.25})])

        (point,) = client.points
        assert isinstance(point.vector, dict)
        assert point.vector[DENSE_VECTOR] == [0.1, 0.2]
        lexical = point.vector[SPARSE_VECTOR]
        assert isinstance(lexical, models.SparseVector)
        assert dict(zip(lexical.indices, lexical.values, strict=True)) == {11: 0.5, 7: 1.25}
        assert point.payload is not None
        assert point.payload[PAYLOAD_LEXICAL_VERSION] == "bm25-identifier/1.1.0"

    async def test_a_dense_only_point_is_exactly_what_it_was_before(self) -> None:
        # Phase A changes nothing in scholiast_chunks: lexical weights handed to a
        # dense-only collection are ignored, payload and all.
        with_weights, without = WritingClient(), WritingClient()
        await _writing(with_weights, sparse=False).upsert([_record({11: 0.5})])
        await _writing(without, sparse=False).upsert([_record(None, None)])

        assert with_weights.points == without.points
        (point,) = with_weights.points
        assert point.vector == [0.1, 0.2]
        assert point.payload is not None and PAYLOAD_LEXICAL_VERSION not in point.payload

    async def test_a_chunk_with_no_terms_is_stamped_without_an_empty_sparse_vector(self) -> None:
        # The store rejects an empty sparse vector; the point is still current, so
        # the completeness check must still find it by version.
        client = WritingClient()
        await _writing(client, sparse=True).upsert([_record({})])

        (point,) = client.points
        assert isinstance(point.vector, dict) and SPARSE_VECTOR not in point.vector
        assert point.payload is not None
        assert point.payload[PAYLOAD_LEXICAL_VERSION] == "bm25-identifier/1.1.0"


class TestCoverageIsReadByVersion:
    async def test_chunk_ids_are_collected_across_every_page_for_one_version(self) -> None:
        client = WritingClient(pages=[["a", "b"], ["c"]])
        ids = await _writing(client, sparse=True).lexical_chunk_ids(lexical_version="v2")

        assert ids == {"a", "b", "c"}
        for used in client.filters:
            assert isinstance(used, models.Filter) and used.must is not None
            (condition,) = used.must  # type: ignore[misc]
            assert condition.key == PAYLOAD_LEXICAL_VERSION
            assert condition.match.value == "v2"

    async def test_the_count_is_filtered_to_the_version_too(self) -> None:
        client = WritingClient(counted=42)
        assert await _writing(client, sparse=True).count_lexical(lexical_version="v2") == 42
        (used,) = client.filters
        assert isinstance(used, models.Filter) and used.must is not None

    async def test_a_dense_only_adapter_has_no_lexical_coverage_to_report(self) -> None:
        index = _writing(WritingClient(), sparse=False)
        with pytest.raises(RuntimeError, match="dense-only"):
            await index.lexical_chunk_ids(lexical_version="v2")
        with pytest.raises(RuntimeError, match="dense-only"):
            await index.count_lexical(lexical_version="v2")
