"""Vector index behaviour that is decidable without a running Qdrant.

The point-id mapping (D46) and the mandatory-filter refusal (§5.8 layer 3) are
pure logic, and both are properties that must never quietly change — the first
because it is baked into stored data, the second because it is a tenancy
boundary. `tests/integration/test_vector_index.py` covers the rest against a
real Qdrant.
"""

from __future__ import annotations

import uuid

import pytest

from scholiast.application.ports.vector import EmptyRetrievalScope
from scholiast.infrastructure.vector.qdrant import (
    POINT_NAMESPACE,
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
