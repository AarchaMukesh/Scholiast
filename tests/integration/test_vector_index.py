"""Real Qdrant, and real GPU embedding.

Split by marker so each can be skipped for the right reason: `integration` needs
a Qdrant on localhost, `gpu` needs a CUDA device that CI does not have. The
CPU-only embedding path is exercised unmarked, because §5.7 requires it to keep
working and CI is where that gets verified (DEP-9).

Each test uses its own collection so a failure cannot poison the next run, and
so these never touch a real corpus.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from scholiast.application.ports.ml import Precision, TextRole
from scholiast.application.ports.vector import VectorDimensionMismatch, VectorRecord
from scholiast.domain.paper import SectionKind
from scholiast.domain.retrieval import RetrievalArm
from scholiast.infrastructure.ml.embedder import LocalEmbedder
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex, point_id_for

QDRANT_URL = "http://localhost:6333"
MODEL = "BAAI/bge-base-en-v1.5"


def _qdrant_alive() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            return client.get(f"{QDRANT_URL}/readyz").status_code == 200
    except httpx.HTTPError:
        return False


def _model_cached(model_id: str = MODEL) -> bool:
    """Is the encoder already in the local HuggingFace cache?

    Loading it is a 419 MB download on a cold cache, and NFR-T3 requires the
    default test run to make zero network calls. Skipping when absent keeps that
    true in CI while still exercising the real CPU inference path on any machine
    that has pulled the model.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
    except Exception:  # not cached, or hub unavailable
        return False
    return True


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch may be absent entirely
        return False


needs_qdrant = pytest.mark.skipif(not _qdrant_alive(), reason="Qdrant is not running")
needs_gpu = pytest.mark.skipif(not _has_cuda(), reason="no CUDA device")
needs_model = pytest.mark.skipif(
    not _model_cached(), reason=f"{MODEL} is not in the local HF cache (NFR-T3: no downloads)"
)


def record(chunk_id: str, paper_id: str, vector: list[float]) -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        paper_id=paper_id,
        vector=tuple(vector),
        section_kind=SectionKind.METHOD,
        chunker_version="structure-aware/1.1.0",
        embedder_version="test/fp16/1.0.0",
    )


def unit(index: int, dims: int = 8) -> list[float]:
    """A one-hot vector, so cosine similarity is exactly predictable."""
    vector = [0.0] * dims
    vector[index % dims] = 1.0
    return vector


@pytest_asyncio.fixture
async def index() -> AsyncIterator[QdrantVectorIndex]:
    store = QdrantVectorIndex(QDRANT_URL, collection=f"test_{uuid.uuid4().hex[:10]}", dimensions=8)
    await store.ensure_ready()
    yield store
    await store._client.delete_collection(store.collection)
    await store.close()


@pytest.mark.integration
@needs_qdrant
class TestAgainstRealQdrant:
    async def test_upsert_then_search_returns_the_chunk(self, index: QdrantVectorIndex) -> None:
        await index.upsert([record("p1:p0:c0", "paper-1", unit(0))])
        hits = await index.search(unit(0), paper_ids=["paper-1"], limit=5)

        assert [h.chunk_id for h in hits] == ["p1:p0:c0"]
        assert hits[0].arm is RetrievalArm.DENSE
        assert hits[0].rank == 1
        assert hits[0].score > 0.99

    async def test_the_filter_excludes_papers_out_of_scope(self, index: QdrantVectorIndex) -> None:
        # FR-A4 at the vector layer: two workspaces holding identical content
        # must not see each other's chunks. This is the leak §5.8 layer 3 exists
        # to stop, tested against the real filtered-ANN path rather than a mock.
        await index.upsert(
            [
                record("mine:p0:c0", "paper-mine", unit(0)),
                record("theirs:p0:c0", "paper-theirs", unit(0)),
            ]
        )
        hits = await index.search(unit(0), paper_ids=["paper-mine"], limit=10)

        assert [h.paper_id for h in hits] == ["paper-mine"]
        assert "theirs:p0:c0" not in {h.chunk_id for h in hits}

    async def test_identical_vectors_do_not_leak_across_the_filter(
        self, index: QdrantVectorIndex
    ) -> None:
        # The adversarial version: the out-of-scope chunk is a *better* match.
        # A post-filter applied after a top-k search would drop the in-scope
        # result entirely and return nothing.
        await index.upsert(
            [
                record("theirs:p0:c0", "paper-theirs", unit(0)),
                record("mine:p0:c0", "paper-mine", [0.9, 0.1, 0, 0, 0, 0, 0, 0]),
            ]
        )
        hits = await index.search(unit(0), paper_ids=["paper-mine"], limit=1)
        assert [h.chunk_id for h in hits] == ["mine:p0:c0"]

    async def test_reindexing_the_same_chunks_does_not_duplicate_them(
        self, index: QdrantVectorIndex
    ) -> None:
        # D46's whole purpose. Chunk ids are deterministic, so re-chunking must
        # converge on the same points — otherwise every re-index doubles the
        # corpus and the two stores drift exactly where §5.6.4's sweep is trying
        # to keep them together.
        records = [record("p1:p0:c0", "paper-1", unit(0))]
        await index.upsert(records)
        await index.upsert(records)
        assert await index.count(paper_ids=["paper-1"]) == 1

    async def test_reindexing_replaces_the_vector(self, index: QdrantVectorIndex) -> None:
        # Upsert must update, not merely skip: after a re-embed the new vector
        # has to win, or a model change would leave stale vectors in place while
        # the version stamp claimed otherwise.
        await index.upsert([record("p1:p0:c0", "paper-1", unit(0))])
        await index.upsert([record("p1:p0:c0", "paper-1", unit(3))])

        hits = await index.search(unit(3), paper_ids=["paper-1"], limit=1)
        assert hits[0].score > 0.99

    async def test_deleting_a_paper_removes_only_its_vectors(
        self, index: QdrantVectorIndex
    ) -> None:
        # FR-W2: removal cleans up vectors with no orphans, and without taking
        # a neighbouring paper's chunks with it.
        await index.upsert(
            [
                record("a:p0:c0", "paper-a", unit(0)),
                record("b:p0:c0", "paper-b", unit(1)),
            ]
        )
        removed = await index.delete_by_papers(["paper-a"])

        assert removed == 1
        assert await index.count(paper_ids=["paper-a"]) == 0
        assert await index.count(paper_ids=["paper-b"]) == 1

    async def test_the_payload_index_on_paper_id_exists(self, index: QdrantVectorIndex) -> None:
        # Without it the mandatory filter degrades to a scan, and AD-4's whole
        # argument for choosing Qdrant — selective filtered ANN — evaporates
        # while every test still passes.
        info = await index._client.get_collection(index.collection)
        assert "paper_id" in (info.payload_schema or {})

    async def test_ensure_ready_is_idempotent(self, index: QdrantVectorIndex) -> None:
        # Called on every startup; a second call must not fail or wipe data.
        await index.upsert([record("p1:p0:c0", "paper-1", unit(0))])
        await index.ensure_ready()
        assert await index.count(paper_ids=["paper-1"]) == 1

    async def test_a_dimension_mismatch_fails_at_startup(self, index: QdrantVectorIndex) -> None:
        # Changing the embedding model changes the vector width, and a Qdrant
        # collection is fixed at creation. Without this check there is no error
        # here and then every upsert fails deep inside ingestion, far from the
        # cause — so the mismatch is caught where both numbers are still known.
        wrong = QdrantVectorIndex(QDRANT_URL, collection=index.collection, dimensions=384)
        try:
            with pytest.raises(VectorDimensionMismatch) as caught:
                await wrong.ensure_ready()
            assert caught.value.existing == 8
            assert caught.value.configured == 384
        finally:
            await wrong.close()

    async def test_matching_dimensions_pass_validation(self, index: QdrantVectorIndex) -> None:
        # Positive control: the check must not reject the collection this
        # embedder actually built, or every startup would fail.
        same = QdrantVectorIndex(QDRANT_URL, collection=index.collection, dimensions=8)
        try:
            await same.ensure_ready()
        finally:
            await same.close()

    async def test_version_stamps_survive_into_the_payload(self, index: QdrantVectorIndex) -> None:
        # NFR-M5 wants "re-embed only chunks from embedder v1" to be a query.
        # That requires the stamp to be present in the store holding the
        # vectors, not only in Postgres.
        await index.upsert([record("p1:p0:c0", "paper-1", unit(0))])
        points = await index._client.retrieve(
            index.collection, ids=[point_id_for("p1:p0:c0")], with_payload=True
        )
        payload = points[0].payload or {}
        assert payload["embedder_version"] == "test/fp16/1.0.0"
        assert payload["chunker_version"] == "structure-aware/1.1.0"
        assert payload["section_kind"] == SectionKind.METHOD.value


@needs_model
class TestEmbeddingOnCpu:
    """The CPU fallback path §5.7 requires, exercised wherever the model exists.

    Skipped rather than downloading: a 419 MB fetch is a network call, and
    NFR-T3 makes the default test run offline. That means CI does not verify
    this path — an accepted, recorded limitation rather than a hidden one.
    """

    async def test_cpu_embedding_produces_normalised_vectors(self) -> None:
        embedder = LocalEmbedder(MODEL, device="cpu")
        vectors = await embedder.embed(["retrieval augmented generation"], role=TextRole.PASSAGE)

        assert len(vectors) == 1
        assert len(vectors[0]) == 768
        norm = sum(value * value for value in vectors[0]) ** 0.5
        assert abs(norm - 1.0) < 1e-3, "BGE needs L2-normalised vectors for cosine"

    async def test_the_query_prefix_changes_the_vector(self) -> None:
        # Proof the prefix reaches the model rather than being applied to a copy
        # and discarded. If it were dropped, both roles would embed identically
        # and retrieval would silently degrade with nothing to show for it.
        embedder = LocalEmbedder(MODEL, device="cpu")
        text = "what is reciprocal rank fusion"
        as_query = await embedder.embed([text], role=TextRole.QUERY)
        as_passage = await embedder.embed([text], role=TextRole.PASSAGE)
        assert as_query[0] != as_passage[0]


@pytest.mark.gpu
@needs_gpu
@needs_model
class TestEmbeddingOnGpu:
    """D18 and NFR-P7, asserted on the real device rather than assumed."""

    async def test_the_model_runs_in_fp16_on_cuda(self) -> None:
        # §6.16: "precision is set explicitly in the adapter and asserted in
        # tests". Reporting fp16 while running fp32 is exactly the undisclosed
        # degradation NFR-R4 forbids, and it is invisible from the outside.
        import torch

        embedder = LocalEmbedder(MODEL, device="cuda")
        await embedder.embed(["warm up the model"], role=TextRole.PASSAGE)

        assert embedder.precision is Precision.FP16
        assert "cuda" in embedder.device
        model = embedder._load()
        assert next(model.parameters()).dtype is torch.float16

    async def test_gpu_and_cpu_agree_on_direction(self) -> None:
        # fp16 and fp32 differ numerically, which is why they carry different
        # version stamps — but they must still describe the same semantics, or
        # the CPU fallback would silently return different neighbours.
        text = "structure-aware chunking preserves section boundaries"
        gpu = (await LocalEmbedder(MODEL, device="cuda").embed([text], role=TextRole.PASSAGE))[0]
        cpu = (await LocalEmbedder(MODEL, device="cpu").embed([text], role=TextRole.PASSAGE))[0]

        cosine = sum(a * b for a, b in zip(gpu, cpu, strict=True))
        assert cosine > 0.99, f"fp16 and fp32 disagree materially (cosine {cosine:.4f})"

    async def test_throughput_clears_the_derived_requirement(self) -> None:
        # NFR-P7 is derived from NFR-P6's 90 s/paper: >= 10 chunks/s. M0
        # measured 75.8 on this GPU. The assertion is against the requirement,
        # not the measurement, so a slower machine is not a test failure.
        import time

        embedder = LocalEmbedder(MODEL, device="cuda")
        chunks = [f"Chunk number {i} about retrieval quality and evaluation." for i in range(64)]
        await embedder.embed(chunks[:4], role=TextRole.PASSAGE)  # warm up

        started = time.perf_counter()
        vectors = await embedder.embed(chunks, role=TextRole.PASSAGE)
        rate = len(chunks) / (time.perf_counter() - started)

        assert len(vectors) == 64
        assert rate >= 10, f"{rate:.1f} chunks/s is below NFR-P7's derived floor of 10"


@pytest.mark.gpu
@pytest.mark.integration
@needs_gpu
@needs_qdrant
@needs_model
class TestRealEmbeddingsIntoRealQdrant:
    """The Phase 4 slice end to end: encode on the GPU, index, retrieve."""

    async def test_semantic_search_ranks_the_relevant_chunk_first(self) -> None:
        embedder = LocalEmbedder(MODEL, device="cuda")
        store = QdrantVectorIndex(
            QDRANT_URL, collection=f"test_e2e_{uuid.uuid4().hex[:8]}", dimensions=768
        )
        try:
            await store.ensure_ready()
            passages = {
                "paper:p0:c0": "Reciprocal rank fusion combines ranked lists using ranks, "
                "which makes it immune to score-scale mismatch between retrievers.",
                "paper:p0:c1": "The dataset was collected from arXiv preprints published "
                "between 2018 and 2023 across several subject areas.",
                "paper:p0:c2": "Training used a batch size of 32 with a learning rate of "
                "3e-4 and cosine decay over ten epochs.",
            }
            vectors = await embedder.embed(list(passages.values()), role=TextRole.PASSAGE)
            await store.upsert(
                [
                    record(chunk_id, "paper-1", vector)
                    for chunk_id, vector in zip(passages, vectors, strict=True)
                ]
            )

            query = (await embedder.embed(["how does rank fusion work?"], role=TextRole.QUERY))[0]
            hits = await store.search(query, paper_ids=["paper-1"], limit=3)

            assert hits[0].chunk_id == "paper:p0:c0", (
                "the fusion passage should win a fusion query — "
                f"got {[(h.chunk_id, round(h.score, 3)) for h in hits]}"
            )
            assert hits[0].score > hits[1].score
        finally:
            await store._client.delete_collection(store.collection)
            await store.close()
