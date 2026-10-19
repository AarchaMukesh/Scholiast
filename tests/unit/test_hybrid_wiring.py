"""How production's processes are wired for the hybrid collection (D172).

Phase A's promise is that merging it changes nothing: with today's settings every
process builds exactly the dense collection it built before. Its guards are that
no API or worker ever creates the hybrid collection, that a hybrid read is refused
until that collection is complete, that every write reaches both collections once
one is configured, and that readiness notices a writer that does not mirror. Each
is tested here at the place it is wired, with stand-ins for Qdrant and the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from scholiast.api import app as app_module
from scholiast.api import deps
from scholiast.api.app import api_lifespan, create_app
from scholiast.api.routers import health
from scholiast.application.use_cases.verify_hybrid_index import (
    HybridCoverage,
    HybridIndexIncomplete,
    VerifyHybridIndexUseCase,
)
from scholiast.config.settings import Settings
from scholiast.domain.lexical import LEXICAL_VERSION
from scholiast.domain.retrieval import RetrievalMode
from scholiast.infrastructure.vector import factory as factory_module
from scholiast.infrastructure.vector.factory import build_vector_indexes
from scholiast.infrastructure.vector.mirrored import MirroredVectorIndex
from scholiast.infrastructure.vector.qdrant import CollectionNotFound, QdrantVectorIndex
from scholiast.workers import maintenance as maintenance_module
from scholiast.workers import reindex, tasks
from scholiast.workers.maintenance import MaintenanceWorker
from tests.unit.test_observability import ops_client

HYBRID = "scholiast_chunks_hybrid"


def settings(**overrides: Any) -> Settings:
    """Settings from these arguments alone: no `.env`, so a developer's file cannot leak in."""
    return Settings(
        _env_file=None, celery_enabled=False, embedding_warm_at_startup=False, **overrides
    )  # type: ignore[call-arg]


# --- settings ----------------------------------------------------------------------


class TestTheDefaultsAreTodaysDeployment:
    def test_nothing_names_a_second_collection_and_qa_reads_dense(self) -> None:
        fields = Settings.model_fields
        assert fields["qdrant_collection"].default == "scholiast_chunks"
        assert fields["qdrant_hybrid_collection"].default is None
        assert fields["retrieval_mode"].default == "dense"

    def test_hybrid_mode_without_a_hybrid_collection_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="no collection to read the lexical arm"):
            settings(retrieval_mode="hybrid")

    def test_one_collection_named_as_both_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="cannot be both"):
            settings(qdrant_hybrid_collection="scholiast_chunks")

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            settings(retrieval_mode="rerank")


# --- the factory -------------------------------------------------------------------


class TestEveryProcessBuildsTheSameIndexes:
    def test_by_default_there_is_one_dense_collection_for_reading_and_writing(self) -> None:
        built = build_vector_indexes(settings())
        assert isinstance(built.write, QdrantVectorIndex)
        assert built.write.collection == "scholiast_chunks" and built.write.sparse is False
        assert built.hybrid is None and built.lexical is None
        assert built.read is built.write and built.mode is RetrievalMode.DENSE

    def test_a_configured_hybrid_collection_is_mirrored_to_while_qa_stays_dense(self) -> None:
        built = build_vector_indexes(settings(qdrant_hybrid_collection=HYBRID))
        assert isinstance(built.write, MirroredVectorIndex)
        assert built.write.primary.collection == "scholiast_chunks"
        assert built.hybrid is not None and built.write.mirror is built.hybrid
        assert built.hybrid.collection == HYBRID and built.hybrid.sparse is True
        # Dense mode reads the collection it always read: rollback is this.
        assert built.read is built.primary and built.lexical is None

    def test_hybrid_mode_reads_both_arms_from_the_hybrid_collection(self) -> None:
        built = build_vector_indexes(
            settings(qdrant_hybrid_collection=HYBRID, retrieval_mode="hybrid")
        )
        assert built.read is built.hybrid and built.lexical is built.hybrid

    def test_the_factory_refuses_hybrid_mode_settings_validation_was_bypassed_for(self) -> None:
        bypassed = Settings.model_construct(retrieval_mode="hybrid", qdrant_hybrid_collection=None)
        with pytest.raises(ValueError, match="qdrant_hybrid_collection"):
            build_vector_indexes(bypassed)

    def test_the_celery_worker_writes_through_the_mirror_and_sweeps_the_hybrid(self) -> None:
        # The process most likely to drift from the API's configuration is the
        # one built elsewhere; it builds through the same factory.
        _, maintenance, index = tasks._build(settings(qdrant_hybrid_collection=HYBRID))
        assert isinstance(index, MirroredVectorIndex)
        assert maintenance._hybrid is index.mirror

    def test_the_celery_worker_by_default_is_exactly_what_it_was(self) -> None:
        _, maintenance, index = tasks._build(settings())
        assert isinstance(index, QdrantVectorIndex) and index.collection == "scholiast_chunks"
        assert maintenance._hybrid is None


# --- the mirror ---------------------------------------------------------------------


class Collection:
    """A store stand-in that records what is asked of it."""

    def __init__(self, name: str, *, orphans: int = 0) -> None:
        self.collection = name
        self.calls: list[str] = []
        self._orphans = orphans

    async def ensure_ready(self) -> None:
        self.calls.append("ensure_ready")

    async def require_existing(self) -> None:
        self.calls.append("require_existing")

    async def upsert(self, records: Sequence[object]) -> int:
        self.calls.append("upsert")
        return len(records)

    async def search(
        self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int
    ) -> list[object]:
        self.calls.append("search")
        return []

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        self.calls.append("delete_by_papers")
        return 3

    async def count(self, *, paper_ids: Sequence[str] | None = None) -> int:
        self.calls.append("count")
        return 7

    async def ping(self) -> bool:
        self.calls.append("ping")
        return True

    async def count_orphans(self, live: Sequence[str]) -> int:
        self.calls.append("count_orphans")
        return self._orphans

    async def delete_orphans(self, live: Sequence[str]) -> int:
        self.calls.append("delete_orphans")
        return self._orphans

    async def close(self) -> None:
        self.calls.append("close")


class Holding:
    """A collection holding paper p's two chunks, which can refuse a delete."""

    def __init__(self, name: str, order: list[str], *, fails: bool = False) -> None:
        self.collection = name
        self.points = {"p:c0", "p:c1"}
        self._order = order
        self._fails = fails

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        self._order.append(self.collection)
        if self._fails:
            raise ConnectionError(f"{self.collection} refused the delete")
        removed = {c for c in self.points if c.split(":")[0] in paper_ids}
        self.points -= removed
        return len(removed)

    async def lexical_chunk_ids(self, *, lexical_version: str) -> set[str]:
        return set(self.points)


class _StampedRows:
    def __init__(self, ids: set[str]) -> None:
        self._ids = ids

    async def embedded_chunk_ids(self) -> set[str]:
        return set(self._ids)


def mirrored(**orphans: int) -> tuple[MirroredVectorIndex, Collection, Collection]:
    primary = Collection("dense", orphans=orphans.get("primary", 0))
    mirror = Collection("hybrid", orphans=orphans.get("mirror", 0))
    return MirroredVectorIndex(primary, mirror), primary, mirror  # type: ignore[arg-type]


class TestEveryWriteReachesBothCollections:
    async def test_the_mirror_is_validated_never_created(self) -> None:
        index, primary, mirror = mirrored()
        await index.ensure_ready()
        assert primary.calls == ["ensure_ready"]
        assert mirror.calls == ["require_existing"]

    async def test_an_upsert_reaches_both(self) -> None:
        index, primary, mirror = mirrored()
        assert await index.upsert(["r1", "r2"]) == 2  # type: ignore[list-item]
        assert primary.calls == mirror.calls == ["upsert"]

    async def test_a_delete_reaches_both(self) -> None:
        # Otherwise the hybrid collection would keep a deleted paper's chunks,
        # retrievable after its owner removed it (FR-W2).
        index, primary, mirror = mirrored()
        assert await index.delete_by_papers(["p"]) == 3
        assert primary.calls == mirror.calls == ["delete_by_papers"]

    async def test_a_delete_reaches_the_mirror_before_the_primary(self) -> None:
        order: list[str] = []
        primary, mirror = Holding("dense", order), Holding("hybrid", order)
        await MirroredVectorIndex(primary, mirror).delete_by_papers(["p"])  # type: ignore[arg-type]
        assert order == ["hybrid", "dense"]

    async def test_a_failed_delete_leaves_production_whole_and_its_gap_detectable(
        self,
    ) -> None:
        # The purge deletes vectors before the paper row, and a failure rolls the
        # row back: the paper stays, with whatever was deleted gone from under it.
        # Mirror-first puts that gap where the completeness check sees it, and
        # leaves the dense collection production reads intact.
        order: list[str] = []
        primary = Holding("dense", order, fails=True)
        mirror = Holding("hybrid", order)
        with pytest.raises(ConnectionError):
            await MirroredVectorIndex(primary, mirror).delete_by_papers(["p"])  # type: ignore[arg-type]

        assert primary.points == {"p:c0", "p:c1"}
        coverage = await VerifyHybridIndexUseCase(
            chunks=_StampedRows({"p:c0", "p:c1"}),  # type: ignore[arg-type]
            lexical=mirror,  # type: ignore[arg-type]
        ).execute()
        assert coverage.missing == ("p:c0", "p:c1")

    async def test_the_sweep_removes_and_reports_residue_in_either(self) -> None:
        index, _, mirror = mirrored(primary=1, mirror=2)
        assert await index.count_orphans([]) == 3
        assert await index.delete_orphans([]) == 3
        assert mirror.calls == ["count_orphans", "delete_orphans"]

    async def test_reads_come_from_the_primary_alone(self) -> None:
        index, primary, mirror = mirrored()
        await index.search([0.1], paper_ids=["p"], limit=5)
        assert await index.count() == 7
        assert mirror.calls == [] and primary.calls == ["search", "count"]

    async def test_a_store_is_reachable_only_if_both_collections_are(self) -> None:
        index, primary, mirror = mirrored()
        assert await index.ping() is True
        assert primary.calls == mirror.calls == ["ping"]

    def test_a_collection_cannot_mirror_itself(self) -> None:
        with pytest.raises(ValueError, match="cannot mirror itself"):
            MirroredVectorIndex(Collection("same"), Collection("same"))  # type: ignore[arg-type]


# --- API startup --------------------------------------------------------------------


class FakeQdrant:
    """Stands in for `QdrantVectorIndex` inside the factory. Has no way to create anything."""

    built: ClassVar[list[FakeQdrant]] = []
    hybrid_exists: ClassVar[bool] = True
    hybrid_points: ClassVar[set[str]] = set()

    def __init__(self, url: str, *, collection: str, dimensions: int, sparse: bool = False) -> None:
        self.collection = collection
        self.sparse = sparse
        self.calls: list[str] = []
        FakeQdrant.built.append(self)

    async def ensure_ready(self) -> None:
        self.calls.append("ensure_ready")

    async def require_existing(self) -> None:
        self.calls.append("require_existing")
        if not FakeQdrant.hybrid_exists:
            raise CollectionNotFound(self.collection)

    async def lexical_chunk_ids(self, *, lexical_version: str) -> set[str]:
        self.calls.append(f"lexical_chunk_ids:{lexical_version}")
        return set(FakeQdrant.hybrid_points)

    async def close(self) -> None:
        self.calls.append("close")


class FakeRows:
    embedded: ClassVar[set[str]] = set()

    def __init__(self, session: object, scope: object) -> None:
        self.scope = scope

    async def embedded_chunk_ids(self) -> set[str]:
        return set(FakeRows.embedded)


class _Encoder:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.fixture
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> type[FakeQdrant]:
    FakeQdrant.built = []
    FakeQdrant.hybrid_exists = True
    FakeQdrant.hybrid_points = set()
    FakeRows.embedded = set()
    monkeypatch.setattr(factory_module, "QdrantVectorIndex", FakeQdrant)
    monkeypatch.setattr(app_module, "SqlChunkRepository", FakeRows)
    monkeypatch.setattr(app_module, "LocalEmbedder", _Encoder)
    return FakeQdrant


def by_collection() -> dict[str, FakeQdrant]:
    return {index.collection: index for index in FakeQdrant.built}


class TestStartup:
    async def test_by_default_only_scholiast_chunks_is_touched(
        self, stand_ins: type[FakeQdrant]
    ) -> None:
        async with api_lifespan(create_app(settings())):
            pass
        (only,) = FakeQdrant.built
        assert only.collection == "scholiast_chunks" and only.sparse is False
        assert only.calls[0] == "ensure_ready"

    async def test_a_configured_hybrid_collection_is_validated_not_created(
        self, stand_ins: type[FakeQdrant]
    ) -> None:
        app = create_app(settings(qdrant_hybrid_collection=HYBRID))
        async with api_lifespan(app):
            assert isinstance(app.state.index, MirroredVectorIndex)
        hybrid = by_collection()[HYBRID]
        assert "ensure_ready" not in hybrid.calls and "require_existing" in hybrid.calls
        # Dense mode reads nothing from it, so it waits on no completeness check.
        assert not any(call.startswith("lexical_chunk_ids") for call in hybrid.calls)

    async def test_a_missing_hybrid_collection_stops_the_boot(
        self, stand_ins: type[FakeQdrant]
    ) -> None:
        stand_ins.hybrid_exists = False
        with pytest.raises(CollectionNotFound):
            async with api_lifespan(create_app(settings(qdrant_hybrid_collection=HYBRID))):
                pytest.fail("the API started without its hybrid collection")

    @pytest.mark.parametrize(
        ("points", "case"),
        [(set(), "empty"), ({"a"}, "one chunk missing")],
    )
    async def test_hybrid_mode_refuses_an_incomplete_collection(
        self, stand_ins: type[FakeQdrant], points: set[str], case: str
    ) -> None:
        FakeRows.embedded = {"a", "b"}
        stand_ins.hybrid_points = points
        hybrid_settings = settings(qdrant_hybrid_collection=HYBRID, retrieval_mode="hybrid")
        with pytest.raises(HybridIndexIncomplete):
            async with api_lifespan(create_app(hybrid_settings)):
                pytest.fail(f"the API started reading a hybrid collection that was {case}")

    async def test_hybrid_mode_checks_the_current_lexical_version(
        self, stand_ins: type[FakeQdrant]
    ) -> None:
        FakeRows.embedded = {"a"}
        stand_ins.hybrid_points = {"a"}
        app = create_app(settings(qdrant_hybrid_collection=HYBRID, retrieval_mode="hybrid"))
        async with api_lifespan(app):
            assert app.state.retrieval.mode is RetrievalMode.HYBRID
        assert f"lexical_chunk_ids:{LEXICAL_VERSION}" in by_collection()[HYBRID].calls


# --- readiness ----------------------------------------------------------------------


class Rows:
    count: ClassVar[int] = 0

    def __init__(self, session: object, scope: object) -> None:
        pass

    async def count_embedded(self) -> int:
        return Rows.count


class Points:
    def __init__(self, count: int, *, fails: bool = False) -> None:
        self.count = count
        self.fails = fails
        self.asked = 0

    async def count_lexical(self, *, lexical_version: str) -> int:
        self.asked += 1
        if self.fails:
            raise ConnectionError("qdrant down")
        return self.count


class _Reachable:
    async def ping(self) -> bool:
        return True


@pytest.fixture
def hybrid_probe(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(health, "SqlChunkRepository", Rows)

    def build(points: Points) -> Any:
        client = ops_client(index=_Reachable())
        client.app.state.retrieval = SimpleNamespace(lexical=points)  # type: ignore[attr-defined]
        return client

    return build


class TestReadinessInHybridMode:
    def test_dense_mode_reports_no_hybrid_check(self) -> None:
        body = ops_client(index=_Reachable()).get("/health/ready").json()
        assert "hybrid_index" not in body["checks"]

    def test_fewer_hybrid_points_than_embedded_rows_is_not_ready(self, hybrid_probe: Any) -> None:
        Rows.count = 2
        response = hybrid_probe(Points(1)).get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["hybrid_index"] == "incomplete"

    def test_a_complete_collection_is_ready(self, hybrid_probe: Any) -> None:
        Rows.count = 2
        response = hybrid_probe(Points(2)).get("/health/ready")
        assert response.status_code == 200
        assert response.json()["checks"]["hybrid_index"] == "ok"

    def test_a_store_that_cannot_answer_is_not_ready(self, hybrid_probe: Any) -> None:
        Rows.count = 2
        response = hybrid_probe(Points(2, fails=True)).get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["hybrid_index"] == "unavailable"

    def test_the_answer_is_reused_for_a_minute_then_asked_again(
        self, hybrid_probe: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = [1000.0]
        monkeypatch.setattr(health.time, "monotonic", lambda: clock[0])
        Rows.count = 2
        points = Points(1)
        client = hybrid_probe(points)
        assert client.get("/health/ready").status_code == 503
        points.count = 2
        clock[0] += health.HYBRID_PROBE_SECONDS - 1
        assert client.get("/health/ready").status_code == 503 and points.asked == 1
        clock[0] += 2
        assert client.get("/health/ready").status_code == 200 and points.asked == 2


# --- scholiast-reindex ------------------------------------------------------------------


class TestTheReindexCommand:
    def test_hybrid_work_needs_a_configured_hybrid_collection(self) -> None:
        with pytest.raises(SystemExit, match="qdrant_hybrid_collection is not set"):
            reindex._hybrid_index(settings())

    def test_the_hybrid_collection_is_built_sparse_as_every_process_builds_it(self) -> None:
        built = reindex._hybrid_index(settings(qdrant_hybrid_collection=HYBRID))
        assert built.collection == HYBRID and built.sparse is True

    def test_modes_are_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            reindex._parser().parse_args(["--hybrid", "--verify"])

    async def test_hybrid_rebuild_creates_and_fills_only_the_hybrid_collection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The one sanctioned creation site (D172): `ensure_ready` here, on the
        # hybrid adapter. Stopped at that call so nothing else runs.
        class Stop(Exception):
            pass

        chosen: list[QdrantVectorIndex] = []

        async def ensure_ready(self: QdrantVectorIndex) -> None:
            chosen.append(self)
            raise Stop

        monkeypatch.setattr(QdrantVectorIndex, "ensure_ready", ensure_ready)
        with pytest.raises(Stop):
            await reindex._rebuild(settings(qdrant_hybrid_collection=HYBRID), hybrid=True)
        assert [(i.collection, i.sparse) for i in chosen] == [(HYBRID, True)]

    @pytest.mark.parametrize(("missing", "code"), [((), 0), (("b",), 1)])
    def test_verify_exits_nonzero_unless_complete(
        self, monkeypatch: pytest.MonkeyPatch, missing: tuple[str, ...], code: int
    ) -> None:
        coverage = HybridCoverage(
            lexical_version=LEXICAL_VERSION,
            expected=2,
            present=2 - len(missing),
            missing=missing,
            unaccounted=0,
        )

        async def verify(_: Settings) -> HybridCoverage:
            return coverage

        monkeypatch.setattr(reindex, "_verify", verify)
        monkeypatch.setattr(reindex, "get_settings", settings)
        assert reindex.main(["--verify"]) == code


# --- what QA is handed -----------------------------------------------------------


def _request(retrieval: object | None) -> Any:
    state = SimpleNamespace() if retrieval is None else SimpleNamespace(retrieval=retrieval)
    return SimpleNamespace(app=SimpleNamespace(state=state))


class TestQaIsHandedTheConfiguredRetrieval:
    def test_hybrid_mode_hands_qa_both_arms_of_the_hybrid_collection(self) -> None:
        built = build_vector_indexes(
            settings(qdrant_hybrid_collection=HYBRID, retrieval_mode="hybrid")
        )
        use_case = deps.streaming_answer_use_case(
            _request(built),
            session=SimpleNamespace(),  # type: ignore[arg-type]
            scope=SimpleNamespace(),  # type: ignore[arg-type]
            embedder=SimpleNamespace(),  # type: ignore[arg-type]
            index=built.write,
            providers=[],
        )
        assert use_case._mode is RetrievalMode.HYBRID
        assert use_case._index is built.hybrid and use_case._lexical is built.hybrid

    def test_dense_mode_hands_qa_the_dense_collection_even_while_mirroring(self) -> None:
        built = build_vector_indexes(settings(qdrant_hybrid_collection=HYBRID))
        read, mode, lexical = deps._retrieval(_request(built), built.write)
        assert (read, mode, lexical) == (built.primary, RetrievalMode.DENSE, None)

    def test_a_root_that_wired_only_an_index_gets_todays_dense_path(self) -> None:
        index = object()
        assert deps._retrieval(_request(None), index) == (index, RetrievalMode.DENSE, None)  # type: ignore[arg-type]


# --- the sweep's wiring -----------------------------------------------------------


class _Session:
    async def commit(self) -> None:
        return None


class _Sessions:
    def __call__(self) -> _Sessions:
        return self

    async def __aenter__(self) -> _Session:
        return _Session()

    async def __aexit__(self, *exc: object) -> None:
        return None


class _NoResidue:
    def __init__(self, session: object, scope: object) -> None:
        pass

    async def list_deleting(self) -> list[str]:
        return []

    async def orphaned_paper_ids(self) -> list[str]:
        return []

    async def all_paper_ids(self) -> tuple[str, ...]:
        return ()


class _EmbeddedRows:
    def __init__(self, session: object, scope: object) -> None:
        pass

    async def embedded_chunk_ids(self) -> set[str]:
        return {"a", "b"}


class _HybridPoints:
    async def lexical_chunk_ids(self, *, lexical_version: str) -> set[str]:
        return {"a"} if lexical_version == LEXICAL_VERSION else set()


class _EmptyIndex:
    async def delete_orphans(self, live: Sequence[str]) -> int:
        return 0


class _EmptyStore:
    async def list_keys(self, prefix: str) -> list[str]:
        return []


class TestTheSweepIsHandedTheHybridCollection:
    async def test_the_maintenance_sweep_reports_a_hybrid_shortfall(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(maintenance_module, "SqlWorkspaceRepository", _NoResidue)
        monkeypatch.setattr(maintenance_module, "SqlPaperRepository", _NoResidue)
        monkeypatch.setattr(maintenance_module, "SqlChunkRepository", _EmbeddedRows)
        worker = MaintenanceWorker(
            sessions=_Sessions(),  # type: ignore[arg-type]
            index=_EmptyIndex(),  # type: ignore[arg-type]
            store=_EmptyStore(),  # type: ignore[arg-type]
            hybrid=_HybridPoints(),  # type: ignore[arg-type]
        )
        assert (await worker.reconcile()).hybrid_missing == 1
