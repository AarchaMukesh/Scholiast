"""The discovery cache against a real Redis (D127, D128).

These are the claims a fake cannot make: that every entry really expires, that a
Redis which is down or hung costs a bounded wait and then nothing, that the cache
Redis compose runs is really capped and evicting, and, end to end through the real
adapters, that a user's second identical search sends nothing (criterion 6) while
another user's sends everything. Plus NFR-P1's cached budget, measured.

They use database 14 of the Redis on 6379, which only tests use, and skip when no
Redis is running; CI runs one (D128).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import struct
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
import respx
from fastapi import FastAPI

from scholiast.api import deps
from scholiast.api.errors import register_error_handlers
from scholiast.api.routers import discovery
from scholiast.application.ports.discovery import CacheKey, SourcePage, VectorKey
from scholiast.application.ports.ml import Precision
from scholiast.application.ports.tenancy import TenantContext
from scholiast.application.use_cases.discover_papers import (
    ConfiguredSource,
    DiscoverPapersUseCase,
)
from scholiast.config.settings import Settings
from scholiast.domain.discovery import PaperCandidate
from scholiast.infrastructure.circuit import Circuit
from scholiast.infrastructure.discovery import arxiv as arxiv_adapter
from scholiast.infrastructure.discovery import openalex as openalex_adapter
from scholiast.infrastructure.discovery.breaker import SourceBreaker
from scholiast.infrastructure.discovery.cache import (
    RedisDiscoveryCache,
    redis_key,
    vector_redis_key,
)
from scholiast.infrastructure.discovery.sources import build_discovery_sources
from scholiast.infrastructure.ml.embedder import LocalEmbedder
from scholiast.infrastructure.observability.metrics import REGISTRY
from scholiast.infrastructure.security.ratelimit import FixedWindowLimiter

pytestmark = [pytest.mark.integration]

REDIS_URL = os.environ.get("SCHOLIAST_TEST_REDIS_URL", "redis://localhost:6379/14")
CACHE_REDIS_URL = "redis://localhost:6380/0"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "discovery"
A3_POOLS = Path(__file__).resolve().parents[2] / "data" / "eval" / "discovery-pools-a3.json.gz"
OPENALEX_WORKS = "https://api.openalex.org/works"
ARXIV_QUERY = "https://export.arxiv.org/api/query"
KEY = "test-key-0123456789"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


async def _alive(url: str) -> bool:
    try:
        client = aioredis.from_url(url, socket_connect_timeout=0.5)
        await client.ping()
        await client.aclose()
    except Exception:
        return False
    return True


needs_redis = pytest.mark.skipif(not asyncio.run(_alive(REDIS_URL)), reason="Redis is not running")


@pytest_asyncio.fixture
async def redis() -> AsyncIterator[aioredis.Redis]:
    client = aioredis.from_url(REDIS_URL)
    yield client
    async for name in client.scan_iter(match="scholiast:discovery:*"):
        await client.delete(name)
    await client.aclose()


def cache_on(client: aioredis.Redis, **kwargs: object) -> RedisDiscoveryCache:
    return RedisDiscoveryCache(client, **kwargs)  # type: ignore[arg-type]


def page() -> SourcePage:
    candidate = PaperCandidate(
        source="openalex", source_id="W1", rank=1, title="A paper", adapter_version="openalex-2"
    )
    return SourcePage("openalex", (candidate,), total_hits=1)


def a_key() -> CacheKey:
    return CacheKey(
        scope=uuid.uuid4().hex,
        source="openalex",
        version="openalex-2",
        page_size=25,
        query="lora",
    )


def count(operation: str, outcome: str, source: str = "openalex") -> float:
    value = REGISTRY.get_sample_value(
        "scholiast_discovery_cache_total",
        {"source": source, "operation": operation, "outcome": outcome},
    )
    return value or 0.0


@needs_redis
class TestAgainstRealRedis:
    async def test_a_page_survives_the_round_trip(self, redis: aioredis.Redis) -> None:
        cache, k = cache_on(redis), a_key()
        await cache.put(k, page(), NOW)

        hit = await cache.get(k)

        assert hit is not None
        assert (hit.page, hit.fetched_at) == (page(), NOW)

    async def test_every_entry_expires_within_the_day(self, redis: aioredis.Redis) -> None:
        # D127, and what keeps D128's eviction honest: an entry without an expiry
        # would be served forever if memory never filled.
        cache, k = cache_on(redis), a_key()
        await cache.put(k, page(), NOW)

        ttl = await redis.ttl(redis_key(k))

        assert 0 < ttl <= 24 * 3600

    async def test_an_entry_this_build_cannot_read_is_a_miss_and_is_overwritten(
        self, redis: aioredis.Redis
    ) -> None:
        cache, k = cache_on(redis), a_key()
        await redis.set(redis_key(k), b"written by something else")
        before = count("get", "unreadable")

        assert await cache.get(k) is None
        assert count("get", "unreadable") == before + 1

        await cache.put(k, page(), NOW)
        assert await cache.get(k) is not None


class TestAMissingOrSickRedisCostsAMissNotASearch:
    async def test_redis_down_is_a_quick_miss_and_nothing_stored(self) -> None:
        # Nothing listens on port 1. Connection refused locally is immediate.
        cache = cache_on(aioredis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2))

        started = time.perf_counter()
        assert await cache.get(a_key()) is None
        await cache.put(a_key(), page(), NOW)

        assert time.perf_counter() - started < 1.0

    async def test_a_hung_redis_costs_a_bounded_wait_then_nothing(self) -> None:
        """A server that accepts and never answers: the worst case, and the one a
        plain timeout would pay on every search. The breaker is what stops that."""
        connections = 0

        async def accept_and_ignore(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            nonlocal connections
            connections += 1
            await asyncio.sleep(30)

        server = await asyncio.start_server(accept_and_ignore, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            # No socket timeouts on the client: the bound must be the cache's own,
            # whatever a deployment configures the client with.
            client = aioredis.from_url(f"redis://127.0.0.1:{port}/0")
            cache = cache_on(client, timeout_seconds=0.1, circuit=Circuit(failure_threshold=3))
            for _ in range(3):
                started = time.perf_counter()
                assert await cache.get(a_key()) is None
                assert time.perf_counter() - started < 0.5, "one call was not bounded"
            seen = connections
            before = count("get", "bypassed")

            started = time.perf_counter()
            assert await cache.get(a_key()) is None

            assert time.perf_counter() - started < 0.05, "the open breaker still waited on Redis"
            assert connections == seen, "the open breaker still dialled Redis"
            assert count("get", "bypassed") == before + 1
        finally:
            server.close()


@pytest.mark.skipif(
    not asyncio.run(_alive(CACHE_REDIS_URL)), reason="compose's redis-cache is not running"
)
class TestTheCacheRedisIsCapped:
    async def test_it_has_a_memory_cap_evicts_any_key_and_persists_nothing(self) -> None:
        # D128's whole point, checked against the container compose runs.
        client = aioredis.from_url(CACHE_REDIS_URL)
        try:
            policy = (await client.config_get("maxmemory-policy"))["maxmemory-policy"]
            cap = int((await client.config_get("maxmemory"))["maxmemory"])
            save = (await client.config_get("save"))["save"]
        finally:
            await client.aclose()

        assert policy == "allkeys-lru"
        assert 0 < cap <= 512 * 1024 * 1024
        assert save == ""


# ---------------------------------------------------------------------------
# End to end: the real adapters, a real Redis, respx in place of the sources.
# ---------------------------------------------------------------------------


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def routes() -> tuple[respx.Route, respx.Route]:
    return (
        respx.get(OPENALEX_WORKS).mock(
            return_value=httpx.Response(200, text=fixture("openalex_search.json"))
        ),
        respx.get(ARXIV_QUERY).mock(
            return_value=httpx.Response(200, text=fixture("arxiv_search.xml"))
        ),
    )


def unpaced() -> list[ConfiguredSource]:
    """The real adapters behind real breakers, without arXiv's pacer.

    Pacing is tested on its own; here it would refuse a second search within 3 s
    and hide whether the cache or the pacer kept arXiv from being asked.
    """
    return [
        ConfiguredSource(
            SourceBreaker(openalex_adapter.OpenAlexSource(KEY)),
            25,
            version=openalex_adapter.ADAPTER_VERSION,
        ),
        ConfiguredSource(
            SourceBreaker(arxiv_adapter.ArxivSource()), 50, version=arxiv_adapter.ADAPTER_VERSION
        ),
    ]


@needs_redis
class TestCriterionSixEndToEnd:
    @respx.mock
    async def test_a_users_second_identical_search_sends_nothing(
        self, redis: aioredis.Redis
    ) -> None:
        # Through the app's own wiring, pacer included.
        openalex, arxiv = routes()
        sources = build_discovery_sources(Settings(openalex_api_key=KEY))
        search = DiscoverPapersUseCase(sources, cache=cache_on(redis))
        user = uuid.uuid4().hex

        await search.execute("dense passage retrieval", scope=user)
        sent = (openalex.call_count, arxiv.call_count)
        second = await search.execute("Dense  PASSAGE retrieval ", scope=user)

        assert sent == (1, 1)
        assert (openalex.call_count, arxiv.call_count) == sent, "a cached search went out"
        assert all(o.cached for o in second.sources)
        assert second.partial is False

    @respx.mock
    async def test_a_source_that_failed_is_asked_again_and_the_other_is_not(
        self, redis: aioredis.Redis
    ) -> None:
        openalex, _ = routes()
        arxiv = respx.get(ARXIV_QUERY).mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(200, text=fixture("arxiv_search.xml")),
            ]
        )
        search = DiscoverPapersUseCase(unpaced(), cache=cache_on(redis))
        user = uuid.uuid4().hex

        first = await search.execute("dense passage retrieval", scope=user)
        second = await search.execute("dense passage retrieval", scope=user)

        assert first.partial is True
        assert (openalex.call_count, arxiv.call_count) == (1, 2)
        assert {o.source: o.cached for o in second.sources} == {"openalex": True, "arxiv": False}
        assert second.partial is False

    @respx.mock
    async def test_another_users_identical_search_is_not_served_from_mine(
        self, redis: aioredis.Redis
    ) -> None:
        openalex, arxiv = routes()
        search = DiscoverPapersUseCase(unpaced(), cache=cache_on(redis))

        await search.execute("dense passage retrieval", scope=uuid.uuid4().hex)
        other = await search.execute("dense passage retrieval", scope=uuid.uuid4().hex)

        assert (openalex.call_count, arxiv.call_count) == (2, 2)
        assert not any(o.cached for o in other.sources)


def app_with(cache: RedisDiscoveryCache, user: str) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(discovery.router)
    app.state.settings = Settings(rate_limit_enabled=False)
    app.state.rate_limiter = FixedWindowLimiter()
    app.state.discovery_sources = build_discovery_sources(Settings(openalex_api_key=KEY))
    app.state.discovery_cache = cache
    app.dependency_overrides[deps.current_tenant] = lambda: TenantContext(user_id=user)
    return app


@needs_redis
class TestThroughTheEndpoint:
    @respx.mock
    async def test_the_response_says_which_pages_were_cached_and_how_old(
        self, redis: aioredis.Redis
    ) -> None:
        routes()
        app = app_with(cache_on(redis), uuid.uuid4().hex)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            first = (await client.post("/v1/discovery/search", json={"query": "dpr"})).json()
            second = (await client.post("/v1/discovery/search", json={"query": "dpr"})).json()

        assert [s["cached"] for s in first["source_status"]] == [False, False]
        assert [s["cached"] for s in second["source_status"]] == [True, True]
        assert all(0 <= s["age_seconds"] < 60 for s in second["source_status"])
        assert second["results"] == first["results"], "a cached search answered differently"

    @respx.mock
    async def test_each_user_searches_as_themselves(self, redis: aioredis.Redis) -> None:
        # The route must pass the caller's identity to the cache, not a constant: a
        # second user's identical search is a miss, as D127 says.
        routes()
        app = app_with(cache_on(redis), "alice-" + uuid.uuid4().hex)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            await client.post("/v1/discovery/search", json={"query": "dpr"})
            bob = TenantContext(user_id="bob-" + uuid.uuid4().hex)
            app.dependency_overrides[deps.current_tenant] = lambda: bob
            second = (await client.post("/v1/discovery/search", json={"query": "dpr"})).json()

        assert [s["cached"] for s in second["source_status"]] == [False, False]

    @respx.mock
    async def test_a_cached_search_meets_nfr_p1s_400_ms(self, redis: aioredis.Redis) -> None:
        """NFR-P1: p95 <= 400 ms cached, for the whole request through the endpoint.

        Measured here against a real Redis on this machine, so it is a regression
        bound, not phase 10's live number. Every search after the first is cached.
        """
        routes()
        app = app_with(cache_on(redis), uuid.uuid4().hex)
        transport = httpx.ASGITransport(app=app)
        timings = []
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            await client.post("/v1/discovery/search", json={"query": "dpr"})
            for _ in range(30):
                started = time.perf_counter()
                response = await client.post("/v1/discovery/search", json={"query": "dpr"})
                timings.append(time.perf_counter() - started)
                assert all(s["cached"] for s in response.json()["source_status"])

        timings.sort()
        p95 = timings[int(0.95 * len(timings)) - 1]
        assert p95 <= 0.4, f"cached p95 {p95 * 1000:.0f} ms"


# -- work vectors (D153) ----------------------------------------------------------


def a_vector_key(record: str = "arxiv:2004.04906", scope: str = "u-1") -> VectorKey:
    return VectorKey(scope=scope, embedder="enc/1", record=record, text_sha256="ab" * 32)


@needs_redis
class TestWorkVectorsAgainstRealRedis:
    async def test_a_vector_survives_the_round_trip_at_float32(self, redis: aioredis.Redis) -> None:
        cache, k = cache_on(redis), a_vector_key()
        vector = [0.1, -0.25, 3.0e-3]

        await cache.put_vectors([(k, vector)])
        [found] = await cache.get_vectors([k])

        assert found == pytest.approx(vector, rel=1e-6)

    async def test_every_vector_expires_within_the_day(self, redis: aioredis.Redis) -> None:
        cache, k = cache_on(redis), a_vector_key()
        await cache.put_vectors([(k, [1.0])])

        ttl = await redis.ttl(vector_redis_key(k))

        assert 0 < ttl <= 24 * 3600

    async def test_hits_and_misses_come_back_in_the_order_asked(
        self, redis: aioredis.Redis
    ) -> None:
        cache = cache_on(redis)
        stored, absent = a_vector_key("arxiv:1"), a_vector_key("arxiv:2")
        await cache.put_vectors([(stored, [1.0, 2.0])])

        found = await cache.get_vectors([absent, stored, absent])

        assert found[0] is None and found[2] is None
        assert found[1] == pytest.approx([1.0, 2.0])

    async def test_another_users_vector_is_a_miss(self, redis: aioredis.Redis) -> None:
        cache = cache_on(redis)
        await cache.put_vectors([(a_vector_key(scope="u-1"), [1.0])])

        assert await cache.get_vectors([a_vector_key(scope="u-2")]) == [None]

    async def test_a_vector_in_another_format_is_a_miss_not_a_misread(
        self, redis: aioredis.Redis
    ) -> None:
        # Whole float32 values under another format tag: readable bytes, wrong meaning.
        cache, k = cache_on(redis), a_vector_key()
        await redis.set(vector_redis_key(k), b"v0:" + struct.pack("<2f", 1.0, 2.0))

        assert await cache.get_vectors([k]) == [None]

    async def test_an_entry_this_build_cannot_read_is_a_miss(self, redis: aioredis.Redis) -> None:
        cache, k = cache_on(redis), a_vector_key()
        await redis.set(vector_redis_key(k), b"written by something else")
        before = count("get", "unreadable", source="vectors")

        assert await cache.get_vectors([k]) == [None]
        assert count("get", "unreadable", source="vectors") == before + 1


class TestWorkVectorsWithRedisDown:
    async def test_redis_down_is_every_vector_a_miss_and_nothing_raised(self) -> None:
        cache = cache_on(aioredis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2))

        started = time.perf_counter()
        assert await cache.get_vectors([a_vector_key(), a_vector_key("arxiv:2")]) == [None, None]
        await cache.put_vectors([(a_vector_key(), [1.0])])

        assert time.perf_counter() - started < 1.0


def _encoder_ready() -> bool:
    """The production encoder, cached locally, on a CUDA device (NFR-T3: no downloads)."""
    settings = Settings()
    try:
        import torch
        from huggingface_hub import snapshot_download

        snapshot_download(settings.embedding_model, local_files_only=True)
        return bool(torch.cuda.is_available()) or not settings.embedding_device.startswith("cuda")
    except Exception:
        return False


@pytest.fixture(scope="module")
def warm_encoder() -> LocalEmbedder:
    """The production encoder, loaded before any test mocks HTTP.

    Loading asks the model hub whether the cached files are current; under respx
    that request is refused, the load fails, and relevance falls back to lexical,
    which is how this test first failed.
    """
    settings = Settings()
    encoder = LocalEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        precision=Precision(settings.embedding_precision),
    )
    assert encoder.dimensions > 0  # loads the model
    return encoder


@needs_redis
@pytest.mark.skipif(not _encoder_ready(), reason="production encoder not cached locally, or no GPU")
class TestDenseRankingThroughTheEndpoint:
    @respx.mock
    async def test_a_cached_dense_search_meets_nfr_p1s_400_ms(
        self, redis: aioredis.Redis, warm_encoder: LocalEmbedder
    ) -> None:
        """D153's claim: with work vectors cached, a repeat search embeds only its query.

        Through the endpoint with the production encoder on this machine; a regression
        bound, not phase 10's live number. The first search is cold and warms the model.
        """
        # A real-sized search: A3's recorded responses for one query, about 60 works.
        # The cached path's cost grows with the works (one vector each), so a
        # five-work fixture would pass this bound without testing it.
        entries = json.loads(gzip.decompress(A3_POOLS.read_bytes()))["entries"]
        bodies = {e["source"]: e["body"] for e in entries if e["query_id"] == "t03-keyword"}
        respx.get(OPENALEX_WORKS).mock(return_value=httpx.Response(200, text=bodies["openalex"]))
        respx.get(ARXIV_QUERY).mock(return_value=httpx.Response(200, text=bodies["arxiv"]))
        query = {"query": "dense passage retrieval open-domain question answering"}
        app = app_with(cache_on(redis), uuid.uuid4().hex)
        app.state.embedder = warm_encoder
        transport = httpx.ASGITransport(app=app)
        timings = []
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            cold = (await client.post("/v1/discovery/search", json=query)).json()
            hits_before = count("get", "hit", source="vectors")
            for _ in range(30):
                started = time.perf_counter()
                response = await client.post("/v1/discovery/search", json=query)
                timings.append(time.perf_counter() - started)

        assert len(cold["results"]) >= 50
        assert cold["ranking"]["relevance"] == "dense"
        assert response.json()["ranking"]["relevance"] == "dense"
        assert count("get", "hit", source="vectors") == hits_before + 30, (
            "a cached search embedded its works again"
        )
        timings.sort()
        p95 = timings[int(0.95 * len(timings)) - 1]
        assert p95 <= 0.4, f"cached dense p95 {p95 * 1000:.0f} ms"
