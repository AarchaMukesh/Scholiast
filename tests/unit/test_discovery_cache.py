"""The discovery cache's rules, without Redis (D127).

What fails silently in a cache is serving the wrong thing: a failure stored as an
answer, one user's page served to another, a page from an older adapter build, a
field dropped on the way through, a query that should have hit and missed. And the
one thing a cache must never do here: turn a search that would have worked into one
that does not. The Redis adapter's behaviour against a real server is in
tests/integration/test_discovery_cache_redis.py.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scholiast.application.ports.discovery import (
    CachedPage,
    CacheKey,
    DiscoveryCache,
    SourceFailure,
    SourcePage,
    SourceUnavailable,
)
from scholiast.application.use_cases.discover_papers import (
    ConfiguredSource,
    DiscoverPapersUseCase,
    SourceStatus,
)
from scholiast.domain.discovery import PaperCandidate
from scholiast.infrastructure.discovery.arxiv import parse_feed
from scholiast.infrastructure.discovery.cache import (
    FORMAT_VERSION,
    _candidate_to_json,
    decode,
    encode,
    redis_key,
)
from scholiast.infrastructure.discovery.openalex import parse_works

DATA = Path(__file__).resolve().parents[2] / "data" / "eval"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
#: "LoRA" in full-width letters, which NFKC folds to the ASCII spelling.
FULL_WIDTH_LORA = "".join(map(chr, (0xFF2C, 0xFF4F, 0xFF32, 0xFF21)))


def recorded_pages() -> list[SourcePage]:
    """Every OpenAlex page A2 recorded at the shipped size, and every P1 arXiv page."""
    a2 = json.loads(gzip.decompress((DATA / "discovery-pools-openalex-a2.json.gz").read_bytes()))
    pools = json.loads(gzip.decompress((DATA / "discovery-pools.json.gz").read_bytes()))
    pages = [
        parse_works(json.loads(e["body"]))
        for e in a2["entries"]
        if e["per_page"] == 25 and e["status"] == 200
    ]
    pages += [
        parse_feed(e["body"])
        for e in pools["entries"]
        if e["source"] == "arxiv" and e["pass"] == "P1" and e["status"] == 200
    ]
    return pages


class TestTheCodec:
    def test_every_recorded_page_comes_back_identical(self) -> None:
        pages = recorded_pages()
        assert len(pages) == 40, "the recorded pools changed"

        for page in pages:
            back = decode(encode(page, NOW))
            assert back.page == page
            assert back.fetched_at == NOW

    def test_every_candidate_field_is_stored(self) -> None:
        # A field added to `PaperCandidate` and forgotten here would be silently
        # missing from every cached page, while fresh pages had it.
        candidate = recorded_pages()[0].candidates[0]
        stored = set(_candidate_to_json(candidate))
        assert stored == {f.name for f in dataclasses.fields(PaperCandidate)}

    @pytest.mark.parametrize(
        "corrupt",
        [
            b"not zlib at all",
            zlib.compress(b"{not json"),
            "a text entry",
        ],
        ids=["not compressed", "not json", "text, not bytes"],
    )
    def test_an_entry_this_build_cannot_read_is_refused(self, corrupt: bytes | str) -> None:
        with pytest.raises(Exception):  # noqa: B017 - any refusal; the adapter turns it into a miss
            decode(corrupt)

    def test_a_complete_entry_in_an_older_format_is_refused(self) -> None:
        # Complete and otherwise valid, so the format check is the only thing that
        # can refuse it: a partial document would be refused for missing fields and
        # prove nothing (a mutation run caught exactly that).
        document = json.loads(zlib.decompress(encode(recorded_pages()[0], NOW)))
        document["format"] = "0"
        with pytest.raises(ValueError, match="format"):
            decode(zlib.compress(json.dumps(document).encode()))

    def test_a_tampered_candidate_is_refused_by_its_own_invariants(self) -> None:
        document = json.loads(zlib.decompress(encode(recorded_pages()[0], NOW)))
        document["candidates"][0]["rank"] = 0
        with pytest.raises(Exception):  # noqa: B017 - InvariantViolation, re-checked on the way out
            decode(zlib.compress(json.dumps(document).encode()))

    def test_a_fetch_time_without_a_zone_is_refused(self) -> None:
        # An age computed against a naive time would be off by the server's offset.
        document = json.loads(zlib.decompress(encode(recorded_pages()[0], NOW)))
        document["fetched_at"] = "2026-10-04T12:00:00"
        with pytest.raises(ValueError):
            decode(zlib.compress(json.dumps(document).encode()))

    def test_a_search_stores_tens_of_kilobytes_not_hundreds(self) -> None:
        # D127's sizing: ~40 KB for a search's two pages. A codec that stopped
        # compressing would multiply what D128's capped Redis can hold by three.
        sizes = [len(encode(page, NOW)) for page in recorded_pages()]
        assert max(sizes) < 40_000


def key(**changes: object) -> CacheKey:
    base = CacheKey(
        scope="u-1", source="openalex", version="openalex-2", page_size=25, query="lora"
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


class TestTheKey:
    def test_it_shows_neither_the_query_nor_the_user(self) -> None:
        # A key listing must not say what anyone searched for (D127).
        k = redis_key(key(scope="alice@example.org", query="my unpublished idea"))
        assert "alice" not in k and "unpublished" not in k
        assert k.startswith(f"scholiast:discovery:v{FORMAT_VERSION}:")

    def test_the_same_request_is_the_same_key(self) -> None:
        assert redis_key(key()) == redis_key(key())

    @pytest.mark.parametrize(
        "change",
        [
            {"scope": "u-2"},
            {"source": "arxiv"},
            {"version": "openalex-1"},
            {"page_size": 50},
            {"query": "lora adapters"},
        ],
        ids=[
            "another user",
            "another source",
            "another adapter build",
            "another page size",
            "another query",
        ],
    )
    def test_any_difference_is_another_key(self, change: dict[str, object]) -> None:
        assert redis_key(key(**change)) != redis_key(key())


class FakeCache:
    """A dict, with switches to make it fail, and a record of what was stored."""

    def __init__(self) -> None:
        self.store: dict[CacheKey, CachedPage] = {}
        self.fail_get = False
        self.fail_put = False

    async def get(self, key: CacheKey) -> CachedPage | None:
        if self.fail_get:
            raise RuntimeError("cache bug")
        return self.store.get(key)

    async def put(self, key: CacheKey, page: SourcePage, fetched_at: datetime) -> None:
        if self.fail_put:
            raise RuntimeError("cache bug")
        self.store[key] = CachedPage(page, fetched_at)


class Source:
    def __init__(self, name: str = "openalex", *, raises: BaseException | None = None) -> None:
        self._name = name
        self.raises = raises
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def search(self, query: str, *, limit: int) -> SourcePage:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        candidate = PaperCandidate(
            source=self._name, source_id="x1", rank=1, title="A paper", adapter_version="v"
        )
        return SourcePage(self._name, (candidate,), total_hits=1)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


def searcher(
    *sources: Source, cache: FakeCache, clock: Clock | None = None, version: str = "v1"
) -> DiscoverPapersUseCase:
    return DiscoverPapersUseCase(
        [ConfiguredSource(s, 25, version=version) for s in sources],
        cache=cache,
        now=clock or Clock(),
    )


class TestTheUseCaseCaches:
    async def test_a_repeat_asks_nobody_and_says_how_old_its_answer_is(self) -> None:
        source, cache, clock = Source(), FakeCache(), Clock()
        await searcher(source, cache=cache, clock=clock).execute("LoRA", scope="u-1")
        clock.now += timedelta(hours=2)

        result = await searcher(source, cache=cache, clock=clock).execute("LoRA", scope="u-1")

        assert source.calls == 1, "a cached search asked the source again"
        outcome = result.sources[0]
        assert (outcome.status, outcome.cached, outcome.age_seconds) == (
            SourceStatus.OK,
            True,
            7200.0,
        )
        assert [c.source_id for c in result.candidates] == ["x1"]

    @pytest.mark.parametrize(
        "query",
        ["lora", "  LoRA ", "LORA", FULL_WIDTH_LORA],
        ids=["lower", "spaced", "upper", "full-width"],
    )
    async def test_queries_that_normalise_alike_share_an_entry(self, query: str) -> None:
        source, cache = Source(), FakeCache()
        await searcher(source, cache=cache).execute("LoRA", scope="u-1")
        await searcher(source, cache=cache).execute(query, scope="u-1")

        assert source.calls == 1

    async def test_a_hit_does_not_refresh_the_entry(self) -> None:
        # Re-storing on every hit would reset the day's expiry, and a query searched
        # daily would then be served from a page that never ages out (D127).
        source, cache, clock = Source(), FakeCache(), Clock()
        await searcher(source, cache=cache, clock=clock).execute("LoRA", scope="u-1")
        clock.now += timedelta(hours=5)
        await searcher(source, cache=cache, clock=clock).execute("LoRA", scope="u-1")

        assert [entry.fetched_at for entry in cache.store.values()] == [NOW]

    async def test_another_user_never_gets_my_entry(self) -> None:
        # D127: per user. A shared entry would tell the second user someone else
        # searched this exact text today.
        source, cache = Source(), FakeCache()
        await searcher(source, cache=cache).execute("LoRA", scope="alice")

        result = await searcher(source, cache=cache).execute("LoRA", scope="bob")

        assert source.calls == 2
        assert result.sources[0].cached is False

    async def test_a_page_from_another_adapter_build_is_not_served(self) -> None:
        source, cache = Source(), FakeCache()
        await searcher(source, cache=cache, version="openalex-1").execute("LoRA", scope="u-1")
        await searcher(source, cache=cache, version="openalex-2").execute("LoRA", scope="u-1")

        assert source.calls == 2

    @pytest.mark.parametrize(
        "failure",
        [
            SourceUnavailable("openalex", SourceFailure.ERROR, "500"),
            SourceUnavailable("openalex", SourceFailure.RATE_LIMITED, "429", retry_after=5.0),
            SourceUnavailable("openalex", SourceFailure.RATE_LIMITED, "paced", sent=False),
            SourceUnavailable("openalex", SourceFailure.CIRCUIT_OPEN, "open", sent=False),
            KeyError("crash"),
        ],
        ids=["error", "429", "our pacing", "breaker open", "crash"],
    )
    async def test_a_failure_is_never_stored(self, failure: BaseException) -> None:
        # A stored failure would turn one bad second into a day of partial results.
        failing, healthy, cache = Source("openalex", raises=failure), Source("arxiv"), FakeCache()
        await searcher(failing, healthy, cache=cache).execute("LoRA", scope="u-1")
        failing.raises = None

        result = await searcher(failing, healthy, cache=cache).execute("LoRA", scope="u-1")

        assert failing.calls == 2, "the failed source was not asked again"
        assert healthy.calls == 1, "the source that answered was asked again"
        assert {o.source: o.cached for o in result.sources} == {"openalex": False, "arxiv": True}
        assert result.partial is False

    async def test_a_timeout_is_never_stored(self) -> None:
        slow, cache = Source(), FakeCache()

        async def dawdle(query: str, *, limit: int) -> SourcePage:
            import asyncio

            await asyncio.sleep(30)
            raise AssertionError("unreachable")

        slow.search = dawdle  # type: ignore[method-assign]
        fast = Source("arxiv")
        use_case = DiscoverPapersUseCase(
            [ConfiguredSource(slow, 25), ConfiguredSource(fast, 50)],
            cache=cache,
            deadline_seconds=0.1,
        )
        await use_case.execute("LoRA", scope="u-1")

        assert [k.source for k in cache.store] == ["arxiv"]

    async def test_an_empty_page_is_an_answer_and_is_stored(self) -> None:
        class Empty(Source):
            async def search(self, query: str, *, limit: int) -> SourcePage:
                self.calls += 1
                return SourcePage(self.name, ())

        source, cache = Empty(), FakeCache()
        await searcher(source, cache=cache).execute("an obscure topic", scope="u-1")
        await searcher(source, cache=cache).execute("an obscure topic", scope="u-1")

        assert source.calls == 1


class TestTheCacheCanOnlyEverMiss:
    async def test_a_cache_that_raises_on_lookup_is_a_miss_not_a_failed_search(self) -> None:
        source, cache = Source(), FakeCache()
        cache.fail_get = True

        result = await searcher(source, cache=cache).execute("LoRA", scope="u-1")

        assert result.sources[0].status is SourceStatus.OK
        assert source.calls == 1

    async def test_a_cache_that_cannot_store_loses_nothing_but_the_next_hit(self) -> None:
        source, cache = Source(), FakeCache()
        cache.fail_put = True

        result = await searcher(source, cache=cache).execute("LoRA", scope="u-1")

        assert result.sources[0].status is SourceStatus.OK
        assert [c.source_id for c in result.candidates] == ["x1"]

    async def test_no_cache_at_all_is_correct_not_broken(self) -> None:
        source = Source()
        use_case = DiscoverPapersUseCase([ConfiguredSource(source, 25)])
        await use_case.execute("LoRA", scope="u-1")
        await use_case.execute("LoRA", scope="u-1")

        assert source.calls == 2

    def test_the_fake_honours_the_port(self) -> None:
        assert isinstance(FakeCache(), DiscoveryCache)
