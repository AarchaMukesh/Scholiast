"""Discovery's cache, in its own Redis (D127, D128).

**It can only ever turn a source call into a miss** (the port's rule). Every Redis
call has a short timeout, and every failure, Redis down, hung or full, or an entry
that no longer decodes, is a miss. After three failures in a row the cache's own
breaker (the shared core, D122) stops calling Redis until its window passes, so a
hung Redis costs a search a few hundred milliseconds once, not on every search.

**Keys are hashed.** The key material is the port's `CacheKey` (user, source,
adapter version, page size, normalised query) and this module's format version, and
none of it appears in Redis: a key listing does not show what anyone searched for.

**Values are compressed JSON of the candidates**, about 40 KB for a search's two
pages, measured on the recorded responses. The codec names every `PaperCandidate`
field explicitly, and a test checks it covers them all, so a field added to the
candidate cannot be silently dropped from cached pages.

**The Redis is the cache's alone** (D128): capped, evicting least-recently-used
keys, no persistence. Every write still carries the day's expiry, so an entry is
never older than D127 allows even when memory is plentiful.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import zlib
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from typing import Any, TypeVar

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from scholiast.application.ports.discovery import CachedPage, CacheKey, SourcePage
from scholiast.domain.discovery import LinkKind, OpenAccessLink, PaperCandidate
from scholiast.infrastructure.circuit import Circuit, Refused
from scholiast.infrastructure.observability import metrics

#: Bumped whenever the stored shape changes, so old entries miss instead of failing.
FORMAT_VERSION = "1"
KEY_PREFIX = "scholiast:discovery"
#: D127.
DEFAULT_TTL_SECONDS = 24 * 3600
#: A local Redis answers in well under a millisecond; this only bounds a sick one.
DEFAULT_TIMEOUT_SECONDS = 0.25

log = structlog.get_logger(__name__)
T = TypeVar("T")


def redis_key(key: CacheKey) -> str:
    """The Redis key for a cache key: the format version and a hash of everything else."""
    material = json.dumps(
        [FORMAT_VERSION, key.scope, key.source, key.version, key.page_size, key.query],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}:v{FORMAT_VERSION}:{digest}"


def _candidate_to_json(c: PaperCandidate) -> dict[str, Any]:
    return {
        "source": c.source,
        "source_id": c.source_id,
        "rank": c.rank,
        "title": c.title,
        "authors": list(c.authors),
        "year": c.year,
        "published": c.published.isoformat() if c.published is not None else None,
        "venue": c.venue,
        "abstract": c.abstract,
        "citation_count": c.citation_count,
        "doi": c.doi,
        "arxiv_id": c.arxiv_id,
        "oa_links": [[link.url, link.kind.value, link.licence] for link in c.oa_links],
        "source_score": c.source_score,
        "adapter_version": c.adapter_version,
    }


def _candidate_from_json(d: dict[str, Any]) -> PaperCandidate:
    # The candidate's own invariants run again here, so an entry that was tampered
    # with or written by a buggy build is refused rather than served.
    return PaperCandidate(
        source=d["source"],
        source_id=d["source_id"],
        rank=d["rank"],
        title=d["title"],
        authors=tuple(d["authors"]),
        year=d["year"],
        published=date.fromisoformat(d["published"]) if d["published"] is not None else None,
        venue=d["venue"],
        abstract=d["abstract"],
        citation_count=d["citation_count"],
        doi=d["doi"],
        arxiv_id=d["arxiv_id"],
        oa_links=tuple(
            OpenAccessLink(url, LinkKind(kind), licence) for url, kind, licence in d["oa_links"]
        ),
        source_score=d["source_score"],
        adapter_version=d["adapter_version"],
    )


def encode(page: SourcePage, fetched_at: datetime) -> bytes:
    document = {
        "format": FORMAT_VERSION,
        "fetched_at": fetched_at.isoformat(),
        "source": page.source,
        "total_hits": page.total_hits,
        "skipped": page.skipped,
        "candidates": [_candidate_to_json(c) for c in page.candidates],
    }
    return zlib.compress(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def decode(raw: bytes | str) -> CachedPage:
    """A stored entry back to a page. Raises on anything malformed; the caller misses."""
    if not isinstance(raw, bytes):
        # The client never decodes responses, so text here was not written by `encode`.
        raise ValueError("a cached entry must be bytes")
    document = json.loads(zlib.decompress(raw))
    if document.get("format") != FORMAT_VERSION:
        raise ValueError(f"cached entry has format {document.get('format')!r}")
    fetched_at = datetime.fromisoformat(document["fetched_at"])
    if fetched_at.tzinfo is None:
        raise ValueError("cached entry's fetch time has no timezone")
    page = SourcePage(
        source=document["source"],
        candidates=tuple(_candidate_from_json(c) for c in document["candidates"]),
        total_hits=document["total_hits"],
        skipped=document["skipped"],
    )
    return CachedPage(page=page, fetched_at=fetched_at)


class RedisDiscoveryCache:
    """The `DiscoveryCache` port over Redis. Never raises for anything Redis does."""

    def __init__(
        self,
        client: Redis,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        circuit: Circuit | None = None,
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._timeout = timeout_seconds
        self._circuit = circuit or Circuit()

    async def get(self, key: CacheKey) -> CachedPage | None:
        found, raw = await self._call(key.source, "get", lambda: self._client.get(redis_key(key)))
        if not found:
            return None
        if raw is None:
            metrics.record_discovery_cache(key.source, "get", "miss")
            return None
        try:
            hit = decode(raw)
        except Exception as exc:
            # An entry this build cannot read is a miss; the next answer overwrites it.
            metrics.record_discovery_cache(key.source, "get", "unreadable")
            log.warning("discovery.cache_unreadable", source=key.source, error=type(exc).__name__)
            return None
        metrics.record_discovery_cache(key.source, "get", "hit")
        return hit

    async def put(self, key: CacheKey, page: SourcePage, fetched_at: datetime) -> None:
        value = encode(page, fetched_at)
        stored, _ = await self._call(
            key.source, "put", lambda: self._client.set(redis_key(key), value, ex=self._ttl)
        )
        if stored:
            metrics.record_discovery_cache(key.source, "put", "stored")

    async def close(self) -> None:
        await self._client.aclose()

    async def _call(
        self, source: str, operation: str, command: Callable[[], Awaitable[T]]
    ) -> tuple[bool, T | None]:
        """Run one Redis command under the timeout and the breaker: (succeeded, result)."""
        admission = self._circuit.admit()
        if isinstance(admission, Refused):
            metrics.record_discovery_cache(source, operation, "bypassed")
            return False, None
        settled = False
        try:
            async with asyncio.timeout(self._timeout):
                result = await command()
        except (RedisError, OSError, TimeoutError) as exc:
            settled = True
            self._circuit.failed(None)
            metrics.record_discovery_cache(source, operation, "error")
            log.warning(
                "discovery.cache_unavailable",
                source=source,
                operation=operation,
                error=type(exc).__name__,
            )
            return False, None
        else:
            settled = True
            self._circuit.succeeded()
            return True, result
        finally:
            if not settled:
                # Cancelled from outside, or a bug: no verdict on Redis's health.
                self._circuit.abandoned(admission)
