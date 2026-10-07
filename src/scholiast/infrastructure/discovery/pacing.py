"""Keeping to a source's published request rate, without making a search wait (D123).

arXiv's API terms ask for no more than one request every three seconds, on one
connection. `PacedSource` holds a source to that: one request at a time, each
starting at least `min_gap_seconds` after the previous one ended.

**A search never waits for the slot.** If the slot is not free when the search
asks, the source is reported `rate_limited` at once, with the time until it is free
as `retry_after`, and the search goes on with the other sources. Waiting would make
arXiv's pace every user's latency, and a request sent after a wait would usually be
cut off at D122's deadline anyway, spending arXiv's capacity on an answer nobody
receives. A refusal here is marked `sent=False`, so the source's breaker does not
count it: arXiv did nothing wrong.

**Per process** (D123, as D104 and D112). One API process keeps arXiv's terms; a
deployment that runs two must share the pacer in the same change, or N processes
send N times the rate. That is recorded as an M6 precondition.

**A request cut off at the deadline counts as ended when it was cut:** the gap is
measured from then, because that is the last moment this process knows about.

**A lookup by id may wait, a little** (D133). Adding papers is an explicit action
with no 2 s deadline, and refusing it because a search just used the slot would turn
a click into an error. So a lookup waits for the slot when it will be free within
`lookup_wait_seconds` (3 s), and is refused like a search otherwise. While a request
is in flight its end is unknown, but the slot is then at least a full gap away, so a
wait no longer than the gap is refused at once rather than spent finding that out.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TypeVar

from scholiast.application.ports.discovery import (
    AcademicSource,
    SourceFailure,
    SourcePage,
    SourceUnavailable,
)
from scholiast.domain.discovery import PaperCandidate

#: How long a lookup may wait for the slot (D133).
LOOKUP_WAIT_SECONDS = 3.0
#: How often a waiting lookup looks again while a request is in flight.
_POLL_SECONDS = 0.1

T = TypeVar("T")


class PacedSource:
    """An `AcademicSource` held to one request at a time and a minimum gap after each."""

    def __init__(
        self,
        inner: AcademicSource,
        *,
        min_gap_seconds: float,
        lookup_wait_seconds: float = LOOKUP_WAIT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._inner = inner
        self._gap = max(0.0, min_gap_seconds)
        self._lookup_wait = max(0.0, lookup_wait_seconds)
        self._clock = clock
        self._sleep = sleep
        self._busy = False
        #: When the next request may start. Monotonic, for the breaker's reason: a
        #: clock correction must not shorten a gap a third party's terms set.
        self._free_at = float("-inf")

    @property
    def name(self) -> str:
        return self._inner.name

    async def search(self, query: str, *, limit: int) -> SourcePage:
        # Check and claim with no await between them, so two searches on the event
        # loop cannot both see the slot free.
        now = self._clock()
        if self._busy or now < self._free_at:
            # While a request is in flight its end is unknown; the gap is the least wait.
            raise self._refusal(self._gap if self._busy else self._free_at - now, "D123")
        return await self._paced(lambda: self._inner.search(query, limit=limit))

    def is_valid_id(self, source_id: str) -> bool:
        return self._inner.is_valid_id(source_id)

    async def lookup(self, source_ids: Sequence[str]) -> Mapping[str, PaperCandidate]:
        give_up_at = self._clock() + self._lookup_wait
        while self._busy or self._clock() < self._free_at:
            now = self._clock()
            if self._busy:
                # In flight: free a full gap after it ends, so strictly later than
                # `now + gap`. Refused once that cannot fit; otherwise look again soon.
                if now + self._gap >= give_up_at:
                    raise self._refusal(self._gap, "D133")
                await self._sleep(min(_POLL_SECONDS, give_up_at - now))
            elif self._free_at > give_up_at:
                raise self._refusal(self._free_at - now, "D133")
            else:
                await self._sleep(self._free_at - now)
        # The loop's last check and the claim inside `_paced` have no await between
        # them, so two waiting lookups cannot both take the slot.
        return await self._paced(lambda: self._inner.lookup(source_ids))

    def _refusal(self, wait: float, decision: str) -> SourceUnavailable:
        return SourceUnavailable(
            self.name,
            SourceFailure.RATE_LIMITED,
            f"paced: the next request may start in {wait:.1f}s ({decision})",
            retry_after=wait,
            sent=False,
        )

    async def _paced(self, call: Callable[[], Awaitable[T]]) -> T:
        self._busy = True
        sent = True
        try:
            return await call()
        except ValueError:
            # The port's "nothing searchable in this query" or "not this source's
            # id", refused before sending: no request reached the source, so no
            # slot was used.
            sent = False
            raise
        except SourceUnavailable as exc:
            sent = exc.sent
            raise
        finally:
            self._busy = False
            if sent:
                self._free_at = self._clock() + self._gap
