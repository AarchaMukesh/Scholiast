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
"""

from __future__ import annotations

import time
from collections.abc import Callable

from scholiast.application.ports.discovery import (
    AcademicSource,
    SourceFailure,
    SourcePage,
    SourceUnavailable,
)


class PacedSource:
    """An `AcademicSource` held to one request at a time and a minimum gap after each."""

    def __init__(
        self,
        inner: AcademicSource,
        *,
        min_gap_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._gap = max(0.0, min_gap_seconds)
        self._clock = clock
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
            wait = self._gap if self._busy else self._free_at - now
            raise SourceUnavailable(
                self.name,
                SourceFailure.RATE_LIMITED,
                f"paced: the next request may start in {wait:.1f}s (D123)",
                retry_after=wait,
                sent=False,
            )
        self._busy = True
        sent = True
        try:
            return await self._inner.search(query, limit=limit)
        except ValueError:
            # The port's "nothing searchable in this query", refused before sending:
            # no request reached the source, so no slot was used.
            sent = False
            raise
        except SourceUnavailable as exc:
            sent = exc.sent
            raise
        finally:
            self._busy = False
            if sent:
                self._free_at = self._clock() + self._gap
