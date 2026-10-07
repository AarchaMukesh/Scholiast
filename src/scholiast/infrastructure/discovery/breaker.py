"""A circuit breaker over one academic source (D104, D122).

The same state machine as the LLM providers' (`scholiast.infrastructure.circuit`),
adapted to the discovery port. What it buys is the same, too: a source that is down
is not asked again until its window passes, so a dead source costs a search nothing
instead of its share of the deadline, and an outage costs OpenAlex no credits.

**What counts as a failure is the part that differs from the providers':**

  a failure        `SourceUnavailable` for a request that was sent, any other
                   exception, and the deadline cancelling the call. A source that
                   cannot answer inside the budget is down for this purpose (D122).

  not a failure    our own refusals, where nothing reached the source: arXiv's pacing
                   (D123) and a query the source cannot express, which the port says
                   is the caller's mistake. Counting them would let a burst of
                   searches, or three stopword-only queries, open arXiv's breaker
                   for something arXiv never saw.

A lookup by id (D133) goes through the same gate with the same rules, so an add
never asks a source the breaker knows is down, and a source that fails lookups is
as down for searches as one that fails searches.
"""

from __future__ import annotations

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
from scholiast.infrastructure.circuit import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_FAILURE_THRESHOLD,
    BreakerState,
    Circuit,
    Refused,
)

T = TypeVar("T")


class SourceBreaker:
    """An `AcademicSource` that stops asking a source it knows is down."""

    def __init__(
        self,
        inner: AcademicSource,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._circuit = Circuit(
            failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds, clock=clock
        )

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def state(self) -> BreakerState:
        return self._circuit.state

    @property
    def failures(self) -> int:
        return self._circuit.failures

    async def search(self, query: str, *, limit: int) -> SourcePage:
        return await self._guarded(lambda: self._inner.search(query, limit=limit))

    def is_valid_id(self, source_id: str) -> bool:
        return self._inner.is_valid_id(source_id)

    async def lookup(self, source_ids: Sequence[str]) -> Mapping[str, PaperCandidate]:
        return await self._guarded(lambda: self._inner.lookup(source_ids))

    async def _guarded(self, call: Callable[[], Awaitable[T]]) -> T:
        admission = self._circuit.admit()
        if isinstance(admission, Refused):
            raise SourceUnavailable(
                self.name,
                SourceFailure.CIRCUIT_OPEN,
                admission.reason,
                retry_after=admission.retry_after,
                sent=False,
            )
        settled = False
        try:
            result = await call()
        except SourceUnavailable as exc:
            if exc.sent:
                settled = True
                self._circuit.failed(exc.retry_after)
            raise
        except ValueError:
            # The port's "nothing searchable in this query" or "not this source's
            # id": refused before sending, so nothing was sent.
            raise
        except BaseException:
            # A crash in the adapter, or the deadline cancelling a request still in
            # flight: either way the source did not answer in time (D122).
            settled = True
            self._circuit.failed(None)
            raise
        else:
            settled = True
            self._circuit.succeeded()
        finally:
            if not settled:
                self._circuit.abandoned(admission)
        return result
