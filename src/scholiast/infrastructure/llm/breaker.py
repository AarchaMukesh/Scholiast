"""A circuit breaker over one provider (§5.11, D103, D104).

**What it buys is latency on the ladder, not fewer failures.** Without it, a
provider whose endpoint is unreachable costs its full timeout — 60 s for Groq —
on *every* request before the chain moves to the next rung, so one dead provider
makes every answer slow rather than making one answer fail. An open breaker
turns that 60 s into microseconds, and the failover is disclosed exactly as it
would have been: the use case still sees `GenerationUnavailable` and still flags
`provider_failover`, so the breaker changes what an outage costs and not what
the user is told.

**The open window comes from the server when the server said** (D103). A 429's
`Retry-After` is the authority on its own limit, so it sets the window exactly;
a generic failure gets the configured cooldown, because nothing told us
otherwise. That is D98's reactive-detection argument applied to a second
resource — in both cases the thing enforcing the limit is the thing that knows
it.

**An explicit `Retry-After` opens the breaker immediately, before the failure
threshold.** This follows from what D103 says the mechanism is *for*: "every
subsequent request inside the window is refused locally at no cost". Waiting for
three failures would spend two more calls inside a window the server had already
declared, which is the saving the decision was written to get.

**In-process, per provider instance** (D104). Each process learns independently,
so N processes each spend their own probe discovering the same outage; at this
scale that is bounded at two, and generation happens only in the API process, so
a shared-state version would coordinate between one participant and nothing.
Redis-backed was rejected: it would make answering a question depend on Redis
being reachable, and D97 is the recorded precedent for shipping coordination
nothing needed.

**Three states, not two, because of the risk the breaker itself introduces:**
with a single provider configured, an open breaker means `generation_unavailable`
for the whole cooldown even if the provider recovered a second later. The
HALF-OPEN probe is what bounds that.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from enum import StrEnum, unique

from scholiast.application.ports.llm import (
    GenerationChunk,
    GenerationUnavailable,
    LLMProvider,
)

#: D48's number, for D48's reason: three consecutive failures survives a restart
#: of the thing being called without sitting through an outage.
DEFAULT_FAILURE_THRESHOLD = 3
#: Only used when the provider said nothing. A 429 overrides it (D103).
DEFAULT_COOLDOWN_SECONDS = 30.0


@unique
class BreakerState(StrEnum):
    """Derived from the counters rather than stored alongside them.

    A stored state plus the data it summarises is two representations of one
    fact, and they drift: every transition then has to remember to update both.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """An `LLMProvider` that stops calling a provider it knows is down.

    Wraps by delegation rather than inheritance so it composes with any adapter
    and adds nothing to the port: `name` and `model` pass straight through, which
    is what keeps the chain's identity assertions — and §6.10's per-message
    attribution — true of the wrapped provider rather than of the wrapper.
    """

    def __init__(
        self,
        inner: LLMProvider,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._threshold = max(1, failure_threshold)
        self._cooldown = max(0.0, cooldown_seconds)
        #: Monotonic on purpose. Wall-clock would let an NTP correction close a
        #: breaker early or hold it open past its window, and the window is the
        #: one thing this class is supposed to get right.
        self._clock = clock
        self._failures = 0
        self._opened_until: float | None = None
        #: True while one HALF-OPEN probe is in flight, so concurrent requests
        #: do not all become probes and all pay the timeout the breaker exists
        #: to avoid. D104 says "a single HALF-OPEN probe"; this is that word.
        self._probing = False

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def state(self) -> BreakerState:
        """Where the breaker is now — for tests, and for phase 7's metrics."""
        if self._opened_until is None:
            return BreakerState.CLOSED
        if self._clock() < self._opened_until:
            return BreakerState.OPEN
        return BreakerState.HALF_OPEN

    @property
    def failures(self) -> int:
        """Consecutive failures. A success resets it, so it measures an outage."""
        return self._failures

    async def stream(self, *, system: str, user: str) -> AsyncIterator[GenerationChunk]:
        """Stream from the inner provider, unless it is known to be down.

        An async generator, so the call is cheap and the refusal surfaces on
        first iteration — the same shape every other adapter has, which is what
        the port's docstring requires and what lets the chain treat a breaker
        and a bare provider identically.
        """
        self._before_call()
        try:
            async for chunk in self._inner.stream(system=system, user=user):
                yield chunk
        except GenerationUnavailable as exc:
            self._record_failure(exc.retry_after)
            raise
        # Completion counts as success even if nothing was yielded. A provider
        # that streams no tokens without raising is handled by the use case,
        # which fails over on empty output; making it a breaker failure too
        # would put one decision in two places.
        self._record_success()

    def _before_call(self) -> None:
        state = self.state
        if state is BreakerState.OPEN:
            raise GenerationUnavailable(
                self.name,
                f"circuit open for another {self._remaining():.1f}s",
                # Pass the remaining window on, so a caller that stacks breakers
                # or logs the reason sees the same number this one is holding.
                retry_after=self._remaining(),
            )
        if state is BreakerState.HALF_OPEN:
            if self._probing:
                raise GenerationUnavailable(self.name, "circuit half-open, probe in flight")
            self._probing = True

    def _remaining(self) -> float:
        if self._opened_until is None:
            return 0.0
        return max(0.0, self._opened_until - self._clock())

    def _record_success(self) -> None:
        self._failures = 0
        self._opened_until = None
        self._probing = False

    def _record_failure(self, retry_after: float | None) -> None:
        was_probing = self._probing
        self._probing = False
        self._failures += 1

        # A failed probe re-opens at once, whatever the count: the half-open
        # call exists precisely to answer "is it back", and it said no.
        # An explicit `Retry-After` also opens at once — see the module
        # docstring on why waiting for the threshold would spend the calls the
        # window was meant to save.
        if was_probing or retry_after is not None or self._failures >= self._threshold:
            window = self._cooldown if retry_after is None else retry_after
            self._opened_until = self._clock() + window
