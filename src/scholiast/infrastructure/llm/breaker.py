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

**The state machine is `scholiast.infrastructure.circuit`'s** (D122), shared with
the discovery sources' breakers. This class adapts it to the provider port: what
counts as a failure, and which exception a refusal becomes.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable

from scholiast.application.ports.llm import (
    GenerationChunk,
    GenerationUnavailable,
    LLMProvider,
)
from scholiast.infrastructure.circuit import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_FAILURE_THRESHOLD,
    BreakerState,
    Circuit,
    Refused,
)

__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "DEFAULT_FAILURE_THRESHOLD",
    "BreakerState",
    "CircuitBreaker",
]


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
        self._circuit = Circuit(
            failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds, clock=clock
        )

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def state(self) -> BreakerState:
        """Where the breaker is now — for tests, and for phase 7's metrics."""
        return self._circuit.state

    @property
    def failures(self) -> int:
        """Consecutive failures. A success resets it, so it measures an outage."""
        return self._circuit.failures

    async def stream(self, *, system: str, user: str) -> AsyncIterator[GenerationChunk]:
        """Stream from the inner provider, unless it is known to be down.

        An async generator, so the call is cheap and the refusal surfaces on
        first iteration — the same shape every other adapter has, which is what
        the port's docstring requires and what lets the chain treat a breaker
        and a bare provider identically.
        """
        admission = self._circuit.admit()
        if isinstance(admission, Refused):
            raise GenerationUnavailable(
                self.name, admission.reason, retry_after=admission.retry_after
            )
        settled = False
        try:
            async for chunk in self._inner.stream(system=system, user=user):
                yield chunk
        except GenerationUnavailable as exc:
            settled = True
            self._circuit.failed(exc.retry_after)
            raise
        else:
            # Completion counts as success even if nothing was yielded. A provider
            # that streams no tokens without raising is handled by the use case,
            # which fails over on empty output; making it a breaker failure too
            # would put one decision in two places.
            settled = True
            self._circuit.succeeded()
        finally:
            # A caller that stopped reading, a cancellation, or any other exception
            # says nothing about the provider, so it is not a failure; but a probe
            # that ends this way must free the probe slot (D122).
            if not settled:
                self._circuit.abandoned(admission)
