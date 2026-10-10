"""The circuit-breaker state machine, shared by every breaker in the system (D104, D122).

M3 built it inside the LLM provider breaker; M4's academic sources need exactly the
same machine, so it lives here once and each breaker adapts it to its own port and
its own exception. Two copies of a three-state machine with a `Retry-After` window
would drift, and the window is the one thing a breaker must get right.

**Three outcomes for a call, not two** (D122). A call that was admitted ends in a
success, a failure, or neither: a stream its caller stops reading, a call cancelled
from outside, a request that was never sent. M3's breaker had no third outcome, so
a HALF-OPEN probe that ended without a verdict kept the probe slot forever and the
breaker refused every later call until the process restarted. Discovery cancels
slow sources at its deadline as a matter of course, which would have made that
routine. `abandoned` releases the slot and leaves the breaker HALF-OPEN, so the
next call probes again.

**The probe is a ticket, not a flag the next caller can clear.** `admit` says
whether this call holds the probe, and only that call can release it: a call
admitted before the breaker opened, finishing late, cannot hand the probe slot to
a second concurrent probe.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique

#: D48's number, for D48's reason: three consecutive failures survives a restart
#: of the thing being called without sitting through an outage.
DEFAULT_FAILURE_THRESHOLD = 3
#: Only used when the callee said nothing. A stated `Retry-After` overrides it (D103).
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


@dataclass(frozen=True, slots=True)
class Admitted:
    """The call may go ahead. `probe` is True when it is the one HALF-OPEN probe."""

    probe: bool


@dataclass(frozen=True, slots=True)
class Refused:
    """The call must not be made. Each breaker turns this into its port's exception."""

    reason: str
    #: The rest of the open window, so a caller that stacks breakers or logs the
    #: reason sees the same number this one is holding. None while a probe is in
    #: flight: nothing is known about when the probe will answer.
    retry_after: float | None = None


class Circuit:
    """CLOSED until a failure threshold, OPEN for a window, then one HALF-OPEN probe."""

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = max(1, failure_threshold)
        self._cooldown = max(0.0, cooldown_seconds)
        #: Monotonic on purpose. Wall-clock would let an NTP correction close a
        #: breaker early or hold it open past its window, and the window is the
        #: one thing this class is supposed to get right.
        self._clock = clock
        self._failures = 0
        self._opened_until: float | None = None
        #: True while one HALF-OPEN probe is in flight, so concurrent calls do not
        #: all become probes and all pay the timeout the breaker exists to avoid.
        #: D104 says "a single HALF-OPEN probe"; this is that word.
        self._probing = False

    @property
    def state(self) -> BreakerState:
        if self._opened_until is None:
            return BreakerState.CLOSED
        if self._clock() < self._opened_until:
            return BreakerState.OPEN
        return BreakerState.HALF_OPEN

    @property
    def failures(self) -> int:
        """Consecutive failures. A success resets it, so it measures an outage."""
        return self._failures

    def admit(self) -> Admitted | Refused:
        """Whether a call may be made now. HALF-OPEN admits exactly one, as the probe."""
        state = self.state
        if state is BreakerState.OPEN:
            return Refused(
                f"circuit open for another {self._remaining():.1f}s",
                retry_after=self._remaining(),
            )
        if state is BreakerState.HALF_OPEN:
            if self._probing:
                return Refused("circuit half-open, probe in flight")
            self._probing = True
            return Admitted(probe=True)
        return Admitted(probe=False)

    def succeeded(self) -> None:
        self._failures = 0
        self._opened_until = None
        self._probing = False

    def failed(self, retry_after: float | None) -> None:
        was_probing = self._probing
        self._probing = False
        self._failures += 1
        # A failed probe re-opens at once, whatever the count: the half-open call
        # exists precisely to answer "is it back", and it said no. An explicit
        # `Retry-After` also opens at once: waiting for the threshold would spend
        # calls inside a window the callee had already declared (D103).
        if was_probing or retry_after is not None or self._failures >= self._threshold:
            window = self._cooldown if retry_after is None else retry_after
            self._opened_until = self._clock() + window

    def abandoned(self, admission: Admitted) -> None:
        """An admitted call ended with no verdict: release the probe it may hold (D122)."""
        if admission.probe:
            self._probing = False

    def _remaining(self) -> float:
        if self._opened_until is None:
            return 0.0
        return max(0.0, self._opened_until - self._clock())
