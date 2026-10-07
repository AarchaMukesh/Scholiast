"""A fixed-window rate limiter, in process (NFR-SEC7, D112, D113).

**In-process for D104's reason, and the cost is not the same.** There, each
process learning independently was harmless because a breaker is a latency
optimisation. This is a security control, so N processes mean N times the limit:
the number is real today, with one API process, and weakens silently the moment
M6 adds a replica. D112 is where that is written down.

**Fixed window, and its flaw stated:** a caller can send up to twice the limit
across a window boundary. A token bucket smooths that at the cost of per-key
floats and timestamps, and the boundary burst is not what the limit exists for —
it exists so one caller cannot monopolise generation or grind login.

**The limiter must not become the denial of service it prevents.** The map is
keyed by caller and grows with distinct callers, so an anonymous flood of
addresses would otherwise exhaust memory. It is hard-capped with oldest-first
eviction. Eviction can forgive a limit, which is the wrong direction — but
reaching the cap costs the attacker far more requests than it forgives, and an
unbounded map is a hole anyone can drive.
At that volume the answer is network-level, which needs a deployment (M6).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Limit:
    """How many requests per window, for one named bucket."""

    requests: int
    window_seconds: float

    def __post_init__(self) -> None:
        if self.requests < 1 or self.window_seconds <= 0:
            # A limit of zero would refuse every request including the first,
            # which is an outage wearing a control's clothes.
            raise ValueError("a limit needs at least one request and a positive window")


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether the request may proceed, and when to come back if not."""

    allowed: bool
    #: Seconds until the current window ends. Sent as `Retry-After` on a 429,
    #: which is the same courtesy this system expects from Groq (D103) — a
    #: refusal that says when to return costs a client nothing to honour.
    retry_after: float = 0.0


#: Above this many tracked (caller, bucket) pairs, the oldest are evicted. Sized
#: so ordinary use never reaches it: at one entry per caller per bucket, ten
#: thousand is far more distinct callers than NFR-S4's twenty concurrent users.
MAX_TRACKED_KEYS = 10_000


class FixedWindowLimiter:
    """Counts requests per (caller, bucket) inside a wall-clock window."""

    def __init__(
        self,
        *,
        max_keys: int = MAX_TRACKED_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_keys = max(1, max_keys)
        #: Monotonic, for the reason the breaker uses it: a clock correction must
        #: not end a window early or extend one.
        self._clock = clock
        #: key -> (window_started_at, count). Insertion-ordered, which is what
        #: makes oldest-first eviction a `next(iter(...))` rather than a sort.
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, *, key: str, bucket: str, limit: Limit, cost: int = 1) -> Decision:
        """Record one request costing `cost` and say whether it is allowed.

        `cost` is how much of the allowance the request spends: one, except where
        a bucket counts what a request carries rather than the request (D133's add,
        which counts papers). A request that does not fit whole is refused whole.

        Counting and deciding are one operation on purpose. Splitting them into
        `allowed()` then `record()` would leave a window where two concurrent
        requests both read a count below the limit and both proceed — and this
        runs on an event loop where that interleaving is ordinary.
        """
        if cost < 1:
            raise ValueError(f"a request costs at least 1, got {cost}")
        now = self._clock()
        tracked = f"{bucket}:{key}"
        started, count = self._windows.get(tracked, (now, 0))
        if now - started >= limit.window_seconds:
            # The window has rolled. A new one starts at this request rather than
            # on a grid, so a caller's limit does not depend on when they arrived
            # relative to some epoch.
            #
            # **This is the only expiry mechanism, and it used to be dead.** An
            # eager `_prune` ran first and deleted every expired entry, so this
            # branch was unreachable — a mutation that disabled it broke no test,
            # which is how it was found. Worse, that prune applied the CURRENT
            # call's window to every OTHER bucket's entries, so a bucket with a
            # different window would have been expired against the wrong one.
            # Invisible today because all three windows are sixty seconds.
            #
            # Expiring only the key being used removes both problems, and memory
            # is bounded by the cap rather than by eager sweeping. The cost is
            # that idle entries linger until eviction: at most `max_keys` tuples,
            # which is nothing.
            started, count = now, 0

        if count + cost > limit.requests:
            # Not incremented. Counting refused requests would extend the
            # penalty every time a caller retried, which turns a rate limit into
            # an escalating ban nobody configured.
            self._windows[tracked] = (started, count)
            return Decision(
                allowed=False, retry_after=max(0.0, started + limit.window_seconds - now)
            )

        self._windows[tracked] = (started, count + cost)
        self._evict_if_needed()
        return Decision(allowed=True)

    def _evict_if_needed(self) -> None:
        while len(self._windows) > self._max_keys:
            # Insertion order, so this is the least recently started window.
            del self._windows[next(iter(self._windows))]

    @property
    def tracked(self) -> int:
        """How many windows are live. For the eviction test, and for phase 7's eyes."""
        return len(self._windows)
