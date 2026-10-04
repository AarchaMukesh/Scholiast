"""`Retry-After` read as a number, for every adapter that talks to a rate-limited service (D103).

Moved out of the Groq adapter when discovery became the second caller (D119): a
429 or 503 from OpenAlex or arXiv says when to come back exactly as Groq's does,
and two copies of an RFC parser would drift apart in the edge cases that matter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx


def retry_after_seconds(response: httpx.Response) -> float | None:
    """`Retry-After` as seconds, or None when the header is absent or unusable.

    **Both legal forms are accepted** (RFC 9110 §10.2.3): Groq sends a number,
    but the spec also allows an HTTP-date, and a deployment behind a proxy that
    rewrites it would otherwise have the header silently ignored — a breaker
    falling back to its guessed cooldown while the server had stated the answer.

    None rather than zero for an unparseable value, because the two mean
    opposite things to the breaker: None is "nothing was said, use the
    cooldown", zero is "the server says come back now".

    Not clamped at the top. A daily quota legitimately answers with hours, and
    honouring that is the point (D103) — the window is in-process anyway (D104),
    so a restart clears it.
    """
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        # RFC 9110 dates are GMT; a naive one from a sloppy proxy is read as
        # UTC rather than as local time, which would be wrong by the offset.
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())
