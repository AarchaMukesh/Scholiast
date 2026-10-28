"""D179's Rule 1 (`refusal/1.0.0`): whether an answer refused, refused in part, or answered.

FR-Q4's measurement after D178 adopted no tau: production refuses only when the
model does. D179 fixed the labels: refused (the answer states the evidence does
not answer and makes no non-trivial claim), partial (it states that and still
makes one) or answered; D60's structural refusal counts as refused.

The rule reads Rule 2's sentences rather than the raw text, so "states the
evidence does not answer" and "non-trivial claim" mean exactly what they mean for
groundedness: a refusal statement is any sentence with a clause that refuses
(`Claim.refuses`), and a non-trivial claim is any sentence of kind CLAIM.

It is built on the 167 train answers alone and frozen by hash with Rule 2 before
any eval answer exists. Bar 1 is judged on the owner's labels, not on this rule;
its eval numbers are quoted only if its kappa against those labels is at least
0.80 (D179).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum, unique

from scholiast.domain.claims import Claim, ClaimKind

REFUSAL_VERSION = "refusal/1.0.0"


@unique
class Refusal(StrEnum):
    """D179's three labels for one answer."""

    REFUSED = "refused"
    PARTIAL = "partial"
    ANSWERED = "answered"


def label_answer(claims: Sequence[Claim], *, structural: bool) -> Refusal:
    """Rule 1 over Rule 2's sentences; `structural` is D60's no-evidence answer."""
    if structural:
        return Refusal.REFUSED
    if not any(claim.refuses for claim in claims):
        return Refusal.ANSWERED
    if any(claim.kind is ClaimKind.CLAIM for claim in claims):
        return Refusal.PARTIAL
    return Refusal.REFUSED
