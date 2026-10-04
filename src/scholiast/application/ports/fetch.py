"""Fetching a paper's PDF from links a third party supplied (D115, D129).

The caller hands over a paper's open-access links, best first, and gets one PDF or
one failure. Which link worked, which were refused and why, is the adapter's to
decide and to report; the caller decides only what a failure means for the paper,
and the failure carries a class from the ingestion taxonomy so it can.

**One failure class for all the links**, by what the user can do next:

  transient           some link failed in a way a retry may fix (5xx, 429, timeout),
                      so the job is retried
  too_large           a link served a PDF over MAX_PDF_BYTES, so retrying cannot help
                      and uploading cannot either
  source_unavailable  every link was refused, missing or not a PDF; the user is
                      told to upload it
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from scholiast.domain.errors import FailureClass


@dataclass(frozen=True, slots=True)
class FetchAttempt:
    """What happened to one link: for the log, never for a client."""

    url: str
    #: `fetched`, or why not: `refused:<rule>`, `status:<code>`, `not_pdf`,
    #: `too_large`, `encoded`, `timeout`, `error`, `too_many_redirects`.
    outcome: str
    #: Redirects followed before the outcome.
    redirects: int = 0


@dataclass(frozen=True, slots=True)
class FetchedPdf:
    data: bytes
    #: The link that produced it, as given, and the URL that finally served it.
    link: str
    final_url: str
    attempts: tuple[FetchAttempt, ...]


class FetchFailed(Exception):
    """No link produced a PDF."""

    def __init__(self, failure_class: FailureClass, attempts: tuple[FetchAttempt, ...]) -> None:
        super().__init__(
            f"{failure_class}: " + "; ".join(f"{a.url} -> {a.outcome}" for a in attempts)
        )
        self.failure_class = failure_class
        self.attempts = attempts


@runtime_checkable
class PdfFetcher(Protocol):
    async def fetch(self, links: Sequence[str]) -> FetchedPdf:
        """The first link, in order, that yields a PDF. Raises `FetchFailed` otherwise."""
        ...
