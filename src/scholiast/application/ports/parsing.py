"""Document parsing port.

The ingestion use case needs a parser and must not know it is GROBID — that is
the layering claim `import-linter` checks. This is the interface that makes
"swap the parser" an adapter change.

`ParsedDocument` is the boundary DTO. It carries the sections the chunker
consumes, the bibliographic metadata that feeds paper identity, and the raw TEI
retained for reproducible re-chunking (D43).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from scholiast.domain.chunking import ParsedSection
from scholiast.domain.errors import FailureClass


class DocumentUnparseable(Exception):
    """The PDF cannot yield usable structure, and retrying will not help.

    Carries a `FailureClass` rather than a message because two decisions hang off
    it — whether to retry, and what the user is told (FR-I8, FR-I9). A bare
    exception collapses both into "something went wrong".
    """

    def __init__(self, failure_class: FailureClass, detail: str) -> None:
        super().__init__(f"{failure_class.value}: {detail}")
        self.failure_class = failure_class
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Everything one parse produced."""

    sections: tuple[ParsedSection, ...]
    title: str | None = None
    authors: tuple[str, ...] = ()
    abstract: str | None = None
    page_count: int = 0
    #: Stamped onto every chunk's provenance contract (FR-I5): offsets are only
    #: meaningful together with the parser version that produced them.
    parser_version: str = ""
    #: Raw TEI, retained so re-chunking ablations are reproducible across a
    #: GROBID version change rather than silently re-baselined (D43).
    raw_tei: str | None = None
    #: Counters worth reporting rather than discarding — a paper that parsed
    #: with 2 sections is technically a success and practically a failure.
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def has_body(self) -> bool:
        return any(not s.is_empty for s in self.sections)


@runtime_checkable
class DocumentParser(Protocol):
    """PDF bytes in, structured document out."""

    @property
    def version(self) -> str:
        """Stable identifier stamped onto the paper (NFR-M5)."""
        ...

    async def parse(self, pdf: bytes, *, filename: str = "") -> ParsedDocument:
        """Extract structure, or raise `DocumentUnparseable` with a class.

        Implementations must classify failure rather than raising something
        generic: the retry decision and the user-facing message both depend on
        which of §4.4.1's classes applies.
        """
        ...
