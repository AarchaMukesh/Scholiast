"""Does the hybrid collection hold every embedded chunk? (D172)

A hybrid read against an incomplete collection fails silently: a missing point is
not an error, it is a chunk that can never be retrieved, and nothing downstream
can tell that from a chunk that was never relevant. So a hybrid read is refused
until this says the collection is complete, and the same check serves four
callers: API startup in hybrid mode, `scholiast-reindex --verify`, the
reconciliation sweep, and - in its cheap form - the readiness probe.

**Postgres is the source of truth** (NFR-R6). The chunks it records as embedded
are the ones that must be retrievable; a chunk row is stamped only after every
collection it is written to has taken it (D53), so each stamped id should have a
point. **By id, not by count:** one missing chunk and one orphaned point cancel
in a count, and the completeness that matters is per chunk.

**Only points carrying the current `LEXICAL_VERSION` count.** After a change to
the weighting, a point with the old weights is not a current point, so a hybrid
read waits until the reindex has rewritten every one.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholiast.application.ports.repositories import ChunkRepository
from scholiast.application.ports.vector import LexicalIndex
from scholiast.domain.lexical import LEXICAL_VERSION


@dataclass(frozen=True, slots=True)
class HybridCoverage:
    """What the hybrid collection holds against what Postgres says it must."""

    lexical_version: str
    #: Chunks Postgres records as embedded.
    expected: int
    #: Of those, how many have a current-version point.
    present: int
    #: The expected chunk ids without one, sorted.
    missing: tuple[str, ...]
    #: Current-version points no embedded chunk accounts for: orphans, or chunks
    #: written but not yet stamped. Reported, not a failure (the sweep removes the
    #: first; the second resolves when its job stamps the rows).
    unaccounted: int

    @property
    def complete(self) -> bool:
        return not self.missing


class HybridIndexIncomplete(Exception):
    """A hybrid read was asked of a collection that lacks embedded chunks."""

    def __init__(self, coverage: HybridCoverage) -> None:
        self.coverage = coverage
        shown = ", ".join(coverage.missing[:5])
        more = len(coverage.missing) - 5
        super().__init__(
            f"the hybrid collection has {coverage.present} of {coverage.expected} embedded "
            f"chunks at lexical version {coverage.lexical_version!r}; missing {shown}"
            + (f" and {more} more" if more > 0 else "")
            + ". Run `scholiast-reindex --hybrid`, then `--verify` (D172)"
        )


class VerifyHybridIndexUseCase:
    """Compare the hybrid collection with the embedded chunk rows. Reads only."""

    def __init__(
        self,
        *,
        chunks: ChunkRepository,
        lexical: LexicalIndex,
        lexical_version: str = LEXICAL_VERSION,
    ) -> None:
        self._chunks = chunks
        self._lexical = lexical
        self._lexical_version = lexical_version

    async def execute(self) -> HybridCoverage:
        """The id-level check: every embedded chunk, looked for by id."""
        expected = await self._chunks.embedded_chunk_ids()
        present = await self._lexical.lexical_chunk_ids(lexical_version=self._lexical_version)
        return HybridCoverage(
            lexical_version=self._lexical_version,
            expected=len(expected),
            present=len(expected & present),
            missing=tuple(sorted(expected - present)),
            unaccounted=len(present - expected),
        )

    async def require_complete(self) -> HybridCoverage:
        """`execute`, raising `HybridIndexIncomplete` unless nothing is missing."""
        coverage = await self.execute()
        if not coverage.complete:
            raise HybridIndexIncomplete(coverage)
        return coverage

    async def falls_short(self) -> bool:
        """The readiness probe's cheap form: fewer current points than embedded rows.

        Two counts rather than two enumerations. Points are written before rows
        are stamped, so an ingest in flight can only make points outnumber rows -
        never the reverse - and this never reports a false shortfall. It can miss
        one that an orphan elsewhere offsets; the id check at startup and in the
        sweep catches that.
        """
        rows = await self._chunks.count_embedded()
        points = await self._lexical.count_lexical(lexical_version=self._lexical_version)
        return points < rows
