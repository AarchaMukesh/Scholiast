"""Deleting one paper's derived data, once nothing holds it (FR-W2, FR-W3, D62).

Extracted from `remove_paper` so that its two callers cannot drift apart.
`RemovePaperUseCase` runs this once, inside a request, for a paper the user
removed; `DeleteWorkspaceUseCase` runs it in a loop for every paper a deleted
workspace held. D62 split those two operations because their *bounds* differ —
one is work on a known id, the other is unbounded and resumable — but the purge
itself is a single sequence, and its ordering argument is only reliably true if
it is written once.

**Vectors first, Postgres second** — the reverse of ingestion's D53 ordering,
for the same reason. A vector whose chunk row is gone is the orphan §5.6.4's
sweep is built to find and NFR-R6 declares rebuildable; a chunk row whose vector
is gone is silently unretrievable, with nothing anywhere reporting it. Given a
crash between the two steps, this ordering leaves the recoverable residue.

**Idempotent at every step**, because a restarted cascade re-runs the steps its
previous attempt finished (§5.6.4 step 3): the membership check reads what is
there now, the vector delete is a filter rather than a list of ids, the row
delete reports rather than raises, and the blob delete is by prefix over
whatever remains.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholiast.application.ports.repositories import PaperRepository
from scholiast.application.ports.storage import ObjectStore, paper_prefix
from scholiast.application.ports.vector import VectorIndex


@dataclass(frozen=True, slots=True)
class PurgeResult:
    """What the purge actually did, so a caller can report rather than assume."""

    #: False when another workspace still holds the paper (AD-8), which is a
    #: successful outcome and not a failure to delete.
    purged: bool
    vectors_deleted: int = 0
    blobs_deleted: int = 0


class PaperPurger:
    """Delete a paper's vectors, rows and blobs — but only if nothing holds it."""

    def __init__(
        self,
        *,
        papers: PaperRepository,
        index: VectorIndex,
        store: ObjectStore,
    ) -> None:
        self._papers = papers
        self._index = index
        self._store = store

    async def purge_if_unheld(self, paper_id: str) -> PurgeResult:
        """Purge the paper, unless some workspace still has it.

        The membership check is deliberately the first thing and is deliberately
        unscoped (see `PaperRepository.has_any_membership`): §4.4.2's registry
        shares a public paper across tenants, so a scoped answer would say "no
        memberships" whenever the *caller's* workspaces had none — and the
        caller would delete a paper another tenant is still reading from.
        """
        if await self._papers.has_any_membership(paper_id):
            return PurgeResult(purged=False)

        vectors = await self._index.delete_by_papers([paper_id])
        # The paper row cascades to its sections and chunks by foreign key, so
        # this is one statement and there is no window where sections outlive
        # the paper they describe.
        await self._papers.delete(paper_id)
        blobs = await self._purge_blobs(paper_id)
        return PurgeResult(purged=True, vectors_deleted=vectors, blobs_deleted=blobs)

    async def _purge_blobs(self, paper_id: str) -> int:
        """Delete everything under the paper's prefix (D49, D102).

        **By prefix, not by reconstructing the keys.** M1 deleted `source_key`
        and the `tei_key` for the parser version the row recorded, which leaks
        the TEI of every *other* parser generation — and D49 keeps those on
        purpose, so the leak is by design rather than by accident. Listing the
        prefix deletes what is actually there, including anything written by a
        version of this code that predates or postdates it.

        Best-effort by construction: `ObjectStore.delete` reports a missing key
        rather than raising, because a half-finished earlier attempt must not
        make the next one fail. A blob that outlives its row is exactly the
        residue the reconciliation sweep is for.
        """
        deleted = 0
        for key in await self._store.list_keys(paper_prefix(paper_id)):
            if await self._store.delete(key):
                deleted += 1
        return deleted
