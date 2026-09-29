"""The reconciliation sweep — §5.6.4 step 4, D100, D102.

**The safety net that makes AP-4 true rather than aspirational.** Everything
else in the deletion path is a best effort that can be interrupted; this is what
converges the three stores afterwards, and it is the reason the cascade is
allowed to be interruptible in the first place.

Four targets, in an order chosen so each one's output is the next one's input:

  1. **unfinished cascades** — workspaces still marked DELETING. Their sweep
     crashed, or its dispatch never reached the broker. Run first, because
     finishing them is what turns their papers into targets 2-4.
  2. **orphaned papers** — paper rows no workspace holds. Neither §5.6.4 nor
     FR-W3 names them, and D99's race is why they are here: two workspaces
     sharing a paper, both deleted, each seeing the other's membership still
     uncommitted, so both skip the purge. A paper with zero memberships is
     unreachable by every scoped query, so nothing *else* will ever find it —
     while it still holds sections, chunks, vectors and blobs. The largest
     residue and the only invisible one.
  3. **orphaned vectors** — points whose paper is gone from Postgres. §5.6.4's
     named target, and the one NFR-R6 rests on: the relational store is the
     source of truth and the index is derived, which is a claim only a sweep
     can enforce.
  4. **orphaned blobs** — object keys whose paper no longer exists. FR-W3 says
     "zero residue in every store"; §5.6.4 step 4 names only vectors. The test
     wins, because a sweeper that satisfies the document while leaking blobs
     passes the criterion it was written for and fails the requirement it was
     written from.

**The live set is read once, after step 1 and before steps 3-4.** Reading it
earlier would treat papers the cascades just deleted as live and leave their
vectors behind for a whole cycle; reading it per-target would let the two
answers disagree. A paper created *during* the sweep is simply not in the set —
and since the set is only used to decide what to delete, a missing entry can
only make the sweep delete something new. That is why the read happens after the
deletions rather than before: everything in the set at that moment has a
committed paper row, and anything newer is not yet referenced by a vector or a
blob this pass can see.

**Stated cost:** O(unfinished + orphans + index + blobs), with no paging and no
checkpoint. Sized for this project's corpus, not an arbitrary one. A production
version would page and resume; this one scans, and says so (D102).
"""

from __future__ import annotations

from dataclasses import dataclass

from scholiast.application.ports.repositories import PaperRepository, WorkspaceRepository
from scholiast.application.ports.storage import ObjectStore
from scholiast.application.ports.vector import VectorIndex
from scholiast.application.use_cases.delete_workspace import DeleteWorkspaceUseCase
from scholiast.application.use_cases.purge_paper import PaperPurger

#: Everything the object store holds for papers lives under this prefix (D49).
#: Anything outside it is not this sweep's business, which is what keeps a
#: mis-keyed write from being deleted as residue.
_PAPER_ROOT = "papers/"


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """Residue found and removed, reported per store.

    Counted rather than merely acted on, because a sweep that silently fixes
    things is indistinguishable from one that has nothing to fix — and those
    two states mean opposite things about the rest of the system. A non-zero
    count here is a defect somewhere else.
    """

    cascades_resumed: int = 0
    papers_purged: int = 0
    vectors_deleted: int = 0
    blobs_deleted: int = 0


class ReconcileUseCase:
    """Converge Postgres, Qdrant and object storage onto the relational truth."""

    def __init__(
        self,
        *,
        workspaces: WorkspaceRepository,
        papers: PaperRepository,
        index: VectorIndex,
        store: ObjectStore,
    ) -> None:
        self._workspaces = workspaces
        self._papers = papers
        self._index = index
        self._store = store
        self._purger = PaperPurger(papers=papers, index=index, store=store)
        self._cascade = DeleteWorkspaceUseCase(
            workspaces=workspaces, papers=papers, index=index, store=store
        )

    async def execute(self) -> ReconcileResult:
        resumed, from_cascades = await self._resume_cascades()
        purged, from_orphans = await self._purge_orphaned_papers()

        # After the deletions above, so their papers are already gone from it.
        live = await self._papers.all_paper_ids()
        vectors = await self._index.delete_orphans(live)
        blobs = await self._delete_orphaned_blobs(live)

        return ReconcileResult(
            cascades_resumed=resumed,
            papers_purged=purged,
            # A purge's own deletions are counted here too: the point of the
            # number is total residue removed by this pass, not a breakdown of
            # which internal step removed it.
            vectors_deleted=from_cascades[0] + from_orphans[0] + vectors,
            blobs_deleted=from_cascades[1] + from_orphans[1] + blobs,
        )

    async def _resume_cascades(self) -> tuple[int, tuple[int, int]]:
        """Finish every workspace whose sweep did not reach the end (D99)."""
        resumed = 0
        vectors = blobs = 0
        for workspace_id in await self._workspaces.list_deleting():
            result = await self._cascade.execute(workspace_id)
            if result.swept:
                resumed += 1
            vectors += result.vectors_deleted
            blobs += result.blobs_deleted
        return resumed, (vectors, blobs)

    async def _purge_orphaned_papers(self) -> tuple[int, tuple[int, int]]:
        """Delete paper rows no workspace holds, and their derived data."""
        purged = 0
        vectors = blobs = 0
        for paper_id in await self._papers.orphaned_paper_ids():
            result = await self._purger.purge_if_unheld(paper_id)
            if result.purged:
                purged += 1
                vectors += result.vectors_deleted
                blobs += result.blobs_deleted
        return purged, (vectors, blobs)

    async def _delete_orphaned_blobs(self, live_paper_ids: tuple[str, ...]) -> int:
        """Remove object keys whose paper id is not in Postgres (D102).

        The paper id is a path segment (D49), so the test is a string compare
        against a set — nothing is opened, read or hashed. A key that does not
        parse as `papers/<id>/...` is left alone rather than deleted: this sweep
        removes residue it can *prove* is residue, and an unrecognised key is
        evidence of something it does not model.
        """
        live = set(live_paper_ids)
        deleted = 0
        for key in await self._store.list_keys(_PAPER_ROOT):
            paper_id = _paper_id_of(key)
            if paper_id is None or paper_id in live:
                continue
            if await self._store.delete(key):
                deleted += 1
        return deleted


def _paper_id_of(key: str) -> str | None:
    """The paper id in `papers/<id>/...`, or None if the key is not that shape."""
    parts = key.split("/")
    if len(parts) < 3 or parts[0] != "papers" or not parts[1]:
        return None
    return parts[1]
