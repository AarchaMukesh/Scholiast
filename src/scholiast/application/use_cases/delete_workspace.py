"""The workspace cascade — §5.6.4 step 2, FR-W3, D99.

Step 1 already happened somewhere else and is already durable: the API marked
the workspace `DELETING`, stamped `deleted_at` and returned 202, in one
transaction. From that moment every scoped repository, the queue and the
ingestion worker exclude it, so the workspace stops serving reads before any of
this runs. **That mark is the work item** (D99) — there is no deletion-job row,
because a marked workspace already records that work is outstanding, and a
second record of one fact can only disagree with the first.

**Everything here is idempotent, because it will be run again.** Not might:
`list_deleting` re-finds any workspace whose sweep did not reach the end, so the
normal case and the crash case run the same code. The steps are ordered so that
a crash between any two of them leaves the *recoverable* residue:

  memberships   dropped one paper at a time, so a crash leaves the rest to find
  derived data  vectors before rows (D53 inverted — see `PaperPurger`)
  the row       last, because it is what `list_deleting` finds the work by;
                deleting it first would strand every paper still attached

**Retry is unbounded, and `attempts` is absent rather than forgotten** (D99).
D48 bounds ingestion attempts because a broken paper should stop consuming the
queue; a deletion that gives up leaves residue in three stores permanently,
which is the one condition FR-W3 forbids. A transient Qdrant outage must not
become permanently leaked data.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholiast.application.ports.repositories import PaperRepository, WorkspaceRepository
from scholiast.application.ports.storage import ObjectStore
from scholiast.application.ports.vector import VectorIndex
from scholiast.application.use_cases.purge_paper import PaperPurger


@dataclass(frozen=True, slots=True)
class DeleteWorkspaceResult:
    """What the cascade did, in the terms FR-W3's acceptance test asks about."""

    workspace_id: str
    #: False when the workspace was not marked DELETING — already swept, or
    #: never deleted. A no-op, not an error: the sweep may legitimately race
    #: with the task the API dispatched.
    swept: bool
    memberships_dropped: int = 0
    papers_purged: int = 0
    vectors_deleted: int = 0
    blobs_deleted: int = 0


class DeleteWorkspaceUseCase:
    """Sweep one marked workspace out of Postgres, Qdrant and object storage."""

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
        self._purger = PaperPurger(papers=papers, index=index, store=store)

    async def execute(self, workspace_id: str) -> DeleteWorkspaceResult:
        """Run the cascade to completion, or leave it findable for the next pass.

        **Refuses anything not already marked**, and that is a safety check
        rather than an optimisation. This runs under `SystemContext`, so no
        tenant predicate constrains which workspace it can destroy, and its
        argument arrives over a broker — an id naming a live workspace would
        otherwise delete it, along with every paper only it held. Requiring the
        mark means the cascade can only finish work an authenticated owner
        already started (D99).
        """
        if not await self._workspaces.is_deleting(workspace_id):
            return DeleteWorkspaceResult(workspace_id=workspace_id, swept=False)

        paper_ids = await self._papers.membership_paper_ids(workspace_id)

        dropped = purged = vectors = blobs = 0
        for paper_id in paper_ids:
            if await self._papers.remove_from_workspace(workspace_id, paper_id):
                dropped += 1
            # The purge runs whether or not a membership was there. A False
            # means an earlier attempt or a concurrent sweep already dropped it
            # — and that attempt may have crashed between dropping the
            # membership and purging, which is the state this must converge
            # from rather than skip over.
            result = await self._purger.purge_if_unheld(paper_id)
            if result.purged:
                purged += 1
                vectors += result.vectors_deleted
                blobs += result.blobs_deleted

        # Last, and only now: this row is how the work was findable. The
        # database cascades from here to conversations, messages, the remaining
        # memberships and the ingestion jobs, so a paper whose membership the
        # loop above somehow missed still loses its membership row here — it
        # would then be an orphaned PAPER, which is D102's third sweep target.
        swept = await self._workspaces.purge(workspace_id)
        return DeleteWorkspaceResult(
            workspace_id=workspace_id,
            swept=swept,
            memberships_dropped=dropped,
            papers_purged=purged,
            vectors_deleted=vectors,
            blobs_deleted=blobs,
        )
