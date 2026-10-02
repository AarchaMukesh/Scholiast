"""Remove a paper from a workspace, leaving no orphans (FR-W2, D62).

FR-W2's stated test is "orphan-detection test after removal", so dropping the
membership is not the requirement — the requirement is that nothing survives it.

**Bounded, so it runs inside the request.** Removing one paper is work on a known
id with a known extent; deleting a *workspace* (FR-W3) is unbounded and has to be
resumable, which is what buys it §5.6.4's sweep in M3. Splitting them on that
line is why this one needs no state machine.

**Shared papers are not deleted.** §4.4.2's registry means a public paper may be a
member of several workspaces, and AD-8 exists so it is parsed and embedded once.
So removal always drops the membership, and only cascades to derived data when no
workspace holds the paper any more — which is `PaperPurger`'s job, shared with
the workspace cascade so the two cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholiast.application.ports.repositories import PaperRepository, WorkspaceRepository
from scholiast.application.ports.storage import ObjectStore
from scholiast.application.ports.vector import VectorIndex
from scholiast.application.use_cases.purge_paper import PaperPurger
from scholiast.domain.errors import WorkspaceNotAvailable


@dataclass(frozen=True, slots=True)
class RemovePaperResult:
    """What removal actually did, so the caller can say so."""

    paper_id: str
    #: False when the paper was not a member — removal is idempotent, and a
    #: caller retrying a delete should not get an error for succeeding twice.
    removed: bool
    #: True when no workspace held the paper afterwards and its derived data was
    #: deleted. False means another workspace still has it (AD-8).
    purged: bool = False
    vectors_deleted: int = 0


class RemovePaperUseCase:
    """Drop a membership, and the paper's derived data if nothing else holds it."""

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

    async def execute(self, *, workspace_id: str, paper_id: str) -> RemovePaperResult:
        workspace = await self._workspaces.get(workspace_id)
        if workspace is None:
            raise WorkspaceNotAvailable(workspace_id)

        removed = await self._papers.remove_from_workspace(workspace_id, paper_id)
        if not removed:
            # Not a member, or already gone. Idempotent by design (NFR-R5): a
            # retried delete must not fail for having succeeded already.
            return RemovePaperResult(paper_id=paper_id, removed=False)

        result = await self._purger.purge_if_unheld(paper_id)
        return RemovePaperResult(
            paper_id=paper_id,
            removed=True,
            purged=result.purged,
            vectors_deleted=result.vectors_deleted,
        )
