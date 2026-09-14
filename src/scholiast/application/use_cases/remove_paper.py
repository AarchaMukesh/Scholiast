"""Remove a paper from a workspace, leaving no orphans (FR-W2, D62).

FR-W2's stated test is "orphan-detection test after removal", so dropping the
membership is not the requirement — the requirement is that nothing survives it.

**Bounded, so it runs inside the request.** Removing one paper is work on a known
id with a known extent; deleting a *workspace* (FR-W3) is unbounded and has to be
resumable, which is what buys it §5.6.4's job and sweeper in M3. Splitting them
on that line is why this one needs no state machine.

**Shared papers are not deleted.** §4.4.2's registry means a public paper may be a
member of several workspaces, and AD-8 exists so it is parsed and embedded once.
So removal always drops the membership, and only cascades to derived data when no
workspace holds the paper any more.

**Vectors first, Postgres second** — the reverse of ingestion's D53 ordering, for
the same reason. A vector whose chunk row is gone is the orphan §5.6.4's sweep is
built to find and NFR-R6 declares rebuildable; a chunk row whose vector is gone is
silently unretrievable, with nothing anywhere reporting it.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholiast.application.ports.repositories import PaperRepository, WorkspaceRepository
from scholiast.application.ports.storage import ObjectStore, source_key, tei_key
from scholiast.application.ports.vector import VectorIndex
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
        self._index = index
        self._store = store

    async def execute(self, *, workspace_id: str, paper_id: str) -> RemovePaperResult:
        workspace = await self._workspaces.get(workspace_id)
        if workspace is None:
            raise WorkspaceNotAvailable(workspace_id)

        paper = await self._papers.get(paper_id)
        removed = await self._papers.remove_from_workspace(workspace_id, paper_id)
        if not removed:
            # Not a member, or already gone. Idempotent by design (NFR-R5): a
            # retried delete must not fail for having succeeded already.
            return RemovePaperResult(paper_id=paper_id, removed=False)

        if await self._papers.has_any_membership(paper_id):
            # Another workspace still holds it. Deleting the derived data here
            # would break that workspace's retrieval — which is the cost AD-8's
            # shared registry accepts in exchange for parsing once.
            return RemovePaperResult(paper_id=paper_id, removed=True)

        vectors = await self._index.delete_by_papers([paper_id])
        # Postgres second (D53's ordering, inverted for deletion). The paper row
        # cascades to sections and chunks by foreign key, so this is one delete.
        await self._papers.delete(paper_id)
        await self._purge_blobs(paper)
        return RemovePaperResult(
            paper_id=paper_id, removed=True, purged=True, vectors_deleted=vectors
        )

    async def _purge_blobs(self, paper: object) -> None:
        """Delete the stored PDF and TEI, if the paper row told us where they are.

        Best-effort by construction: `ObjectStore.delete` reports a missing key
        rather than raising, because a half-finished earlier attempt must not
        make the next one fail. A blob that outlives its row is exactly the
        residue §5.6.4's sweep is for.
        """
        paper_id = getattr(paper, "id", None)
        if not isinstance(paper_id, str):
            return
        await self._store.delete(source_key(paper_id))
        parser_version = getattr(paper, "parser_version", None)
        if isinstance(parser_version, str) and parser_version:
            await self._store.delete(tei_key(paper_id, parser_version))
