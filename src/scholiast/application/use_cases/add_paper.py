"""Accept an upload and queue it for ingestion (§5.4.2, FR-I2, FR-I7).

The synchronous half of ingestion: everything that must happen while the user is
waiting, and nothing that cannot. It ends with a committed job row, which under
D47's Postgres queue *is* the enqueue — the paper, the membership and the job
land in one transaction, so §5.4.2's durability rule has no window left to lose
work in.

Order matters and is not arbitrary:

  validate   before anything is written. A 60 MB file or a renamed .zip should
             cost the user one response, not a stored blob and a queued job that
             fails a minute later (D55).

  dedup      before the blob is stored. FR-I7 says re-adding a paper never
             re-parses or re-embeds, and §5.4.2 makes that "create membership
             only". A paper already parsed *and* embedded joins the workspace
             INDEXED with no job at all.

  blob       before the rows. The worker reads the PDF by key, so a job whose
             blob does not exist yet is a race it loses. The reverse ordering
             costs a leak instead, and a leak is the cheaper failure.

**The leak is real and is not self-healing.** If the transaction rolls back
after `store.put`, the blob survives with no row pointing at it — and the next
attempt does *not* overwrite it, because the key is `papers/{paper_id}/…` and
`paper_id` is a fresh UUID minted per call (D49). A retry writes a different
key. This is the deliberate cost of paper-scoped keys: content-addressed keys
would make the retry converge, but they would also give two tenants' identical
private uploads one shared blob, which is the §4.4.2 carve-out defeated in the
one case it exists for. Keeping the key and accepting the orphan is the right
trade; pretending the orphan cleans itself up is not.

Cleaning it up belongs to §5.6.4's reconciliation sweep, which already owns the
same shape of problem for vectors: derived storage holding entries the source of
truth has no row for. **Recorded as debt** — the sweep gains an object-store
pass that deletes blobs under `papers/{id}/` for which no `papers` row exists.
Bounded and rare in the meantime: one blob per upload whose transaction failed.

This use case names no vendor. It is given ports, and `import-linter` is what
keeps that true.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from scholiast.application.ports.queue import Queue
from scholiast.application.ports.repositories import (
    ChunkRepository,
    PaperRepository,
    WorkspaceRepository,
)
from scholiast.application.ports.storage import ObjectStore, source_key
from scholiast.domain.errors import WorkspaceNotAvailable
from scholiast.domain.paper import Paper, PaperVisibility
from scholiast.domain.upload import provisional_title, validate_upload
from scholiast.domain.workspace import PaperState, Workspace


@dataclass(frozen=True, slots=True)
class AddPaperResult:
    """What the API returns as `202` (or `200` when nothing had to be done)."""

    paper_id: str
    state: PaperState
    #: None exactly when the paper was already ingested and no work was queued.
    job_id: str | None
    #: True when an existing paper was reused (FR-I7), for the response body and
    #: for the metric that makes the dedup rate observable rather than assumed.
    deduplicated: bool


class AddPaperUseCase:
    """Upload a PDF into a workspace."""

    def __init__(
        self,
        *,
        workspaces: WorkspaceRepository,
        papers: PaperRepository,
        chunks: ChunkRepository,
        store: ObjectStore,
        queue: Queue,
        owner_id: str,
    ) -> None:
        self._workspaces = workspaces
        self._papers = papers
        self._chunks = chunks
        self._store = store
        self._queue = queue
        self._owner_id = owner_id

    async def execute(
        self, *, workspace_id: str, data: bytes, filename: str = ""
    ) -> AddPaperResult:
        validate_upload(data, filename=filename)

        workspace = await self._workspaces.get(workspace_id)
        if workspace is None:
            # Indistinguishable from "does not exist" on purpose — an
            # out-of-scope id must not confirm the workspace is real (FR-A2).
            raise WorkspaceNotAvailable(workspace_id)

        content_hash = hashlib.sha256(data).hexdigest()
        existing = await self._papers.find_by_content_hash(content_hash)
        if existing is not None:
            return await self._attach(workspace, existing)

        paper = Paper(
            id=str(uuid.uuid4()),
            content_hash=content_hash,
            # An upload is never globally shared, whatever its hash collides
            # with: §4.4.2's carve-out exists so an unpublished manuscript does
            # not become a registry entry another tenant can reach.
            visibility=PaperVisibility.PRIVATE,
            owner_id=self._owner_id,
            title=provisional_title(filename),
        )
        await self._store.put(source_key(paper.id), data, content_type="application/pdf")
        stored = await self._papers.add(paper)

        # The domain decides whether this membership may exist and what state it
        # starts in; the repository only writes what it was told.
        membership = workspace.add_paper(stored.id, now=datetime.now(UTC))
        await self._papers.add_to_workspace(workspace_id, stored.id, membership.state)
        job_id = await self._queue.enqueue(workspace_id=workspace_id, paper_id=stored.id)

        return AddPaperResult(
            paper_id=stored.id, state=membership.state, job_id=job_id, deduplicated=False
        )

    async def _attach(self, workspace: Workspace, paper: Paper) -> AddPaperResult:
        """Add an already-known paper to this workspace (FR-I7, §5.4.2).

        Whether work is needed is decided by evidence, not by the parse status
        alone: `parse_status = parsed` says the structure was extracted, and says
        nothing about whether vectors exist. A paper whose chunks are stamped
        with an embedder version has been through the whole pipeline, so the
        membership starts INDEXED and no job is queued; anything else — parsed
        but never embedded, failed, or still in flight — is queued and the
        idempotent pipeline converges on the same rows.
        """
        member = next(
            (
                m
                for m in workspace.memberships
                if m.paper_id == paper.id and m.state is not PaperState.REMOVED
            ),
            None,
        )
        if member is not None:
            # Already here. Re-uploading must not reset a paper mid-pipeline or
            # re-queue one that is finished (FR-I7).
            return AddPaperResult(
                paper_id=paper.id, state=member.state, job_id=None, deduplicated=True
            )

        indexed = any(chunk.is_indexed for chunk in await self._chunks.list_for_paper(paper.id))
        membership = workspace.add_paper(
            paper.id,
            now=datetime.now(UTC),
            state=PaperState.INDEXED if indexed else PaperState.QUEUED,
        )
        await self._papers.add_to_workspace(workspace.id, paper.id, membership.state)

        job_id: str | None = None
        if membership.state is PaperState.QUEUED:
            job_id = await self._queue.enqueue(workspace_id=workspace.id, paper_id=paper.id)
        return AddPaperResult(
            paper_id=paper.id, state=membership.state, job_id=job_id, deduplicated=True
        )
