"""Papers in a workspace: upload, list, remove, read (FR-I2, FR-W2, NFR-U3).

The upload endpoint is where §5.4.2's synchronous half ends. It returns 202 with
a job id rather than blocking, because parsing and embedding take seconds and
NFR-P6 budgets ninety of them — a user holding an HTTP connection open for that
is a worse design than a status endpoint they can poll.

**Every handler here reaches the workspace through the scoped repository.** Not
through a session lookup by id: `SqlWorkspaceRepository.get` applies the tenant
predicate and returns None for a workspace someone else owns, which is what
turns FR-A2's 404-not-403 rule into something a route cannot forget.
"""

from __future__ import annotations

import uuid
from http import HTTPStatus

from fastapi import APIRouter, File, UploadFile
from sqlalchemy import select

from scholiast.api import schemas
from scholiast.api.deps import (
    AddPaperDep,
    PapersDep,
    RemovePaperDep,
    SectionsDep,
    SessionDep,
    WorkspacesDep,
)
from scholiast.domain.errors import FailureClass, WorkspaceNotAvailable
from scholiast.domain.upload import MAX_PDF_BYTES
from scholiast.domain.workspace import PaperState, Workspace
from scholiast.infrastructure.db import models

router = APIRouter(prefix="/v1/workspaces/{workspace_id}/papers", tags=["papers"])
jobs_router = APIRouter(prefix="/v1/jobs", tags=["papers"])


async def _scoped_workspace(workspace_id: str, repo: WorkspacesDep) -> Workspace:
    """The workspace, or 404 — through the tenant predicate, never around it."""
    workspace = await repo.get(workspace_id)
    if workspace is None:
        raise WorkspaceNotAvailable(workspace_id)
    return workspace


@router.post("/upload", status_code=HTTPStatus.ACCEPTED, response_model=schemas.UploadAccepted)
async def upload_paper(
    workspace_id: str, use_case: AddPaperDep, file: UploadFile = File(...)
) -> schemas.UploadAccepted:
    """Accept a PDF and queue it (FR-I2, §5.4.2).

    The body is read fully into memory before validation, which is safe only
    because of the cap: `validate_upload` refuses anything over 50 MB (NFR-P8),
    and the read is bounded by the same number plus one — enough to know the
    limit was exceeded without buffering an unbounded body to find out.
    """
    data = await file.read(MAX_PDF_BYTES + 1)
    result = await use_case.execute(
        workspace_id=workspace_id, data=data, filename=file.filename or ""
    )
    return schemas.UploadAccepted(
        paper_id=result.paper_id,
        job_id=result.job_id,
        state=result.state.value,
        deduplicated=result.deduplicated,
    )


@router.get("", response_model=list[schemas.PaperSummary])
async def list_papers(
    workspace_id: str, repo: WorkspacesDep, session: SessionDep
) -> list[schemas.PaperSummary]:
    """Members with their lifecycle state — the workspace view (FR-W4's M1 subset).

    `user_message` is what NFR-U2 asks for: a failed paper tells the user what
    they can *do* about it, not which pipeline stage raised.
    """
    workspace = await _scoped_workspace(workspace_id, repo)
    live = [m for m in workspace.memberships if m.state is not PaperState.REMOVED]
    if not live:
        return []

    rows = (
        await session.execute(
            select(models.Paper.id, models.Paper.title).where(
                models.Paper.id.in_([uuid.UUID(m.paper_id) for m in live])
            )
        )
    ).all()
    titles: dict[uuid.UUID, str] = {row[0]: row[1] for row in rows}
    return [
        schemas.PaperSummary(
            paper_id=m.paper_id,
            title=titles.get(uuid.UUID(m.paper_id), "Untitled"),
            state=m.state.value,
            failure_class=m.failure_class,
            failure_detail=m.failure_detail,
            user_message=(FailureClass(m.failure_class).user_message if m.failure_class else None),
        )
        for m in live
    ]


@router.delete("/{paper_id}", response_model=schemas.RemovalResult)
async def remove_paper(
    workspace_id: str, paper_id: str, use_case: RemovePaperDep
) -> schemas.RemovalResult:
    """Remove a paper, and its derived data if nothing else holds it (FR-W2, D62).

    The use case resolves the workspace through its own scoped repository, so a
    caller cannot delete out of someone else's workspace by guessing two ids.
    """
    result = await use_case.execute(workspace_id=workspace_id, paper_id=paper_id)
    return schemas.RemovalResult(
        paper_id=result.paper_id,
        removed=result.removed,
        purged=result.purged,
        vectors_deleted=result.vectors_deleted,
    )


@router.get("/{paper_id}/content", response_model=schemas.PaperContent)
async def paper_content(
    workspace_id: str,
    paper_id: str,
    repo: WorkspacesDep,
    papers: PapersDep,
    sections: SectionsDep,
) -> schemas.PaperContent:
    """The citation's landing target (NFR-U3, D42).

    Returns `sections.text` — the exact string every chunk's `char_start`/
    `char_end` indexes — so the reader view highlights the cited span rather
    than approximating it. This is the "one join plus a slice" D42 chose
    section-relative offsets to make possible.

    Two scoped reads, not one: the workspace must be the caller's, *and* the
    paper must be reachable through a membership. A paper id from another
    workspace of the same tenant would pass the first check and fail the second.
    """
    await _scoped_workspace(workspace_id, repo)
    paper = await papers.get(paper_id)
    if paper is None:
        raise WorkspaceNotAvailable(paper_id)

    return schemas.PaperContent(
        paper_id=paper.id,
        title=paper.title,
        sections=[
            schemas.SectionOut(
                id=s.id,
                path=s.path,
                raw_title=s.raw_title,
                kind=s.kind.value,
                page_start=s.page_start,
                page_end=s.page_end,
                text=s.text,
            )
            for s in await sections.list_for_paper(paper_id)
        ],
    )


@jobs_router.get("/{job_id}", response_model=schemas.JobStatus)
async def job_status(job_id: str, session: SessionDep, repo: WorkspacesDep) -> schemas.JobStatus:
    """Ingestion progress for one job (NFR-U1).

    Scoped by joining through the job's workspace: `ingestion_jobs` has no tenant
    column, so the predicate has to come from the workspace the job points at.
    Without it, a job id would disclose another tenant's ingestion state — and
    the id is a UUID, which is an obstacle rather than an authorisation check.
    """
    try:
        target = uuid.UUID(job_id)
    except ValueError as exc:
        raise WorkspaceNotAvailable(job_id) from exc

    row = (
        await session.execute(select(models.IngestionJob).where(models.IngestionJob.id == target))
    ).scalar_one_or_none()
    if row is None:
        raise WorkspaceNotAvailable(job_id)
    # The scoped read that authorises it. A job whose workspace this tenant
    # cannot reach must be indistinguishable from one that does not exist — so
    # the refusal is raised HERE, naming `job_id`, rather than delegated to
    # `_scoped_workspace`, which would name the job's own workspace instead.
    # That uuid is one the caller never supplied and could not otherwise learn,
    # and it would also make the two 404s tell each other apart (D93).
    if await repo.get(str(row.workspace_id)) is None:
        raise WorkspaceNotAvailable(job_id)

    return schemas.JobStatus(
        job_id=str(row.id),
        paper_id=str(row.paper_id),
        state=row.state,
        attempts=row.attempts,
        last_error=row.last_error,
    )
