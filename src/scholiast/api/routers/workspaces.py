"""Workspace CRUD (FR-W1, §5.5).

All four verbs, because FR-W1 names all four and it is in M1's requirement list.
`DELETE` is the one that looks incomplete and is not: §5.6.4 makes marking the
workspace `DELETING` the *only* synchronous step, with the cascade across three
stores deferred to a resumable job — which is FR-W3 and M3's. Returning 202 here
is the documented shape, not a stub.
"""

from __future__ import annotations

import uuid
from http import HTTPStatus

from fastapi import APIRouter, Response

from scholiast.api import schemas
from scholiast.api.deps import TenantDep, WorkspacesDep
from scholiast.domain.errors import WorkspaceNotAvailable
from scholiast.domain.workspace import Workspace

router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])


def _to_summary(workspace: Workspace) -> schemas.WorkspaceSummary:
    return schemas.WorkspaceSummary(
        id=workspace.id,
        name=workspace.name,
        state=workspace.state.value,
        paper_count=len(workspace.active_memberships),
        queryable_count=len(workspace.queryable_paper_ids),
        created_at=workspace.created_at,
    )


@router.get("", response_model=list[schemas.WorkspaceSummary])
async def list_workspaces(repo: WorkspacesDep) -> list[schemas.WorkspaceSummary]:
    return [_to_summary(w) for w in await repo.list_all()]


@router.post("", status_code=HTTPStatus.CREATED, response_model=schemas.WorkspaceSummary)
async def create_workspace(
    body: schemas.CreateWorkspace, repo: WorkspacesDep, tenant: TenantDep
) -> schemas.WorkspaceSummary:
    created = await repo.create(
        Workspace(id=str(uuid.uuid4()), owner_id=tenant.user_id, name=body.name)
    )
    return _to_summary(created)


@router.get("/{workspace_id}", response_model=schemas.WorkspaceSummary)
async def get_workspace(workspace_id: str, repo: WorkspacesDep) -> schemas.WorkspaceSummary:
    workspace = await repo.get(workspace_id)
    if workspace is None:
        # 404 whether it is missing or someone else's — the repository cannot
        # tell them apart, and FR-A2 says the API must not either.
        raise WorkspaceNotAvailable(workspace_id)
    return _to_summary(workspace)


@router.patch("/{workspace_id}", response_model=schemas.WorkspaceSummary)
async def rename_workspace(
    workspace_id: str, body: schemas.RenameWorkspace, repo: WorkspacesDep
) -> schemas.WorkspaceSummary:
    if not await repo.rename(workspace_id, body.name):
        raise WorkspaceNotAvailable(workspace_id)
    workspace = await repo.get(workspace_id)
    if workspace is None:  # pragma: no cover - renamed then vanished
        raise WorkspaceNotAvailable(workspace_id)
    return _to_summary(workspace)


@router.delete("/{workspace_id}", status_code=HTTPStatus.ACCEPTED)
async def delete_workspace(workspace_id: str, repo: WorkspacesDep) -> Response:
    """Mark the workspace DELETING (§5.6.4 step 1).

    202, not 204, and the difference is honest rather than pedantic: the rows in
    Postgres, the vectors in Qdrant and the blobs on disk are still there. What
    has happened is that the workspace stops serving reads immediately — every
    scoped query already excludes soft-deleted workspaces — and a sweeper will
    finish the job. That sweeper is FR-W3, which is M3's.
    """
    if not await repo.soft_delete(workspace_id):
        raise WorkspaceNotAvailable(workspace_id)
    return Response(status_code=HTTPStatus.ACCEPTED)
