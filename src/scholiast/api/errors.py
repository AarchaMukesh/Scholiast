"""RFC 9457 problem documents (§5.5, NFR-U2, D65).

Every failure the API can produce leaves as `application/problem+json` with a
stable `type`, because §5.5 asks for "machine-readable errors [that] enable
NFR-U2's actionable messages". A client that has to regex a human sentence to
decide what happened does not have a machine-readable error.

**Types are relative URIs.** RFC 9457 §3.1.1 permits them, they are greppable,
and they commit the project to no domain it does not own — an absolute
`https://…` type that 404s is worse than a relative one that resolves against
whatever host serves the API.

**The mapping lives here, once.** Scattering `raise HTTPException(404)` through
routes would mean the same domain error becoming different statuses depending on
which handler caught it, and the 404-not-403 rule below is exactly the kind of
thing that erodes that way.

**404, never 403, for a workspace another tenant owns.** FR-A2 requires it: a
403 confirms the id is real, which is the existence leak §5.8's layer 1 exists to
prevent. `WorkspaceNotAvailable` deliberately cannot distinguish the two cases,
so this handler could not leak it even if it wanted to.
"""

from __future__ import annotations

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from scholiast.domain.errors import (
    DomainError,
    FailureClass,
    IllegalStateTransition,
    InsufficientPapersForComparison,
    InvariantViolation,
    WorkspaceCapacityExceeded,
    WorkspaceNotAvailable,
    WorkspaceNotQueryable,
)
from scholiast.domain.upload import UploadRejected

PROBLEM_CONTENT_TYPE = "application/problem+json"

#: The taxonomy. One slug per failure a client can meaningfully branch on.
TYPE_PREFIX = "/problems"


def problem(
    *,
    status: int,
    slug: str,
    title: str,
    detail: str,
    instance: str | None = None,
    **extra: object,
) -> JSONResponse:
    """One RFC 9457 document.

    `detail` is the human-readable, per-occurrence explanation; `title` is the
    per-type summary. Keeping them distinct is what lets a UI show a stable
    heading with a specific body, which is NFR-U2's "actionable and specific"
    without inventing a message per call site.
    """
    body: dict[str, object] = {
        "type": f"{TYPE_PREFIX}/{slug}",
        "title": title,
        "status": status,
        "detail": detail,
    }
    if instance is not None:
        body["instance"] = instance
    body.update(extra)
    return JSONResponse(status_code=status, content=body, media_type=PROBLEM_CONTENT_TYPE)


#: Upload failures carry a `FailureClass`, and the class decides the status: a
#: file that is too big is a 413, a file that is not a PDF is a 415. Mapping them
#: both to 400 would throw away the only thing the client could act on.
_UPLOAD_STATUS: dict[FailureClass, int] = {
    FailureClass.TOO_LARGE: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    FailureClass.UNPARSEABLE: HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
    FailureClass.SCANNED_IMAGE: HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
}


def register_error_handlers(app: FastAPI) -> None:
    """Wire every domain error to its problem document."""

    @app.exception_handler(WorkspaceNotAvailable)
    async def _not_found(request: Request, exc: WorkspaceNotAvailable) -> JSONResponse:
        # 404, never 403 (FR-A2). "Belongs to someone else" and "does not exist"
        # must be indistinguishable, or the API confirms which ids are real.
        return problem(
            status=HTTPStatus.NOT_FOUND,
            slug="workspace-not-found",
            title="Workspace not found",
            detail=str(exc),
            instance=request.url.path,
        )

    @app.exception_handler(WorkspaceNotQueryable)
    async def _not_queryable(request: Request, exc: WorkspaceNotQueryable) -> JSONResponse:
        # 409 rather than 400: the request is well-formed and will succeed once
        # ingestion finishes, so the client should retry rather than rewrite it.
        return problem(
            status=HTTPStatus.CONFLICT,
            slug="workspace-not-queryable",
            title="No papers are ready to query yet",
            detail=str(exc),
            instance=request.url.path,
            pending=exc.pending,
        )

    @app.exception_handler(WorkspaceCapacityExceeded)
    async def _at_capacity(request: Request, exc: WorkspaceCapacityExceeded) -> JSONResponse:
        return problem(
            status=HTTPStatus.CONFLICT,
            slug="workspace-at-capacity",
            title="Workspace is full",
            detail=str(exc),
            instance=request.url.path,
            cap=exc.cap,
        )

    @app.exception_handler(UploadRejected)
    async def _bad_upload(request: Request, exc: UploadRejected) -> JSONResponse:
        status = _UPLOAD_STATUS.get(exc.failure_class, HTTPStatus.BAD_REQUEST)
        return problem(
            status=status,
            slug="upload-rejected",
            title="Upload rejected",
            # The user-facing half of the taxonomy, not the log half: §4.4.1's
            # message says what to do, while `exc.detail` says what went wrong
            # internally and belongs in the response's diagnostic field only.
            detail=exc.failure_class.user_message,
            instance=request.url.path,
            failure_class=exc.failure_class.value,
            diagnostic=exc.detail,
        )

    @app.exception_handler(InsufficientPapersForComparison)
    async def _too_few(request: Request, exc: InsufficientPapersForComparison) -> JSONResponse:
        return problem(
            status=HTTPStatus.CONFLICT,
            slug="insufficient-papers",
            title="Not enough papers",
            detail=str(exc),
            instance=request.url.path,
        )

    @app.exception_handler(IllegalStateTransition)
    async def _illegal(request: Request, exc: IllegalStateTransition) -> JSONResponse:
        return problem(
            status=HTTPStatus.CONFLICT,
            slug="illegal-state-transition",
            title="Operation not valid in the current state",
            detail=str(exc),
            instance=request.url.path,
        )

    @app.exception_handler(InvariantViolation)
    async def _invariant(
        request: Request,
        exc: InvariantViolation,  # noqa: ARG001 - the message is deliberately not used
    ) -> JSONResponse:
        # A broken invariant is a bug here, not a bad request — so 500, and the
        # message does not reach the client. NFR-SEC8 and NFR-U2 agree on this:
        # a stack trace is neither actionable nor safe to show.
        return problem(
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            slug="internal-error",
            title="Internal error",
            detail="The request could not be completed. This has been logged.",
            instance=request.url.path,
        )

    @app.exception_handler(DomainError)
    async def _domain(request: Request, exc: DomainError) -> JSONResponse:
        # The catch-all for rule violations that have no dedicated handler. 400,
        # because `DomainError` means the request itself was invalid and
        # retrying it unchanged will fail identically.
        return problem(
            status=HTTPStatus.BAD_REQUEST,
            slug="invalid-request",
            title="Invalid request",
            detail=str(exc),
            instance=request.url.path,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default is a bare JSON array under `detail`, which is not a
        # problem document. Reshaped so every error this API emits has one form.
        return problem(
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            slug="validation-failed",
            title="Request failed validation",
            detail="One or more fields were missing or malformed.",
            instance=request.url.path,
            errors=[
                {"location": list(e.get("loc", ())), "message": e.get("msg", "")}
                for e in exc.errors()
            ],
        )
