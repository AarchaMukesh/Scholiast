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

import math
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from scholiast.application.ports.security import AuthenticationFailed
from scholiast.application.use_cases.discover_papers import NoSourceAnswered
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


class RateLimited(Exception):
    """This caller has used its allowance for now (NFR-SEC7, D112).

    Defined here rather than in `domain/` because it is not a domain event: no
    rule about papers or workspaces was broken, and a different deployment with
    a proxy in front might refuse the same request before it ever reached this
    process. It is an API policy, so it lives with the handler that renders it.
    """

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after:.0f}s")
        self.retry_after = retry_after


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

    @app.exception_handler(RateLimited)
    async def _rate_limited(request: Request, exc: RateLimited) -> JSONResponse:
        # 429 with `Retry-After`, which is the same courtesy this system expects
        # from Groq (D103): a refusal that says when to come back costs nothing
        # to honour and saves the caller from guessing.
        #
        # The detail deliberately does NOT say which limit was hit or how much of
        # it remains. That would let a caller map the buckets and pace itself to
        # just under each one, which is the opposite of what the limit is for.
        response = problem(
            status=HTTPStatus.TOO_MANY_REQUESTS,
            slug="rate-limited",
            title="Too many requests",
            detail="Too many requests. Please slow down and try again shortly.",
            instance=request.url.path,
        )
        response.headers["Retry-After"] = str(max(1, int(exc.retry_after) + 1))
        return response

    @app.exception_handler(NoSourceAnswered)
    async def _no_source(request: Request, exc: NoSourceAnswered) -> JSONResponse:
        # 503, never an empty 200 (D119, D122): an empty list would tell the user
        # there are no papers on the topic when the truth is that no source
        # answered. Each source's status goes with it, as on a partial result;
        # why each failed does not, because that text is a third party's.
        response = problem(
            status=HTTPStatus.SERVICE_UNAVAILABLE,
            slug="discovery-unavailable",
            title="No discovery source answered",
            detail="None of the academic sources answered this search. Try again shortly.",
            instance=request.url.path,
            source_status=[
                {
                    "source": o.source,
                    "status": o.status.value,
                    "results": 0,
                    "total_hits": None,
                    "retry_after_seconds": o.retry_after,
                }
                for o in exc.sources
            ],
        )
        # When every source said when it will take work again, the soonest of
        # those is when this search can next succeed. Otherwise nothing is known.
        waits = [o.retry_after for o in exc.sources if o.retry_after is not None]
        if waits and len(waits) == len(exc.sources):
            response.headers["Retry-After"] = str(max(1, math.ceil(min(waits))))
        return response

    @app.exception_handler(AuthenticationFailed)
    async def _unauthenticated(
        request: Request,
        exc: AuthenticationFailed,  # noqa: ARG001 - the reason is logged, never returned
    ) -> JSONResponse:
        # ONE shape for every cause (FR-A5): unknown account, wrong password,
        # expired token, tampered signature, revoked session, reused refresh. The
        # specific reason travels on the exception for the server's logs; letting
        # any of it reach the client hands over an account-enumeration oracle.
        response = problem(
            status=HTTPStatus.UNAUTHORIZED,
            slug="unauthenticated",
            title="Not authenticated",
            detail="Invalid or expired credentials.",
            instance=request.url.path,
        )
        # RFC 9110 requires this on a 401, and it is what tells a client to
        # refresh rather than to re-prompt for a password.
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

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
