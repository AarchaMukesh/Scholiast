"""Register, log in, refresh, log out (FR-A1, FR-A5, §5.8 layer 1).

Four endpoints and no authorisation logic: this router turns HTTP into use-case
calls and use-case results into HTTP. Everything that decides *whether* a caller
is who they claim lives behind `deps.current_tenant`, which every other router
depends on and none of them implements.

**Status codes carry meaning here, so they are chosen rather than defaulted:**

- `201` on register, because an account was created.
- `200` on login and refresh, because a session was issued for something that
  already existed.
- `204` on logout, always — including for a token that was already revoked or
  never existed. Logout is idempotent, and an error would turn it into an oracle
  for which tokens are real.
- `409` on a duplicate address. It reveals that the address is registered; see
  `ports.accounts.EmailAlreadyRegistered` for why that is unavoidable without an
  email path, and stated rather than hidden.
"""

from __future__ import annotations

from datetime import UTC, datetime
from http import HTTPStatus

from fastapi import APIRouter, HTTPException, Response

from scholiast.api import schemas
from scholiast.api.deps import (
    EndSessionDep,
    RefreshSessionDep,
    RegisterUserDep,
    StartSessionDep,
)
from scholiast.application.ports.accounts import EmailAlreadyRegistered
from scholiast.application.use_cases.authentication import SessionTokens

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _to_pair(tokens: SessionTokens) -> schemas.TokenPair:
    # Seconds remaining rather than an absolute timestamp: a client with a skewed
    # clock can still schedule a refresh correctly against a duration.
    remaining = int((tokens.access_expires_at - datetime.now(UTC)).total_seconds())
    return schemas.TokenPair(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        expires_in=max(remaining, 0),
        user_id=tokens.user_id,
    )


@router.post("/register", status_code=HTTPStatus.CREATED)
async def register(body: schemas.RegisterRequest, use_case: RegisterUserDep) -> schemas.TokenPair:
    """Create an account and return its first session."""
    try:
        return _to_pair(await use_case.execute(email=body.email, password=body.password))
    except EmailAlreadyRegistered as error:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT, detail="that email is already registered"
        ) from error


@router.post("/login")
async def login(body: schemas.LoginRequest, use_case: StartSessionDep) -> schemas.TokenPair:
    """Exchange credentials for a token pair. One failure shape for every cause."""
    return _to_pair(await use_case.execute(email=body.email, password=body.password))


@router.post("/refresh")
async def refresh(body: schemas.RefreshRequest, use_case: RefreshSessionDep) -> schemas.TokenPair:
    """Rotate a refresh token. Single-use: the presented one dies here (D84)."""
    return _to_pair(await use_case.execute(refresh_token=body.refresh_token))


@router.post("/logout", status_code=HTTPStatus.NO_CONTENT)
async def logout(body: schemas.RefreshRequest, use_case: EndSessionDep) -> Response:
    """Revoke the presented chain. Idempotent, and silent about what it found."""
    await use_case.execute(refresh_token=body.refresh_token)
    return Response(status_code=HTTPStatus.NO_CONTENT)
