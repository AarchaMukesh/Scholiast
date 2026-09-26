"""SQL adapters for the two authentication tables (D84).

In their own module rather than beside the tenant-scoped repositories, because
they are a different kind of thing: those answer "what may this principal see",
these answer "who is this principal". Keeping them apart means a reader of
`repositories.py` never meets a query without a tenant predicate and has to
wonder whether it is a bug.

Both require a `SystemContext`. That is not decoration — it is the one place the
type system can express "this reads across tenants on purpose", and the
constructor refuses anything else.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.application.ports.accounts import Account, EmailAlreadyRegistered, Session
from scholiast.application.ports.tenancy import AUTHENTICATION_REASON, Scope
from scholiast.infrastructure.db import models


def _require_authentication_scope(scope: Scope) -> None:
    """Refuse a scope that is not the sanctioned authentication bypass.

    A `TenantContext` here would be a category error: these tables are read to
    *produce* a principal. Checking the reason string as well as the type keeps
    the escape hatch enumerable — `AUTHENTICATION_REASON` greps to exactly the
    paths allowed to do this.
    """
    reason = getattr(scope, "reason", None)
    if not scope.is_system or reason != AUTHENTICATION_REASON:
        raise ValueError(
            "account and session repositories require "
            f"SystemContext({AUTHENTICATION_REASON!r}), got {scope!r}"
        )


class SqlAccountRepository:
    """`AccountRepository` over Postgres."""

    def __init__(self, session: AsyncSession, scope: Scope) -> None:
        _require_authentication_scope(scope)
        self._session = session

    async def create(self, *, email: str, password_hash: str) -> Account:
        row = models.User(id=uuid.uuid4(), email=email, password_hash=password_hash)
        self._session.add(row)
        try:
            # Flush rather than commit: the unit of work belongs to the request
            # (deps.auth_session), and registration must be able to fail the whole
            # transaction. Flushing is what surfaces the unique violation here
            # instead of at commit, where the router could no longer translate it.
            await self._session.flush()
        except IntegrityError as error:
            await self._session.rollback()
            raise EmailAlreadyRegistered(email) from error
        return Account(id=str(row.id), email=row.email, password_hash=row.password_hash)

    async def by_email(self, email: str) -> Account | None:
        result = await self._session.execute(select(models.User).where(models.User.email == email))
        row = result.scalar_one_or_none()
        return None if row is None else _to_account(row)

    async def by_id(self, account_id: str) -> Account | None:
        try:
            key = uuid.UUID(account_id)
        except ValueError:
            # A malformed subject in a signed token means the token was issued by
            # something other than this service. Absent, not an error.
            return None
        row = await self._session.get(models.User, key)
        return None if row is None else _to_account(row)

    async def set_password_hash(self, account_id: str, password_hash: str) -> None:
        await self._session.execute(
            update(models.User)
            .where(models.User.id == uuid.UUID(account_id))
            .values(password_hash=password_hash)
        )


class SqlSessionRepository:
    """`SessionRepository` over Postgres."""

    def __init__(self, session: AsyncSession, scope: Scope) -> None:
        _require_authentication_scope(scope)
        self._session = session

    async def record(
        self, *, user_id: str, family_id: str, fingerprint: str, expires_at: datetime
    ) -> Session:
        row = models.RefreshToken(
            id=uuid.uuid4(),
            user_id=uuid.UUID(user_id),
            family_id=uuid.UUID(family_id),
            fingerprint=fingerprint,
            expires_at=expires_at,
        )
        self._session.add(row)
        await self._session.flush()
        return _to_session(row)

    async def by_fingerprint(self, fingerprint: str) -> Session | None:
        result = await self._session.execute(
            select(models.RefreshToken).where(models.RefreshToken.fingerprint == fingerprint)
        )
        row = result.scalar_one_or_none()
        return None if row is None else _to_session(row)

    async def revoke(self, session_id: str, *, at: datetime) -> int:
        """Revoke one live token, reporting whether it was this call that did it.

        `revoked_at IS NULL` in the predicate is what makes this a
        compare-and-swap rather than a blind write: under READ COMMITTED a
        concurrent revoke of the same row makes this one match nothing and
        report 0, which is how the caller learns it lost the race (D87).
        """
        result = await self._session.execute(
            update(models.RefreshToken)
            .where(
                models.RefreshToken.id == uuid.UUID(session_id),
                models.RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=at)
        )
        return cast("CursorResult[Any]", result).rowcount

    async def revoke_family(self, family_id: str, *, at: datetime) -> int:
        result = await self._session.execute(
            update(models.RefreshToken)
            .where(
                models.RefreshToken.family_id == uuid.UUID(family_id),
                models.RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=at)
        )
        # Same narrowing as `repositories._affected_rows`: `execute` is typed as
        # returning `Result`, which exposes no `rowcount`, while DML actually
        # returns a `CursorResult`. Cast rather than silence, so the day this
        # stops being DML the error comes back.
        return cast("CursorResult[Any]", result).rowcount


def _to_account(row: models.User) -> Account:
    return Account(id=str(row.id), email=row.email, password_hash=row.password_hash)


def _to_session(row: models.RefreshToken) -> Session:
    return Session(
        id=str(row.id),
        user_id=str(row.user_id),
        family_id=str(row.family_id),
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
    )
