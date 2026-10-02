"""Register, log in, refresh, log out (FR-A1, FR-A5, D84).

Four use cases in one module because they share one mechanism — the rotating
refresh family — and reading them apart would mean reconstructing that mechanism
four times. Each is still its own class with its own dependencies.

**The rules that are easy to get wrong, and where they live:**

- *Failures are uniform.* Unknown address, wrong password, expired token, revoked
  session and reused token all raise `AuthenticationFailed`. The distinction is
  logged, never returned (FR-A5).
- *Login costs the same whether or not the account exists.* An unknown address
  still pays a full Argon2id verification against a dummy hash, because
  responding faster for unknown accounts is a free account-enumeration oracle —
  a timing side channel that no amount of uniform messaging closes.
- *Refresh is single-use.* The presented token is revoked as its successor is
  issued, both in one transaction. Presenting it again means replay or theft, and
  the whole family dies (D84).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from scholiast.application.ports.accounts import (
    Account,
    AccountRepository,
    SessionRepository,
)
from scholiast.application.ports.security import (
    AuthenticationFailed,
    PasswordHasher,
    TokenService,
)
from scholiast.domain.identity import normalise_email, validate_password

#: Verified against when no account matches, so the work done is the same either
#: way. The value is irrelevant — only the cost is — and it is never a password
#: anyone can present, because no account stores this hash.
_DUMMY_PASSWORD = "scholiast-timing-equalisation"  # noqa: S105 - not a credential


@dataclass(frozen=True, slots=True)
class SessionTokens:
    """What a successful authentication hands back."""

    access_token: str
    access_expires_at: datetime
    refresh_token: str
    refresh_expires_at: datetime
    user_id: str


class _SessionIssuer:
    """Shared machinery: mint a token pair and record the refresh half."""

    def __init__(
        self,
        *,
        sessions: SessionRepository,
        tokens: TokenService,
    ) -> None:
        self._sessions = sessions
        self._tokens = tokens

    async def issue(self, *, user_id: str, family_id: str | None = None) -> SessionTokens:
        access = self._tokens.issue_access(subject=user_id)
        refresh = self._tokens.issue_refresh()
        await self._sessions.record(
            user_id=user_id,
            # A new family for a fresh login, the existing one for a rotation:
            # that is what makes "this chain was stolen" a statement the schema
            # can express.
            family_id=family_id or str(uuid.uuid4()),
            fingerprint=refresh.fingerprint,
            expires_at=refresh.expires_at,
        )
        return SessionTokens(
            access_token=access.token,
            access_expires_at=access.expires_at,
            refresh_token=refresh.secret,
            refresh_expires_at=refresh.expires_at,
            user_id=user_id,
        )


class RegisterUserUseCase:
    """Create an account and log it straight in (FR-A1)."""

    def __init__(
        self,
        *,
        accounts: AccountRepository,
        sessions: SessionRepository,
        hasher: PasswordHasher,
        tokens: TokenService,
    ) -> None:
        self._accounts = accounts
        self._hasher = hasher
        self._issuer = _SessionIssuer(sessions=sessions, tokens=tokens)

    async def execute(self, *, email: str, password: str) -> SessionTokens:
        """Register, then issue the first session.

        Policy runs before hashing: rejecting a short password after paying 64 MB
        of Argon2id would be free work for anyone wanting to load the server.
        """
        address = normalise_email(email)
        validate_password(password)
        account = await self._accounts.create(
            email=address, password_hash=self._hasher.hash(password)
        )
        return await self._issuer.issue(user_id=account.id)


class StartSessionUseCase:
    """Exchange credentials for a token pair (FR-A1)."""

    def __init__(
        self,
        *,
        accounts: AccountRepository,
        sessions: SessionRepository,
        hasher: PasswordHasher,
        tokens: TokenService,
    ) -> None:
        self._accounts = accounts
        self._hasher = hasher
        self._issuer = _SessionIssuer(sessions=sessions, tokens=tokens)
        self._dummy_hash = hasher.hash(_DUMMY_PASSWORD)

    async def execute(self, *, email: str, password: str) -> SessionTokens:
        try:
            address = normalise_email(email)
        except Exception as error:
            # A malformed address is a failed login, not a validation error: a
            # 422 here would tell a prober that the address format mattered.
            raise AuthenticationFailed("login: malformed email") from error

        account = await self._accounts.by_email(address)
        if account is None:
            # The equalising verification. Its result is discarded; its COST is
            # the point.
            self._hasher.verify(password, self._dummy_hash)
            raise AuthenticationFailed("login: no such account")

        if not self._hasher.verify(password, account.password_hash):
            raise AuthenticationFailed("login: password mismatch")

        await self._upgrade_hash_if_stale(account, password)
        return await self._issuer.issue(user_id=account.id)

    async def _upgrade_hash_if_stale(self, account: Account, password: str) -> None:
        """Rehash under current parameters, if the stored hash predates them.

        Login is the only moment the plaintext exists, so it is the only moment
        an upgrade is possible without asking the user for anything.
        """
        if self._hasher.needs_rehash(account.password_hash):
            await self._accounts.set_password_hash(account.id, self._hasher.hash(password))


class RefreshSessionUseCase:
    """Rotate a refresh token for a new pair (FR-A5, D84)."""

    def __init__(
        self,
        *,
        sessions: SessionRepository,
        tokens: TokenService,
    ) -> None:
        self._sessions = sessions
        self._tokens = tokens
        self._issuer = _SessionIssuer(sessions=sessions, tokens=tokens)

    async def execute(self, *, refresh_token: str) -> SessionTokens:
        now = datetime.now(UTC)
        session = await self._sessions.by_fingerprint(self._tokens.fingerprint(refresh_token))
        if session is None:
            raise AuthenticationFailed("refresh: unknown token")

        if session.revoked_at is not None:
            # Reuse. Either this token was replayed or the chain was stolen, and
            # nothing distinguishes the two from here — so the whole family goes,
            # which logs out the legitimate holder too. That is the intended
            # trade: a forced re-login beats an attacker holding a live chain.
            await self._sessions.revoke_family(session.family_id, at=now)
            raise AuthenticationFailed("refresh: token reuse — family revoked")

        if not session.is_usable_at(now):
            raise AuthenticationFailed("refresh: token expired")

        if not await self._sessions.revoke(session.id, at=now):
            # Nothing changed, so someone else revoked this exact row between the
            # read above and this write — a second presentation of a single-use
            # token, which is reuse by another name and gets the same answer
            # (D87). Without the count both racers would issue a successor and
            # one token would become two live chains.
            await self._sessions.revoke_family(session.family_id, at=now)
            raise AuthenticationFailed("refresh: lost the rotation race — family revoked")
        return await self._issuer.issue(user_id=session.user_id, family_id=session.family_id)


class EndSessionUseCase:
    """Log out by revoking the presented chain (FR-A5)."""

    def __init__(self, *, sessions: SessionRepository, tokens: TokenService) -> None:
        self._sessions = sessions
        self._tokens = tokens

    async def execute(self, *, refresh_token: str) -> int:
        """Revoke the family and report how many tokens it held.

        **Never raises for an unknown token.** Logout is idempotent by design: a
        client retrying after a dropped response, or presenting a token that was
        already revoked, has achieved what it asked for. Raising would also make
        logout an oracle for which tokens exist.

        The whole family goes, not just the presented token: a chain has exactly
        one live member, and revoking by family says what was meant even if a
        rotation raced the logout.
        """
        session = await self._sessions.by_fingerprint(self._tokens.fingerprint(refresh_token))
        if session is None:
            return 0
        return await self._sessions.revoke_family(session.family_id, at=datetime.now(UTC))
