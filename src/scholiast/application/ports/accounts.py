"""Account and session ports — the two tables authentication owns (D84).

Separate from `ports.repositories` because these are the **bootstrap** case, and
the distinction is the interesting part. Every repository there is constructed
with a `Scope` and answers "what may this principal see"; these two answer "who
is this principal", which is a question asked before one exists. They reach
`users` and `refresh_tokens` and nothing else — no workspace, paper or chunk is
reachable through either — so the `SystemContext` they require (spelled
`AUTHENTICATION_REASON`) stays as narrow as the escape hatch it uses.

Both speak small frozen records rather than ORM rows, for the same reason the
other ports do: a use case that could touch a SQLAlchemy model would make
"swap the database" untrue.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class EmailAlreadyRegistered(Exception):
    """Raised when a registration collides with an existing address.

    A distinct type because this is the one place the answer cannot be uniform.
    Everywhere else, existence is hidden — FR-A2 asks for 404 rather than 403 on
    someone else's resources, and login reports one failure for every cause. A
    registration form cannot: it must either say the address is taken, or accept
    the registration and send a confirmation email that quietly says nothing.
    **There is no mail path in M3**, so the honest option is the first, and the
    cost is stated rather than hidden: `POST /v1/auth/register` reveals whether an
    address is registered. Closing that needs email, and email is not this phase.
    """


@dataclass(frozen=True, slots=True)
class Account:
    """A user, as authentication sees one."""

    id: str
    email: str
    password_hash: str


@dataclass(frozen=True, slots=True)
class Session:
    """One issued refresh token's row, without the secret.

    `is_usable_at` is a method rather than a property because usability depends
    on *when* you ask, and a property that silently reads the clock is how a
    stale value gets cached across a request.
    """

    id: str
    user_id: str
    family_id: str
    expires_at: datetime
    revoked_at: datetime | None

    def is_usable_at(self, moment: datetime) -> bool:
        return self.revoked_at is None and self.expires_at > moment


class AccountRepository(Protocol):
    """`users`, reachable only under `AUTHENTICATION_REASON`."""

    async def create(self, *, email: str, password_hash: str) -> Account:
        """Insert an account, or raise `EmailAlreadyRegistered`.

        The uniqueness check belongs to the database, not to a prior SELECT: two
        simultaneous registrations for one address both pass a check-then-insert
        and one of them loses at the constraint anyway.
        """
        ...

    async def by_email(self, email: str) -> Account | None: ...

    async def by_id(self, account_id: str) -> Account | None: ...

    async def set_password_hash(self, account_id: str, password_hash: str) -> None:
        """Replace a stored hash — used to upgrade Argon2id parameters at login."""
        ...


class SessionRepository(Protocol):
    """`refresh_tokens`, reachable only under `AUTHENTICATION_REASON`."""

    async def record(
        self, *, user_id: str, family_id: str, fingerprint: str, expires_at: datetime
    ) -> Session: ...

    async def by_fingerprint(self, fingerprint: str) -> Session | None:
        """The session a refresh secret names, whether or not it is still usable.

        Deliberately returns revoked and expired rows: refusing to look them up
        would make a reused token indistinguishable from an unknown one, and
        reuse is exactly what has to trigger family revocation (D84).
        """
        ...

    async def revoke(self, session_id: str, *, at: datetime) -> int:
        """Revoke one token if it is still live; return the rows changed (D87).

        The count is the point. A caller that read the row as usable and then
        revokes it has a gap between the two, and a second presentation of the
        same token can fall into it — so "did my UPDATE actually change
        anything" is the only honest test of single use.
        """
        ...

    async def revoke_family(self, family_id: str, *, at: datetime) -> int:
        """Revoke every unrevoked token in a rotation chain. Returns the count."""
        ...
