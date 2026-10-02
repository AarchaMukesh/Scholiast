"""Password hashing and token issue/verify, as ports (FR-A1, FR-A5, D84).

Two primitives, both behind protocols for the same reason every other adapter is:
the use cases decide *what* authentication means, and the adapters decide which
library performs it. Argon2id parameters and a JWT algorithm are exactly the kind
of choice that gets revised — by a CVE, by a hardware change, by a hosting
constraint — and revising it should not reach into a use case.

**Two token shapes, on purpose (D84).**

- The **access token** is a signed JWT verified by signature alone, because §5.8
  requires stateless verification at the edge so replicas stay independent. It is
  short-lived precisely because nothing can revoke it before it expires.
- The **refresh token** is an opaque random secret, stored only as a fingerprint.
  It is never a JWT: every refresh hits the database anyway to enforce
  single-use, so claims inside the token would buy nothing and would put a second
  copy of the expiry somewhere it could disagree with the row.

**Fingerprints use a fast hash, and that is deliberate.** A refresh secret is 256
bits of `secrets` randomness, so there is no dictionary to attack and no work
factor worth paying; Argon2id here would cost every refresh ~64 MB of memory to
defend against an attack that cannot happen. Passwords are the opposite case, and
get the opposite treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from scholiast.domain.errors import DomainError


class AuthenticationFailed(DomainError):
    """Credentials, tokens or sessions that did not check out.

    **One error for every cause, by design (FR-A5's "rejected uniformly").** A
    caller must not be able to tell "no such user" from "wrong password" from
    "expired token" from "revoked session" — each distinction is a free oracle
    for enumerating accounts or probing session state. The *reason* is carried
    for the server's logs; the HTTP surface renders one 401.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class IssuedAccess:
    """A signed access token and the moment it stops being valid."""

    token: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class IssuedRefresh:
    """A refresh secret, the fingerprint to store, and its expiry.

    The secret is returned to the caller once and never persisted; the
    fingerprint is what the database keeps. A leaked database therefore yields no
    usable refresh token.
    """

    secret: str
    fingerprint: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class AccessClaims:
    """What a verified access token asserts."""

    subject: str
    expires_at: datetime


class PasswordHasher(Protocol):
    """Argon2id in production; anything deterministic and fast in tests."""

    def hash(self, password: str) -> str: ...

    def verify(self, password: str, encoded: str) -> bool:
        """True when the password matches. Never raises for a wrong password."""
        ...

    def needs_rehash(self, encoded: str) -> bool:
        """True when `encoded` was produced with weaker parameters than current.

        Checked at login, where the plaintext is in hand and rehashing is free —
        the only moment a stored hash can be upgraded without asking the user to
        do anything.
        """
        ...

    @property
    def version(self) -> str:
        """Algorithm and parameters, for the record (NFR-M5)."""
        ...


class TokenService(Protocol):
    """Issue and verify the two token shapes."""

    def issue_access(self, *, subject: str) -> IssuedAccess: ...

    def verify_access(self, token: str) -> AccessClaims:
        """Return the claims, or raise `AuthenticationFailed`.

        Must reject a token whose algorithm header is not the expected one:
        accepting whatever the token names is the `alg: none` and
        RS256-verified-as-HS256 family of forgeries.
        """
        ...

    def issue_refresh(self) -> IssuedRefresh: ...

    def fingerprint(self, secret: str) -> str:
        """The stored form of a refresh secret. Must be deterministic."""
        ...
