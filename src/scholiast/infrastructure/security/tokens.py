"""JWT access tokens and opaque refresh secrets (FR-A1, FR-A5, D84).

**The access token is verified by signature and nothing else**, which is what
§5.8 means by stateless verification at the edge: no database read, so replicas
stay independent and a request costs one HMAC.

**The algorithm is pinned on the way in.** `jwt.decode` is told which algorithms
are acceptable rather than trusting the token's own header, because the header is
attacker-controlled. That one argument is the difference between this and the
`alg: none` forgery, and the whole algorithm-confusion family — of which
RS256-verified-as-HS256, where a public key is used as an HMAC secret, is the
famous member. The pin is tested directly: a token naming a different algorithm
is refused *even when it is signed with the correct secret*, which is the
property the confusion attacks exploit. RS256 itself is not exercised, because
signing one needs a backend this project does not install.

**The refresh token is not a JWT** — see `ports/security`. It is 256 bits from
`secrets`, stored as a SHA-256 fingerprint, and single-use (D84).
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import jwt

from scholiast.application.ports.security import (
    AccessClaims,
    AuthenticationFailed,
    IssuedAccess,
    IssuedRefresh,
)

#: Symmetric, because one service issues and verifies. RS256 buys independent
#: verifiers a public key, which matters when someone else validates your tokens;
#: nobody does here, and a keypair would be ceremony without a beneficiary.
ALGORITHM = "HS256"

#: Claims a token must carry to be considered at all. Listed explicitly so a
#: token missing `exp` is rejected rather than treated as eternal — PyJWT only
#: enforces an expiry that is present.
REQUIRED_CLAIMS = ("exp", "iat", "sub", "iss")

#: Distinguishes an access token from anything else signed with the same secret.
ACCESS_TYPE = "access"

#: 32 bytes -> 43 URL-safe characters. Far past any brute-force concern, which is
#: what lets the fingerprint be a fast hash (D84).
REFRESH_SECRET_BYTES = 32


class JwtTokenService:
    """`TokenService` over PyJWT."""

    def __init__(
        self,
        *,
        secret: str,
        issuer: str,
        access_ttl: timedelta,
        refresh_ttl: timedelta,
    ) -> None:
        if not secret or len(secret) < 32:
            # A short HMAC secret is brute-forceable offline from one token, and
            # every token this process ever issued becomes forgeable. Refused at
            # construction rather than validated at issue, so a misconfigured
            # deployment fails to start instead of failing open.
            raise ValueError("the JWT secret must be at least 32 characters")
        self._secret = secret
        self._issuer = issuer
        self._access_ttl = access_ttl
        self._refresh_ttl = refresh_ttl

    def issue_access(self, *, subject: str) -> IssuedAccess:
        now = datetime.now(UTC)
        expires_at = now + self._access_ttl
        payload = {
            "sub": subject,
            "iss": self._issuer,
            "typ": ACCESS_TYPE,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        return IssuedAccess(
            token=jwt.encode(payload, self._secret, algorithm=ALGORITHM),
            expires_at=expires_at,
        )

    def verify_access(self, token: str) -> AccessClaims:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                # The pin. Not `algorithms=jwt.algorithms.get_default_algorithms()`,
                # and never derived from the token's own header.
                algorithms=[ALGORITHM],
                issuer=self._issuer,
                options={"require": list(REQUIRED_CLAIMS)},
            )
        except jwt.PyJWTError as error:
            # Every PyJWT failure collapses to one message: expired, tampered,
            # wrong issuer and malformed must be indistinguishable to a caller
            # (FR-A5). The specific error is the server's to log.
            raise AuthenticationFailed(f"access token rejected: {type(error).__name__}") from error

        if payload.get("typ") != ACCESS_TYPE:
            raise AuthenticationFailed("token is not an access token")
        subject = str(payload["sub"])
        if not subject:
            raise AuthenticationFailed("access token carries no subject")
        return AccessClaims(
            subject=subject,
            expires_at=datetime.fromtimestamp(int(payload["exp"]), tz=UTC),
        )

    def issue_refresh(self) -> IssuedRefresh:
        secret = secrets.token_urlsafe(REFRESH_SECRET_BYTES)
        return IssuedRefresh(
            secret=secret,
            fingerprint=self.fingerprint(secret),
            expires_at=datetime.now(UTC) + self._refresh_ttl,
        )

    def fingerprint(self, secret: str) -> str:
        """SHA-256 of the secret — see `ports/security` for why a fast hash."""
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()
