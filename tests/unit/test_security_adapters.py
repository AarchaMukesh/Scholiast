"""Argon2id hashing and JWT issue/verify (D84, NFR-SEC1).

Doc 7 chose to hand-roll these rather than adopt a framework, on the grounds that
the security reasoning is the interesting part. The bill for that choice is paid
here: every forgery a framework would have refused on your behalf has to be
refused by code in this repository, and shown to be refused by a test.

The forgeries that matter are the ones that look like valid tokens: `alg: none`,
a token signed with a different secret, one minted by another issuer, and one
that is simply past its expiry.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from scholiast.application.ports.security import AuthenticationFailed
from scholiast.infrastructure.security.passwords import Argon2idHasher
from scholiast.infrastructure.security.tokens import ALGORITHM, JwtTokenService

SECRET = "a" * 32
OTHER_SECRET = "b" * 32


def _service(**overrides: object) -> JwtTokenService:
    settings: dict[str, object] = {
        "secret": SECRET,
        "issuer": "scholiast",
        "access_ttl": timedelta(minutes=15),
        "refresh_ttl": timedelta(days=14),
    }
    settings.update(overrides)
    return JwtTokenService(**settings)  # type: ignore[arg-type]


class TestPasswordHashing:
    def test_a_password_verifies_against_its_own_hash(self) -> None:
        hasher = Argon2idHasher()
        encoded = hasher.hash("correct horse battery staple")
        assert hasher.verify("correct horse battery staple", encoded)

    def test_a_wrong_password_does_not(self) -> None:
        hasher = Argon2idHasher()
        assert not hasher.verify("wrong", hasher.hash("right-and-long-enough"))

    def test_the_same_password_hashes_differently_every_time(self) -> None:
        # The salt. Without it, identical passwords share a hash and one rainbow
        # table serves every account in the table at once.
        hasher = Argon2idHasher()
        assert hasher.hash("a-repeated-password") != hasher.hash("a-repeated-password")

    def test_a_malformed_hash_is_a_failed_verification_not_an_exception(self) -> None:
        # A corrupt row must read as "wrong password" to the caller, not as a 500
        # that announces the row is corrupt.
        assert not Argon2idHasher().verify("anything", "not-an-argon2-hash")

    def test_a_hash_from_weaker_parameters_is_flagged_for_upgrade(self) -> None:
        # The upgrade path: login is the only moment the plaintext exists, so it
        # is the only moment a stored hash can be strengthened.
        weak = Argon2idHasher(time_cost=1, memory_cost_kib=8192, parallelism=1)
        current = Argon2idHasher()
        assert current.needs_rehash(weak.hash("a-long-enough-password"))
        assert not current.needs_rehash(current.hash("a-long-enough-password"))

    def test_a_malformed_hash_also_needs_rehashing(self) -> None:
        assert Argon2idHasher().needs_rehash("garbage")

    def test_the_parameters_are_reported(self) -> None:
        # NFR-M5: the cost is the security property, so it is recorded rather
        # than left to whatever the library defaulted to that year.
        assert Argon2idHasher().version == "argon2id/t=3,m=65536,p=4/1.0.0"


class TestAccessTokens:
    def test_a_freshly_issued_token_verifies_to_its_subject(self) -> None:
        service = _service()
        issued = service.issue_access(subject="user-1")
        assert service.verify_access(issued.token).subject == "user-1"
        assert issued.expires_at > datetime.now(UTC)

    def test_an_expired_token_is_rejected(self) -> None:
        expired = _service(access_ttl=timedelta(seconds=-1)).issue_access(subject="user-1")
        with pytest.raises(AuthenticationFailed):
            _service().verify_access(expired.token)

    def test_a_token_signed_with_another_secret_is_rejected(self) -> None:
        # The whole security of a symmetric token is the secret; a verifier that
        # accepts another signer accepts anyone who can run the library.
        foreign = _service(secret=OTHER_SECRET).issue_access(subject="user-1")
        with pytest.raises(AuthenticationFailed):
            _service().verify_access(foreign.token)

    def test_an_unsigned_alg_none_token_is_rejected(self) -> None:
        # THE classic JWT forgery: hand back a token whose header says no
        # signature is needed. It works against any verifier that trusts the
        # token's own header instead of pinning the algorithm.
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "none", "typ": "JWT"}).encode()
        ).rstrip(b"=")
        payload = base64.urlsafe_b64encode(
            json.dumps(
                {
                    "sub": "attacker",
                    "iss": "scholiast",
                    "typ": "access",
                    "iat": int(datetime.now(UTC).timestamp()),
                    "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
                }
            ).encode()
        ).rstrip(b"=")
        forged = f"{header.decode()}.{payload.decode()}."

        with pytest.raises(AuthenticationFailed):
            _service().verify_access(forged)

    # PyJWT warns that a 32-byte key is short for SHA-512. Expected, and rather
    # the point: the token really is HS512-signed with our own secret.
    @pytest.mark.filterwarnings("ignore:The HMAC key is")
    def test_an_algorithm_the_header_names_is_not_accepted(self) -> None:
        # The pin, tested on its own terms: this token is signed with the RIGHT
        # secret and is rejected anyway, because it names an algorithm the
        # verifier does not accept. That is the property the confusion attacks
        # exploit — a verifier that reads `alg` from the token and obliges.
        other_algorithm = jwt.encode(
            {
                "sub": "user-1",
                "iss": "scholiast",
                "typ": "access",
                "iat": int(datetime.now(UTC).timestamp()),
                "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            },
            SECRET,
            algorithm="HS512",
        )
        with pytest.raises(AuthenticationFailed):
            _service().verify_access(other_algorithm)

    def test_a_tampered_payload_is_rejected(self) -> None:
        issued = _service().issue_access(subject="user-1")
        header, _payload, signature = issued.token.split(".")
        swapped = base64.urlsafe_b64encode(
            json.dumps(
                {
                    "sub": "someone-else",
                    "iss": "scholiast",
                    "typ": "access",
                    "iat": int(datetime.now(UTC).timestamp()),
                    "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
                }
            ).encode()
        ).rstrip(b"=")

        with pytest.raises(AuthenticationFailed):
            _service().verify_access(f"{header}.{swapped.decode()}.{signature}")

    def test_a_token_from_another_issuer_is_rejected(self) -> None:
        # Matters when a secret is shared across services: the signature checks
        # out and the token still is not ours.
        foreign = _service(issuer="someone-else").issue_access(subject="user-1")
        with pytest.raises(AuthenticationFailed):
            _service().verify_access(foreign.token)

    def test_a_token_without_an_expiry_is_rejected(self) -> None:
        # PyJWT only enforces an `exp` that is present, so a token minted without
        # one would otherwise be eternal.
        eternal = jwt.encode(
            {"sub": "user-1", "iss": "scholiast", "typ": "access", "iat": 0},
            SECRET,
            algorithm=ALGORITHM,
        )
        with pytest.raises(AuthenticationFailed):
            _service().verify_access(eternal)

    def test_a_non_access_token_type_is_rejected(self) -> None:
        pretender = jwt.encode(
            {
                "sub": "user-1",
                "iss": "scholiast",
                "typ": "refresh",
                "iat": int(datetime.now(UTC).timestamp()),
                "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            },
            SECRET,
            algorithm=ALGORITHM,
        )
        with pytest.raises(AuthenticationFailed):
            _service().verify_access(pretender)

    def test_rubbish_is_rejected_as_an_authentication_failure(self) -> None:
        # Not a ValueError escaping into a 500: everything that is not a valid
        # token is one failure shape (FR-A5).
        with pytest.raises(AuthenticationFailed):
            _service().verify_access("not.a.token")

    def test_a_deeply_nested_header_is_refused_not_raised_as_a_recursion_error(self) -> None:
        # PYSEC-2026-4142 (D121): before PyJWT 2.14, a header nested deeply enough
        # made `jwt.decode` raise RecursionError. That is not a PyJWTError, so it
        # escaped the one-failure-shape handler (FR-A5) as a 500, and any client
        # that can send a bearer token could trigger it without authenticating.
        depth = 100_000
        header = base64.urlsafe_b64encode(
            ('{"alg":"HS256","typ":"JWT","x":' + "[" * depth + "]" * depth + "}").encode()
        ).rstrip(b"=")
        payload = base64.urlsafe_b64encode(b'{"sub":"user-1"}').rstrip(b"=")
        with pytest.raises(AuthenticationFailed):
            _service().verify_access(f"{header.decode()}.{payload.decode()}.c2lnbmF0dXJl")

    def test_a_short_secret_is_refused_at_construction(self) -> None:
        # Fail to boot rather than fail open: a brute-forceable HMAC key makes
        # every token this process issues forgeable.
        with pytest.raises(ValueError, match="at least 32 characters"):
            _service(secret="too-short")


class TestRefreshSecrets:
    def test_each_secret_is_unique_and_long(self) -> None:
        service = _service()
        secrets = {service.issue_refresh().secret for _ in range(50)}
        assert len(secrets) == 50
        assert all(len(secret) >= 43 for secret in secrets)

    def test_the_fingerprint_is_deterministic_and_not_the_secret(self) -> None:
        # What the database stores. A leaked table must not yield usable tokens.
        service = _service()
        issued = service.issue_refresh()
        assert service.fingerprint(issued.secret) == issued.fingerprint
        assert issued.fingerprint != issued.secret
        assert len(issued.fingerprint) == 64

    def test_different_secrets_fingerprint_differently(self) -> None:
        service = _service()
        assert service.fingerprint("one") != service.fingerprint("two")

    def test_the_expiry_is_the_configured_lifetime(self) -> None:
        issued = _service(refresh_ttl=timedelta(days=14)).issue_refresh()
        remaining = issued.expires_at - datetime.now(UTC)
        assert timedelta(days=13, hours=23) < remaining <= timedelta(days=14)
