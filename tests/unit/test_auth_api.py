"""The HTTP surface of authentication (FR-A1, FR-A5, §5.8 layer 1).

Hermetic, in the manner of `test_api_errors`: the router and the real token
service, with fakes behind the repositories. No database, so this runs in CI —
which matters, because the properties checked here are the security-relevant
ones and a suite that skips without Postgres would check them nowhere.

Real `JwtTokenService`, though, not a fake. The point of most of these tests is
what a *client* can observe, and a token that is not really signed cannot be
really forged.

**The central property is uniformity (FR-A5).** Nine different causes of failure
— unknown account, wrong password, short password, unknown refresh, reused
refresh, missing header, wrong scheme, forged token, expired token — must produce
one indistinguishable response. Any difference in status, body or shape is an
oracle, and an oracle is how account enumeration starts.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scholiast.api import deps
from scholiast.api.errors import PROBLEM_CONTENT_TYPE, register_error_handlers
from scholiast.api.routers import auth
from scholiast.application.use_cases.authentication import (
    EndSessionUseCase,
    RefreshSessionUseCase,
    RegisterUserUseCase,
    StartSessionUseCase,
)
from scholiast.config.settings import Settings
from scholiast.infrastructure.security.ratelimit import FixedWindowLimiter
from scholiast.infrastructure.security.tokens import JwtTokenService
from tests.unit.test_authentication import FakeAccounts, FakeHasher, FakeSessions

PASSWORD = "a-perfectly-adequate-password"
SECRET = "unit-test-secret-of-adequate-length"


@dataclass
class Harness:
    client: TestClient
    accounts: FakeAccounts
    sessions: FakeSessions
    tokens: JwtTokenService


@pytest.fixture
def harness() -> Harness:
    accounts = FakeAccounts()
    sessions = FakeSessions()
    hasher = FakeHasher()
    tokens = JwtTokenService(
        secret=SECRET,
        issuer="scholiast",
        access_ttl=timedelta(minutes=15),
        refresh_ttl=timedelta(days=14),
    )

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(auth.router)
    app.state.tokens = tokens
    # This suite collapses twelve failure causes into one shape, which is more
    # failing logins than NFR-SEC7's limit allows — so the limiter is off here.
    # That it FIRES on the auth routes is asserted in `test_rate_limits.py`;
    # keeping the two apart stops each test from depending on the other's number.
    app.state.settings = Settings(rate_limit_enabled=False)
    app.state.rate_limiter = FixedWindowLimiter()

    @app.get("/protected")
    async def _protected(tenant: deps.TenantDep) -> dict[str, str]:
        # Stands in for every real router: none of them verifies anything, they
        # just depend on `current_tenant`.
        return {"user_id": tenant.user_id}

    app.dependency_overrides[deps.register_user_use_case] = lambda: RegisterUserUseCase(
        accounts=accounts,  # type: ignore[arg-type]
        sessions=sessions,  # type: ignore[arg-type]
        hasher=hasher,  # type: ignore[arg-type]
        tokens=tokens,
    )
    app.dependency_overrides[deps.start_session_use_case] = lambda: StartSessionUseCase(
        accounts=accounts,  # type: ignore[arg-type]
        sessions=sessions,  # type: ignore[arg-type]
        hasher=hasher,  # type: ignore[arg-type]
        tokens=tokens,
    )
    app.dependency_overrides[deps.refresh_session_use_case] = lambda: RefreshSessionUseCase(
        sessions=sessions,  # type: ignore[arg-type]
        tokens=tokens,
    )
    app.dependency_overrides[deps.end_session_use_case] = lambda: EndSessionUseCase(
        sessions=sessions,  # type: ignore[arg-type]
        tokens=tokens,
    )

    return Harness(
        client=TestClient(app, raise_server_exceptions=False),
        accounts=accounts,
        sessions=sessions,
        tokens=tokens,
    )


def _register(harness: Harness, email: str = "ada@example.com") -> dict[str, str]:
    response = harness.client.post("/v1/auth/register", json={"email": email, "password": PASSWORD})
    assert response.status_code == 201, response.text
    return dict(response.json())


class TestTheWholeFlow:
    def test_register_then_call_then_refresh_then_logout(self, harness: Harness) -> None:
        """The phase's acceptance test, end to end over HTTP."""
        first = _register(harness)

        authenticated = harness.client.get(
            "/protected", headers={"Authorization": f"Bearer {first['access_token']}"}
        )
        assert authenticated.status_code == 200
        assert authenticated.json()["user_id"] == first["user_id"]

        rotated = harness.client.post(
            "/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
        )
        assert rotated.status_code == 200
        second = rotated.json()
        assert second["refresh_token"] != first["refresh_token"]
        assert second["user_id"] == first["user_id"]

        # The rotated-away token is dead, and using it kills the family — so the
        # successor stops working too.
        assert (
            harness.client.post(
                "/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
            ).status_code
            == 401
        )
        assert (
            harness.client.post(
                "/v1/auth/refresh", json={"refresh_token": second["refresh_token"]}
            ).status_code
            == 401
        )

    def test_login_after_register_and_logout_ends_the_session(self, harness: Harness) -> None:
        _register(harness)
        signed_in = harness.client.post(
            "/v1/auth/login", json={"email": "ADA@example.com", "password": PASSWORD}
        )
        assert signed_in.status_code == 200
        issued = signed_in.json()

        assert (
            harness.client.post(
                "/v1/auth/logout", json={"refresh_token": issued["refresh_token"]}
            ).status_code
            == 204
        )
        assert (
            harness.client.post(
                "/v1/auth/refresh", json={"refresh_token": issued["refresh_token"]}
            ).status_code
            == 401
        )

    def test_logout_is_idempotent_and_silent(self, harness: Harness) -> None:
        # Repeated logout, and logout of something never issued, both 204 with an
        # empty body. An error would say which tokens are real.
        issued = _register(harness)
        for token in (issued["refresh_token"], issued["refresh_token"], "never-issued"):
            response = harness.client.post("/v1/auth/logout", json={"refresh_token": token})
            assert response.status_code == 204
            assert response.content == b""

    def test_the_response_carries_a_bearer_pair_and_a_lifetime(self, harness: Harness) -> None:
        issued = _register(harness)
        assert issued["token_type"] == "bearer"
        assert 0 < int(issued["expires_in"]) <= 15 * 60
        assert PASSWORD not in json.dumps(issued)


class TestRegistration:
    def test_a_duplicate_address_is_a_conflict(self, harness: Harness) -> None:
        # 409, and the docstring on `EmailAlreadyRegistered` owns the fact that
        # this reveals the address is taken. Stated, not hidden.
        _register(harness)
        response = harness.client.post(
            "/v1/auth/register", json={"email": "ADA@example.com", "password": PASSWORD}
        )
        assert response.status_code == 409

    def test_a_short_password_is_refused_at_the_boundary(self, harness: Harness) -> None:
        # 422 here is correct: registration is the one place where telling the
        # user their password is too short is the entire point.
        response = harness.client.post(
            "/v1/auth/register", json={"email": "ada@example.com", "password": "short"}
        )
        assert response.status_code == 422

    def test_an_oversized_password_never_reaches_the_hasher(self, harness: Harness) -> None:
        # Argon2id reads the whole input; an unbounded password is an unbounded
        # amount of server work per request.
        response = harness.client.post(
            "/v1/auth/register", json={"email": "ada@example.com", "password": "a" * 5000}
        )
        assert response.status_code == 422
        assert harness.accounts.rows == {}

    def test_unexpected_fields_are_refused(self, harness: Harness) -> None:
        # `extra="forbid"`: a request naming a field the server does not know is
        # a client bug or a probe, and silently ignoring it hides both.
        response = harness.client.post(
            "/v1/auth/register",
            json={"email": "ada@example.com", "password": PASSWORD, "is_admin": True},
        )
        assert response.status_code == 422


class TestUniformRejection:
    """FR-A5: every cause of failure produces one indistinguishable response."""

    def _failures(self, harness: Harness) -> list[tuple[str, httpx.Response]]:
        _register(harness, email="ada@example.com")
        issued = _register(harness, email="grace@example.com")
        harness.client.post("/v1/auth/refresh", json={"refresh_token": issued["refresh_token"]})

        expired = JwtTokenService(
            secret=SECRET,
            issuer="scholiast",
            access_ttl=timedelta(seconds=-1),
            refresh_ttl=timedelta(days=14),
        ).issue_access(subject="ada")
        foreign = JwtTokenService(
            secret="a-different-secret-of-adequate-length",
            issuer="scholiast",
            access_ttl=timedelta(minutes=15),
            refresh_ttl=timedelta(days=14),
        ).issue_access(subject="ada")

        return [
            (
                "unknown account",
                harness.client.post(
                    "/v1/auth/login", json={"email": "nobody@example.com", "password": PASSWORD}
                ),
            ),
            (
                "wrong password",
                harness.client.post(
                    "/v1/auth/login",
                    json={"email": "ada@example.com", "password": "not-the-password"},
                ),
            ),
            (
                "short password",
                harness.client.post(
                    "/v1/auth/login", json={"email": "ada@example.com", "password": "x"}
                ),
            ),
            (
                "malformed address",
                harness.client.post(
                    "/v1/auth/login", json={"email": "not-an-address", "password": PASSWORD}
                ),
            ),
            (
                "unknown refresh",
                harness.client.post("/v1/auth/refresh", json={"refresh_token": "never-issued"}),
            ),
            (
                "reused refresh",
                harness.client.post(
                    "/v1/auth/refresh", json={"refresh_token": issued["refresh_token"]}
                ),
            ),
            ("no header", harness.client.get("/protected")),
            (
                "wrong scheme",
                harness.client.get(
                    "/protected", headers={"Authorization": "Basic YWRhOnNlY3JldA=="}
                ),
            ),
            (
                "not a token",
                harness.client.get("/protected", headers={"Authorization": "Bearer nonsense"}),
            ),
            (
                "expired token",
                harness.client.get(
                    "/protected", headers={"Authorization": f"Bearer {expired.token}"}
                ),
            ),
            (
                "foreign secret",
                harness.client.get(
                    "/protected", headers={"Authorization": f"Bearer {foreign.token}"}
                ),
            ),
            (
                "alg none",
                harness.client.get(
                    "/protected", headers={"Authorization": f"Bearer {_alg_none_token()}"}
                ),
            ),
        ]

    def test_every_cause_is_401(self, harness: Harness) -> None:
        for name, response in self._failures(harness):
            assert response.status_code == 401, f"{name}: {response.status_code}"

    def test_every_cause_produces_the_same_problem_document(self, harness: Harness) -> None:
        # The oracle test. If any branch ever gains its own detail string, this
        # collapses to more than one element and fails.
        shapes = {
            (
                response.json()["type"],
                response.json()["title"],
                response.json()["detail"],
                response.json()["status"],
                tuple(sorted(response.json())),
            )
            for _, response in self._failures(harness)
        }
        assert len(shapes) == 1

    def test_every_cause_carries_the_bearer_challenge(self, harness: Harness) -> None:
        # RFC 9110 requires it on a 401, and it is what tells a client to refresh
        # rather than re-prompt for a password.
        for name, response in self._failures(harness):
            assert response.headers["WWW-Authenticate"] == "Bearer", name
            assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)

    def test_no_rejection_leaks_the_reason(self, harness: Harness) -> None:
        # The reason travels on the exception for the logs; none of those phrases
        # may travel to the client. (The shared detail does say "expired
        # credentials" — that word is in EVERY rejection, so it distinguishes
        # nothing. What must not appear is any phrasing specific to one cause.)
        leaks = (
            "no such account",
            "password mismatch",
            "token reuse",
            "family revoked",
            "malformed email",
            "unknown token",
            "no bearer credentials",
        )
        for name, response in self._failures(harness):
            body = response.text.lower()
            assert not any(leak in body for leak in leaks), name


class TestTheTenantSeam:
    def test_a_valid_token_becomes_the_principal(self, harness: Harness) -> None:
        # §5.8 layer 1: the subject of the token IS the tenant every repository
        # below will be scoped to.
        issued = harness.tokens.issue_access(subject="user-42")
        response = harness.client.get(
            "/protected", headers={"Authorization": f"Bearer {issued.token}"}
        )
        assert response.status_code == 200
        assert response.json()["user_id"] == "user-42"

    def test_a_refresh_secret_is_not_a_bearer_token(self, harness: Harness) -> None:
        # The two token shapes are not interchangeable: a refresh secret is an
        # opaque random string with no signature to verify, and presenting one
        # here must not authenticate anything.
        issued = _register(harness)
        response = harness.client.get(
            "/protected", headers={"Authorization": f"Bearer {issued['refresh_token']}"}
        )
        assert response.status_code == 401

    def test_the_header_is_case_insensitive_in_its_scheme(self, harness: Harness) -> None:
        # RFC 9110: the scheme is case-insensitive. Rejecting `bearer` would be a
        # spec violation that looks exactly like a credential failure.
        issued = harness.tokens.issue_access(subject="user-42")
        response = harness.client.get(
            "/protected", headers={"Authorization": f"bearer {issued.token}"}
        )
        assert response.status_code == 200


def _alg_none_token() -> str:
    """A syntactically valid JWT claiming it needs no signature."""
    now = datetime.now(UTC)
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(
        b"="
    )
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "sub": "attacker",
                "iss": "scholiast",
                "typ": "access",
                "iat": int(now.timestamp()),
                "exp": int((now + timedelta(hours=1)).timestamp()),
            }
        ).encode()
    ).rstrip(b"=")
    return f"{header.decode()}.{payload.decode()}."
