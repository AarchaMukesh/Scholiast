"""Rate limiting and the ops-endpoint gate (NFR-SEC7, D112-D114).

Three things here would fail silently and be discovered by an incident:

  the wiring     a dependency every route has to remember is one some route will
                 forget, and nothing fails when it does — the request just goes
                 unlimited. Phases 5 and 7 both hit this trap. The completeness
                 test converts it into a test failure.

  the key        reading `X-Forwarded-For` would make the limiter bypassable with
                 one header (D113). A limiter that is bypassable reads as
                 protection, which is worse than none.

  the bound      the limiter's own map grows per caller, so an anonymous flood of
                 addresses would exhaust memory — the control becoming the denial
                 of service it exists to prevent.
"""

from __future__ import annotations

from http import HTTPStatus

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from scholiast.api import deps
from scholiast.api.app import create_app
from scholiast.api.errors import PROBLEM_CONTENT_TYPE, RateLimited, register_error_handlers
from scholiast.application.ports.tenancy import TenantContext
from scholiast.config.settings import Settings
from scholiast.infrastructure.security.ratelimit import (
    Decision,
    FixedWindowLimiter,
    Limit,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def limiter(*, max_keys: int = 1000) -> tuple[FixedWindowLimiter, FakeClock]:
    clock = FakeClock()
    return FixedWindowLimiter(max_keys=max_keys, clock=clock), clock


def hit(limiter_: FixedWindowLimiter, *, key: str = "k", times: int, allowance: int) -> Decision:
    decision = Decision(allowed=True)
    for _ in range(times):
        decision = limiter_.check(
            key=key, bucket="b", limit=Limit(requests=allowance, window_seconds=60.0)
        )
    return decision


class TestTheWindowCountsAndRolls:
    def test_requests_up_to_the_limit_are_allowed(self) -> None:
        subject, _ = limiter()
        assert hit(subject, times=3, allowance=3).allowed is True

    def test_the_next_one_is_refused(self) -> None:
        subject, _ = limiter()
        assert hit(subject, times=4, allowance=3).allowed is False

    def test_the_refusal_says_when_to_come_back(self) -> None:
        # The same courtesy this system expects from Groq (D103): a refusal that
        # states when to return costs nothing to honour.
        subject, clock = limiter()
        hit(subject, times=3, allowance=3)
        clock.advance(20.0)

        refused = hit(subject, times=1, allowance=3)

        assert refused.allowed is False
        assert refused.retry_after == pytest.approx(40.0)

    def test_the_window_rolls_and_the_allowance_returns(self) -> None:
        subject, clock = limiter()
        hit(subject, times=3, allowance=3)
        clock.advance(61.0)

        assert hit(subject, times=1, allowance=3).allowed is True

    def test_a_refused_request_does_not_extend_the_penalty(self) -> None:
        # Counting refusals would turn a rate limit into an escalating ban nobody
        # configured: a client retrying in a loop would never be let back in.
        subject, clock = limiter()
        hit(subject, times=10, allowance=3)  # seven of them refused
        clock.advance(61.0)

        assert hit(subject, times=1, allowance=3).allowed is True

    def test_buckets_are_independent(self) -> None:
        # Spending the QA allowance must not lock a caller out of uploading.
        subject, _ = limiter()
        for _ in range(3):
            subject.check(key="k", bucket="qa", limit=Limit(3, 60.0))

        other = subject.check(key="k", bucket="upload", limit=Limit(3, 60.0))

        assert other.allowed is True

    def test_callers_are_independent(self) -> None:
        subject, _ = limiter()
        hit(subject, key="a", times=3, allowance=3)

        assert hit(subject, key="b", times=1, allowance=3).allowed is True

    @pytest.mark.parametrize(("requests", "window"), [(0, 60.0), (-1, 60.0), (5, 0.0), (5, -1.0)])
    def test_an_impossible_limit_is_refused_at_construction(
        self, requests: int, window: float
    ) -> None:
        # A limit of zero would refuse the first request too: an outage wearing a
        # control's clothes, and a typo in configuration is how it would arrive.
        with pytest.raises(ValueError, match="at least one request"):
            Limit(requests=requests, window_seconds=window)


class TestTheLimiterIsNotItselfADenialOfService:
    def test_an_idle_callers_entry_is_reused_not_multiplied(self) -> None:
        """Expiry happens per key, in `check`, and is bounded by the cap.

        An earlier version pruned every entry eagerly on each call, which made the
        roll branch in `check` dead code — a mutation disabling it broke nothing,
        which is how that was found — and applied one call's window to every other
        bucket. Now a returning caller's stale entry is replaced rather than added
        to, so the map does not grow per idle caller.
        """
        subject, clock = limiter()
        for index in range(50):
            subject.check(key=f"k{index}", bucket="b", limit=Limit(5, 60.0))
        assert subject.tracked == 50

        clock.advance(61.0)
        for index in range(50):
            subject.check(key=f"k{index}", bucket="b", limit=Limit(5, 60.0))

        assert subject.tracked == 50

    def test_a_bucket_with_a_different_window_is_not_expired_against_another(self) -> None:
        # The latent bug the dead-code finding exposed. All three production
        # windows are sixty seconds, so an eager prune using the caller's window
        # was invisible — this fails against that implementation.
        subject, clock = limiter()
        subject.check(key="k", bucket="slow", limit=Limit(1, 600.0))
        clock.advance(61.0)

        # A short-window bucket is checked, which must not reset the long one.
        subject.check(key="k", bucket="fast", limit=Limit(1, 60.0))
        still_limited = subject.check(key="k", bucket="slow", limit=Limit(1, 600.0))

        assert still_limited.allowed is False

    def test_the_map_is_hard_capped(self) -> None:
        # An anonymous flood of source addresses must not grow memory without
        # bound. Eviction can forgive a limit, which is the wrong direction — but
        # reaching the cap costs far more requests than it forgives, and an
        # unbounded map is a hole anyone can drive.
        subject, _ = limiter(max_keys=10)

        for index in range(200):
            subject.check(key=f"k{index}", bucket="b", limit=Limit(5, 60.0))

        assert subject.tracked <= 10


class TestTheKeyCannotBeForged:
    def build(self, **settings: object) -> TestClient:
        app = FastAPI()
        register_error_handlers(app)
        app.state.settings = Settings(**settings)  # type: ignore[arg-type]
        app.state.rate_limiter = FixedWindowLimiter()

        @app.get("/peer", dependencies=[deps.AuthRateLimit])
        async def _peer() -> dict[str, bool]:
            return {"ok": True}

        return TestClient(app, raise_server_exceptions=False)

    def test_a_forwarded_for_header_does_not_mint_a_fresh_bucket(self) -> None:
        """D113, and the reason the header is never read.

        Nothing sits in front of this app, so honouring `X-Forwarded-For` would
        let any caller opt out of the limit by varying one header — and would let
        them attribute their traffic to somebody else's address. If the header
        were trusted, every request below would land in its own bucket and none
        would be refused.
        """
        client = self.build(rate_limit_auth_per_minute=3)

        statuses = [
            client.get("/peer", headers={"X-Forwarded-For": f"10.0.0.{index}"}).status_code
            for index in range(5)
        ]

        assert HTTPStatus.TOO_MANY_REQUESTS in statuses

    def test_the_limiter_can_be_turned_off(self) -> None:
        # Present so a load test measures the system rather than the limiter, and
        # so a deployment with a real limiter in front does not stack two.
        client = self.build(rate_limit_enabled=False, rate_limit_auth_per_minute=1)

        statuses = [client.get("/peer").status_code for _ in range(5)]

        assert set(statuses) == {HTTPStatus.OK}

    def test_the_configured_allowance_is_the_one_applied(self) -> None:
        # A limit captured at import would ignore the setting and nothing else
        # would notice — the mutation the completeness test cannot catch.
        client = self.build(rate_limit_auth_per_minute=1)

        first = client.get("/peer").status_code
        second = client.get("/peer").status_code

        assert (first, second) == (HTTPStatus.OK, HTTPStatus.TOO_MANY_REQUESTS)


class TestTheRefusalIsAProblemDocument:
    def build(self) -> TestClient:
        app = FastAPI()
        register_error_handlers(app)

        @app.get("/limited")
        async def _limited() -> None:
            raise RateLimited(retry_after=17.4)

        return TestClient(app, raise_server_exceptions=False)

    def test_it_is_429_with_the_rfc_9457_media_type(self) -> None:
        response = self.build().get("/limited")
        assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
        assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
        assert response.json()["type"].endswith("/rate-limited")

    def test_it_carries_retry_after_rounded_up(self) -> None:
        # Rounded UP and floored at 1: telling a client to come back in 0 seconds
        # invites an immediate retry that is refused again.
        assert self.build().get("/limited").headers["Retry-After"] == "18"

    def test_it_does_not_reveal_the_limit_or_what_remains(self) -> None:
        # Publishing the numbers would let a caller pace itself to just under
        # each bucket, which is the opposite of what the limit is for.
        body = self.build().get("/limited").json()
        assert "17" not in body["detail"]
        assert "limit" not in body["detail"].lower()


#: Every route that must carry a limiter, and which bucket it must use. Written
#: out rather than derived, so adding an expensive endpoint fails here until
#: somebody decides what its allowance is.
#:
#: `/v1/auth/logout` is deliberately absent: it takes a token the caller already
#: holds, revoking your own session is not an attack, and limiting it would leave
#: a caller unable to log out.
PROTECTED: dict[tuple[str, str], str] = {
    ("POST", "/v1/auth/register"): "auth",
    ("POST", "/v1/auth/login"): "auth",
    ("POST", "/v1/auth/refresh"): "auth",
    ("POST", "/v1/workspaces/{workspace_id}/papers/upload"): "upload",
    (
        "POST",
        "/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
    ): "qa",
}


def buckets_for(route: object) -> set[str]:
    """Which rate-limit buckets a route's dependency graph carries."""
    dependant = getattr(route, "dependant", None)
    found: set[str] = set()
    for dependency in getattr(dependant, "dependencies", []) or []:
        bucket = getattr(dependency.call, "rate_limit_bucket", None)
        if isinstance(bucket, str):
            found.add(bucket)
    return found


class TestEveryExpensiveRouteIsLimited:
    """NFR-SEC7's wiring, asserted rather than trusted (D112).

    The failure this prevents is silent: a new expensive endpoint arrives without
    a limiter and nothing breaks — the requests simply are not counted. Phase 2
    used the same instrument on the isolation suite, and phase 5's `app.routes`
    lesson applies: read the route table, and guard against the assertion passing
    vacuously.
    """

    def routes(self) -> dict[tuple[str, str], object]:
        """Every real route, walked recursively.

        **`app.routes` does not contain them at the top level.** An included
        router appears as an `_IncludedRouter` with no `.path`, holding its real
        `APIRoute`s inside — the trap phase 5 hit, and hit again here: the first
        version of this returned only the four docs routes and a mount, and the
        non-vacuity guard below was weak enough to pass anyway.
        """
        table: dict[tuple[str, str], object] = {}

        def walk(routes: object) -> None:
            for route in routes or ():  # type: ignore[union-attr]
                path = getattr(route, "path", None)
                methods = getattr(route, "methods", None)
                if isinstance(path, str) and methods:
                    for method in methods:
                        table[(method, path)] = route
                # An `_IncludedRouter` holds its real routes behind
                # `original_router`, not `routes` — the second layer of the same
                # trap. The prefix is already on those paths, because this app
                # sets prefixes on the routers rather than at include time.
                inner = getattr(route, "original_router", None)
                nested = getattr(inner, "routes", None) or getattr(route, "routes", None)
                if nested:
                    walk(nested)

        walk(create_app(Settings()).routes)
        return table

    def test_the_route_table_was_actually_read(self) -> None:
        """Non-vacuity, and the weak version of this let a real bug through.

        `len(routes) > len(PROTECTED)` passed against a table holding only
        `/docs` and `/openapi.json`. Naming a route that must be there is what
        makes the guard mean "the application's routes were found".
        """
        table = self.routes()
        assert ("GET", "/health/live") in table
        assert ("POST", "/v1/auth/logout") in table
        assert len(table) > len(PROTECTED) + 4

    @pytest.mark.parametrize(
        ("method", "path"), list(PROTECTED), ids=[f"{m} {p}" for m, p in PROTECTED]
    )
    def test_the_route_exists_and_carries_its_bucket(self, method: str, path: str) -> None:
        table = self.routes()
        assert (method, path) in table, "the route moved; update PROTECTED or the route"
        assert buckets_for(table[(method, path)]) == {PROTECTED[(method, path)]}

    def test_logout_is_deliberately_unlimited(self) -> None:
        # The negative control. Without it, a change that limited every auth route
        # would pass every test above while leaving a caller unable to log out.
        table = self.routes()
        assert buckets_for(table[("POST", "/v1/auth/logout")]) == set()


class TestTheOpsEndpointsHonourAScrapeToken:
    """D114, closing D110."""

    def build(self, token: str | None) -> TestClient:
        from scholiast.api.routers import health

        app = FastAPI()
        app.include_router(health.router)
        app.state.settings = Settings(metrics_token=token)
        app.state.sessions = None
        return TestClient(app, raise_server_exceptions=False)

    def test_no_token_configured_leaves_them_open(self) -> None:
        # The development default. Requiring a token unconditionally would break
        # `docker compose up` plus one command for an endpoint exposed to nobody.
        assert self.build(None).get("/metrics").status_code == HTTPStatus.OK

    def test_a_configured_token_is_required(self) -> None:
        response = self.build("s3cret").get("/metrics")
        assert response.status_code == HTTPStatus.UNAUTHORIZED
        # RFC 9110, and the same convention phase 1's handler uses.
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_the_right_token_is_accepted(self) -> None:
        response = self.build("s3cret").get("/metrics", headers={"Authorization": "Bearer s3cret"})
        assert response.status_code == HTTPStatus.OK

    @pytest.mark.parametrize(
        "header", ["Bearer wrong", "s3cret", "Basic s3cret", "Bearer ", "Bearer s3cre"]
    )
    def test_a_wrong_or_malformed_credential_is_refused(self, header: str) -> None:
        # Includes a prefix of the real token: a comparison that short-circuited
        # would accept nothing extra, but `compare_digest` is what stops the
        # length and prefix leaking to anyone willing to measure.
        response = self.build("s3cret").get("/metrics", headers={"Authorization": header})
        assert response.status_code == HTTPStatus.UNAUTHORIZED

    def test_readiness_is_gated_too(self) -> None:
        # It names which dependency is unhealthy, which is the weaker of the two
        # exposures D110 described.
        assert self.build("s3cret").get("/health/ready").status_code == (HTTPStatus.UNAUTHORIZED)

    def test_liveness_is_never_gated(self) -> None:
        # An orchestrator's restart decision must not need a credential, and it
        # answers the same thing whatever is wrong.
        assert self.build("s3cret").get("/health/live").status_code == HTTPStatus.OK


def test_the_dependency_objects_are_distinct_per_bucket() -> None:
    # Three `Depends` built from the same factory: if the module had reused one,
    # every route would share a bucket and the completeness test above would
    # still pass.
    assert isinstance(deps.AuthRateLimit, type(Depends(lambda: None)))
    assert deps.AuthRateLimit.dependency is not deps.QaRateLimit.dependency
    assert deps.QaRateLimit.dependency is not deps.UploadRateLimit.dependency


class TestTheUserKeyedLimitIsWiredToo:
    """The gap a mutation found: only the PEER-keyed path was covered.

    `TestTheKeyCannotBeForged` builds its route from `deps.AuthRateLimit`, which
    is peer-keyed — so hardcoding the allowance inside `user_rate_limit` left
    every test green. The QA and upload limits are the expensive ones, and they
    are the two that go through this factory.
    """

    def build(self, *, allowance: int, user: str = "u-1") -> TestClient:
        app = FastAPI()
        register_error_handlers(app)
        app.state.settings = Settings(rate_limit_qa_per_minute=allowance)
        app.state.rate_limiter = FixedWindowLimiter()

        # `user_rate_limit` declares `TenantDep`, so the tenant is overridden
        # rather than faked — which also proves the ordering claim: the principal
        # is resolved before the bucket is touched.
        app.dependency_overrides[deps.current_tenant] = lambda: TenantContext(user_id=user)

        @app.get("/qa", dependencies=[deps.QaRateLimit])
        async def _qa() -> dict[str, bool]:
            return {"ok": True}

        return TestClient(app, raise_server_exceptions=False)

    def test_the_configured_qa_allowance_is_the_one_applied(self) -> None:
        client = self.build(allowance=2)

        statuses = [client.get("/qa").status_code for _ in range(3)]

        assert statuses == [
            HTTPStatus.OK,
            HTTPStatus.OK,
            HTTPStatus.TOO_MANY_REQUESTS,
        ]

    def test_the_bucket_is_the_user_not_the_address(self) -> None:
        # Every request below comes from the same TestClient address, so if the
        # key were the peer these would share a bucket. Two users, one allowance
        # each, both served.
        first = self.build(allowance=1, user="alice")
        second = self.build(allowance=1, user="bob")
        # Share the limiter, the way one process does.
        second.app.state.rate_limiter = first.app.state.rate_limiter  # type: ignore[attr-defined]

        assert first.get("/qa").status_code == HTTPStatus.OK
        assert second.get("/qa").status_code == HTTPStatus.OK
        assert first.get("/qa").status_code == HTTPStatus.TOO_MANY_REQUESTS
