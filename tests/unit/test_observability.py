"""Logging, correlation and metrics (FR-O1, FR-O2, FR-O3, D107-D111).

What is worth testing here is not "does structlog work" but the three things that
would fail silently and be discovered by reading a log file months later:

  redaction     a processor that let an email or a chunk of paper text through
                looks identical to one that works, until the disclosure. D17 and
                D30 are why content is on the list and not just credentials.

  label shape   a metric labelled by raw path would put workspace and paper ids
                into an unauthenticated endpoint (D110) — the existence oracle
                FR-A2 closes — and grow the series count without bound. Nothing
                fails; `/metrics` just quietly becomes a leak.

  the gates     a readiness probe that reports ready without checking is "a green
                light nobody verified", and one that goes red on a *degradable*
                dependency takes the instance out of rotation for a condition
                §5.11 is designed to serve through.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scholiast.api.middleware import UNMATCHED_ROUTE, ObservabilityMiddleware
from scholiast.application.ports.llm import TokenUsage
from scholiast.infrastructure.observability import metrics
from scholiast.infrastructure.observability.logging import (
    CORRELATION_HEADER,
    REDACTED,
    _add_correlation_id,
    _redact,
    bind_correlation_id,
    current_correlation_id,
    new_correlation_id,
)


def redact(**fields: Any) -> dict[str, Any]:
    return _redact(None, "info", dict(fields))


class TestRedactionHappensAtTheBoundary:
    @pytest.mark.parametrize(
        "key",
        [
            "password",
            "password_hash",
            "access_token",
            "refresh_token",
            "Authorization",
            "api_key",
            "GROQ_API_KEY",
            "jwt_secret",
            "credential",
        ],
    )
    def test_credentials_are_replaced(self, key: str) -> None:
        assert redact(**{key: "hunter2"})[key] == REDACTED

    @pytest.mark.parametrize("key", ["email", "question", "answer", "prompt", "system", "user"])
    def test_content_and_pii_are_replaced(self, key: str) -> None:
        # D17/D30: this system ingests possibly-unpublished manuscripts, so a
        # logged chunk of paper text is a disclosure rather than a debugging aid.
        # `system` and `user` are `LLMProvider.stream`'s kwargs — the whole
        # prompt, question plus every retrieved chunk.
        assert redact(**{key: "sensitive"})[key] == REDACTED

    def test_the_key_survives_so_the_record_stays_readable(self) -> None:
        # `email: "[redacted]"` still says the line was about a specific user.
        # Dropping the field would make a redacted record indistinguishable from
        # one that never had it.
        assert "email" in redact(email="a@b.test")

    @pytest.mark.parametrize(
        "key", ["context", "next_visible_at", "job_id", "paper_id", "failure_class", "attempts"]
    )
    def test_operational_fields_are_not_over_redacted(self, key: str) -> None:
        # The negative control, and the reason the content list is exact rather
        # than substring-matched: "text" as a substring would swallow "context",
        # and a redactor that ate the diagnostics would be quietly useless.
        assert redact(**{key: "value"})[key] == "value"

    def test_nested_mappings_are_walked(self) -> None:
        cleaned = redact(outer={"password": "x", "job_id": "j1"})
        assert cleaned["outer"] == {"password": REDACTED, "job_id": "j1"}


class TestTheCorrelationIdIsAmbient:
    def test_it_is_absent_rather_than_null_when_unbound(self) -> None:
        # A `correlation_id: null` on every line from the eval harness would be
        # noise in every record to describe a property of none of them.
        bind_correlation_id(None)
        assert "correlation_id" not in _add_correlation_id(None, "info", {})

    def test_a_bound_id_reaches_the_event(self) -> None:
        bind_correlation_id("abc123")
        try:
            assert _add_correlation_id(None, "info", {})["correlation_id"] == "abc123"
        finally:
            bind_correlation_id(None)

    def test_a_fresh_id_is_url_and_grep_safe(self) -> None:
        generated = new_correlation_id()
        assert generated.isalnum()
        assert len(generated) == 32


class TestMetricLabelsCarryNoIdentifiers:
    """D110. The privacy property and the cardinality property are the same rule."""

    @pytest.mark.parametrize(
        ("status", "expected"), [(200, "2xx"), (404, "4xx"), (403, "4xx"), (500, "5xx")]
    )
    def test_status_is_bucketed_by_class(self, status: int, expected: str) -> None:
        # A 404 and a 403 differing in the metric would reintroduce at the
        # metrics layer the existence oracle FR-A2 closes at the HTTP layer.
        assert metrics._status_class(status) == expected

    def test_unreported_usage_is_not_counted(self) -> None:
        """`reported=False` means nobody counted, which is not "spent nothing".

        **Non-zero counts on purpose**, and a mutation run is why. The first
        version of this passed `TokenUsage(0, 0, reported=False)` and asserted
        the counter was unchanged — which it is either way, because incrementing
        by zero changes nothing. Deleting the `reported` guard left the test
        green. Only a value that WOULD move the counter can tell the guard from
        its absence.

        The pairing is defensive rather than reachable today: the port's
        convention is zeros-when-unreported, so this combination should not
        arise. That is exactly why it needs a test — nothing else exercises it.
        """
        counter = metrics.GENERATION_TOKENS.labels(provider="p", direction="prompt")
        before = counter._value.get()

        metrics.record_generation(
            provider="p", usage=TokenUsage(prompt_tokens=11, completion_tokens=13, reported=False)
        )

        assert counter._value.get() == before

    def test_a_missing_provider_records_nothing(self) -> None:
        # `generation_unavailable` leaves provider None. Labelling that "None"
        # would create a series for the absence of a provider.
        metrics.record_generation(provider=None, usage=TokenUsage(1, 2))
        assert 'provider="None"' not in metrics.render().decode()


def build_client() -> TestClient:
    app = FastAPI()

    @app.get("/v1/things/{thing_id}")
    async def _thing(thing_id: str) -> dict[str, str]:
        return {"seen": thing_id, "correlation_id": current_correlation_id() or ""}

    @app.get("/boom")
    async def _boom() -> None:
        raise RuntimeError("deliberate")

    app.add_middleware(ObservabilityMiddleware)
    return TestClient(app, raise_server_exceptions=False)


class TestTheMiddlewareCorrelatesAndLabels:
    def test_a_fresh_id_is_minted_and_echoed(self) -> None:
        response = build_client().get("/v1/things/abc")
        assert response.status_code == 200
        assert response.headers[CORRELATION_HEADER]
        # The id the handler saw is the id the caller is told about; otherwise a
        # log line and a response could not be joined.
        assert response.json()["correlation_id"] == response.headers[CORRELATION_HEADER]

    def test_a_safe_inbound_id_is_reused(self) -> None:
        response = build_client().get("/v1/things/abc", headers={CORRELATION_HEADER: "req-42_A"})
        assert response.headers[CORRELATION_HEADER] == "req-42_A"

    @pytest.mark.parametrize(
        "hostile",
        ["with space", "semi;colon", 'quote"mark', "x" * 65, "", "tab\tchar"],
    )
    def test_an_unsafe_inbound_id_is_replaced_not_echoed(self, hostile: str) -> None:
        # Echoing arbitrary header content into a response and into every log
        # line is how a caller writes into someone else's logs. The request is
        # not refused — only its label is — because failing an upload over a
        # malformed debug header would be the worse outcome.
        response = build_client().get("/v1/things/abc", headers={CORRELATION_HEADER: hostile})
        assert response.status_code == 200
        assert response.headers[CORRELATION_HEADER] != hostile

    def test_the_route_label_is_the_template_not_the_path(self) -> None:
        # The load-bearing one (D110). `/v1/things/{thing_id}`, never
        # `/v1/things/<a real id>`.
        build_client().get("/v1/things/secret-workspace-id")
        exposition = metrics.render().decode()
        assert 'route="/v1/things/{thing_id}"' in exposition
        assert "secret-workspace-id" not in exposition

    def test_an_unmatched_request_cannot_mint_a_label(self) -> None:
        # Otherwise a 404 probe writes one series per URL it tries.
        build_client().get("/no/such/route/xyzzy")
        exposition = metrics.render().decode()
        assert f'route="{UNMATCHED_ROUTE}"' in exposition
        assert "xyzzy" not in exposition

    def test_an_unhandled_error_is_still_recorded(self) -> None:
        # A 500 that vanished from the metrics would make the error rate in RED
        # report zero during exactly the incident it exists for.
        build_client().get("/boom")
        assert 'route="/boom",status="5xx"' in metrics.render().decode()


class TestTheExpositionIsWellFormed:
    def test_it_renders_and_carries_the_prometheus_content_type(self) -> None:
        assert metrics.CONTENT_TYPE.startswith("text/plain")
        assert b"scholiast_http_requests_total" in metrics.render()

    def test_gauges_are_named_without_a_total_suffix(self) -> None:
        # Prometheus convention: `_total` means counter. A gauge wearing it would
        # be summed by a rate() that has no business summing it.
        exposition = metrics.render().decode()
        for line in exposition.splitlines():
            if line.startswith("# TYPE ") and line.endswith(" gauge"):
                assert not line.split()[2].endswith("_total"), line


class TestBreakerStateIsReportedWithoutADialTone:
    def test_a_breaker_state_becomes_a_gauge(self) -> None:
        class FakeBreaker:
            name = "fakeprov"
            state = "open"

        metrics.refresh_breaker_states([FakeBreaker()])
        assert 'scholiast_provider_breaker_state{provider="fakeprov"} 2.0' in (
            metrics.render().decode()
        )

    def test_a_bare_provider_without_a_breaker_is_skipped_not_fatal(self) -> None:
        # Test apps and the auth-only lifespan wire fakes with no `state`. A
        # refresh that raised on them would make `/metrics` a liability in
        # exactly the configuration used to test it.
        class Bare:
            name = "bare"

        metrics.refresh_breaker_states([Bare()])
        assert 'provider="bare"' not in metrics.render().decode()

    def test_a_non_list_is_tolerated(self) -> None:
        metrics.refresh_breaker_states(None)


class TestLogOutputIsJson:
    def test_a_configured_logger_emits_one_json_object_per_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # FR-O1's actual claim. structlog's default renderer is built for a human
        # reading a terminal; a collector needs a document per line.
        import structlog

        from scholiast.infrastructure.observability.logging import (
            configure_logging,
            reset_for_tests,
        )

        reset_for_tests()
        configure_logging(level="INFO", json_format=True)
        bind_correlation_id("cid-1")
        try:
            structlog.get_logger("test").info("probe.event", job_id="j1", password="leak")
        finally:
            bind_correlation_id(None)

        line = capsys.readouterr().out.strip().splitlines()[-1]
        record = json.loads(line)
        assert record["event"] == "probe.event"
        assert record["job_id"] == "j1"
        assert record["correlation_id"] == "cid-1"
        # Redaction is in the same chain, so it holds for real output and not
        # only for the processor called directly.
        assert record["password"] == REDACTED
        assert record["level"] == "info"
        assert record["timestamp"].endswith("Z")


class FakeSession:
    """Answers `SELECT 1` and the metric aggregates, or refuses to connect."""

    def __init__(self, *, fails: bool) -> None:
        self._fails = fails

    async def execute(self, statement: object, *args: object, **kwargs: object) -> object:
        if self._fails:
            raise RuntimeError("connection refused")
        return _FakeResult()


class _FakeResult:
    def all(self) -> list[object]:
        return []

    def scalar_one(self) -> int:
        return 0


class FakeSessions:
    def __init__(self, *, fails: bool = False) -> None:
        self._fails = fails

    def __call__(self) -> FakeSessions:
        return self

    async def __aenter__(self) -> FakeSession:
        if self._fails:
            raise RuntimeError("connection refused")
        return FakeSession(fails=False)

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeIndex:
    def __init__(self, *, reachable: bool) -> None:
        self._reachable = reachable

    async def ping(self) -> bool:
        return self._reachable


def ops_client(
    *, db_ok: bool = True, index: object | None = None, providers: object | None = None
) -> TestClient:
    from scholiast.api.routers import health

    app = FastAPI()
    app.include_router(health.router)
    app.state.sessions = FakeSessions(fails=not db_ok)
    if index is not None:
        app.state.index = index
    if providers is not None:
        app.state.providers = providers
    app.add_middleware(ObservabilityMiddleware)
    return TestClient(app, raise_server_exceptions=False)


class TestReadinessGatesOnlyOnWhatCannotDegrade:
    def test_a_healthy_system_is_ready(self) -> None:
        response = ops_client(index=FakeIndex(reachable=True)).get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_postgres_down_is_503(self) -> None:
        # §5.11: "no meaningful degradation for the source of truth".
        response = ops_client(db_ok=False, index=FakeIndex(reachable=True)).get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["postgres"] == "unavailable"

    def test_qdrant_down_is_503(self) -> None:
        # Losing it loses retrieval entirely — both arms live there (D105) — so
        # an instance without it cannot answer at all.
        response = ops_client(index=FakeIndex(reachable=False)).get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["qdrant"] == "unavailable"

    def test_a_dead_provider_does_not_make_the_instance_unready(self) -> None:
        """The one that makes the degradation ladder possible.

        §5.11 degrades a generation outage to evidence-only. A readiness probe
        that went red would pull this instance out of rotation, so no instance
        would ever be routed to while degraded and the rung could never fire in a
        load-balanced deployment.
        """

        class OpenBreaker:
            name = "ollama"
            state = "open"

        response = ops_client(index=FakeIndex(reachable=True), providers=[OpenBreaker()]).get(
            "/health/ready"
        )

        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["checks"]["provider:ollama"] == "open"

    def test_an_unconfigured_index_is_not_reported_as_unhealthy(self) -> None:
        # The auth-only and isolation lifespans wire no index. Calling that
        # "unavailable" would make those apps permanently not-ready.
        response = ops_client().get("/health/ready")
        assert response.status_code == 200
        assert response.json()["checks"]["qdrant"] == "not_configured"


class TestTheMetricsEndpointSurvivesAnIncident:
    def test_it_serves_the_exposition_format(self) -> None:
        response = ops_client(index=FakeIndex(reachable=True)).get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "scholiast_http_requests_total" in response.text

    def test_a_postgres_outage_still_returns_the_process_metrics(self) -> None:
        # A monitoring endpoint that fails completely during an incident is a
        # monitoring endpoint that is absent during every incident (D111).
        response = ops_client(db_ok=False, index=FakeIndex(reachable=True)).get("/metrics")
        assert response.status_code == 200
        assert "scholiast_http_requests_total" in response.text

    def test_the_route_label_for_metrics_itself_is_a_template(self) -> None:
        client = ops_client(index=FakeIndex(reachable=True))
        client.get("/metrics")
        assert 'route="/metrics"' in client.get("/metrics").text


class TestStaleGaugeLabelsAreCleared:
    """A label that stops appearing must stop reporting, not freeze.

    Without the `clear()` in `refresh_work_gauges`, an emptied queue would go on
    reporting the depth it had when it drained — the exact failure D111 says a
    pushed gauge has, reintroduced in the pulled one.
    """

    async def test_a_state_that_disappears_stops_being_reported(self) -> None:
        metrics.QUEUE_DEPTH.labels(state="queued").set(7)
        assert 'scholiast_ingestion_queue_depth{state="queued"} 7.0' in (metrics.render().decode())

        # The fake returns no rows, i.e. an empty queue.
        await metrics.refresh_work_gauges(FakeSessions())

        assert 'scholiast_ingestion_queue_depth{state="queued"}' not in (metrics.render().decode())
