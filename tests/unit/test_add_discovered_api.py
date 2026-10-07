"""`POST /v1/workspaces/{id}/papers`: what a client is told about an add (D133, D137).

The use case is a stub here, because the claims are about the HTTP surface: the
body cannot carry a URL or metadata, every refused paper is named, a source that
did not answer is a 503 that says when to come back, and a third party's error
text never reaches the client. The real use case runs against Postgres in the
integration suite.
"""

from __future__ import annotations

import uuid
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scholiast.api import deps
from scholiast.api.errors import PROBLEM_CONTENT_TYPE, register_error_handlers
from scholiast.api.routers import papers
from scholiast.application.ports.discovery import SourceFailure
from scholiast.application.ports.tenancy import TenantContext
from scholiast.application.use_cases.add_discovered import (
    AddedPaper,
    AddRefusal,
    AddRefused,
    DiscoveredRef,
    LookupOutcome,
    RefusedRef,
    SourceLookupFailed,
)
from scholiast.application.use_cases.add_paper import AddPaperResult
from scholiast.config.settings import Settings
from scholiast.domain.workspace import PaperState
from scholiast.infrastructure.security.ratelimit import FixedWindowLimiter

WORKSPACE = "22222222-2222-4222-8222-222222222222"
ADD = f"/v1/workspaces/{WORKSPACE}/papers"


class Stub:
    """Answers every add the same way, and records what it was asked."""

    def __init__(self, outcome: list[AddedPaper] | BaseException) -> None:
        self._outcome = outcome
        self.calls: list[list[DiscoveredRef]] = []

    async def execute(self, *, workspace_id: str, refs: list[DiscoveredRef]) -> list[AddedPaper]:
        self.calls.append(list(refs))
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def build(stub: Stub) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(papers.router)
    app.state.settings = Settings()
    app.state.rate_limiter = FixedWindowLimiter()
    app.dependency_overrides[deps.current_tenant] = lambda: TenantContext(user_id="u-1")
    app.dependency_overrides[deps.add_discovered_use_case] = lambda: stub
    return TestClient(app, raise_server_exceptions=False)


def body(*refs: tuple[str, str]) -> dict[str, list[dict[str, str]]]:
    return {"papers": [{"source": s, "source_id": i} for s, i in refs]}


def added(ref: DiscoveredRef, *, deduplicated: bool = False) -> AddedPaper:
    return AddedPaper(
        ref,
        AddPaperResult(
            paper_id=str(uuid.uuid4()),
            state=PaperState.QUEUED,
            job_id=None if deduplicated else str(uuid.uuid4()),
            deduplicated=deduplicated,
        ),
    )


class TestAnAcceptedAdd:
    def test_it_is_a_202_with_one_entry_per_paper_in_the_order_sent(self) -> None:
        first = DiscoveredRef("arxiv", "2004.04906v3")
        second = DiscoveredRef("openalex", "W3015883388")
        stub = Stub([added(first), added(second, deduplicated=True)])

        response = build(stub).post(
            ADD, json=body(("arxiv", "2004.04906v3"), ("openalex", "W3015883388"))
        )

        assert response.status_code == HTTPStatus.ACCEPTED
        entries = response.json()["papers"]
        assert [(e["source"], e["source_id"]) for e in entries] == [
            ("arxiv", "2004.04906v3"),
            ("openalex", "W3015883388"),
        ]
        assert entries[0]["state"] == "queued" and entries[0]["job_id"] is not None
        assert entries[1]["deduplicated"] is True and entries[1]["job_id"] is None
        assert stub.calls == [[first, second]]


class TestTheBodyCarriesIdentifiersAndNothingElse:
    def test_a_url_smuggled_into_a_paper_is_refused_before_anything_runs(self) -> None:
        # D133 and D115: the server never fetches a URL a client named.
        stub = Stub([])
        payload = {
            "papers": [
                {
                    "source": "arxiv",
                    "source_id": "2004.04906v3",
                    "url": "http://169.254.169.254/latest/meta-data/",
                }
            ]
        }

        response = build(stub).post(ADD, json=payload)

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert stub.calls == []

    def test_client_metadata_is_refused_rather_than_trusted(self) -> None:
        stub = Stub([])
        payload = {"papers": [{"source": "arxiv", "source_id": "2004.04906v3", "title": "Mine"}]}

        assert build(stub).post(ADD, json=payload).status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert stub.calls == []

    def test_a_batch_is_one_to_ten_papers(self) -> None:
        stub = Stub([])
        client = build(stub)

        too_many = body(*[("arxiv", f"2004.0490{i}") for i in range(11)])
        assert client.post(ADD, json=too_many).status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert client.post(ADD, json={"papers": []}).status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert stub.calls == []


class TestARefusedAdd:
    def test_every_refused_paper_is_named_with_what_to_do_about_it(self) -> None:
        refused = AddRefused(
            (
                RefusedRef(DiscoveredRef("arxiv", "2004.99999"), AddRefusal.NOT_FOUND),
                RefusedRef(DiscoveredRef("openalex", "W12"), AddRefusal.NO_FETCHABLE_PDF),
            )
        )

        response = build(Stub(refused)).post(
            ADD, json=body(("arxiv", "2004.99999"), ("openalex", "W12"))
        )

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
        problem = response.json()
        assert problem["type"] == "/problems/add-refused"
        assert [(p["source_id"], p["reason"]) for p in problem["papers"]] == [
            ("2004.99999", "not_found"),
            ("W12", "no_fetchable_pdf"),
        ]
        assert "Upload its PDF" in problem["papers"][1]["message"]

    def test_every_refusal_has_a_message(self) -> None:
        # A reason without a message would 500 inside the error handler.
        for reason in AddRefusal:
            refused = AddRefused((RefusedRef(DiscoveredRef("arxiv", "x"), reason),))
            response = build(Stub(refused)).post(ADD, json=body(("arxiv", "x")))
            assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, reason
            assert response.json()["papers"][0]["message"], reason


class TestASourceThatDidNotAnswer:
    def failed(self, *outcomes: LookupOutcome) -> TestClient:
        return build(Stub(SourceLookupFailed(outcomes)))

    def test_it_is_a_503_naming_each_sources_status(self) -> None:
        client = self.failed(
            LookupOutcome("arxiv"),
            LookupOutcome("openalex", SourceFailure.RATE_LIMITED, retry_after=7.0),
        )

        response = client.post(ADD, json=body(("openalex", "W12")))

        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert response.json()["source_status"] == [
            {"source": "arxiv", "status": "ok", "retry_after_seconds": None},
            {"source": "openalex", "status": "rate_limited", "retry_after_seconds": 7.0},
        ]

    def test_retry_after_is_when_the_slowest_failed_source_takes_work_again(self) -> None:
        # The add needs every source it asked, so the soonest wait would come back
        # to a source that is still refusing.
        client = self.failed(
            LookupOutcome("arxiv", SourceFailure.RATE_LIMITED, retry_after=2.2),
            LookupOutcome("openalex", SourceFailure.RATE_LIMITED, retry_after=7.0),
        )

        assert client.post(ADD, json=body(("openalex", "W12"))).headers["Retry-After"] == "7"

    def test_no_retry_after_when_a_failed_source_said_nothing(self) -> None:
        client = self.failed(
            LookupOutcome("arxiv", SourceFailure.TIMEOUT),
            LookupOutcome("openalex", SourceFailure.RATE_LIMITED, retry_after=7.0),
        )

        assert "Retry-After" not in client.post(ADD, json=body(("arxiv", "x"))).headers

    def test_a_third_partys_error_text_is_never_shown(self) -> None:
        leak = RuntimeError("upstream said: internal host db-7.openalex.internal refused")
        client = self.failed(LookupOutcome("openalex", SourceFailure.ERROR, error=leak))

        response = client.post(ADD, json=body(("openalex", "W12")))

        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "db-7" not in response.text and "upstream" not in response.text
