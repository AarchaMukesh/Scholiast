"""Problem documents at the HTTP boundary (§5.5, NFR-U2, D65).

Hermetic: a bare app with routes that raise, so the mapping is tested without a
database. What matters here is not that errors become responses — it is *which*
response, because the status code is the only part a client can branch on
without parsing prose.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scholiast.api.errors import PROBLEM_CONTENT_TYPE, register_error_handlers
from scholiast.api.schemas import CreateWorkspace
from scholiast.domain.errors import (
    DomainError,
    FailureClass,
    IllegalStateTransition,
    InvariantViolation,
    WorkspaceCapacityExceeded,
    WorkspaceNotAvailable,
    WorkspaceNotQueryable,
)
from scholiast.domain.upload import UploadRejected
from scholiast.domain.workspace import PaperState


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/raise/{kind}")
    async def _raise(kind: str) -> None:
        raise {
            "not-found": WorkspaceNotAvailable("ws-1"),
            "not-queryable": WorkspaceNotQueryable(3),
            "capacity": WorkspaceCapacityExceeded(100),
            "too-large": UploadRejected(FailureClass.TOO_LARGE, "60 MB"),
            "not-pdf": UploadRejected(FailureClass.UNPARSEABLE, "missing %PDF-"),
            "scanned": UploadRejected(FailureClass.SCANNED_IMAGE, "no text layer"),
            "illegal": IllegalStateTransition(
                "PaperMembership(p)", PaperState.INDEXED, PaperState.QUEUED
            ),
            "invariant": InvariantViolation("chunk text cannot be blank"),
            "domain": DomainError("some other rule"),
        }[kind]

    @app.post("/validated")
    async def _validated(body: CreateWorkspace) -> dict[str, str]:
        return {"name": body.name}

    return TestClient(app, raise_server_exceptions=False)


class TestEveryErrorIsAProblemDocument:
    @pytest.mark.parametrize(
        "kind",
        [
            "not-found",
            "not-queryable",
            "capacity",
            "too-large",
            "not-pdf",
            "illegal",
            "invariant",
            "domain",
        ],
    )
    def test_the_media_type_and_required_members_are_present(
        self, client: TestClient, kind: str
    ) -> None:
        # RFC 9457 requires the media type; without it a client cannot tell a
        # problem document from whatever else a proxy might have returned.
        response = client.get(f"/raise/{kind}")
        assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
        body = response.json()
        assert body["type"].startswith("/problems/")
        assert body["title"] and body["detail"]
        assert body["status"] == response.status_code

    def test_the_type_is_stable_per_failure_not_per_occurrence(self, client: TestClient) -> None:
        # §5.5 asks for "a stable `type` taxonomy": the URI identifies the kind
        # of failure, so a client can branch on it. Two occurrences of the same
        # failure must agree.
        first = client.get("/raise/not-found").json()["type"]
        second = client.get("/raise/not-found").json()["type"]
        assert first == second == "/problems/workspace-not-found"


class TestTheStatusCodeCarriesTheMeaning:
    def test_an_unreachable_workspace_is_404_never_403(self, client: TestClient) -> None:
        # FR-A2. A 403 confirms the id is real, which is the existence leak
        # §5.8's layer 1 exists to prevent — so a workspace someone else owns
        # and one that never existed must produce the same response.
        assert client.get("/raise/not-found").status_code == 404

    def test_nothing_indexed_yet_is_409_not_400(self, client: TestClient) -> None:
        # The request is well-formed and will succeed once ingestion finishes,
        # so the client should retry rather than rewrite it.
        response = client.get("/raise/not-queryable")
        assert response.status_code == 409
        assert response.json()["pending"] == 3

    def test_an_oversized_upload_is_413(self, client: TestClient) -> None:
        assert client.get("/raise/too-large").status_code == 413

    def test_a_non_pdf_is_415_not_413(self, client: TestClient) -> None:
        # The two upload failures are different actions for the user — shrink
        # the file, or send a different one. Collapsing both to 400 would throw
        # away the only thing they could act on.
        assert client.get("/raise/not-pdf").status_code == 415
        assert client.get("/raise/scanned").status_code == 415

    def test_a_broken_invariant_is_500_and_says_nothing(self, client: TestClient) -> None:
        # A violated invariant is a bug, not a bad request. NFR-SEC8 and NFR-U2
        # agree the message must not reach the client.
        response = client.get("/raise/invariant")
        assert response.status_code == 500
        assert "chunk text cannot be blank" not in response.text

    def test_an_unclassified_domain_error_is_400(self, client: TestClient) -> None:
        # `DomainError` means the request was invalid and retrying it unchanged
        # will fail identically.
        assert client.get("/raise/domain").status_code == 400


class TestTheUserFacingHalfIsWhatReachesTheUser:
    def test_an_upload_rejection_shows_the_actionable_message(self, client: TestClient) -> None:
        # NFR-U2: "actionable and specific", never a stack trace. §4.4.1's
        # message says what to do; the diagnostic says what broke, and they are
        # different fields on purpose.
        body = client.get("/raise/too-large").json()
        assert body["detail"] == FailureClass.TOO_LARGE.user_message
        assert body["failure_class"] == "too_large"
        assert body["diagnostic"] == "60 MB"


class TestValidationFailuresAreProblemsToo:
    def test_a_malformed_body_is_a_problem_document(self, client: TestClient) -> None:
        # FastAPI's default is a bare array under `detail`, which is not a
        # problem document — so one response shape would have been the exception.
        response = client.post("/validated", json={})
        assert response.status_code == 422
        assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)
        assert response.json()["type"] == "/problems/validation-failed"

    def test_an_unknown_field_is_rejected_not_ignored(self, client: TestClient) -> None:
        # NFR-SEC2's reject-by-default. A client sending {"nmae": "typo"} should
        # be told, not given a workspace called nothing.
        response = client.post("/validated", json={"name": "ok", "nmae": "typo"})
        assert response.status_code == 422

    def test_the_offending_fields_are_named(self, client: TestClient) -> None:
        body = client.post("/validated", json={}).json()
        assert body["errors"]
        assert any("name" in e["location"] for e in body["errors"])
