"""The browser UI's guards: what can break silently without a browser to notice (D154, D155).

What fails silently here is third-party text becoming markup (a title from an
OpenAlex record running as script in a user's session), a link with a
`javascript:` scheme, the Content-Security-Policy quietly dropped or spread to
`/docs` (which it would break), and the page calling an endpoint or reading a
field the API no longer has. None of these fails a Python test on its own, and
the UI has no JavaScript test runner (D155), so these are read from the files.

The UI's behaviour itself is verified by a recorded live walkthrough (D155).
"""

from __future__ import annotations

import re
from functools import cache
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from scholiast.api import schemas
from scholiast.api.app import STATIC_DIR, create_app
from scholiast.api.ui import CONTENT_SECURITY_POLICY
from scholiast.config.settings import Settings

APP_JS = STATIC_DIR / "app.js"
INDEX = STATIC_DIR / "index.html"


@cache
def _js() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _js_code() -> str:
    """app.js with comments removed, so a rule about code is not tripped by prose."""
    without_blocks = re.sub(r"/\*.*?\*/", "", _js(), flags=re.DOTALL)
    return re.sub(r"(?m)^\s*//.*$|\s//.*$", "", without_blocks)


@cache
def _client() -> TestClient:
    return TestClient(create_app(Settings()))


class TestTheUIsHeaders:
    @pytest.mark.parametrize("path", ["/", "/index.html", "/app.js", "/app.css"])
    def test_every_ui_file_carries_the_policy(self, path: str) -> None:
        response = _client().get(path)

        assert response.status_code == HTTPStatus.OK
        assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        # Without it a browser runs a cached app.js after a deploy (the walkthrough).
        assert response.headers["cache-control"] == "no-cache"

    def test_the_policy_refuses_inline_script_and_framing(self) -> None:
        directives = dict(part.strip().split(" ", 1) for part in CONTENT_SECURITY_POLICY.split(";"))

        assert directives["default-src"] == "'none'"
        assert directives["script-src"] == "'self'"
        assert "unsafe-inline" not in CONTENT_SECURITY_POLICY
        assert directives["frame-ancestors"] == "'none'"

    @pytest.mark.parametrize("path", ["/docs", "/health/live"])
    def test_the_api_and_its_docs_do_not_carry_it(self, path: str) -> None:
        # /docs loads its scripts from a CDN; the policy would break it (D154).
        response = _client().get(path)

        assert "content-security-policy" not in response.headers


class TestThirdPartyTextNeverBecomesMarkup:
    def test_innerhtml_is_only_ever_cleared(self) -> None:
        writes = re.findall(r"\.innerHTML\s*=\s*([^;]+);", _js_code())

        assert writes, "the guard found no innerHTML writes at all; has the file moved?"
        assert all(w.strip() == '""' for w in writes), writes

    @pytest.mark.parametrize(
        "sink",
        [
            "insertAdjacentHTML",
            "outerHTML",
            "document.write",
            "eval(",
            "new Function",
            "createContextualFragment",
            "srcdoc",
            'setAttribute("on',
            'setAttribute("href',
            'setAttribute("style',
        ],
    )
    def test_no_markup_or_code_sink_is_used(self, sink: str) -> None:
        assert sink not in _js_code()

    def test_a_link_is_built_only_by_the_http_checking_helper(self) -> None:
        code = _js_code()
        helper = re.search(r"function safeLink\(.*?\n}\n", code, flags=re.DOTALL)
        assert helper is not None, "safeLink is gone"
        assert 'parsed.protocol !== "https:" && parsed.protocol !== "http:"' in helper.group()

        outside = code.replace(helper.group(), "")
        assert ".href =" not in outside, "a link was built without checking its scheme"
        assert 'createElement("a")' not in outside
        assert 'el("a"' not in outside

    def test_the_page_has_no_inline_script_style_or_handler(self) -> None:
        html = INDEX.read_text(encoding="utf-8")

        assert re.findall(r"<script\b[^>]*>", html) == ['<script src="/app.js" type="module">']
        assert re.search(r"<script\b[^>]*>\s*(?!</script>)\S", html) is None
        assert re.search(r"\sstyle\s*=", html) is None
        assert re.search(r"\son[a-z]+\s*=", html) is None


def _normalise(path: str) -> str:
    return re.sub(r"\{[^}]*\}|\$\{[^}]*\}", "{}", path)


class TestThePageAndTheApiAgree:
    def test_every_endpoint_the_page_calls_exists(self) -> None:
        routes = {_normalise(path) for path in _api_paths(create_app(Settings()).routes)}
        called = {_normalise(p) for p in re.findall(r"[`\"](/v1/[^`\"]*)[`\"]", _js())}

        assert called, "the guard found no API calls; has the file moved?"
        assert called <= routes, sorted(called - routes)

    @pytest.mark.parametrize(
        ("model", "path"),
        [
            (schemas.DiscoveryResults, "results.title"),
            (schemas.DiscoveryResults, "results.authors"),
            (schemas.DiscoveryResults, "results.year"),
            (schemas.DiscoveryResults, "results.venue"),
            (schemas.DiscoveryResults, "results.citation_count"),
            (schemas.DiscoveryResults, "results.abstract"),
            (schemas.DiscoveryResults, "results.members.source"),
            (schemas.DiscoveryResults, "results.add.source"),
            (schemas.DiscoveryResults, "results.add.source_id"),
            (schemas.DiscoveryResults, "results.open_access.url"),
            (schemas.DiscoveryResults, "results.open_access.kind"),
            (schemas.DiscoveryResults, "results.score"),
            (schemas.DiscoveryResults, "results.score_parts.relevance"),
            (schemas.DiscoveryResults, "results.score_parts.citation_impact"),
            (schemas.DiscoveryResults, "results.score_parts.recency"),
            (schemas.DiscoveryResults, "results.score_parts.imputed"),
            (schemas.DiscoveryResults, "ranking.relevance"),
            (schemas.DiscoveryResults, "ranking.version"),
            (schemas.DiscoveryResults, "ranking.weights.relevance"),
            (schemas.DiscoveryResults, "ranking.weights.citation_impact"),
            (schemas.DiscoveryResults, "ranking.weights.recency"),
            (schemas.DiscoveryResults, "excluded.year"),
            (schemas.DiscoveryResults, "excluded.citations"),
            (schemas.DiscoveryResults, "excluded.venue"),
            (schemas.DiscoveryResults, "excluded.unknown_citations"),
            (schemas.DiscoveryResults, "source_status.source"),
            (schemas.DiscoveryResults, "source_status.status"),
            (schemas.DiscoveryResults, "source_status.retry_after_seconds"),
            (schemas.DiscoveryResults, "partial"),
            (schemas.TokenPair, "access_token"),
            (schemas.TokenPair, "refresh_token"),
            (schemas.AddDiscoveredAccepted, "papers.deduplicated"),
        ],
    )
    def test_every_field_the_page_reads_is_one_the_api_returns(
        self, model: type[Any], path: str
    ) -> None:
        schema = model.model_json_schema()
        node: dict[str, Any] = schema
        for name in path.split("."):
            node = _resolve(schema, node)
            assert name in node.get("properties", {}), f"{model.__name__}.{path}"
            node = node["properties"][name]
        assert path.split(".")[-1] in _js(), f"the page no longer reads {path}"

    def test_the_search_body_the_page_sends_is_one_the_api_accepts(self) -> None:
        accepted = set(schemas.DiscoverySearch.model_json_schema()["properties"])
        sent = set(re.findall(r"body\.([a-z_]+)\s*=", _js())) | {"query", "sort"}

        assert sent <= accepted, sorted(sent - accepted)


def _api_paths(routes: Any) -> set[str]:
    """Every API path, including those inside included routers (`original_router`)."""
    found: set[str] = set()
    for route in routes or ():
        if isinstance(route, APIRoute):
            found.add(route.path)
        inner = getattr(route, "original_router", None)
        nested = getattr(inner, "routes", None) or getattr(route, "routes", None)
        if nested:
            found |= _api_paths(nested)
    return found


def _resolve(root: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Follow $ref, array items and nullable unions to the object that has properties."""
    while True:
        if "$ref" in node:
            node = root["$defs"][node["$ref"].rsplit("/", 1)[-1]]
        elif node.get("type") == "array":
            node = node["items"]
        elif "anyOf" in node:
            node = next(n for n in node["anyOf"] if n.get("type") != "null")
        else:
            return node


def test_the_static_directory_is_the_one_served() -> None:
    assert Path(STATIC_DIR, "index.html").is_file()


class TestEveryApiCallCarriesTheSession:
    def test_a_direct_fetch_is_either_the_session_wrapper_or_an_auth_call(self) -> None:
        # A call that skips `authFetch` gets a 401 in production and no refresh,
        # which no Python test would see (D154).
        code = _js_code()
        raw = [m.start() for m in re.finditer(r"(?<![A-Za-z])fetch\(", code)]
        assert raw, "the guard found no fetch calls; has the file moved?"
        for start in raw:
            argument = code[start + len("fetch(") :].lstrip()
            assert argument.startswith(("path,", '"/v1/auth/')), code[start : start + 80]

    def test_the_session_wrapper_sends_the_bearer_token(self) -> None:
        helper = re.search(r"function withAuth\(.*?\n}\n", _js_code(), flags=re.DOTALL)

        assert helper is not None, "withAuth is gone"
        assert 'headers.set("Authorization", `Bearer ${session.access}`)' in helper.group()
