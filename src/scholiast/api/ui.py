"""The browser UI's files, served with the headers third-party text calls for (D154).

Phase 9 is the first time a title, abstract or link written by whoever deposited
a record at OpenAlex or arXiv reaches a browser. The page renders all of it as
text; these headers are the second line, so a mistake in the page is not a script
running in a user's session.

**Only the UI's files carry them.** FastAPI's `/docs` loads its scripts from a CDN,
so a site-wide policy would break it, and JSON responses gain nothing from one.
Subclassing the static mount rather than adding a middleware keeps the scope
exactly "what the UI serves", with no path list to fall out of date.
"""

from __future__ import annotations

from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

#: D154. `default-src 'none'` and then only what the page uses: its own script, its
#: own stylesheet, calls to its own API. No inline script or style, no frames, no
#: `<base>` rewriting, and forms post only to this origin.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
)

UI_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    # A file is what its Content-Type says, never what its bytes look like.
    "X-Content-Type-Options": "nosniff",
    # A third-party link opened from a result learns nothing about this page.
    "Referrer-Policy": "no-referrer",
    # Revalidate on every load (the ETag makes an unchanged file a cheap 304).
    # Without it a browser caches the page's script by heuristic, and after a
    # deploy runs the old app.js against the new API; the walkthrough found this.
    "Cache-Control": "no-cache",
}


class UIFiles(StaticFiles):
    """`StaticFiles` whose every response carries `UI_HEADERS`."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(UI_HEADERS)
        return response
