"""Academic source adapters behind `AcademicSource` (D118, D119): OpenAlex and arXiv."""

from __future__ import annotations

import functools
import ssl

import httpx

#: Sent to every academic source. Both ask clients to identify themselves, and a
#: link back is what lets an operator reach us before blocking us.
USER_AGENT = "scholiast (+https://github.com/AarchaMukesh/Scholiast)"


@functools.cache
def tls_context() -> ssl.SSLContext:
    """The one certificate-verifying TLS context every discovery request shares (D122).

    Building one loads the CA bundle: 183 ms on the development machine, and
    synchronous, so a context per search blocked the event loop for every request in
    the process. Shared, a search builds its client in 0.1 ms and still opens the
    fresh connection addendum A2 measured. An `SSLContext` is safe to share.
    """
    return httpx.create_ssl_context()
