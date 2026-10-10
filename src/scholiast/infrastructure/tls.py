"""The one certificate-verifying TLS context outbound requests share (D122, D129).

Building one loads the CA bundle: 183 ms on the development machine, and synchronous,
so a context per request blocks the event loop for every request in the process.
Shared, a client builds in 0.1 ms. An `SSLContext` is safe to share.
"""

from __future__ import annotations

import functools
import ssl

import httpx


@functools.cache
def tls_context() -> ssl.SSLContext:
    """Built on first use and kept; the API warms it at startup (D122)."""
    return httpx.create_ssl_context()
