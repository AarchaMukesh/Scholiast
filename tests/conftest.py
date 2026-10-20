"""Test-wide configuration: no test reads the developer's `.env`.

`Settings` reads a `.env` in the working directory, which is how a local deployment
is configured - since D173 the repository root's carries
`QDRANT_HYBRID_COLLECTION`, so every process started from the checkout mirrors its
writes to the hybrid collection. A test that built `Settings(...)` would pick that
up too: its writes would be mirrored into the real `scholiast_chunks_hybrid`, and
its results would depend on one machine's deployment file rather than on the test.

So the file is switched off here, before any test builds a `Settings`. Real
environment variables still apply, as they do in CI, and a test that wants a
setting passes it explicitly. This module is imported before collection, which is
what makes it the place for this rather than a fixture: settings are also built at
import time and in module-scoped fixtures.
"""

from __future__ import annotations

from scholiast.config.settings import Settings

Settings.model_config["env_file"] = None
