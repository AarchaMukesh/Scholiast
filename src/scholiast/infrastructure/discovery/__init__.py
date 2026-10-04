"""Academic source adapters behind `AcademicSource` (D118, D119): OpenAlex and arXiv."""

from __future__ import annotations

from scholiast.infrastructure.tls import tls_context

__all__ = ["USER_AGENT", "tls_context"]

#: Sent to every academic source. Both ask clients to identify themselves, and a
#: link back is what lets an operator reach us before blocking us.
USER_AGENT = "scholiast (+https://github.com/AarchaMukesh/Scholiast)"
