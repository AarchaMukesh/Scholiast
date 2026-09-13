"""PyMuPDF pre-flight: reject what GROBID would waste 3 seconds failing on.

D39 narrowed PyMuPDF's role. It is no longer the provenance source — TEI
coordinates supply page numbers at 100% coverage — but it remains better than
GROBID at one thing: deciding *quickly and locally* that a PDF is hopeless.

The three classes this catches map onto §4.4.1, and each was measured in M0
against a synthetic fixture, so the taxonomy is grounded rather than guessed:

  encrypted / corrupt        -> UNPARSEABLE   (_broken-encrypted.pdf)
  image-only or empty        -> SCANNED_IMAGE (_broken-scanned, _broken-empty)
  over the size limit        -> TOO_LARGE     (NFR-P8)

Running this first matters for more than latency: sending an encrypted PDF to
GROBID yields a 4xx whose body has to be reverse-engineered into a class, which
is exactly the stringly-typed error handling FR-I8 forbids.
"""

from __future__ import annotations

import fitz

from scholiast.application.ports.parsing import DocumentUnparseable
from scholiast.domain.errors import FailureClass
from scholiast.domain.upload import MAX_PDF_BYTES

__all__ = ["MAX_PDF_BYTES", "MIN_EXTRACTABLE_CHARS", "preflight"]

# NFR-P8. Re-exported from the domain rather than redeclared (D55): the edge
# check and this one must refuse the same byte, and two constants that agree by
# coincidence agree only until one of them is edited.

#: Below this many extractable characters a PDF has no usable text layer. M0
#: measured both pathological fixtures at exactly 0; the margin is for PDFs that
#: carry a handful of stray glyphs over an otherwise scanned page.
MIN_EXTRACTABLE_CHARS = 200


def preflight(pdf: bytes, *, filename: str = "") -> int:
    """Validate a PDF and return its page count.

    Raises `DocumentUnparseable` with the right `FailureClass` rather than
    returning a flag, because every caller would otherwise have to remember to
    check it — and the one that forgets sends a scan down the whole pipeline.
    """
    if len(pdf) > MAX_PDF_BYTES:
        raise DocumentUnparseable(
            FailureClass.TOO_LARGE,
            f"{filename or 'upload'} is {len(pdf) / 1024 / 1024:.1f} MB, "
            f"limit is {MAX_PDF_BYTES // 1024 // 1024} MB",
        )

    try:
        document = fitz.open(stream=pdf, filetype="pdf")
    except Exception as exc:
        raise DocumentUnparseable(
            FailureClass.UNPARSEABLE, f"could not open: {type(exc).__name__}: {exc}"[:200]
        ) from exc

    try:
        if document.needs_pass:
            # Encrypted is UNPARSEABLE, not TRANSIENT: no amount of retrying
            # supplies the password.
            raise DocumentUnparseable(FailureClass.UNPARSEABLE, "encrypted / password required")

        page_count = document.page_count
        if page_count == 0:
            raise DocumentUnparseable(FailureClass.UNPARSEABLE, "no pages")

        extractable = sum(len(page.get_text().strip()) for page in document)
        if extractable < MIN_EXTRACTABLE_CHARS:
            # An image-only PDF is a *different* failure from a corrupt one: OCR
            # would fix it (FR-I12, stretch), so the user gets told that rather
            # than "corrupted", which would be both wrong and unactionable.
            raise DocumentUnparseable(
                FailureClass.SCANNED_IMAGE,
                f"only {extractable} extractable characters across {page_count} page(s)",
            )
        return int(page_count)
    finally:
        document.close()
