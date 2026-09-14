"""What an upload must satisfy before anything is written (D55, NFR-SEC).

Two checks, both pure functions of the bytes, so they run at the edge before a
blob is stored, a row is created or a job is queued. A 60 MB file rejected
synchronously tells the user the same thing four steps earlier than a queued job
that fails asynchronously — and an upload that is not a PDF at all should never
reach the parser to find that out.

Everything requiring the document to be *opened* — encrypted, image-only, zero
extractable text — stays in the worker's pre-flight, where PyMuPDF lives. The
split is by what it costs to know, not by what is worth knowing.

`MAX_PDF_BYTES` lives here, and the pre-flight imports it, so the limit the code
enforces and the limit `FailureClass.TOO_LARGE.user_message` quotes cannot
drift apart.
"""

from __future__ import annotations

from scholiast.domain.errors import DomainError, FailureClass

#: NFR-P8.
MAX_PDF_BYTES = 50 * 1024 * 1024

#: Every PDF begins with this, per the specification. Checking it is what stops a
#: renamed .exe or .zip from being stored and handed to a parser — the file
#: extension is attacker-controlled and the magic bytes are not.
PDF_MAGIC = b"%PDF-"


class UploadRejected(DomainError):
    """An upload failed a check that can be made without opening the file.

    Carries a `FailureClass` for the same reason `DocumentUnparseable` does: the
    user-facing message and the retry decision both come from the class, and a
    bare exception collapses them into "something went wrong" (FR-I8, FR-I9).
    """

    def __init__(self, failure_class: FailureClass, detail: str) -> None:
        super().__init__(f"{failure_class.value}: {detail}")
        self.failure_class = failure_class
        self.detail = detail


def validate_upload(data: bytes, *, filename: str = "") -> None:
    """Raise `UploadRejected` unless these bytes could be a PDF of legal size.

    Size is checked first: a 2 GB body should be refused on its length rather
    than after inspecting its contents.
    """
    if len(data) > MAX_PDF_BYTES:
        raise UploadRejected(
            FailureClass.TOO_LARGE,
            f"{filename or 'upload'} is {len(data) / 1024 / 1024:.1f} MB, "
            f"limit is {MAX_PDF_BYTES // 1024 // 1024} MB",
        )
    if not data.startswith(PDF_MAGIC):
        # UNPARSEABLE rather than a new class: from the user's side this is the
        # same fact as a corrupt PDF — the file cannot be read, and retrying the
        # same bytes will not change that.
        raise UploadRejected(
            FailureClass.UNPARSEABLE,
            f"{filename or 'upload'} is not a PDF (missing %PDF- signature)",
        )


def provisional_title(filename: str) -> str:
    """The title a paper carries until its parse supplies a real one (D55).

    Nothing knows a paper's title before GROBID has run, and `Paper.title` is
    NOT NULL with a blank-title invariant. A visible filename is the honest
    placeholder: a fabricated title would be indistinguishable from a parsed one
    the moment it reached the UI.
    """
    stem = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith(".pdf"):
        stem = stem[: -len(".pdf")]
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem or "Untitled upload"
