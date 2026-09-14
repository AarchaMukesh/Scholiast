"""Edge validation of an upload (D55, NFR-SEC, FR-I2).

These checks exist to stop work, not to describe files. Each test therefore
names the thing that would happen without it — a stored blob, a queued job and
an asynchronous failure for a fact that was knowable in microseconds.
"""

from __future__ import annotations

import pytest

from scholiast.domain.errors import FailureClass
from scholiast.domain.upload import (
    MAX_PDF_BYTES,
    UploadRejected,
    provisional_title,
    validate_upload,
)


class TestOversizeUploadsAreRefusedBeforeAnythingIsWritten:
    def test_a_file_over_the_limit_is_rejected(self) -> None:
        with pytest.raises(UploadRejected) as caught:
            validate_upload(b"%PDF-" + b"x" * MAX_PDF_BYTES, filename="huge.pdf")
        assert caught.value.failure_class is FailureClass.TOO_LARGE

    def test_the_limit_itself_is_accepted(self) -> None:
        # Positive control: an off-by-one here would reject legal uploads and
        # every "rejects oversize" test above would still pass.
        validate_upload(b"%PDF-" + b"x" * (MAX_PDF_BYTES - 5))

    def test_the_message_quotes_the_limit_the_code_enforces(self) -> None:
        # Prevents the drift D55 exists to stop: a user told "the limit is 50 MB"
        # by a message while the code enforces something else.
        assert str(MAX_PDF_BYTES // 1024 // 1024) in FailureClass.TOO_LARGE.user_message


class TestNonPdfUploadsNeverReachTheParser:
    @pytest.mark.parametrize(
        "payload",
        [
            b"PK\x03\x04zip content",
            b"MZ\x90\x00executable",
            b"<!DOCTYPE html><html></html>",
            b"",
        ],
    )
    def test_a_file_without_the_pdf_signature_is_rejected(self, payload: bytes) -> None:
        # The extension is attacker-controlled; the magic bytes are not. Without
        # this, a renamed archive is stored, queued, and only found out about by
        # PyMuPDF a minute later.
        with pytest.raises(UploadRejected) as caught:
            validate_upload(payload, filename="paper.pdf")
        assert caught.value.failure_class is FailureClass.UNPARSEABLE

    def test_a_real_pdf_signature_passes(self) -> None:
        validate_upload(b"%PDF-1.7\nrest of the file")

    def test_size_is_judged_before_content(self) -> None:
        # A 2 GB body should be refused on its length rather than after its
        # contents are inspected.
        with pytest.raises(UploadRejected) as caught:
            validate_upload(b"not a pdf at all" * MAX_PDF_BYTES)
        assert caught.value.failure_class is FailureClass.TOO_LARGE


class TestProvisionalTitleIsVisiblyProvisional:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("attention-is-all-you-need.pdf", "attention is all you need"),
            ("C:\\papers\\lost_in_the_middle.PDF", "lost in the middle"),
            ("/home/ada/papers/bert.pdf", "bert"),
            ("bert.pdf", "bert"),
        ],
    )
    def test_the_filename_becomes_the_title(self, filename: str, expected: str) -> None:
        assert provisional_title(filename) == expected

    def test_a_nameless_upload_still_gets_a_title(self) -> None:
        # `Paper.title` has a blank-title invariant, so returning "" here would
        # turn a missing filename into a 500 at the entity boundary.
        assert provisional_title("") == "Untitled upload"
        assert provisional_title(".pdf") == "Untitled upload"
