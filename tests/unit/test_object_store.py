"""Local object store (D49, §5.9).

Three properties are worth a test each, and the key-safety one is worth several:
a key reaches this adapter from `papers.tei_key` — a database column — so it is
untrusted input the moment anything writes to that column, and a traversal here
would let a paper row address arbitrary files on the worker's disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scholiast.application.ports.storage import ObjectNotFound, source_key, tei_key
from scholiast.infrastructure.storage.local import LocalObjectStore, UnsafeObjectKey


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "objects")


class TestKeysAreScopedToThePaperThatOwnsThem:
    def test_two_papers_with_identical_bytes_get_different_keys(self) -> None:
        # The D49 property: content-addressed keys would give two tenants'
        # byte-identical private uploads one shared blob, and deleting one
        # paper would take the other's bytes with it — the §4.4.2 carve-out
        # defeated in exactly the case it exists for.
        assert source_key("paper-a") != source_key("paper-b")

    def test_the_tei_key_carries_the_parser_version(self) -> None:
        # Without this, upgrading GROBID overwrites the TEI that produced
        # today's chunks and a re-chunk ablation silently re-baselines (D43).
        first = tei_key("p1", "grobid-crf-0.8.2.1/tei/1.1.0")
        second = tei_key("p1", "grobid-crf-0.9.0/tei/2.0.0")
        assert first != second

    def test_a_path_shaped_version_does_not_become_directories(self) -> None:
        key = tei_key("p1", "grobid-crf-0.8.2.1/tei/1.1.0")
        assert key.count("/") == 3  # papers/<id>/tei/<file>
        assert key.endswith(".xml")


class TestKeysCannotEscapeTheRoot:
    @pytest.mark.parametrize(
        "key",
        [
            "../outside.pdf",
            "papers/../../outside.pdf",
            "/etc/passwd",
            "\\windows\\system32\\config",
            "papers/./../../x",
            "",
        ],
    )
    async def test_a_traversing_key_is_refused(self, store: LocalObjectStore, key: str) -> None:
        with pytest.raises(UnsafeObjectKey):
            await store.put(key, b"%PDF-", content_type="application/pdf")

    async def test_a_legitimate_nested_key_is_allowed(self, store: LocalObjectStore) -> None:
        # Positive control: a check that refuses everything passes every test
        # above while making the store useless.
        await store.put("papers/abc/tei/v1.xml", b"<TEI/>", content_type="application/xml")
        assert await store.exists("papers/abc/tei/v1.xml")

    async def test_the_bytes_land_under_the_root(
        self, store: LocalObjectStore, tmp_path: Path
    ) -> None:
        await store.put("papers/abc/source.pdf", b"%PDF-1.7", content_type="application/pdf")
        written = list((tmp_path / "objects").rglob("*.pdf"))
        assert written == [tmp_path / "objects" / "papers" / "abc" / "source.pdf"]


class TestRoundTrip:
    async def test_what_goes_in_comes_out_byte_identical(self, store: LocalObjectStore) -> None:
        payload = bytes(range(256)) * 40
        await store.put("papers/x/source.pdf", payload, content_type="application/pdf")
        assert await store.get("papers/x/source.pdf") == payload

    async def test_a_missing_key_raises_rather_than_returning_empty(
        self, store: LocalObjectStore
    ) -> None:
        # Returning b"" would send an empty "PDF" into the parser, which would
        # classify it as unparseable — a two-store inconsistency reported as a
        # bad document.
        with pytest.raises(ObjectNotFound):
            await store.get("papers/ghost/source.pdf")

    async def test_a_rewrite_replaces_rather_than_appends(self, store: LocalObjectStore) -> None:
        # Retries re-store the same key. Appending would corrupt the object and
        # leave nothing to notice it.
        await store.put("papers/x/source.pdf", b"first", content_type="application/pdf")
        await store.put("papers/x/source.pdf", b"second", content_type="application/pdf")
        assert await store.get("papers/x/source.pdf") == b"second"

    async def test_no_partial_files_survive_a_write(
        self, store: LocalObjectStore, tmp_path: Path
    ) -> None:
        await store.put("papers/x/source.pdf", b"payload", content_type="application/pdf")
        assert not list((tmp_path / "objects").rglob("*.partial"))


class TestDeleteIsIdempotent:
    async def test_deleting_twice_does_not_raise(self, store: LocalObjectStore) -> None:
        # §5.6.4's cascade is restartable, so the second attempt must not fail
        # on the blobs the first one already removed.
        await store.put("papers/x/source.pdf", b"bytes", content_type="application/pdf")
        assert await store.delete("papers/x/source.pdf") is True
        assert await store.delete("papers/x/source.pdf") is False
