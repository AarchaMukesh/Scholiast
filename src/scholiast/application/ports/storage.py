"""Object store port — original PDFs and retained TEI (§5.6.1, D43, D49).

Blobs do not belong in the relational store: they bloat backups and complicate
replication, and the two things kept here are both *rebuildable* (the PDF from
the user or the source, the TEI from GROBID) while Postgres is not. Local
filesystem in dev, S3-compatible in prod, one port.

**Keys are paper-scoped, never content-scoped** (D49). A content-addressed key
would give two tenants' byte-identical private uploads one shared blob — and
§4.4.2's carve-out deliberately gives them two paper rows, so deleting one
paper's blob would silently take the other's with it. The key builders below are
the only place keys are constructed, so that property is enforceable rather than
conventional.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ObjectNotFound(Exception):
    """A key that should exist does not.

    Distinct from a storage outage: this is the two-store inconsistency
    §5.6.4 exists to reconcile, so the caller can classify it rather than
    retrying forever against an object that was never written.
    """

    def __init__(self, key: str) -> None:
        super().__init__(f"no object stored at {key!r}")
        self.key = key


def source_key(paper_id: str) -> str:
    """Where this paper's original PDF lives."""
    return f"papers/{paper_id}/source.pdf"


def paper_prefix(paper_id: str) -> str:
    """Everything this paper owns in the store (D49, D102).

    The cascade deletes by prefix rather than by naming `source_key` and each
    `tei_key`, and the difference is not tidiness. A paper may hold TEI for
    SEVERAL parser generations on purpose, so a purge that deletes the version
    the row happens to record leaks every other one — and FR-W3's "zero residue"
    would then mean "zero residue among the keys we remembered to list".
    Prefix deletion is only possible because D49 made keys paper-scoped; under
    content-addressed keys this function could not exist.
    """
    return f"papers/{paper_id}/"


def tei_key(paper_id: str, parser_version: str) -> str:
    """Where this paper's retained TEI lives, for one parser version (D43).

    The version is part of the key so two parser generations can coexist:
    re-chunking ablations must read the bytes that produced today's chunks, and
    a single overwritten key would silently re-baseline the comparison the
    moment GROBID is upgraded — the failure FR-E4 forbids. `parser_version` is
    path-shaped ("grobid-crf-0.8.2.1/tei/1.1.0"), so its separators are folded
    rather than allowed to become directories.
    """
    return f"papers/{paper_id}/tei/{parser_version.replace('/', '_')}.xml"


@runtime_checkable
class ObjectStore(Protocol):
    """Bytes in, bytes out, addressed by an opaque key."""

    async def put(self, key: str, data: bytes, *, content_type: str) -> None:
        """Write (or overwrite) an object.

        Overwriting is deliberate: every key this system builds is a pure
        function of identity that is already fixed, so a second write of the
        same key is a retry of the same content, not a different object.
        """
        ...

    async def get(self, key: str) -> bytes:
        """Read an object, or raise `ObjectNotFound`."""
        ...

    async def exists(self, key: str) -> bool: ...

    async def delete(self, key: str) -> bool:
        """Remove an object. Returns False if it was already gone.

        Idempotent because the deletion sweep is (§5.6.4): a restarted cascade
        must not fail on the blobs its previous attempt already removed.
        """
        ...

    async def list_keys(self, prefix: str = "") -> list[str]:
        """Every key under `prefix`, in no particular order.

        The reconciliation sweep is the only caller, and it needs this only
        because an object store has no query language to ask "which of these
        have no paper" with (D102). **D49 is what keeps the answer decidable:**
        keys are paper-scoped, so the paper id is a path segment and a key can
        be matched against the live set without opening or hashing anything.
        Content-addressed keys would have made the same question unanswerable
        without a reference count.
        """
        ...
