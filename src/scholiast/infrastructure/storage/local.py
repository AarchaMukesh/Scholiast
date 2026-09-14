"""Local-filesystem `ObjectStore` — the M1 adapter (doc 8, §7.3).

A directory tree under one configured root. S3 replaces it in M3 with no caller
changes; that is the entire point of the port.

Two things here are load-bearing rather than tidy:

**Key validation.** Keys reach this adapter from code today and from a database
column tomorrow (`papers.tei_key`), so a key that escapes the root — `..`, an
absolute path, a Windows drive letter — must be refused rather than resolved. It
is checked by *resolving* the final path and confirming it is still under the
root, not by pattern-matching the key: pattern lists are where traversal bugs
live, and the filesystem already knows the answer.

**Writes are atomic.** Content is written to a temporary file in the same
directory and renamed into place. A half-written PDF that a retry then finds
"present" is a silent corruption; `os.replace` is atomic on both POSIX and
Windows within a filesystem, so a reader sees either the old bytes or the new
ones.

Blocking file I/O is run on a thread. The pipeline is async end to end, and a
40 MB synchronous write on the event loop would stall every other task in the
worker — including the lease heartbeats that keep this job claimed.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from scholiast.application.ports.storage import ObjectNotFound


class UnsafeObjectKey(ValueError):
    """A key that would resolve outside the store root."""

    def __init__(self, key: str) -> None:
        super().__init__(f"object key {key!r} escapes the store root")
        self.key = key


class LocalObjectStore:
    """The `ObjectStore` adapter backed by a directory."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, key: str) -> Path:
        if not key or key.startswith("/") or key.startswith("\\"):
            raise UnsafeObjectKey(key)
        candidate = (self._root / key).resolve()
        # `is_relative_to` compares resolved paths, so `a/../../b` is caught
        # after normalisation rather than by spotting the "..".
        if not candidate.is_relative_to(self._root):
            raise UnsafeObjectKey(key)
        return candidate

    async def put(self, key: str, data: bytes, *, content_type: str = "") -> None:  # noqa: ARG002
        # `content_type` is part of the port because S3 needs it on write; a
        # filesystem has nowhere to put it, and inventing a sidecar file to hold
        # metadata nothing reads would be storage for its own sake.
        await asyncio.to_thread(self._write, self._path_for(key), data)

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".partial")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
            Path(temporary).replace(path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    async def get(self, key: str) -> bytes:
        path = self._path_for(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise ObjectNotFound(key) from exc

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._path_for(key).is_file)

    async def delete(self, key: str) -> bool:
        path = self._path_for(key)

        def _unlink() -> bool:
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            return True

        return await asyncio.to_thread(_unlink)
