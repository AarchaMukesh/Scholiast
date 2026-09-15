"""The vendored QASPER snapshot (D20, REV-6, D68).

D-RET's source is QASPER's validation split. M0 established that the default
`load_dataset("allenai/qasper")` path is broken — `datasets` v4 dropped
script-based loaders — and that the auto-converted parquet branch works. REV-6
required two mitigations before that could be relied on, **both of them**:

1. **Pin the revision**, not just the branch name.
2. **Vendor a snapshot into the repo**, so FR-E4 reproducibility does not depend
   on a third party's conversion branch continuing to exist.

Neither had been done: M0 recorded that 281 papers were reachable and stopped
there. This module is both halves.

**No `datasets`, no `pyarrow`.** D25 is explicit that installing `datasets` breaks
the ML stack on this machine — a Windows Application Control policy blocks
pyarrow's `_dataset` DLL, and `sentence-transformers` imports `datasets` at module
load, so the embedder would go down with it. The rows are fetched as **JSON** from
the datasets-server API instead, which reads the same parquet conversion and
serves it over HTTP. The trap is avoided rather than worked around: nothing here
imports either package.

**The pin fails loudly or not at all.** `refs/convert/parquet` is auto-generated
infrastructure, and HF may re-convert at any time. Fetching asserts the live
revision still equals `QASPER_PARQUET_REVISION`; a mismatch raises with both SHAs
rather than silently vendoring different data under the same filename. Re-pinning
is then a deliberate act with a diff, which is the only version of "pinned" that
means anything.

**Loading verifies the payload.** The manifest carries a SHA-256 of the
compressed snapshot, checked on every load. A truncated or hand-edited vendored
file is the kind of thing that produces slightly wrong numbers for weeks.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: The pinned conversion revision (REV-6 mitigation 1). Resolved 2026-09-15 from
#: `GET /api/datasets/allenai/qasper/revision/refs%2Fconvert%2Fparquet`.
QASPER_PARQUET_REVISION: Final = "06806e4608976fc2fac0a090ac425d5b2b29caf4"

QASPER_DATASET: Final = "allenai/qasper"
QASPER_CONFIG: Final = "qasper"
QASPER_SPLIT: Final = "validation"

#: What M0 measured, asserted on fetch so a silently smaller split is caught.
QASPER_VALIDATION_ROWS: Final = 281

_HUB_API: Final = "https://huggingface.co/api/datasets"
_ROWS_API: Final = "https://datasets-server.huggingface.co/rows"
#: The rows endpoint caps a page at 100.
_PAGE: Final = 100

#: Repo-relative, because the snapshot is committed (it is the whole point) and
#: an absolute path would make the harness machine-specific.
_DATA_DIR: Final = Path(__file__).resolve().parents[3] / "data" / "eval"
SNAPSHOT_PATH: Final = _DATA_DIR / "qasper-validation.json.gz"
MANIFEST_PATH: Final = _DATA_DIR / "qasper-validation.manifest.json"


class SnapshotError(RuntimeError):
    """The vendored snapshot is absent, corrupt, or pinned to a revision that moved."""


@dataclass(frozen=True, slots=True)
class Manifest:
    """What was vendored, from where, and what it should hash to."""

    dataset: str
    config: str
    split: str
    revision: str
    rows: int
    sha256: str
    #: Bytes of the compressed payload. Cheap sanity check, and it documents the
    #: repo cost of the snapshot without anyone having to go and measure it.
    compressed_bytes: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset": self.dataset,
                "config": self.config,
                "split": self.split,
                "revision": self.revision,
                "rows": self.rows,
                "sha256": self.sha256,
                "compressed_bytes": self.compressed_bytes,
            },
            indent=2,
        )

    @staticmethod
    def from_json(raw: str) -> Manifest:
        d = json.loads(raw)
        return Manifest(
            dataset=d["dataset"],
            config=d["config"],
            split=d["split"],
            revision=d["revision"],
            rows=int(d["rows"]),
            sha256=d["sha256"],
            compressed_bytes=int(d["compressed_bytes"]),
        )


def _trim(row: dict[str, Any]) -> dict[str, Any]:
    """Keep what D-RET needs; drop what it does not.

    `figures_and_tables` goes because the corpus is built from `full_text` and
    QASPER's float captions are not part of it — carrying them would vendor 4% more
    bytes that nothing reads. The `qas` metadata fields (`nlp_background`,
    `topic_background`, `paper_read`, `search_query`, `question_writer`) describe
    the *annotator*, not the question, and D-RET judges retrieval.

    Trimming is recorded here rather than done inline at fetch, so what the
    snapshot contains is one function rather than an argument to reconstruct.
    """
    qas = row["qas"]
    return {
        "id": row["id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "full_text": row["full_text"],
        "qas": {
            "question": qas["question"],
            "question_id": qas["question_id"],
            "answers": qas["answers"],
        },
    }


def fetch_snapshot(*, limit: int | None = None) -> Manifest:
    """Download QASPER's validation split and vendor it. Requires network.

    Run once, deliberately, when re-pinning. Never called by tests or by the eval
    run — NFR-T3 requires the default test run to make zero network calls, and an
    eval that silently re-downloads its own dataset is not reproducible, it is
    merely repeatable while the endpoint is up.

    `limit` exists for probing the endpoint without pulling the whole split; a
    limited fetch refuses to write, because a partial snapshot that looks like a
    complete one is exactly the failure vendoring exists to prevent.
    """
    import httpx  # local: the eval harness must not drag HTTP into module import

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        live = client.get(f"{_HUB_API}/{QASPER_DATASET}/revision/refs%2Fconvert%2Fparquet")
        live.raise_for_status()
        sha = str(live.json().get("sha", ""))
        if sha != QASPER_PARQUET_REVISION:
            raise SnapshotError(
                f"the conversion branch moved: pinned {QASPER_PARQUET_REVISION}, "
                f"live {sha}. Re-vendoring is a deliberate act — inspect the diff, "
                f"update QASPER_PARQUET_REVISION, and re-run. Do not silently accept it."
            )

        wanted = limit if limit is not None else QASPER_VALIDATION_ROWS
        rows: list[dict[str, Any]] = []
        while len(rows) < wanted:
            page = client.get(
                _ROWS_API,
                params={
                    "dataset": QASPER_DATASET,
                    "config": QASPER_CONFIG,
                    "split": QASPER_SPLIT,
                    "offset": len(rows),
                    "length": min(_PAGE, wanted - len(rows)),
                },
            )
            page.raise_for_status()
            batch = page.json().get("rows", [])
            if not batch:
                break
            rows.extend(_trim(item["row"]) for item in batch)

    if limit is not None:
        raise SnapshotError(f"probe fetched {len(rows)} rows and will not write a partial snapshot")
    if len(rows) != QASPER_VALIDATION_ROWS:
        raise SnapshotError(
            f"expected {QASPER_VALIDATION_ROWS} validation papers (M0's measurement), "
            f"got {len(rows)} — the split changed shape, which invalidates the pin"
        )

    payload = gzip.compress(
        json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8"), mtime=0
    )
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_bytes(payload)
    manifest = Manifest(
        dataset=QASPER_DATASET,
        config=QASPER_CONFIG,
        split=QASPER_SPLIT,
        revision=QASPER_PARQUET_REVISION,
        rows=len(rows),
        sha256=hashlib.sha256(payload).hexdigest(),
        compressed_bytes=len(payload),
    )
    MANIFEST_PATH.write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest


def load_snapshot() -> list[dict[str, Any]]:
    """Read the vendored snapshot, verifying it against its manifest.

    Offline and deterministic: `mtime=0` on compression plus `sort_keys` on the
    JSON means the same rows always produce byte-identical output, so the hash
    checks the *content* rather than the moment it was written.
    """
    if not SNAPSHOT_PATH.exists() or not MANIFEST_PATH.exists():
        raise SnapshotError(
            f"no vendored snapshot at {SNAPSHOT_PATH}. Run "
            f"`python -m scholiast.eval.snapshot` once to fetch it (needs network)."
        )
    manifest = Manifest.from_json(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = SNAPSHOT_PATH.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest.sha256:
        raise SnapshotError(
            f"snapshot does not match its manifest (sha256 {digest[:16]}… vs "
            f"{manifest.sha256[:16]}…). It was truncated or edited; re-fetch it."
        )
    rows: list[dict[str, Any]] = json.loads(gzip.decompress(payload).decode("utf-8"))
    if len(rows) != manifest.rows:
        raise SnapshotError(f"manifest claims {manifest.rows} rows, payload has {len(rows)}")
    return rows


if __name__ == "__main__":  # pragma: no cover - a one-shot vendoring command
    result = fetch_snapshot()
    print(f"vendored {result.rows} papers, {result.compressed_bytes / 1e6:.2f} MB")
    print(f"revision {result.revision}")
    print(f"sha256   {result.sha256}")
