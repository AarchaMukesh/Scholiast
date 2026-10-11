"""D-RET is a dataset, so the tests are about the properties a dataset must have.

Not "does `build_dret` return something" — whether the thing it returns can carry
a published number. Four properties do that work: it loads without a network call,
every judgment points at a chunk that exists, the split leaks nothing at the paper
level, and building it twice produces identical ids.

The last one is the quiet one. FR-E4 requires two runs to agree, and chunk ids are
derived from ordinals — so a change in section ordering, or a `set` iterated where
a list was meant, reorders ids and silently detaches every judgment from the chunk
it was written for. Nothing raises; Recall@k just drops, and it looks like the
retriever got worse.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from scholiast.domain.chunking import ChunkingConfig
from scholiast.domain.errors import InvariantViolation
from scholiast.domain.paper import ChunkRole
from scholiast.eval import dret as dret_module
from scholiast.eval.dret import (
    CORE_SIZE_BAND,
    Grade,
    Judgment,
    Query,
    Split,
    assign_core,
    assign_split,
    build_dret,
    verify_core,
    verify_split,
)
from scholiast.eval.qasper import build_paper, normalise_for_match
from scholiast.eval.snapshot import SnapshotError, fetch_snapshot, load_snapshot


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return load_snapshot()


@pytest.fixture(scope="module")
def dataset(rows: list[dict[str, Any]]) -> dret_module.DRet:
    return build_dret(rows)


def _synthetic_rows(count: int) -> list[dict[str, Any]]:
    """QASPER-shaped papers for growth tests. Not fixtures — corpus *additions*.

    Each carries one answerable question whose evidence is a verbatim paragraph,
    so it survives judgment resolution and reaches `queries` like a real one.
    """
    made: list[dict[str, Any]] = []
    for i in range(count):
        body = (
            f"Synthetic paper {i} reports a result on a held-out set. "
            f"The measured value was {i / 7:.3f} under the stated protocol."
        )
        made.append(
            {
                "id": f"9999.{i:05d}",
                "title": f"Synthetic Paper {i}",
                "abstract": f"An abstract for synthetic paper {i}.",
                "full_text": {"section_name": ["Results"], "paragraphs": [[body]]},
                "qas": {
                    "question": [f"What did synthetic paper {i} measure?"],
                    "question_id": [f"synthetic-q-{i:05d}"],
                    "answers": [
                        {"answer": [{"unanswerable": False, "evidence": [body]}]},
                    ],
                },
            }
        )
    return made


class TestTheSnapshotIsVendoredNotFetched:
    """NFR-T3: the default test run makes zero network calls."""

    def test_loading_the_snapshot_opens_no_socket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Blunt on purpose. Patching httpx would only prove httpx was not used;
        # patching the socket constructor proves nothing reached the network by
        # any route, including a transitive import deciding to phone home.
        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("the vendored snapshot must load without a network call")

        monkeypatch.setattr(socket, "socket", forbidden)
        monkeypatch.setattr(socket, "create_connection", forbidden)

        loaded = load_snapshot()
        assert len(loaded) == 281

    def test_a_corrupt_snapshot_is_refused_rather_than_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A silently edited snapshot produces slightly wrong numbers for weeks. The
        # manifest hash is the only thing standing between that and a report.
        #
        # The tampering has to be VALID GZIP, or this proves nothing. The first
        # draft appended b"tampered" to the payload, which kills the mutation —
        # but via `gzip.BadGzipFile`, not via the hash. That test would still have
        # passed with the integrity check deleted, which is precisely the case it
        # exists to cover. Here the payload decompresses cleanly and differs only
        # in content, so ONLY the hash can catch it.
        import gzip
        import json

        from scholiast.eval import snapshot as snapshot_module

        genuine = json.loads(gzip.decompress(snapshot_module.SNAPSHOT_PATH.read_bytes()))
        genuine[0]["title"] = "a plausible edit nobody would notice"
        forged = gzip.compress(
            json.dumps(genuine, ensure_ascii=False, sort_keys=True).encode("utf-8"), mtime=0
        )
        assert json.loads(gzip.decompress(forged)), "the forgery must be readable gzip"

        monkeypatch.setattr(
            snapshot_module.SNAPSHOT_PATH.__class__, "read_bytes", lambda self: forged
        )
        with pytest.raises(SnapshotError, match="does not match its manifest"):
            load_snapshot()

    def test_a_partial_fetch_refuses_to_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A partial snapshot that looks complete is the failure vendoring exists
        # to prevent, so `limit` must never be able to produce one.
        #
        # Offline (NFR-T3). Until 2026-10-11 this called the real hub: two live
        # requests on every default run, and a failure whenever Hugging Face
        # answered 502 or moved its conversion branch, neither of which says
        # anything about this code. Both endpoints are now served by a mock
        # transport, and any real socket fails the test, as in the test above.
        import httpx

        from scholiast.eval import snapshot as snapshot_module

        seen: list[httpx.Request] = []

        def hub(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.host == "huggingface.co":
                return httpx.Response(200, json={"sha": snapshot_module.QASPER_PARQUET_REVISION})
            row = {
                "id": "x",
                "title": "t",
                "abstract": "a",
                "full_text": {},
                "qas": {"question": [], "question_id": [], "answers": []},
            }
            return httpx.Response(200, json={"rows": [{"row": row}]})

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(hub), **kwargs),
        )

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("this test must not reach the network")

        monkeypatch.setattr(socket, "socket", forbidden)
        monkeypatch.setattr(socket, "create_connection", forbidden)
        monkeypatch.setattr(
            snapshot_module.SNAPSHOT_PATH.__class__,
            "write_bytes",
            lambda self, data: pytest.fail("a limited fetch must not write the snapshot"),
        )

        with pytest.raises(SnapshotError, match="will not write a partial snapshot"):
            fetch_snapshot(limit=1)
        # The probe path really ran, both endpoints asking for exactly one row, so
        # the refusal above is the limit's and not an earlier failure's.
        assert [r.url.host for r in seen] == ["huggingface.co", "datasets-server.huggingface.co"]
        assert seen[1].url.params["length"] == "1"


class TestEveryJudgmentPointsAtSomethingReal:
    def test_each_judged_chunk_exists_in_the_corpus(self, dataset: dret_module.DRet) -> None:
        missing = [j.chunk_id for j in dataset.judgments if j.chunk_id not in dataset.chunks]
        assert not missing, f"{len(missing)} judgments point at chunks that do not exist"

    def test_each_judgment_belongs_to_a_query_that_exists(self, dataset: dret_module.DRet) -> None:
        ids = {q.id for q in dataset.queries}
        orphans = {j.query_id for j in dataset.judgments} - ids
        assert not orphans, f"judgments for unknown queries: {sorted(orphans)[:3]}"

    def test_every_query_has_at_least_one_judged_chunk(self, dataset: dret_module.DRet) -> None:
        # A query with an empty qrel set depresses Recall@k identically for every
        # configuration, which reads as a hard dataset rather than a broken one.
        judged = {j.query_id for j in dataset.judgments}
        empty = [q.id for q in dataset.queries if q.id not in judged]
        assert not empty, f"{len(empty)} queries carry no judgment and must have been dropped"

    def test_judged_chunks_are_ones_retrieval_can_actually_return(
        self, dataset: dret_module.DRet
    ) -> None:
        # Parent windows are returned as context, never matched. Judging one would
        # be judging something the retriever cannot surface.
        parents = [
            j.chunk_id
            for j in dataset.judgments
            if dataset.chunks[j.chunk_id].role is ChunkRole.PARENT
        ]
        assert not parents, "judgments must target indexed chunks, not parent windows"

    def test_no_evidence_string_is_silently_unresolved(self, dataset: dret_module.DRet) -> None:
        # The accounting exists so nothing is rounded away. `unresolved` is the
        # bucket for evidence the logic cannot explain, and it must stay empty —
        # anything landing here is a defect, not a dataset property.
        assert dataset.evidence.unresolved == 0
        assert dataset.evidence.total == (
            dataset.evidence.exact
            + dataset.evidence.whitespace_normalised
            + dataset.evidence.float_reference
            + dataset.evidence.section_heading
            + dataset.evidence.unresolved
        ), "the evidence accounting must partition every string exactly once"


class TestTheSplitLeaksNothing:
    def test_no_paper_appears_in_both_splits(self, dataset: dret_module.DRet) -> None:
        train = {q.paper_id for q in dataset.queries if q.split is Split.TRAIN}
        evaluation = {q.paper_id for q in dataset.queries if q.split is Split.EVAL}
        assert not (train & evaluation)

    def test_a_query_level_check_would_not_have_caught_it(self, dataset: dret_module.DRet) -> None:
        # The point of doc 9's warning: query ids are unique across the whole set,
        # so a query-level disjointness check passes even when the same PAPER sits
        # on both sides. This asserts the weaker check really is weaker, so nobody
        # later "simplifies" the paper-level one into it.
        ids = [q.id for q in dataset.queries]
        assert len(ids) == len(set(ids)), "query ids are unique — so uniqueness proves nothing"

    def test_verify_split_rejects_a_planted_cross_paper_judgment(
        self, dataset: dret_module.DRet
    ) -> None:
        # Mutation in test form: if this passes with the check removed, the check
        # is decorative.
        victim = dataset.queries[0]
        foreign = next(
            cid
            for cid, chunk in dataset.chunks.items()
            if chunk.provenance.paper_id != victim.paper_id
        )
        poisoned = dret_module.DRet(
            queries=list(dataset.queries),
            judgments=[*dataset.judgments, Judgment(victim.id, foreign, Grade.DIRECTLY_ANSWERS)],
            chunks=dict(dataset.chunks),
            papers=dict(dataset.papers),
        )
        with pytest.raises(InvariantViolation, match="cross-paper judgment"):
            verify_split(poisoned)

    def test_the_real_dataset_passes_verification(self, dataset: dret_module.DRet) -> None:
        verify_split(dataset)

    def test_split_assignment_is_stable_under_corpus_growth(self) -> None:
        # Hashing rather than sampling: adding papers must never move an existing
        # one across the boundary, or every number published before it silently
        # stops describing the same eval set.
        before = {pid: assign_split(pid) for pid in ("1901.00001", "2002.12345", "1810.04805")}
        after = {pid: assign_split(pid) for pid in before}
        assert before == after
        assert assign_split("1810.04805") is assign_split("1810.04805")


class TestTheDatasetIsReproducible:
    def test_two_builds_produce_identical_chunk_ids(
        self, rows: list[dict[str, Any]], dataset: dret_module.DRet
    ) -> None:
        # FR-E4's real precondition. Ids are derived from ordinals, so any
        # reordering detaches judgments from their chunks with nothing raising.
        rebuilt = build_dret(rows)
        assert sorted(rebuilt.chunks) == sorted(dataset.chunks)
        assert [(j.query_id, j.chunk_id, j.grade) for j in rebuilt.judgments] == [
            (j.query_id, j.chunk_id, j.grade) for j in dataset.judgments
        ]

    def test_chunk_text_is_identical_across_builds(
        self, rows: list[dict[str, Any]], dataset: dret_module.DRet
    ) -> None:
        # Same ids with different text would pass the check above and still
        # invalidate every judgment.
        rebuilt = build_dret(rows)
        assert all(rebuilt.chunks[cid].text == chunk.text for cid, chunk in dataset.chunks.items())

    def test_the_dataset_declares_every_version_that_changes_its_numbers(
        self, dataset: dret_module.DRet
    ) -> None:
        assert set(dataset.versions) == {"dret", "builder", "chunker"}
        assert all(dataset.versions.values())

    def test_a_different_chunking_config_changes_the_dataset(
        self, rows: list[dict[str, Any]], dataset: dret_module.DRet
    ) -> None:
        # Guards the claim that `chunker` belongs in the version dict: if chunking
        # config did not move the chunk set, stamping the version would be theatre.
        coarser = build_dret(rows[:40], config=ChunkingConfig(child_target_tokens=600))
        baseline = build_dret(rows[:40])
        assert sorted(coarser.chunks) != sorted(baseline.chunks)


class TestTheCorpusMatchesProductionShape:
    def test_offsets_index_their_own_section_text(self, rows: list[dict[str, Any]]) -> None:
        # D42/D45's round trip. `ParsedSection` asserts it at construction, so this
        # would fail at build time — the test states the guarantee explicitly
        # because the eval corpus is where a normalisation shortcut is tempting.
        for row in rows[:25]:
            paper = build_paper(row)
            for section in paper.document.sections:
                for block in section.blocks:
                    assert section.text[block.char_start : block.char_end] == block.text

    def test_matching_normalisation_never_reaches_stored_text(
        self, rows: list[dict[str, Any]]
    ) -> None:
        # The normalisation used to look evidence up must not touch what is stored,
        # or offsets drift from the text they claim to index.
        #
        # Written to FIND a block that normalisation would change rather than to
        # hope the first one does. The first draft of this test guarded its
        # assertions behind `if normalise_for_match(text) != text` and the
        # condition was False for the block it happened to pick, so it asserted
        # nothing at all while passing. Measured: 149 of 1,876 blocks in the first
        # 40 papers differ under normalisation, so the premise is real — and the
        # assertion below makes its disappearance a failure rather than a silent
        # return to vacuity.
        cases = [
            (section, block)
            for row in rows[:40]
            for section in build_paper(row).document.sections
            for block in section.blocks
            if normalise_for_match(block.text) != block.text
        ]
        assert cases, (
            "no block in the sample is changed by whitespace normalisation, so this "
            "test can no longer distinguish stored text from matched text"
        )
        for section, block in cases[:50]:
            # The stored text is the raw paragraph, and the offsets index it.
            assert section.text[block.char_start : block.char_end] == block.text
            # The normalised form is strictly a lookup key: it is shorter, and it
            # is not what any block stores.
            assert len(normalise_for_match(block.text)) < len(block.text)
            assert normalise_for_match(block.text) != block.text

    def test_section_paths_are_unique_within_a_paper(self, rows: list[dict[str, Any]]) -> None:
        # D51: a duplicate path silently attributes one section's chunks to
        # another's text. The rule is imported from the domain, not reimplemented.
        for row in rows[:40]:
            paper = build_paper(row)
            paths = [s.path for s in paper.document.sections]
            assert len(paths) == len(set(paths)), f"{paper.paper_id} has duplicate section paths"


class TestCoreIsAUsablePrimaryBenchmark:
    """D-RET-CORE carries every headline number (D69), so its properties are the
    ones a published result rests on."""

    def test_core_is_drawn_only_from_the_eval_split(self, dataset: dret_module.DRet) -> None:
        # The failure this prevents is invisible in the numbers: CORE would still
        # look like a valid benchmark while containing papers M7 fine-tuned on.
        assert dataset.core, "CORE is empty"
        assert all(q.split is Split.EVAL for q in dataset.core)

    def test_core_is_a_subset_of_full_not_a_disjoint_sample(
        self, dataset: dret_module.DRet
    ) -> None:
        # FULL answers "does the CORE result hold at 5x the sample", which requires
        # the same queries inside it. Disjoint sets would answer a different and
        # much weaker question.
        assert {q.id for q in dataset.core} <= {q.id for q in dataset.full}

    def test_core_size_sits_in_the_documented_band(self, dataset: dret_module.DRet) -> None:
        low, high = CORE_SIZE_BAND
        assert low <= len(dataset.core) <= high, (
            f"CORE holds {len(dataset.core)}, outside doc 8's {low}-{high} band — "
            "doc 9 §9.4's calibration is written around n≈50"
        )

    def test_assign_core_is_a_pure_function_of_the_query_id(self) -> None:
        # Necessary but nowhere near sufficient — see the test below, which is the
        # one that actually protects the property.
        assert assign_core("q0001") is assign_core("q0001")
        assert {q for q in (f"q{i:04d}" for i in range(259)) if assign_core(q)}

    def test_core_membership_survives_the_corpus_growing(
        self, rows: list[dict[str, Any]], dataset: dret_module.DRet
    ) -> None:
        # THE property the threshold rule exists for (D69): growing the corpus must
        # never change which queries the primary benchmark contains, or every number
        # published against the old CORE quietly describes a different set.
        #
        # Measured against this test — threshold 0 evictions, proportional lowest-N
        # 0, fixed-size lowest-N 5 of 50. So it does not discriminate every rule,
        # only the ones that genuinely destabilise; the threshold is chosen because
        # membership is a function of the id alone, which no measurement can be
        # contingent on.
        #
        # This must run through `build_dret`, not against `assign_core` alone. The
        # first version of this test grew a list of synthetic ids and checked
        # `assign_core` — which is a pure function of one id and therefore CANNOT
        # fail the property, so the test passed while proving nothing. Mutation
        # testing caught it: a genuine lowest-N selection written into `build_dret`
        # — the realistic regression, and the exact rule D69 rejected — survived
        # the whole suite. Growing the real corpus is what exercises the path the
        # dataset is actually built by.
        grown = build_dret([*rows, *_synthetic_rows(90)])
        before = {q.id for q in dataset.core}
        after = {q.id for q in grown.core}
        evicted = before - after
        assert not evicted, (
            f"{len(evicted)} of {len(before)} CORE queries were evicted when the corpus "
            f"grew, e.g. {sorted(evicted)[:3]} — membership must be stable or every "
            "number published against the old CORE describes a different benchmark"
        )
        assert len(after) > len(before), (
            "the grown corpus added no CORE members, so this test would pass even if "
            "selection were frozen — it is not exercising growth"
        )

    def test_core_selection_does_not_prefer_well_evidenced_queries(
        self, dataset: dret_module.DRet
    ) -> None:
        # Selecting on anything correlated with difficulty would make the primary
        # benchmark a set chosen to flatter the pipeline. `assign_core` reads only
        # the id; this asserts the consequence rather than the implementation, so
        # it still fails if someone later "improves" the selection.
        counts = {q.id: len(dataset.qrels(q.id)) for q in dataset.full}
        core_mean = sum(counts[q.id] for q in dataset.core) / len(dataset.core)
        pool_mean = sum(counts.values()) / len(counts)
        assert abs(core_mean - pool_mean) < 0.5, (
            f"CORE averages {core_mean:.2f} judged chunks per query against the pool's "
            f"{pool_mean:.2f} — the selection looks correlated with how well evidenced "
            "a query is, which biases every headline number"
        )

    def test_verify_core_rejects_a_train_query_smuggled_into_core(
        self, dataset: dret_module.DRet
    ) -> None:
        trainee = next(q for q in dataset.queries if q.split is Split.TRAIN)
        poisoned = dret_module.DRet(
            queries=[
                *(q for q in dataset.queries if q.id != trainee.id),
                Query(trainee.id, trainee.text, trainee.paper_id, Split.TRAIN, in_core=True),
            ],
            judgments=list(dataset.judgments),
            chunks=dict(dataset.chunks),
            papers=dict(dataset.papers),
        )
        with pytest.raises(InvariantViolation, match="not in the eval split"):
            verify_core(poisoned)

    def test_verify_core_rejects_a_core_that_drifted_out_of_band(
        self, rows: list[dict[str, Any]]
    ) -> None:
        # The threshold is fixed while the corpus grows, so the count moves on its
        # own. Drifting past the band must fail loudly and demand a re-baseline,
        # not quietly produce numbers doc 9 §9.4 no longer describes.
        oversized = build_dret(rows, core_threshold=0.9)
        with pytest.raises(InvariantViolation, match="outside doc 8's"):
            verify_core(oversized)

    def test_the_real_dataset_passes_core_verification(self, dataset: dret_module.DRet) -> None:
        verify_core(dataset)


class TestTheGradeScaleIsStatedNotImplied:
    """D69: every report must say the scale is 2/0 rather than imply 3 points."""

    def test_only_grade_two_is_populated(self, dataset: dret_module.DRet) -> None:
        assert dataset.populated_grades == frozenset({Grade.DIRECTLY_ANSWERS})
        assert not dataset.is_fully_graded

    def test_the_scale_string_says_grade_one_is_missing(self, dataset: dret_module.DRet) -> None:
        assert "2/0" in dataset.grade_scale
        assert "unpopulated" in dataset.grade_scale

    def test_the_scale_string_is_derived_from_the_data_not_hardcoded(
        self, dataset: dret_module.DRet
    ) -> None:
        # If it were a constant it would keep claiming 2/0 after the hand-review
        # populated grade 1 — a report that lies in the opposite direction.
        victim = dataset.core[0]
        chunk_id = next(iter(dataset.qrels(victim.id)))
        reviewed = dret_module.DRet(
            queries=list(dataset.queries),
            judgments=[*dataset.judgments, Judgment(victim.id, chunk_id, Grade.CONTEXT)],
            chunks=dict(dataset.chunks),
            papers=dict(dataset.papers),
        )
        assert reviewed.is_fully_graded
        assert "3-point" in reviewed.grade_scale
        assert "unpopulated" not in reviewed.grade_scale
