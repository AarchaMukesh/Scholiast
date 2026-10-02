"""The cascade and the sweep, against fakes (FR-W3, §5.6.4, D99-D102).

Fakes rather than mocks, each keeping a shared journal — the same convention as
`test_ingestion`, and for the same reason. The interesting correctness here is
**ordering and convergence**, not "did every step happen":

  the workspace row goes LAST, because it is what makes the work findable;
  vectors go before the paper row, because a vector with no row is recoverable
  and a row with no vector is silently unretrievable (D62);
  the live set is read AFTER the cascades, or a resumed cascade's papers look
  live for one more cycle and their vectors survive it.

A test that only asserts the end state passes for every one of those orderings
while one of them loses data on a crash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from scholiast.application.use_cases.delete_workspace import DeleteWorkspaceUseCase
from scholiast.application.use_cases.purge_paper import PaperPurger
from scholiast.application.use_cases.reconcile import ReconcileUseCase

WORKSPACE = "11111111-1111-4111-8111-111111111111"
OTHER_WORKSPACE = "22222222-2222-4222-8222-222222222222"
PRIVATE_PAPER = "33333333-3333-4333-8333-333333333333"
SHARED_PAPER = "44444444-4444-4444-8444-444444444444"


@dataclass
class Journal:
    events: list[str] = field(default_factory=list)

    def record(self, event: str) -> None:
        self.events.append(event)

    def index_of(self, event: str) -> int:
        return self.events.index(event)


class FakeWorkspaces:
    def __init__(self, journal: Journal, *, deleting: set[str], live: set[str]) -> None:
        self._journal = journal
        self.deleting = set(deleting)
        self.live = set(live)

    async def is_deleting(self, workspace_id: str) -> bool:
        return workspace_id in self.deleting

    async def list_deleting(self) -> tuple[str, ...]:
        self._journal.record("list_deleting")
        return tuple(sorted(self.deleting))

    async def purge(self, workspace_id: str) -> bool:
        self._journal.record(f"purge_workspace:{workspace_id}")
        existed = workspace_id in self.live
        self.live.discard(workspace_id)
        self.deleting.discard(workspace_id)
        # Reports whether a row was actually removed, like the real adapter's
        # rowcount. Returning True unconditionally would make the "swept" flag
        # meaningless and every assertion on it vacuous.
        return existed


class FakePapers:
    """Memberships as a real many-to-many, so AD-8's sharing is exercised."""

    def __init__(
        self,
        journal: Journal,
        *,
        memberships: dict[str, set[str]],
        papers: set[str] | None = None,
    ) -> None:
        self._journal = journal
        self.memberships = {k: set(v) for k, v in memberships.items()}
        self.papers = (
            set(papers)
            if papers is not None
            else {paper for held in memberships.values() for paper in held}
        )

    async def membership_paper_ids(self, workspace_id: str) -> tuple[str, ...]:
        return tuple(sorted(self.memberships.get(workspace_id, set())))

    async def remove_from_workspace(self, workspace_id: str, paper_id: str) -> bool:
        held = self.memberships.get(workspace_id, set())
        if paper_id not in held:
            return False
        held.discard(paper_id)
        self._journal.record(f"drop_membership:{workspace_id}:{paper_id}")
        return True

    async def has_any_membership(self, paper_id: str) -> bool:
        return any(paper_id in held for held in self.memberships.values())

    async def delete(self, paper_id: str) -> bool:
        self._journal.record(f"delete_paper:{paper_id}")
        if paper_id not in self.papers:
            return False
        self.papers.discard(paper_id)
        return True

    async def orphaned_paper_ids(self) -> tuple[str, ...]:
        self._journal.record("orphaned_paper_ids")
        held = {paper for papers in self.memberships.values() for paper in papers}
        return tuple(sorted(self.papers - held))

    async def all_paper_ids(self) -> tuple[str, ...]:
        self._journal.record("all_paper_ids")
        return tuple(sorted(self.papers))


class FakeIndex:
    def __init__(self, journal: Journal, *, vectors: dict[str, int]) -> None:
        self._journal = journal
        self.vectors = dict(vectors)

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        removed = 0
        for paper_id in paper_ids:
            removed += self.vectors.pop(paper_id, 0)
            self._journal.record(f"delete_vectors:{paper_id}")
        return removed

    async def count_orphans(self, live_paper_ids: Sequence[str]) -> int:
        live = set(live_paper_ids)
        return sum(count for paper, count in self.vectors.items() if paper not in live)

    async def delete_orphans(self, live_paper_ids: Sequence[str]) -> int:
        self._journal.record("delete_orphan_vectors")
        live = set(live_paper_ids)
        orphans = [paper for paper in self.vectors if paper not in live]
        return sum(self.vectors.pop(paper) for paper in orphans)


class FakeStore:
    def __init__(self, journal: Journal, *, keys: Sequence[str]) -> None:
        self._journal = journal
        self.keys = set(keys)

    async def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(key for key in self.keys if key.startswith(prefix))

    async def delete(self, key: str) -> bool:
        if key not in self.keys:
            return False
        self.keys.discard(key)
        self._journal.record(f"delete_blob:{key}")
        return True


def blobs_for(paper_id: str) -> list[str]:
    """What a fully ingested paper actually leaves behind.

    Two TEI generations on purpose: D49 keeps them so a re-chunk ablation can
    read the bytes that produced today's chunks, which means a purge that
    reconstructs one `tei_key` leaks the other.
    """
    return [
        f"papers/{paper_id}/source.pdf",
        f"papers/{paper_id}/tei/grobid-crf-0.8.2.1_tei_1.1.0.xml",
        f"papers/{paper_id}/tei/grobid-crf-0.8.3.0_tei_1.1.0.xml",
    ]


@dataclass
class World:
    journal: Journal
    workspaces: FakeWorkspaces
    papers: FakePapers
    index: FakeIndex
    store: FakeStore

    @property
    def cascade(self) -> DeleteWorkspaceUseCase:
        return DeleteWorkspaceUseCase(
            workspaces=self.workspaces,  # type: ignore[arg-type]
            papers=self.papers,  # type: ignore[arg-type]
            index=self.index,  # type: ignore[arg-type]
            store=self.store,  # type: ignore[arg-type]
        )

    @property
    def sweep(self) -> ReconcileUseCase:
        return ReconcileUseCase(
            workspaces=self.workspaces,  # type: ignore[arg-type]
            papers=self.papers,  # type: ignore[arg-type]
            index=self.index,  # type: ignore[arg-type]
            store=self.store,  # type: ignore[arg-type]
        )


def build_world(
    *,
    deleting: set[str] | None = None,
    memberships: dict[str, set[str]] | None = None,
    extra_keys: Sequence[str] = (),
    papers: set[str] | None = None,
) -> World:
    journal = Journal()
    deleting = {WORKSPACE} if deleting is None else deleting
    memberships = (
        memberships
        if memberships is not None
        else {WORKSPACE: {PRIVATE_PAPER, SHARED_PAPER}, OTHER_WORKSPACE: {SHARED_PAPER}}
    )
    keys = [*extra_keys]
    for paper in {paper for held in memberships.values() for paper in held} | (papers or set()):
        keys.extend(blobs_for(paper))
    return World(
        journal=journal,
        workspaces=FakeWorkspaces(
            journal, deleting=set(deleting), live={WORKSPACE, OTHER_WORKSPACE}
        ),
        papers=FakePapers(journal, memberships=memberships, papers=papers),
        index=FakeIndex(journal, vectors={PRIVATE_PAPER: 7, SHARED_PAPER: 5}),
        store=FakeStore(journal, keys=keys),
    )


class TestTheCascadeRefusesWhatWasNotMarked:
    """D99's safety check — the one that stops a stray id destroying live data.

    The cascade runs under `SystemContext`, so no tenant predicate constrains
    which workspace it can delete, and its argument arrives over a broker. If
    the mark were not required, a `delete_workspace` task naming any live
    workspace would wipe it along with every paper only it held.
    """

    async def test_a_live_workspace_is_refused(self) -> None:
        world = build_world(deleting=set())

        result = await world.cascade.execute(WORKSPACE)

        assert result.swept is False
        assert result.papers_purged == 0

    async def test_a_refused_cascade_touches_nothing_at_all(self) -> None:
        # Not just "reports nothing": a cascade that deleted vectors and then
        # decided not to finish would still have destroyed a live workspace's
        # retrieval. The journal is the assertion, not the return value.
        world = build_world(deleting=set())

        await world.cascade.execute(WORKSPACE)

        assert world.journal.events == []
        assert world.papers.papers == {PRIVATE_PAPER, SHARED_PAPER}
        assert world.index.vectors == {PRIVATE_PAPER: 7, SHARED_PAPER: 5}

    async def test_an_unknown_workspace_is_refused(self) -> None:
        world = build_world()

        result = await world.cascade.execute("99999999-9999-4999-8999-999999999999")

        assert result.swept is False
        assert world.journal.events == []


class TestASharedPaperSurvivesTheCascade:
    """AD-8: a public paper is parsed and embedded once and shared.

    Deleting one workspace must not break another workspace's retrieval, which
    is the cost §4.4.2 accepted in exchange for parsing once.
    """

    async def test_the_shared_paper_row_is_kept(self) -> None:
        world = build_world()

        await world.cascade.execute(WORKSPACE)

        assert SHARED_PAPER in world.papers.papers

    async def test_the_shared_paper_keeps_its_vectors(self) -> None:
        world = build_world()

        await world.cascade.execute(WORKSPACE)

        # The other workspace can still retrieve from it. A count of zero here
        # would be a silent retrieval outage for a tenant who deleted nothing.
        assert world.index.vectors.get(SHARED_PAPER) == 5

    async def test_the_shared_paper_keeps_its_blobs(self) -> None:
        world = build_world()

        await world.cascade.execute(WORKSPACE)

        assert set(blobs_for(SHARED_PAPER)) <= world.store.keys

    async def test_the_other_workspace_still_holds_it(self) -> None:
        world = build_world()

        await world.cascade.execute(WORKSPACE)

        assert world.papers.memberships[OTHER_WORKSPACE] == {SHARED_PAPER}


class TestAnUnsharedPaperLeavesNoResidue:
    """FR-W3's actual test: zero residue in every store."""

    async def test_the_row_is_gone(self) -> None:
        world = build_world()
        await world.cascade.execute(WORKSPACE)
        assert PRIVATE_PAPER not in world.papers.papers

    async def test_the_vectors_are_gone(self) -> None:
        world = build_world()
        await world.cascade.execute(WORKSPACE)
        assert PRIVATE_PAPER not in world.index.vectors

    async def test_every_blob_under_the_prefix_goes_not_only_the_named_ones(self) -> None:
        # The regression this exists for: M1 purged `source_key` plus the
        # `tei_key` for the parser version the row recorded, which leaks the TEI
        # of every other generation — and D49 keeps those deliberately, so the
        # leak was by design rather than by accident.
        world = build_world()

        await world.cascade.execute(WORKSPACE)

        leaked = [key for key in world.store.keys if key.startswith(f"papers/{PRIVATE_PAPER}/")]
        assert leaked == [], f"blobs survived the purge: {leaked}"

    async def test_the_workspace_row_is_gone(self) -> None:
        world = build_world()
        await world.cascade.execute(WORKSPACE)
        assert WORKSPACE not in world.workspaces.live


class TestTheOrderingIsCrashSafe:
    """Where a crash lands decides whether the residue is recoverable."""

    async def test_vectors_go_before_the_paper_row(self) -> None:
        # D62's ordering, inverted from ingestion's D53 for the same reason: a
        # vector whose chunk row is gone is the orphan the sweep is built to
        # find, while a chunk row whose vector is gone is silently
        # unretrievable with nothing anywhere reporting it.
        world = build_world()

        await world.cascade.execute(WORKSPACE)

        journal = world.journal
        assert journal.index_of(f"delete_vectors:{PRIVATE_PAPER}") < journal.index_of(
            f"delete_paper:{PRIVATE_PAPER}"
        )

    async def test_the_workspace_row_goes_last(self) -> None:
        # The mark on this row is what makes the work findable (D99). Deleting
        # it first would strand every paper still attached, with nothing left
        # to say a cascade was ever owed.
        world = build_world()

        await world.cascade.execute(WORKSPACE)

        journal = world.journal
        assert journal.events[-1] == f"purge_workspace:{WORKSPACE}"


class TestTheCascadeConverges:
    """§5.6.4 step 3: every step idempotent and restartable."""

    async def test_running_it_twice_changes_nothing_the_second_time(self) -> None:
        world = build_world()
        first = await world.cascade.execute(WORKSPACE)

        # The mark is gone after the first run, so the second is refused — which
        # is itself the idempotence: a redelivered task cannot double-delete.
        second = await world.cascade.execute(WORKSPACE)

        assert first.swept is True
        assert second.swept is False
        assert SHARED_PAPER in world.papers.papers

    async def test_a_cascade_resumed_after_the_memberships_went_still_purges(self) -> None:
        # The crash case: a previous attempt dropped the membership and died
        # before purging. The paper is now held by nobody and reachable by no
        # scoped query, so if the resumed cascade skipped it on "not a member"
        # it would be stranded forever.
        world = build_world(
            memberships={WORKSPACE: set(), OTHER_WORKSPACE: set()},
            papers={PRIVATE_PAPER},
        )

        await world.cascade.execute(WORKSPACE)

        # The cascade itself cannot see it — it iterates memberships, and there
        # are none — so it survives here and the SWEEP is what must catch it.
        assert PRIVATE_PAPER in world.papers.papers

        await world.sweep.execute()

        assert PRIVATE_PAPER not in world.papers.papers


class TestTheSweepFindsWhatNothingElseCan:
    """D102's three targets, plus the cascades that never finished."""

    async def test_an_orphaned_paper_is_purged(self) -> None:
        world = build_world(deleting=set(), memberships={WORKSPACE: set()}, papers={PRIVATE_PAPER})

        result = await world.sweep.execute()

        assert result.papers_purged == 1
        assert PRIVATE_PAPER not in world.papers.papers

    async def test_an_orphaned_papers_blobs_go_with_it(self) -> None:
        world = build_world(deleting=set(), memberships={WORKSPACE: set()}, papers={PRIVATE_PAPER})

        await world.sweep.execute()

        assert not [k for k in world.store.keys if k.startswith(f"papers/{PRIVATE_PAPER}/")]

    async def test_vectors_whose_paper_is_gone_are_removed(self) -> None:
        # Postgres knows about no papers at all; the index still holds two
        # papers' worth of points. That is precisely NFR-R6's claim under test:
        # the relational store is the source of truth.
        world = build_world(deleting=set(), memberships={}, papers=set())

        result = await world.sweep.execute()

        assert result.vectors_deleted == 12
        assert world.index.vectors == {}

    async def test_blobs_whose_paper_is_gone_are_removed(self) -> None:
        world = build_world(deleting=set(), memberships={}, papers=set())

        await world.sweep.execute()

        assert world.store.keys == set()

    async def test_a_key_that_is_not_paper_shaped_is_left_alone(self) -> None:
        # The sweep removes residue it can PROVE is residue. An unrecognised key
        # is evidence of something it does not model, and deleting it would make
        # this a garbage collector for the whole bucket.
        world = build_world(
            deleting=set(),
            memberships={},
            papers=set(),
            extra_keys=["exports/report.csv", "papers/", "papers/stray-file"],
        )

        await world.sweep.execute()

        assert world.store.keys == {"exports/report.csv", "papers/", "papers/stray-file"}

    async def test_an_unfinished_cascade_is_resumed(self) -> None:
        # The workspace was marked but its dispatch never reached the broker.
        # Nothing else will ever run it: `list_deleting` is the only thing that
        # knows the work is owed (D99).
        world = build_world()

        result = await world.sweep.execute()

        assert result.cascades_resumed == 1
        assert WORKSPACE not in world.workspaces.live
        assert PRIVATE_PAPER not in world.papers.papers

    async def test_the_live_set_is_read_after_the_cascades_run(self) -> None:
        # Ordering, and it is load-bearing: reading the live set first would
        # treat papers the cascade is about to delete as live, so their vectors
        # would survive the pass that deleted their rows — the exact
        # inconsistency this sweep exists to remove, created by the sweep.
        world = build_world()

        await world.sweep.execute()

        journal = world.journal
        assert journal.index_of("all_paper_ids") > journal.index_of(f"purge_workspace:{WORKSPACE}")

    async def test_a_clean_system_reports_nothing(self) -> None:
        # The number is the point: non-zero means a defect upstream. A sweep
        # that always reported activity would make residue unnoticeable.
        world = build_world(
            deleting=set(),
            memberships={WORKSPACE: {PRIVATE_PAPER}, OTHER_WORKSPACE: {SHARED_PAPER}},
        )

        result = await world.sweep.execute()

        assert result.cascades_resumed == 0
        assert result.papers_purged == 0
        assert result.vectors_deleted == 0
        assert result.blobs_deleted == 0


class TestTheEmptyLiveSetIsNotNothingToDo:
    """The inverted-guard case, called out in the port contract.

    An empty live set means Postgres holds no papers, so EVERY vector and blob
    is an orphan. An implementation that reads empty as "no filter, skip" would
    invert the guarantee exactly when the index is most wrong.
    """

    async def test_no_papers_means_every_vector_is_an_orphan(self) -> None:
        world = build_world(deleting=set(), memberships={}, papers=set())

        deleted = await world.index.delete_orphans(())

        assert deleted == 12
        assert world.index.vectors == {}


class TestThePurgerRefusesToDeleteAHeldPaper:
    """The check that makes AD-8 survive every caller, not just the two here."""

    @pytest.mark.parametrize("held_by", [WORKSPACE, OTHER_WORKSPACE])
    async def test_a_paper_any_workspace_holds_is_untouched(self, held_by: str) -> None:
        world = build_world(memberships={held_by: {SHARED_PAPER}})
        purger = PaperPurger(
            papers=world.papers,  # type: ignore[arg-type]
            index=world.index,  # type: ignore[arg-type]
            store=world.store,  # type: ignore[arg-type]
        )

        result = await purger.purge_if_unheld(SHARED_PAPER)

        assert result.purged is False
        assert world.journal.events == []
        assert SHARED_PAPER in world.papers.papers
