"""Adding discovered papers by identifier, against fakes (D131, D133, D137, FR-I7).

What fails silently here is what these tests are about: an add that asks a source
before it knows the workspace is the caller's, a batch that half-commits, a paper
created from a record nobody can fetch, a work added twice under two spellings of
its identity, and a job that carries links the fetcher will refuse.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from scholiast.application.ports.discovery import SourceFailure, SourceUnavailable
from scholiast.application.use_cases.add_discovered import (
    AddDiscoveredPapersUseCase,
    AddRefusal,
    AddRefused,
    DiscoveredRef,
    SourceLookupFailed,
)
from scholiast.domain.discovery import LinkKind, OpenAccessLink, PaperCandidate
from scholiast.domain.errors import WorkspaceCapacityExceeded, WorkspaceNotAvailable
from scholiast.domain.fetch_policy import host_allowlist
from scholiast.domain.paper import Paper, PaperOrigin, PaperVisibility, ParseStatus
from scholiast.domain.workspace import (
    MAX_PAPERS_PER_WORKSPACE,
    PaperMembership,
    PaperState,
    Workspace,
)

OWNER = "11111111-1111-4111-8111-111111111111"
WORKSPACE = "22222222-2222-4222-8222-222222222222"
ALLOWED = host_allowlist(["arxiv.org", "aclanthology.org", "www.aclweb.org"])
ARXIV_PDF = "https://arxiv.org/pdf/2004.04906v3"


def record(
    source: str = "arxiv",
    source_id: str = "2004.04906v3",
    *,
    title: str = "Dense Passage Retrieval for Open-Domain Question Answering",
    doi: str | None = None,
    arxiv_id: str | None = "2004.04906",
    links: Sequence[OpenAccessLink] = (OpenAccessLink(ARXIV_PDF, LinkKind.PDF),),
) -> PaperCandidate:
    return PaperCandidate(
        source=source,
        source_id=source_id,
        rank=1,
        title=title,
        authors=("Vladimir Karpukhin", "Barlas Oguz"),
        year=2020,
        venue="EMNLP",
        abstract="Open-domain question answering relies on efficient passage retrieval.",
        citation_count=None,
        doi=doi,
        arxiv_id=arxiv_id,
        oa_links=tuple(links),
        adapter_version=f"{source}-2",
    )


class FakeSource:
    """A source that answers from a fixed set of records and counts every lookup."""

    def __init__(self, name: str, *records: PaperCandidate) -> None:
        self.name = name
        self.records = {r.source_id: r for r in records}
        self.lookups: list[list[str]] = []
        self.raises: BaseException | None = None
        self.hang = False

    def is_valid_id(self, source_id: str) -> bool:
        return not source_id.startswith("bad")

    async def search(self, query: str, *, limit: int) -> object:
        raise AssertionError("an add never searches")

    async def lookup(self, source_ids: Sequence[str]) -> Mapping[str, PaperCandidate]:
        self.lookups.append(list(source_ids))
        if self.hang:
            await asyncio.Event().wait()
        if self.raises is not None:
            raise self.raises
        return {i: self.records[i] for i in source_ids if i in self.records}


class FakeWorkspaces:
    def __init__(self, workspace: Workspace | None) -> None:
        self.workspace = workspace

    async def get(self, workspace_id: str) -> Workspace | None:
        if self.workspace is None or self.workspace.id != workspace_id:
            return None
        return self.workspace


class FakePapers:
    def __init__(self, *existing: Paper) -> None:
        self.by_id = {p.id: p for p in existing}
        self.added: list[Paper] = []
        self.memberships: list[tuple[str, str, PaperState]] = []

    async def find_by_identity(self, *, arxiv_id: str | None, doi: str | None) -> Paper | None:
        mine = [
            p
            for p in self.by_id.values()
            if p.owner_id == OWNER and p.visibility is PaperVisibility.PRIVATE
        ]
        for key, value in (("arxiv_id", arxiv_id), ("doi", doi)):
            if value is not None:
                for paper in mine:
                    if getattr(paper, key) == value:
                        return paper
        return None

    async def add(self, paper: Paper) -> Paper:
        self.added.append(paper)
        self.by_id[paper.id] = paper
        return paper

    async def add_to_workspace(
        self, workspace_id: str, paper_id: str, state: PaperState
    ) -> PaperMembership:
        self.memberships.append((workspace_id, paper_id, state))
        return PaperMembership(
            workspace_id=workspace_id, paper_id=paper_id, state=state, added_at=datetime.now(UTC)
        )


class FakeChunks:
    def __init__(self, indexed: Sequence[str] = ()) -> None:
        self._indexed = set(indexed)

    async def list_for_paper(self, paper_id: str) -> list[object]:
        return [SimpleNamespace(is_indexed=True)] if paper_id in self._indexed else []


class FakeQueue:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, str, tuple[str, ...]]] = []

    async def enqueue(
        self, *, workspace_id: str, paper_id: str, source_links: Sequence[str] = ()
    ) -> str:
        self.jobs.append((workspace_id, paper_id, tuple(source_links)))
        return f"job-{len(self.jobs)}"


class World:
    def __init__(
        self,
        *,
        sources: Sequence[FakeSource] = (),
        existing: Sequence[Paper] = (),
        indexed: Sequence[str] = (),
        workspace: Workspace | None = None,
        deadline: float = 10.0,
    ) -> None:
        self.sources = {s.name: s for s in sources}
        self.workspace = workspace or Workspace(id=WORKSPACE, owner_id=OWNER, name="Reading")
        self.workspaces = FakeWorkspaces(self.workspace)
        self.papers = FakePapers(*existing)
        self.queue = FakeQueue()
        self.use_case = AddDiscoveredPapersUseCase(
            workspaces=self.workspaces,
            papers=self.papers,
            chunks=FakeChunks(indexed),
            queue=self.queue,
            sources=self.sources,
            allowed_hosts=ALLOWED,
            owner_id=OWNER,
            lookup_deadline_seconds=deadline,
        )

    async def add(self, *refs: tuple[str, str], workspace_id: str = WORKSPACE):  # type: ignore[no-untyped-def]
        return await asyncio.wait_for(
            self.use_case.execute(
                workspace_id=workspace_id, refs=[DiscoveredRef(s, i) for s, i in refs]
            ),
            timeout=2.0,
        )

    @property
    def nothing_written(self) -> bool:
        return not self.papers.added and not self.papers.memberships and not self.queue.jobs


def owned_paper(**overrides: object) -> Paper:
    values: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "content_hash": "a" * 64,
        "visibility": PaperVisibility.PRIVATE,
        "owner_id": OWNER,
        "title": "already mine",
    }
    values.update(overrides)
    return Paper(**values)  # type: ignore[arg-type]


def member(paper_id: str, state: PaperState = PaperState.INDEXED) -> PaperMembership:
    return PaperMembership(
        workspace_id=WORKSPACE, paper_id=paper_id, state=state, added_at=datetime.now(UTC)
    )


class TestNothingIsAskedOrWrittenForAnAddThatCannotSucceed:
    async def test_an_unknown_source_or_malformed_id_is_refused_before_any_source_is_asked(
        self,
    ) -> None:
        arxiv = FakeSource("arxiv", record())
        world = World(sources=[arxiv])

        with pytest.raises(AddRefused) as caught:
            await world.add(("crossref", "10.1/x"), ("arxiv", "bad-id"), ("arxiv", "2004.04906v3"))

        assert [(r.ref.source, r.reason) for r in caught.value.refused] == [
            ("crossref", AddRefusal.UNKNOWN_SOURCE),
            ("arxiv", AddRefusal.INVALID_ID),
        ]
        assert arxiv.lookups == [], "a malformed id cost the source a request"
        assert world.nothing_written

    async def test_another_tenants_workspace_is_a_404_that_asks_no_source(self) -> None:
        # FR-A2: the refusal precedes every outbound call, so an add aimed at a
        # workspace that is not yours cannot make this server query anyone.
        arxiv = FakeSource("arxiv", record())
        world = World(sources=[arxiv])

        with pytest.raises(WorkspaceNotAvailable):
            await world.add(("arxiv", "2004.04906v3"), workspace_id=str(uuid.uuid4()))

        assert arxiv.lookups == []
        assert world.nothing_written

    async def test_one_source_down_adds_nothing_even_from_the_source_that_answered(self) -> None:
        arxiv = FakeSource("arxiv", record())
        openalex = FakeSource("openalex", record("openalex", "W3015883388"))
        openalex.raises = SourceUnavailable(
            "openalex", SourceFailure.RATE_LIMITED, "HTTP 429", retry_after=7.0
        )
        world = World(sources=[arxiv, openalex])

        with pytest.raises(SourceLookupFailed) as caught:
            await world.add(("arxiv", "2004.04906v3"), ("openalex", "W3015883388"))

        outcomes = {o.source: (o.failure, o.retry_after) for o in caught.value.sources}
        assert outcomes == {
            "arxiv": (None, None),
            "openalex": (SourceFailure.RATE_LIMITED, 7.0),
        }
        assert world.nothing_written

    async def test_a_source_that_hangs_is_cut_off_at_the_deadline(self) -> None:
        arxiv = FakeSource("arxiv", record())
        arxiv.hang = True
        world = World(sources=[arxiv], deadline=0.05)

        with pytest.raises(SourceLookupFailed) as caught:
            await world.add(("arxiv", "2004.04906v3"))

        assert [o.failure for o in caught.value.sources] == [SourceFailure.TIMEOUT]
        assert world.nothing_written

    async def test_a_crash_in_one_adapter_is_that_source_failing_not_a_500(self) -> None:
        arxiv = FakeSource("arxiv", record())
        arxiv.raises = KeyError("entry")
        world = World(sources=[arxiv])

        with pytest.raises(SourceLookupFailed) as caught:
            await world.add(("arxiv", "2004.04906v3"))

        assert [o.failure for o in caught.value.sources] == [SourceFailure.ERROR]


class TestOneRefusedPaperRefusesTheWholeBatch:
    async def test_an_id_the_source_does_not_know_refuses_the_papers_it_does(self) -> None:
        world = World(sources=[FakeSource("arxiv", record())])

        with pytest.raises(AddRefused) as caught:
            await world.add(("arxiv", "2004.04906v3"), ("arxiv", "2110.01599"))

        assert [(r.ref.source_id, r.reason) for r in caught.value.refused] == [
            ("2110.01599", AddRefusal.NOT_FOUND)
        ]
        assert world.nothing_written, "a batch half-committed"

    async def test_a_record_with_no_identifier_cannot_be_tracked_so_is_refused(self) -> None:
        untrackable = record("openalex", "W12", doi=None, arxiv_id=None)
        world = World(sources=[FakeSource("openalex", untrackable)])

        with pytest.raises(AddRefused) as caught:
            await world.add(("openalex", "W12"))

        assert [r.reason for r in caught.value.refused] == [AddRefusal.NO_IDENTIFIER]
        assert world.nothing_written

    @pytest.mark.parametrize(
        "links",
        [
            (),
            # The spike never found a PDF behind a landing page (D137).
            (OpenAccessLink("https://aclanthology.org/2020.emnlp-main.550/", LinkKind.LANDING),),
            # A PDF, but on a host the fetcher refuses (D130).
            (OpenAccessLink("https://ojs.aaai.org/paper.pdf", LinkKind.PDF),),
            # The allowlist matches hosts exactly, so a lookalike is no exception.
            (OpenAccessLink("https://arxiv.org.evil.test/pdf/1", LinkKind.PDF),),
        ],
    )
    async def test_a_paper_nothing_could_fetch_is_refused_rather_than_queued_to_fail(
        self, links: tuple[OpenAccessLink, ...]
    ) -> None:
        world = World(sources=[FakeSource("arxiv", record(links=links))])

        with pytest.raises(AddRefused) as caught:
            await world.add(("arxiv", "2004.04906v3"))

        assert [r.reason for r in caught.value.refused] == [AddRefusal.NO_FETCHABLE_PDF]
        assert world.nothing_written

    async def test_every_refused_paper_is_named_at_once(self) -> None:
        source = FakeSource(
            "openalex",
            record("openalex", "W12", doi=None, arxiv_id=None),
            record("openalex", "W13", doi="10.1/a", arxiv_id=None, links=()),
        )
        world = World(sources=[source])

        with pytest.raises(AddRefused) as caught:
            await world.add(("openalex", "W12"), ("openalex", "W13"), ("openalex", "W14"))

        assert [(r.ref.source_id, r.reason) for r in caught.value.refused] == [
            ("W12", AddRefusal.NO_IDENTIFIER),
            ("W13", AddRefusal.NO_FETCHABLE_PDF),
            ("W14", AddRefusal.NOT_FOUND),
        ]


class TestANewDiscoveredPaper:
    async def test_it_is_private_to_the_adder_and_awaits_its_pdf(self) -> None:
        world = World(sources=[FakeSource("arxiv", record())])

        (added,) = await world.add(("arxiv", "2004.04906v3"))

        (paper,) = world.papers.added
        assert added.result.paper_id == paper.id
        assert paper.visibility is PaperVisibility.PRIVATE and paper.owner_id == OWNER
        assert paper.content_hash is None and paper.parse_status is ParseStatus.PENDING
        assert paper.origin is PaperOrigin.DISCOVERY
        assert added.result.state is PaperState.QUEUED and added.result.deduplicated is False

    async def test_its_metadata_is_the_sources_record_not_the_clients(self) -> None:
        world = World(sources=[FakeSource("arxiv", record())])

        await world.add(("arxiv", "2004.04906v3"))

        (paper,) = world.papers.added
        assert paper.title == "Dense Passage Retrieval for Open-Domain Question Answering"
        assert paper.authors == ("Vladimir Karpukhin", "Barlas Oguz")
        assert (paper.year, paper.venue) == (2020, "EMNLP")
        assert paper.arxiv_id == "2004.04906" and paper.doi is None

    async def test_arxivs_own_doi_becomes_the_arxiv_id_so_one_work_has_one_identity(
        self,
    ) -> None:
        # D132: OpenAlex keeps arXiv's DataCite DOI as the DOI. Kept as one, the same
        # work added through arXiv would not be found by it, and the user would hold it
        # twice.
        datacite = record(
            "openalex",
            "W4389984066",
            doi="10.48550/arxiv.2312.10997",
            arxiv_id=None,
            links=(OpenAccessLink("https://arxiv.org/pdf/2312.10997", LinkKind.PDF),),
        )
        world = World(sources=[FakeSource("openalex", datacite)])

        await world.add(("openalex", "W4389984066"))

        (paper,) = world.papers.added
        assert (paper.doi, paper.arxiv_id) == (None, "2312.10997")

    async def test_its_job_carries_only_the_links_the_fetcher_would_follow_in_order(
        self,
    ) -> None:
        links = (
            OpenAccessLink("https://aclanthology.org/2020.emnlp-main.550/", LinkKind.LANDING),
            OpenAccessLink(
                "https://www.aclweb.org/anthology/2020.emnlp-main.550.pdf", LinkKind.PDF
            ),
            OpenAccessLink("https://ojs.aaai.org/paper.pdf", LinkKind.PDF),
            OpenAccessLink("http://arxiv.org/pdf/2004.04906", LinkKind.PDF),
        )
        world = World(
            sources=[FakeSource("openalex", record("openalex", "W3015883388", links=links))]
        )

        (added,) = await world.add(("openalex", "W3015883388"))

        assert world.queue.jobs == [
            (
                WORKSPACE,
                added.result.paper_id,
                # As the source gave them: the fetcher re-checks, and upgrades, each hop.
                (
                    "https://www.aclweb.org/anthology/2020.emnlp-main.550.pdf",
                    "http://arxiv.org/pdf/2004.04906",
                ),
            )
        ]

    async def test_a_batch_costs_each_source_one_lookup_and_each_id_once(self) -> None:
        arxiv = FakeSource(
            "arxiv",
            record(),
            record(source_id="2110.01599v1", title="Another", arxiv_id="2110.01599"),
        )
        world = World(sources=[arxiv])

        added = await world.add(
            ("arxiv", "2004.04906v3"), ("arxiv", "2110.01599v1"), ("arxiv", "2004.04906v3")
        )

        assert arxiv.lookups == [["2004.04906v3", "2110.01599v1"]]
        assert [a.ref.source_id for a in added] == ["2004.04906v3", "2110.01599v1", "2004.04906v3"]
        assert len(world.papers.added) == 2 and len(world.queue.jobs) == 2
        # The repeat is the first paper again, already here, with nothing queued.
        assert added[2].result.paper_id == added[0].result.paper_id
        assert (added[2].result.deduplicated, added[2].result.job_id) == (True, None)


class TestReAddingByIdentifierDoesNoWork:
    """FR-I7's external-ID half, moved to M4 by D136."""

    async def test_a_paper_already_in_this_workspace_is_left_alone(self) -> None:
        mine = owned_paper(arxiv_id="2004.04906")
        world = World(sources=[FakeSource("arxiv", record())], existing=[mine])
        world.workspace.memberships.append(member(mine.id))

        (added,) = await world.add(("arxiv", "2004.04906v3"))

        assert added.result.paper_id == mine.id
        assert (added.result.deduplicated, added.result.job_id) == (True, None)
        assert world.nothing_written

    async def test_an_indexed_paper_from_another_workspace_joins_indexed_with_no_job(
        self,
    ) -> None:
        mine = owned_paper(arxiv_id="2004.04906")
        world = World(sources=[FakeSource("arxiv", record())], existing=[mine], indexed=[mine.id])

        (added,) = await world.add(("arxiv", "2004.04906v3"))

        assert added.result.state is PaperState.INDEXED
        assert world.queue.jobs == [] and world.papers.added == []

    async def test_the_arxiv_id_is_matched_before_the_doi(self) -> None:
        # D132 and `find_by_identity`: the arXiv id is what the fetch path runs on.
        by_doi = owned_paper(doi="10.18653/v1/2020.emnlp-main.550")
        by_arxiv = owned_paper(arxiv_id="2004.04906")
        both = record("openalex", "W3015883388", doi="10.18653/v1/2020.emnlp-main.550")
        world = World(sources=[FakeSource("openalex", both)], existing=[by_doi, by_arxiv])

        (added,) = await world.add(("openalex", "W3015883388"))

        assert added.result.paper_id == by_arxiv.id

    async def test_a_paper_already_fetched_needs_no_fetchable_link_to_be_added_again(
        self,
    ) -> None:
        mine = owned_paper(arxiv_id="2004.04906")
        world = World(sources=[FakeSource("arxiv", record(links=()))], existing=[mine])

        (added,) = await world.add(("arxiv", "2004.04906v3"))

        assert added.result.paper_id == mine.id
        # Its job parses the PDF already stored, so it carries no links at all.
        assert world.queue.jobs == [(WORKSPACE, mine.id, ())]

    async def test_a_job_for_a_paper_that_has_its_pdf_carries_no_links(self) -> None:
        # Links on a job mean "this job fetches" (D134). A paper with its PDF parses
        # what is stored, and links on its job would only mislead whoever reads it.
        mine = owned_paper(arxiv_id="2004.04906")
        world = World(sources=[FakeSource("arxiv", record())], existing=[mine])

        await world.add(("arxiv", "2004.04906v3"))

        assert world.queue.jobs == [(WORKSPACE, mine.id, ())]

    async def test_a_paper_still_awaiting_its_pdf_joins_with_links_for_its_own_job(self) -> None:
        pending = owned_paper(content_hash=None, arxiv_id="2004.04906")
        world = World(sources=[FakeSource("arxiv", record())], existing=[pending])

        (added,) = await world.add(("arxiv", "2004.04906v3"))

        assert added.result.paper_id == pending.id
        assert world.queue.jobs == [(WORKSPACE, pending.id, (ARXIV_PDF,))]

    async def test_a_paper_awaiting_its_pdf_with_nothing_to_fetch_it_from_is_refused(
        self,
    ) -> None:
        pending = owned_paper(content_hash=None, arxiv_id="2004.04906")
        world = World(sources=[FakeSource("arxiv", record(links=()))], existing=[pending])

        with pytest.raises(AddRefused) as caught:
            await world.add(("arxiv", "2004.04906v3"))

        assert [r.reason for r in caught.value.refused] == [AddRefusal.NO_FETCHABLE_PDF]

    async def test_one_work_named_twice_in_one_batch_becomes_one_paper(self) -> None:
        # The arXiv record and OpenAlex's record of the same work, picked together.
        arxiv = FakeSource("arxiv", record())
        openalex = FakeSource(
            "openalex",
            record(
                "openalex",
                "W3015883388",
                doi="10.18653/v1/2020.emnlp-main.550",
                links=(OpenAccessLink("https://arxiv.org/pdf/2004.04906", LinkKind.PDF),),
            ),
        )
        world = World(sources=[arxiv, openalex])

        first, second = await world.add(("arxiv", "2004.04906v3"), ("openalex", "W3015883388"))

        assert len(world.papers.added) == 1 and len(world.queue.jobs) == 1
        assert second.result.paper_id == first.result.paper_id
        assert second.result.deduplicated is True


class TestTheCapIsCheckedForTheWholeBatch:
    def _full_to(self, world: World, count: int) -> None:
        world.workspace.memberships.extend(member(str(uuid.uuid4())) for _ in range(count))

    async def test_a_batch_that_would_pass_the_cap_adds_none_of_its_papers(self) -> None:
        arxiv = FakeSource(
            "arxiv",
            record(),
            record(source_id="2110.01599v1", arxiv_id="2110.01599"),
            record(source_id="2010.08191v2", arxiv_id="2010.08191"),
        )
        world = World(sources=[arxiv])
        self._full_to(world, MAX_PAPERS_PER_WORKSPACE - 2)

        with pytest.raises(WorkspaceCapacityExceeded):
            await world.add(
                ("arxiv", "2004.04906v3"), ("arxiv", "2110.01599v1"), ("arxiv", "2010.08191v2")
            )

        assert world.nothing_written, "the cap was found part-way through the batch"

    async def test_papers_already_here_take_no_room(self) -> None:
        mine = owned_paper(arxiv_id="2004.04906")
        arxiv = FakeSource(
            "arxiv", record(), record(source_id="2110.01599v1", arxiv_id="2110.01599")
        )
        world = World(sources=[arxiv], existing=[mine])
        world.workspace.memberships.append(member(mine.id))
        self._full_to(world, MAX_PAPERS_PER_WORKSPACE - 2)

        added = await world.add(("arxiv", "2004.04906v3"), ("arxiv", "2110.01599v1"))

        assert len(world.workspace.active_memberships) == MAX_PAPERS_PER_WORKSPACE
        assert [a.result.deduplicated for a in added] == [True, False]
