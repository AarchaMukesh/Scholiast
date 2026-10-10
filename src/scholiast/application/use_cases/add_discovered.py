"""Add discovered papers by identifier: each looked up at its source, then queued to fetch (D133).

The synchronous half, as `AddPaperUseCase` is for uploads: everything the user waits
for when they click add, ending in committed rows — a paper, a membership and a job
in one transaction (D47). The bytes arrive later, in the worker's FETCHING stage
(D134). Order matters and is not arbitrary:

  ids        checked against each source's syntax before anything else, so a
             malformed id costs the caller a 422 and the source nothing (D133).

  workspace  before any source is asked, so an add aimed at another tenant's
             workspace is a 404 that sent nothing anywhere (FR-A2).

  lookup     one request per source for the whole batch, all sharing one 10 s
             deadline (D137). The record a paper is made from is the source's own,
             fetched now: the client supplies only `{source, source_id}`, never a
             URL and never metadata.

  plan       every item resolved before anything is written: this owner's
             existing paper for the work (arXiv id first, then DOI, D132), or a new
             one. An item with no identifier, or with nothing the fetcher may follow
             when a fetch is needed (D137), refuses the whole batch: all or nothing,
             so resending the same batch is safe (D133). The cap is checked on the
             whole batch here too.

  write      new papers private to the adder (D131), awaiting their PDF; a
             membership through the workspace aggregate; and a job carrying the
             links its fetch will try.

This use case names no vendor. Sources arrive by name, as their adapters report it.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum, unique

from scholiast.application.ports.discovery import (
    MAX_LOOKUP_IDS,
    AcademicSource,
    SourceFailure,
    SourceUnavailable,
)
from scholiast.application.ports.queue import JobPriority, Queue
from scholiast.application.ports.repositories import (
    ChunkRepository,
    PaperRepository,
    WorkspaceRepository,
)
from scholiast.application.use_cases.add_paper import AddPaperResult, attach_existing
from scholiast.domain.discovery import PaperCandidate, paper_identity
from scholiast.domain.errors import WorkspaceNotAvailable
from scholiast.domain.fetch_policy import fetchable_pdf_links
from scholiast.domain.paper import Paper, PaperOrigin, PaperVisibility
from scholiast.domain.workspace import Workspace

#: D137: room for arXiv's 3 s pacing wait and a slow answer, and no more.
LOOKUP_DEADLINE_SECONDS = 10.0
#: D133: one add carries at most this many papers, which is also one lookup's limit.
MAX_ADD_PAPERS = MAX_LOOKUP_IDS


@dataclass(frozen=True, slots=True)
class DiscoveredRef:
    """One paper to add, exactly as search returned it."""

    source: str
    source_id: str


@unique
class AddRefusal(StrEnum):
    """Why one paper in an add was refused (D133, D137). A fixed set, so a UI can enumerate it."""

    #: No source by that name is configured here.
    UNKNOWN_SOURCE = "unknown_source"
    #: Not an id the named source issues.
    INVALID_ID = "invalid_id"
    #: The source has no record with that id.
    NOT_FOUND = "not_found"
    #: The record has neither DOI nor arXiv id, so the paper could not be tracked.
    NO_IDENTIFIER = "no_identifier"
    #: A fetch is needed and none of the record's PDF links passes D115's guard.
    NO_FETCHABLE_PDF = "no_fetchable_pdf"


@dataclass(frozen=True, slots=True)
class RefusedRef:
    ref: DiscoveredRef
    reason: AddRefusal


class AddRefused(Exception):
    """At least one paper in the batch cannot be added, so none is (D133)."""

    def __init__(self, refused: tuple[RefusedRef, ...]) -> None:
        super().__init__(
            "add refused: "
            + ", ".join(f"{r.ref.source}:{r.ref.source_id}={r.reason}" for r in refused)
        )
        self.refused = refused


@dataclass(frozen=True, slots=True)
class LookupOutcome:
    """One source's part in one add's lookups."""

    source: str
    #: None when the source answered.
    failure: SourceFailure | None = None
    retry_after: float | None = None
    #: What went wrong, for the caller's log. Never sent to a client.
    error: BaseException | None = field(default=None, compare=False, repr=False)


class SourceLookupFailed(Exception):
    """A source the batch needs did not answer, so nothing was added (D133): a 503."""

    def __init__(self, sources: tuple[LookupOutcome, ...]) -> None:
        super().__init__(
            "lookup failed: " + ", ".join(f"{o.source}={o.failure or 'ok'}" for o in sources)
        )
        self.sources = sources


@dataclass(frozen=True, slots=True)
class AddedPaper:
    """What happened to one paper of the batch, in the batch's order."""

    ref: DiscoveredRef
    result: AddPaperResult


@dataclass(slots=True)
class _Planned:
    """One item, resolved: the paper it lands on and the links its job would carry."""

    ref: DiscoveredRef
    paper: Paper
    #: True for the first item that creates `paper`; every later one reuses it.
    creates: bool
    links: tuple[str, ...]


class AddDiscoveredPapersUseCase:
    """Add up to `MAX_ADD_PAPERS` discovered papers to a workspace, all or none."""

    def __init__(
        self,
        *,
        workspaces: WorkspaceRepository,
        papers: PaperRepository,
        chunks: ChunkRepository,
        queue: Queue,
        sources: Mapping[str, AcademicSource],
        allowed_hosts: frozenset[str],
        owner_id: str,
        lookup_deadline_seconds: float = LOOKUP_DEADLINE_SECONDS,
    ) -> None:
        self._workspaces = workspaces
        self._papers = papers
        self._chunks = chunks
        self._queue = queue
        self._sources = sources
        self._allowed_hosts = allowed_hosts
        self._owner_id = owner_id
        self._deadline = lookup_deadline_seconds

    async def execute(
        self, *, workspace_id: str, refs: Sequence[DiscoveredRef]
    ) -> list[AddedPaper]:
        if not 1 <= len(refs) <= MAX_ADD_PAPERS:
            raise ValueError(f"an add carries 1..{MAX_ADD_PAPERS} papers, got {len(refs)}")
        malformed = tuple(
            RefusedRef(ref, reason)
            for ref in refs
            if (reason := self._syntax_refusal(ref)) is not None
        )
        if malformed:
            raise AddRefused(malformed)

        workspace = await self._workspaces.get(workspace_id)
        if workspace is None:
            # Indistinguishable from "does not exist" on purpose (FR-A2).
            raise WorkspaceNotAvailable(workspace_id)

        records = await self._look_up(refs)
        plan = await self._plan(workspace, refs, records)
        workspace.ensure_room_for(item.paper.id for item in plan)

        added: list[AddedPaper] = []
        now = datetime.now(UTC)
        for item in plan:
            if item.creates:
                stored = await self._papers.add(item.paper)
                membership = workspace.add_paper(stored.id, now=now)
                await self._papers.add_to_workspace(workspace.id, stored.id, membership.state)
                job_id = await self._queue.enqueue(
                    workspace_id=workspace.id,
                    paper_id=stored.id,
                    # Discovered papers are backfill: an upload goes first (D139).
                    priority=JobPriority.BACKFILL,
                    source_links=item.links,
                )
                result = AddPaperResult(
                    paper_id=stored.id, state=membership.state, job_id=job_id, deduplicated=False
                )
            else:
                result = await attach_existing(
                    workspace,
                    item.paper,
                    papers=self._papers,
                    chunks=self._chunks,
                    queue=self._queue,
                    priority=JobPriority.BACKFILL,
                    source_links=item.links,
                )
            added.append(AddedPaper(item.ref, result))
        return added

    def _syntax_refusal(self, ref: DiscoveredRef) -> AddRefusal | None:
        source = self._sources.get(ref.source)
        if source is None:
            return AddRefusal.UNKNOWN_SOURCE
        if not source.is_valid_id(ref.source_id):
            return AddRefusal.INVALID_ID
        return None

    async def _look_up(self, refs: Sequence[DiscoveredRef]) -> dict[DiscoveredRef, PaperCandidate]:
        """Every source's records for this batch, one request each, under one deadline."""
        wanted: dict[str, list[str]] = {}
        for ref in refs:
            ids = wanted.setdefault(ref.source, [])
            if ref.source_id not in ids:
                ids.append(ref.source_id)
        names = list(wanted)
        deadline = asyncio.get_running_loop().time() + self._deadline
        answers = await asyncio.gather(*(self._ask(name, wanted[name], deadline) for name in names))
        outcomes = tuple(outcome for outcome, _ in answers)
        if any(outcome.failure is not None for outcome in outcomes):
            raise SourceLookupFailed(outcomes)
        found = {name: records for name, (_, records) in zip(names, answers, strict=True)}
        return {
            ref: found[ref.source][ref.source_id]
            for ref in refs
            if ref.source_id in found[ref.source]
        }

    async def _ask(
        self, name: str, ids: list[str], deadline: float
    ) -> tuple[LookupOutcome, Mapping[str, PaperCandidate]]:
        try:
            async with asyncio.timeout_at(deadline):
                records = await self._sources[name].lookup(ids)
        except TimeoutError as exc:
            # The deadline cut the request off; the source's breaker counted it.
            return LookupOutcome(name, SourceFailure.TIMEOUT, error=exc), {}
        except SourceUnavailable as exc:
            return LookupOutcome(name, exc.kind, exc.retry_after, error=exc), {}
        except Exception as exc:
            # A crash in one adapter is that source failing, as in a search (FR-D6).
            return LookupOutcome(name, SourceFailure.ERROR, error=exc), {}
        return LookupOutcome(name), records

    async def _plan(
        self,
        workspace: Workspace,
        refs: Sequence[DiscoveredRef],
        records: Mapping[DiscoveredRef, PaperCandidate],
    ) -> list[_Planned]:
        """Resolve every item before anything is written, refusing the batch if any fails.

        Items are resolved in order, as if each were added alone after the ones
        before it: a work named twice — say its arXiv record and its OpenAlex one —
        lands on one paper, the second item reusing what the first creates.
        """
        members = {m.paper_id for m in workspace.active_memberships}
        by_arxiv: dict[str, Paper] = {}
        by_doi: dict[str, Paper] = {}
        plan: list[_Planned] = []
        refused: list[RefusedRef] = []
        for ref in refs:
            record = records.get(ref)
            if record is None:
                refused.append(RefusedRef(ref, AddRefusal.NOT_FOUND))
                continue
            doi, arxiv_id = paper_identity(record.doi, record.arxiv_id)
            if doi is None and arxiv_id is None:
                refused.append(RefusedRef(ref, AddRefusal.NO_IDENTIFIER))
                continue
            links = self._fetchable(record)
            paper = await self._existing(arxiv_id, doi, by_arxiv, by_doi)
            creates = paper is None
            # A fetch is needed for a paper that will be created, and for one still
            # awaiting its PDF that joins this workspace with a job of its own.
            fetches = paper is None or (paper.content_hash is None and paper.id not in members)
            if fetches and not links:
                refused.append(RefusedRef(ref, AddRefusal.NO_FETCHABLE_PDF))
                continue
            if paper is None:
                paper = self._new_paper(record, doi=doi, arxiv_id=arxiv_id)
            for key, index in ((arxiv_id, by_arxiv), (doi, by_doi)):
                if key is not None:
                    index.setdefault(key, paper)
            members.add(paper.id)
            plan.append(_Planned(ref, paper, creates, links))
        if refused:
            raise AddRefused(tuple(refused))
        return plan

    async def _existing(
        self,
        arxiv_id: str | None,
        doi: str | None,
        by_arxiv: Mapping[str, Paper],
        by_doi: Mapping[str, Paper],
    ) -> Paper | None:
        """This owner's paper for the work, counting papers earlier items will create.

        `find_by_identity`'s precedence exactly, arXiv id and then DOI, with this
        batch's planned papers standing in for rows that do not exist yet.
        """
        if arxiv_id is not None:
            paper = by_arxiv.get(arxiv_id) or await self._papers.find_by_identity(
                arxiv_id=arxiv_id, doi=None
            )
            if paper is not None:
                return paper
        if doi is not None:
            return by_doi.get(doi) or await self._papers.find_by_identity(arxiv_id=None, doi=doi)
        return None

    def _fetchable(self, record: PaperCandidate) -> tuple[str, ...]:
        return fetchable_pdf_links(record, self._allowed_hosts)

    def _new_paper(self, record: PaperCandidate, *, doi: str | None, arxiv_id: str | None) -> Paper:
        return Paper(
            id=str(uuid.uuid4()),
            content_hash=None,
            # Private to the adder, like an upload (D131).
            visibility=PaperVisibility.PRIVATE,
            owner_id=self._owner_id,
            title=record.title,
            authors=record.authors,
            year=record.year,
            venue=record.venue,
            abstract=record.abstract,
            citation_count=record.citation_count,
            doi=doi,
            arxiv_id=arxiv_id,
            # The source's metadata stands; GROBID only fills gaps (D135).
            origin=PaperOrigin.DISCOVERY,
        )
