"""The schema's guarantees, asserted against a real Postgres.

These tests exist because §4.4.2's privacy carve-out is claimed to be enforced
*by the database*. That claim is only worth making if something tries to violate
it and gets rejected — so each test here attempts the violation.

Migrations are what create the schema, never `metadata.create_all()`. Testing
against `create_all` would verify the models and leave the migration — the thing
that actually runs against a real database — unexercised, which is precisely the
path that breaks in production.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.domain.errors import FailureClass

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]


#: Migration 0005 made `users.password_hash` NOT NULL (D85). These rows exist
#: to own data, never to log in, so they get a string that is not a valid
#: Argon2 encoding: `Argon2idHasher.verify` returns False for it against any
#: password, so the account is unreachable by construction rather than by
#: nobody knowing the password.
UNUSABLE_HASH = "not-a-real-argon2-hash"


@pytest.fixture(scope="module")
def postgres() -> Iterator[str]:
    """A throwaway Postgres with the migrations applied."""
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as container:
        url = container.get_connection_url()
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                "-m",
                "alembic",
                "-x",
                f"db_url={url}",
                "upgrade",
                "head",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")
        yield url


@pytest_asyncio.fixture
async def conn(postgres: str) -> AsyncIterator[AsyncConnection]:
    engine = create_async_engine(postgres)
    async with engine.connect() as connection:
        yield connection
    await engine.dispose()


async def _user(conn: AsyncConnection, email: str) -> uuid.UUID:
    user_id = uuid.uuid4()
    await conn.execute(
        text("INSERT INTO users (id, email, password_hash) VALUES (:id, :email, :hash)"),
        {"id": user_id, "email": email, "hash": UNUSABLE_HASH},
    )
    return user_id


async def _paper(
    conn: AsyncConnection,
    *,
    visibility: str,
    content_hash: str,
    owner_id: uuid.UUID | None,
    parse_status: str = "parsed",
    parse_failure_class: str | None = None,
) -> uuid.UUID:
    paper_id = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO papers (id, visibility, owner_id, content_hash, external_ids, "
            "title, authors, parse_status, parse_failure_class) VALUES "
            "(:id, :vis, :owner, :hash, '{}'::jsonb, 'A Paper', '[]'::jsonb, "
            ":status, :failure)"
        ),
        {
            "id": paper_id,
            "vis": visibility,
            "owner": owner_id,
            "hash": content_hash,
            "status": parse_status,
            "failure": parse_failure_class,
        },
    )
    return paper_id


class TestPrivacyCarveOut:
    """§4.4.2, enforced by the database rather than trusted to the application."""

    async def test_the_same_public_paper_cannot_be_registered_twice(
        self, conn: AsyncConnection
    ) -> None:
        # Open-access papers deduplicate globally so they are parsed and
        # embedded once (AD-8). Without this index, a race between two
        # simultaneous ingestions produces two registry entries and doubles the
        # vector index for that paper.
        await _paper(conn, visibility="public", content_hash="hash-shared", owner_id=None)
        with pytest.raises(IntegrityError, match="uq_papers_public_hash"):
            await _paper(conn, visibility="public", content_hash="hash-shared", owner_id=None)
        await conn.rollback()

    async def test_two_tenants_uploading_identical_bytes_get_separate_papers(
        self, conn: AsyncConnection
    ) -> None:
        # THE privacy test. An unpublished manuscript uploaded by two users must
        # produce two rows, because a shared registry entry would be reachable
        # through the other tenant's membership. Global dedup on content hash —
        # the obvious optimisation — is exactly the leak.
        alice = await _user(conn, "alice@example.com")
        bob = await _user(conn, "bob@example.com")

        a = await _paper(conn, visibility="private", content_hash="same", owner_id=alice)
        b = await _paper(conn, visibility="private", content_hash="same", owner_id=bob)

        assert a != b
        rows = (
            (
                await conn.execute(
                    text("SELECT owner_id FROM papers WHERE content_hash = 'same'"),
                )
            )
            .scalars()
            .all()
        )
        assert sorted(str(r) for r in rows) == sorted([str(alice), str(bob)])
        await conn.rollback()

    async def test_one_tenant_uploading_the_same_bytes_twice_is_deduplicated(
        self, conn: AsyncConnection
    ) -> None:
        # The other half: within one owner, re-uploading is idempotent (FR-I7),
        # so a user who uploads the same PDF twice does not pay to parse it twice.
        alice = await _user(conn, "alice2@example.com")
        await _paper(conn, visibility="private", content_hash="dup", owner_id=alice)
        with pytest.raises(IntegrityError, match="uq_papers_private_owner_hash"):
            await _paper(conn, visibility="private", content_hash="dup", owner_id=alice)
        await conn.rollback()

    async def test_a_private_paper_must_have_an_owner(self, conn: AsyncConnection) -> None:
        # An ownerless private paper is unreachable by any tenant filter — it
        # would be invisible data that still consumes an index slot.
        with pytest.raises(IntegrityError, match="ck_papers_private_iff_owned"):
            await _paper(conn, visibility="private", content_hash="orphan", owner_id=None)
        await conn.rollback()

    async def test_a_public_paper_must_not_be_owner_scoped(self, conn: AsyncConnection) -> None:
        # The reverse direction, which a naive NOT NULL check would miss: a
        # public paper carrying an owner_id claims global sharing and private
        # ownership at once.
        alice = await _user(conn, "alice3@example.com")
        with pytest.raises(IntegrityError, match="ck_papers_private_iff_owned"):
            await _paper(conn, visibility="public", content_hash="confused", owner_id=alice)
        await conn.rollback()


class TestTaxonomiesMatchTheDomain:
    """CHECK constraints are generated from the domain enums, so they cannot drift."""

    async def test_an_unknown_section_kind_is_rejected(self, conn: AsyncConnection) -> None:
        # Prevents a parser writing a section kind nothing downstream can
        # interpret — attribution and section priors both switch on this value,
        # so an unrecognised one degrades retrieval silently.
        paper = await _paper(conn, visibility="public", content_hash="kinds", owner_id=None)
        with pytest.raises(IntegrityError, match="ck_sections_kind"):
            await conn.execute(
                text(
                    "INSERT INTO sections (id, paper_id, kind, raw_title, path, order_index, "
                    "page_start, page_end, text) VALUES (:id, :paper, 'methodology', 'M', "
                    "'M', 0, 1, 2, 'body')"
                ),
                {"id": uuid.uuid4(), "paper": paper},
            )
        await conn.rollback()

    async def test_an_unknown_paper_state_is_rejected(self, conn: AsyncConnection) -> None:
        user = await _user(conn, "states@example.com")
        workspace = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO workspaces (id, owner_id, name, state) "
                "VALUES (:id, :owner, 'W', 'active')"
            ),
            {"id": workspace, "owner": user},
        )
        paper = await _paper(conn, visibility="public", content_hash="states", owner_id=None)
        with pytest.raises(IntegrityError, match="ck_workspace_papers_state"):
            await conn.execute(
                text(
                    "INSERT INTO workspace_papers (workspace_id, paper_id, state) "
                    "VALUES (:ws, :paper, 'processing')"
                ),
                {"ws": workspace, "paper": paper},
            )
        await conn.rollback()


class TestParseStatusVocabulary:
    """D44's four constraints, each verified by attempting the violation.

    Written because migration 0002 adds these by hand — `--autogenerate` does
    not compare CHECK constraints, so a constraint that silently failed to be
    created looks exactly like one that is working, right up until something bad
    gets in. Only an attempted violation tells the two apart.
    """

    async def test_an_unknown_parse_status_is_rejected(self, conn: AsyncConnection) -> None:
        # 'indexed' is not hypothetical: it was one of three ad-hoc values in
        # circulation before D44 fixed the vocabulary, and it is what this
        # constraint caught the day it was added.
        with pytest.raises(IntegrityError, match="ck_papers_parse_status"):
            await _paper(
                conn,
                visibility="public",
                content_hash="bad-status",
                owner_id=None,
                parse_status="indexed",
            )
        await conn.rollback()

    async def test_an_unknown_parse_failure_class_is_rejected(self, conn: AsyncConnection) -> None:
        # The class drives the retry decision and the user message (§4.4.1). A
        # value outside the taxonomy produces neither, so the paper would fail
        # with no way to tell the user why or whether to retry.
        with pytest.raises(IntegrityError, match="ck_papers_parse_failure_class"):
            await _paper(
                conn,
                visibility="public",
                content_hash="bad-class",
                owner_id=None,
                parse_status="failed",
                parse_failure_class="something_went_wrong",
            )
        await conn.rollback()

    async def test_a_failed_parse_without_a_class_is_rejected(self, conn: AsyncConnection) -> None:
        with pytest.raises(IntegrityError, match="ck_papers_parse_failure_iff_failed"):
            await _paper(
                conn,
                visibility="public",
                content_hash="failed-no-class",
                owner_id=None,
                parse_status="failed",
            )
        await conn.rollback()

    async def test_a_successful_parse_carrying_a_class_is_rejected(
        self, conn: AsyncConnection
    ) -> None:
        # The reverse direction, which a plain NOT NULL check would miss: a
        # parsed paper claiming a failure class would surface a scary reason on
        # a perfectly healthy paper.
        with pytest.raises(IntegrityError, match="ck_papers_parse_failure_iff_failed"):
            await _paper(
                conn,
                visibility="public",
                content_hash="parsed-with-class",
                owner_id=None,
                parse_status="parsed",
                parse_failure_class="scanned_image",
            )
        await conn.rollback()

    async def test_valid_combinations_are_accepted(self, conn: AsyncConnection) -> None:
        # Positive control. Without it these tests would pass just as happily if
        # the constraints rejected everything, including what they should allow.
        for status in ("pending", "parsing", "parsed"):
            await _paper(
                conn,
                visibility="public",
                content_hash=f"ok-{status}",
                owner_id=None,
                parse_status=status,
            )
        await _paper(
            conn,
            visibility="public",
            content_hash="ok-failed",
            owner_id=None,
            parse_status="failed",
            parse_failure_class="scanned_image",
        )
        await conn.rollback()

    async def test_every_failure_class_the_domain_defines_is_accepted(
        self, conn: AsyncConnection
    ) -> None:
        # The claim in models.py is that CHECKs are generated from the domain
        # enums so the two cannot drift. Rejecting one bad value does not test
        # that: adding a FailureClass member would be silently rejected by the
        # database with every existing test still green. This walks the whole
        # enum, so the drift fails loudly the way the docstring promises.
        for member in FailureClass:
            await _paper(
                conn,
                visibility="public",
                content_hash=f"class-{member.value}",
                owner_id=None,
                parse_status="failed",
                parse_failure_class=member.value,
            )
        await conn.rollback()

    async def test_an_unknown_membership_failure_class_is_rejected(
        self, conn: AsyncConnection
    ) -> None:
        # Same taxonomy, same gap, same fix — on the membership row this time.
        user = await _user(conn, f"membership-class-{uuid.uuid4()}@example.com")
        workspace = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO workspaces (id, owner_id, name, state) "
                "VALUES (:id, :owner, 'W', 'active')"
            ),
            {"id": workspace, "owner": user},
        )
        paper = await _paper(
            conn, visibility="public", content_hash="member-bad-class", owner_id=None
        )
        with pytest.raises(IntegrityError, match="ck_workspace_papers_failure_class"):
            await conn.execute(
                text(
                    "INSERT INTO workspace_papers (workspace_id, paper_id, state, "
                    "failure_class) VALUES (:ws, :paper, 'failed', 'gremlins')"
                ),
                {"ws": workspace, "paper": paper},
            )
        await conn.rollback()

    async def test_every_membership_failure_class_the_domain_defines_is_accepted(
        self, conn: AsyncConnection
    ) -> None:
        user = await _user(conn, f"membership-ok-{uuid.uuid4()}@example.com")
        workspace = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO workspaces (id, owner_id, name, state) "
                "VALUES (:id, :owner, 'W', 'active')"
            ),
            {"id": workspace, "owner": user},
        )
        for member in FailureClass:
            paper = await _paper(
                conn,
                visibility="public",
                content_hash=f"member-{member.value}",
                owner_id=None,
            )
            await conn.execute(
                text(
                    "INSERT INTO workspace_papers (workspace_id, paper_id, state, "
                    "failure_class) VALUES (:ws, :paper, 'failed', :cls)"
                ),
                {"ws": workspace, "paper": paper, "cls": member.value},
            )
        await conn.rollback()


class TestChunkInvariantsSurviveIntoTheDatabase:
    """The domain's chunk rules are re-asserted in SQL, not merely in Python."""

    async def _section(self, conn: AsyncConnection) -> tuple[uuid.UUID, uuid.UUID]:
        paper = await _paper(
            conn, visibility="public", content_hash=f"c-{uuid.uuid4()}", owner_id=None
        )
        section = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO sections (id, paper_id, kind, raw_title, path, order_index, "
                "page_start, page_end, text) VALUES (:id, :paper, 'method', 'Method', "
                "'Method', 0, 1, 3, 'the canonical section text')"
            ),
            {"id": section, "paper": paper},
        )
        return paper, section

    async def _insert_chunk(
        self, conn: AsyncConnection, paper: uuid.UUID, section: uuid.UUID, **overrides: object
    ) -> None:
        values: dict[str, object] = {
            "id": f"chunk-{uuid.uuid4()}",
            "paper": paper,
            "section": section,
            "parent": None,
            "role": "parent",
            "ordinal": 0,
            "text": "body",
            "tokens": 10,
            "page": 1,
            "start": 0,
            "end": 5,
        }
        values.update(overrides)
        await conn.execute(
            text(
                "INSERT INTO chunks (id, paper_id, section_id, parent_id, role, ordinal, "
                "text, token_count, page, char_start, char_end, chunker_version) VALUES "
                "(:id, :paper, :section, :parent, :role, :ordinal, :text, :tokens, :page, "
                ":start, :end, 'structure-aware/1.1.0')"
            ),
            values,
        )

    async def test_a_child_chunk_without_a_parent_is_rejected(self, conn: AsyncConnection) -> None:
        # A child with no parent window cannot be parent-expanded, which is the
        # entire point of small-to-big retrieval — it would retrieve precisely
        # and then answer from a 300-token fragment.
        paper, section = await self._section(conn)
        with pytest.raises(IntegrityError, match="ck_chunks_child_has_parent"):
            await self._insert_chunk(conn, paper, section, role="child", parent=None)
        await conn.rollback()

    async def test_an_empty_character_span_is_rejected(self, conn: AsyncConnection) -> None:
        # A zero-width span resolves to no source text, so the citation it backs
        # would highlight nothing (FR-I5).
        paper, section = await self._section(conn)
        with pytest.raises(IntegrityError, match="ck_chunks_span_non_empty"):
            await self._insert_chunk(conn, paper, section, start=40, end=40)
        await conn.rollback()

    async def test_a_zero_page_is_rejected(self, conn: AsyncConnection) -> None:
        # Pages are 1-indexed; a 0 means a parser forgot to convert and every
        # citation from that paper points one page early.
        paper, section = await self._section(conn)
        with pytest.raises(IntegrityError, match="ck_chunks_page_one_indexed"):
            await self._insert_chunk(conn, paper, section, page=0)
        await conn.rollback()

    async def test_offsets_resolve_against_the_section_text(self, conn: AsyncConnection) -> None:
        # D42's round trip: a chunk's span indexes `sections.text`, so resolving
        # a citation is this one join plus a slice. This is the property the
        # whole provenance design exists to provide.
        paper, section = await self._section(conn)
        await self._insert_chunk(conn, paper, section, start=4, end=13, text="canonical")
        resolved = (
            await conn.execute(
                text(
                    "SELECT substring(s.text FROM c.char_start + 1 FOR c.char_end - c.char_start) "
                    "FROM chunks c JOIN sections s ON s.id = c.section_id "
                    "WHERE c.section_id = :section"
                ),
                {"section": section},
            )
        ).scalar_one()
        assert resolved == "canonical"
        await conn.rollback()


class TestFailureTaxonomy:
    async def test_a_failed_paper_must_carry_a_failure_class(self, conn: AsyncConnection) -> None:
        # A failure without a class cannot produce a retry decision or a user
        # message, which defeats the taxonomy in doc 4 §4.4.1 entirely.
        user = await _user(conn, "fail@example.com")
        workspace = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO workspaces (id, owner_id, name, state) "
                "VALUES (:id, :owner, 'W', 'active')"
            ),
            {"id": workspace, "owner": user},
        )
        paper = await _paper(conn, visibility="public", content_hash="failing", owner_id=None)
        with pytest.raises(IntegrityError, match="ck_workspace_papers_failure_classified"):
            await conn.execute(
                text(
                    "INSERT INTO workspace_papers (workspace_id, paper_id, state) "
                    "VALUES (:ws, :paper, 'failed')"
                ),
                {"ws": workspace, "paper": paper},
            )
        await conn.rollback()


async def _discovered(
    conn: AsyncConnection,
    *,
    owner: uuid.UUID,
    doi: str | None = None,
    arxiv_id: str | None = "2004.04906",
    content_hash: str | None = None,
    parse_status: str = "pending",
    origin: str = "discovery",
) -> uuid.UUID:
    paper_id = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO papers (id, visibility, owner_id, content_hash, external_ids, "
            "title, authors, parse_status, doi, arxiv_id, origin) VALUES "
            "(:id, 'private', :owner, :hash, '{}'::jsonb, 'A Paper', '[]'::jsonb, "
            ":status, :doi, :arxiv, :origin)"
        ),
        {
            "id": paper_id,
            "owner": owner,
            "hash": content_hash,
            "status": parse_status,
            "doi": doi,
            "arxiv": arxiv_id,
            "origin": origin,
        },
    )
    return paper_id


class TestAPaperBeforeItsPdf:
    """D132's CHECKs and indexes, each one attempted.

    Alembic compares neither (CLAUDE.md), so a constraint that silently failed to be
    created would look exactly like one that works until something violated it.
    """

    async def test_a_paper_awaiting_its_pdf_with_an_identifier_is_accepted(
        self, conn: AsyncConnection
    ) -> None:
        owner = await _user(conn, "awaiting@example.com")
        await _discovered(conn, owner=owner)
        await conn.rollback()

    async def test_no_pdf_and_no_identifier_is_rejected(self, conn: AsyncConnection) -> None:
        # A paper that could never be found again by what it is.
        owner = await _user(conn, "lost@example.com")
        with pytest.raises(IntegrityError, match="ck_papers_hash_or_awaiting"):
            await _discovered(conn, owner=owner, arxiv_id=None)
        await conn.rollback()

    async def test_no_pdf_and_already_parsed_is_rejected(self, conn: AsyncConnection) -> None:
        owner = await _user(conn, "parsed@example.com")
        with pytest.raises(IntegrityError, match="ck_papers_hash_or_awaiting"):
            await _discovered(conn, owner=owner, parse_status="parsed")
        await conn.rollback()

    async def test_an_unknown_origin_is_rejected(self, conn: AsyncConnection) -> None:
        owner = await _user(conn, "origin@example.com")
        with pytest.raises(IntegrityError, match="ck_papers_origin"):
            await _discovered(conn, owner=owner, origin="scraped")
        await conn.rollback()

    async def test_a_row_written_without_an_origin_is_an_upload(
        self, conn: AsyncConnection
    ) -> None:
        # Every row before migration 0007 was an upload, which the default says.
        paper = await _paper(conn, visibility="public", content_hash="old-row", owner_id=None)
        origin = (
            await conn.execute(text("SELECT origin FROM papers WHERE id = :id"), {"id": paper})
        ).scalar_one()
        assert origin == "upload"
        await conn.rollback()

    @pytest.mark.parametrize(
        "doi", ["10.1/X", "doi:10.1/x", "11.1/x", "10.1", "10.48550/arxiv.2004.04906"]
    )
    async def test_a_doi_in_any_spelling_but_the_canonical_one_is_rejected(
        self, conn: AsyncConnection, doi: str
    ) -> None:
        owner = await _user(conn, "doi@example.com")
        with pytest.raises(IntegrityError, match="ck_papers_doi_canonical"):
            await _discovered(conn, owner=owner, doi=doi, arxiv_id=None)
        await conn.rollback()

    @pytest.mark.parametrize("arxiv_id", ["2004.04906v3", "HEP-TH/9901001"])
    async def test_an_arxiv_id_in_any_spelling_but_the_canonical_one_is_rejected(
        self, conn: AsyncConnection, arxiv_id: str
    ) -> None:
        owner = await _user(conn, "arxiv@example.com")
        with pytest.raises(IntegrityError, match="ck_papers_arxiv_id_canonical"):
            await _discovered(conn, owner=owner, arxiv_id=arxiv_id)
        await conn.rollback()

    async def test_one_owner_cannot_hold_one_arxiv_id_twice(self, conn: AsyncConnection) -> None:
        owner = await _user(conn, "twice@example.com")
        await _discovered(conn, owner=owner)
        with pytest.raises(IntegrityError, match="uq_papers_private_owner_arxiv"):
            await _discovered(conn, owner=owner)
        await conn.rollback()

    async def test_one_owner_cannot_hold_one_doi_twice(self, conn: AsyncConnection) -> None:
        owner = await _user(conn, "twice-doi@example.com")
        await _discovered(conn, owner=owner, doi="10.1/x", arxiv_id=None)
        with pytest.raises(IntegrityError, match="uq_papers_private_owner_doi"):
            await _discovered(conn, owner=owner, doi="10.1/x", arxiv_id=None)
        await conn.rollback()

    async def test_two_owners_each_hold_their_own_row_for_one_work(
        self, conn: AsyncConnection
    ) -> None:
        # D131: a discovered paper is its adder's, so the same work is two rows.
        alice = await _user(conn, "alice-work@example.com")
        bob = await _user(conn, "bob-work@example.com")
        first = await _discovered(conn, owner=alice, doi="10.1/x")
        second = await _discovered(conn, owner=bob, doi="10.1/x")
        assert first != second
        await conn.rollback()


class TestAJobsPriorityIsOneTheClaimKnows:
    """D139's CHECK, attempted. Alembic compares no CHECK constraints (CLAUDE.md)."""

    async def _job(self, conn: AsyncConnection, priority: int | None) -> None:
        owner = await _user(conn, f"{uuid.uuid4()}@example.com")
        workspace = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO workspaces (id, owner_id, name, state) "
                "VALUES (:id, :owner, 'W', 'active')"
            ),
            {"id": workspace, "owner": owner},
        )
        paper = await _paper(conn, visibility="private", content_hash="job-paper", owner_id=owner)
        columns = "id, workspace_id, paper_id, state" + (
            ", priority" if priority is not None else ""
        )
        values = ":id, :ws, :paper, 'queued'" + (", :priority" if priority is not None else "")
        await conn.execute(
            text(f"INSERT INTO ingestion_jobs ({columns}) VALUES ({values})"),  # noqa: S608 - fixed text
            {"id": uuid.uuid4(), "ws": workspace, "paper": paper, "priority": priority},
        )

    async def test_a_priority_the_claim_does_not_know_is_rejected(
        self, conn: AsyncConnection
    ) -> None:
        with pytest.raises(IntegrityError, match="ck_ingestion_jobs_priority"):
            await self._job(conn, 5)
        await conn.rollback()

    async def test_a_job_written_without_one_is_interactive(self, conn: AsyncConnection) -> None:
        # Every job in flight at migration time was an upload, and a forgotten
        # value must err toward not being deprioritised.
        await self._job(conn, None)
        priority = (await conn.execute(text("SELECT priority FROM ingestion_jobs"))).scalar_one()
        assert priority == 10
        await conn.rollback()
