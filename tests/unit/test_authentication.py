"""The four authentication use cases, against fakes (FR-A1, FR-A5, D84).

Fakes rather than a database because what is under test here is *policy* — what
order things happen in, what is revoked when, and what a caller can learn from a
failure. The real repositories get their own integration test; a Postgres
container would make these tests slower without making them stricter.

The properties worth protecting, each with a test named after the hole it closes:

- every failure is the same failure (`AuthenticationFailed`), so no response
  distinguishes "no such account" from "wrong password";
- an unknown address costs the same as a known one, so timing does not become the
  oracle that uniform messages closed;
- a refresh token works exactly once, and presenting it twice kills the family.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from scholiast.application.ports.accounts import (
    Account,
    EmailAlreadyRegistered,
    Session,
)
from scholiast.application.ports.security import (
    AccessClaims,
    AuthenticationFailed,
    IssuedAccess,
    IssuedRefresh,
)
from scholiast.application.use_cases.authentication import (
    EndSessionUseCase,
    RefreshSessionUseCase,
    RegisterUserUseCase,
    StartSessionUseCase,
)
from scholiast.domain.errors import InvariantViolation

PASSWORD = "a-perfectly-adequate-password"


class FakeHasher:
    """Deterministic and instant. `cost` stands in for Argon2id parameters."""

    def __init__(self, cost: str = "v2") -> None:
        self.cost = cost
        self.hashed: list[str] = []
        self.verified: list[tuple[str, str]] = []

    def hash(self, password: str) -> str:
        self.hashed.append(password)
        return f"{self.cost}:{password}"

    def verify(self, password: str, encoded: str) -> bool:
        self.verified.append((password, encoded))
        _, _, stored = encoded.partition(":")
        return bool(stored) and stored == password

    def needs_rehash(self, encoded: str) -> bool:
        return not encoded.startswith(f"{self.cost}:")

    @property
    def version(self) -> str:
        return f"fake/{self.cost}"


class FakeTokens:
    """Opaque, unique, and trivially inspectable."""

    def __init__(self) -> None:
        self._minted = 0

    def issue_access(self, *, subject: str) -> IssuedAccess:
        self._minted += 1
        return IssuedAccess(
            token=f"access-{subject}-{self._minted}",
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )

    def verify_access(self, token: str) -> AccessClaims:
        if not token.startswith("access-"):
            raise AuthenticationFailed("fake: not an access token")
        return AccessClaims(
            subject=token.split("-")[1], expires_at=datetime.now(UTC) + timedelta(minutes=15)
        )

    def issue_refresh(self) -> IssuedRefresh:
        self._minted += 1
        secret = f"refresh-{self._minted}"
        return IssuedRefresh(
            secret=secret,
            fingerprint=self.fingerprint(secret),
            expires_at=datetime.now(UTC) + timedelta(days=14),
        )

    def fingerprint(self, secret: str) -> str:
        return f"fp:{secret}"


class FakeAccounts:
    def __init__(self) -> None:
        self.rows: dict[str, Account] = {}

    async def create(self, *, email: str, password_hash: str) -> Account:
        if any(row.email == email for row in self.rows.values()):
            raise EmailAlreadyRegistered(email)
        account = Account(id=str(uuid.uuid4()), email=email, password_hash=password_hash)
        self.rows[account.id] = account
        return account

    async def by_email(self, email: str) -> Account | None:
        return next((row for row in self.rows.values() if row.email == email), None)

    async def by_id(self, account_id: str) -> Account | None:
        return self.rows.get(account_id)

    async def set_password_hash(self, account_id: str, password_hash: str) -> None:
        current = self.rows[account_id]
        self.rows[account_id] = Account(
            id=current.id, email=current.email, password_hash=password_hash
        )


class FakeSessions:
    def __init__(self) -> None:
        self.rows: dict[str, Session] = {}

    async def record(
        self, *, user_id: str, family_id: str, fingerprint: str, expires_at: datetime
    ) -> Session:
        session = Session(
            id=str(uuid.uuid4()),
            user_id=user_id,
            family_id=family_id,
            expires_at=expires_at,
            revoked_at=None,
        )
        self.rows[fingerprint] = session
        return session

    async def by_fingerprint(self, fingerprint: str) -> Session | None:
        return self.rows.get(fingerprint)

    async def revoke(self, session_id: str, *, at: datetime) -> int:
        # Returns the rows it changed, like the SQL adapter: revoking an
        # already-revoked row changes nothing and says 0 (D87).
        for fingerprint, row in list(self.rows.items()):
            if row.id == session_id and row.revoked_at is None:
                self.rows[fingerprint] = _revoked(row, at)
                return 1
        return 0

    async def revoke_family(self, family_id: str, *, at: datetime) -> int:
        revoked = 0
        for fingerprint, row in list(self.rows.items()):
            if row.family_id == family_id and row.revoked_at is None:
                self.rows[fingerprint] = _revoked(row, at)
                revoked += 1
        return revoked

    def live(self) -> list[Session]:
        return [row for row in self.rows.values() if row.revoked_at is None]


def _revoked(row: Session, at: datetime) -> Session:
    return Session(
        id=row.id,
        user_id=row.user_id,
        family_id=row.family_id,
        expires_at=row.expires_at,
        revoked_at=at,
    )


@pytest.fixture
def accounts() -> FakeAccounts:
    return FakeAccounts()


@pytest.fixture
def sessions() -> FakeSessions:
    return FakeSessions()


@pytest.fixture
def hasher() -> FakeHasher:
    return FakeHasher()


@pytest.fixture
def tokens() -> FakeTokens:
    return FakeTokens()


@pytest.fixture
def register(
    accounts: FakeAccounts, sessions: FakeSessions, hasher: FakeHasher, tokens: FakeTokens
) -> RegisterUserUseCase:
    return RegisterUserUseCase(
        accounts=accounts,  # type: ignore[arg-type]
        sessions=sessions,  # type: ignore[arg-type]
        hasher=hasher,  # type: ignore[arg-type]
        tokens=tokens,  # type: ignore[arg-type]
    )


@pytest.fixture
def login(
    accounts: FakeAccounts, sessions: FakeSessions, hasher: FakeHasher, tokens: FakeTokens
) -> StartSessionUseCase:
    return StartSessionUseCase(
        accounts=accounts,  # type: ignore[arg-type]
        sessions=sessions,  # type: ignore[arg-type]
        hasher=hasher,  # type: ignore[arg-type]
        tokens=tokens,  # type: ignore[arg-type]
    )


@pytest.fixture
def rotate(sessions: FakeSessions, tokens: FakeTokens) -> RefreshSessionUseCase:
    return RefreshSessionUseCase(
        sessions=sessions,  # type: ignore[arg-type]
        tokens=tokens,  # type: ignore[arg-type]
    )


@pytest.fixture
def logout(sessions: FakeSessions, tokens: FakeTokens) -> EndSessionUseCase:
    return EndSessionUseCase(
        sessions=sessions,  # type: ignore[arg-type]
        tokens=tokens,  # type: ignore[arg-type]
    )


class TestRegistration:
    async def test_the_password_is_never_stored_as_typed(
        self, register: RegisterUserUseCase, accounts: FakeAccounts
    ) -> None:
        await register.execute(email="ada@example.com", password=PASSWORD)
        stored = next(iter(accounts.rows.values())).password_hash
        assert stored != PASSWORD
        assert stored == f"v2:{PASSWORD}"  # the fake's stand-in for a real hash

    async def test_the_address_is_stored_normalised(
        self, register: RegisterUserUseCase, accounts: FakeAccounts
    ) -> None:
        # Otherwise the uniqueness constraint compares raw strings and the
        # case-collision hole reopens one layer down.
        await register.execute(email="  Ada@Example.COM ", password=PASSWORD)
        assert next(iter(accounts.rows.values())).email == "ada@example.com"

    async def test_a_second_registration_for_one_address_is_refused(
        self, register: RegisterUserUseCase
    ) -> None:
        await register.execute(email="ada@example.com", password=PASSWORD)
        with pytest.raises(EmailAlreadyRegistered):
            await register.execute(email="ADA@example.com", password=PASSWORD)

    async def test_a_weak_password_is_rejected_before_any_hashing_happens(
        self, register: RegisterUserUseCase, hasher: FakeHasher
    ) -> None:
        # Ordering, not just outcome: hashing first would let an attacker spend
        # 64 MB of server memory per request with a password they know is
        # invalid. The assertion on `hashed` is the whole point of the test.
        with pytest.raises(InvariantViolation):
            await register.execute(email="ada@example.com", password="short")
        assert hasher.hashed == []

    async def test_registration_returns_a_usable_session(
        self, register: RegisterUserUseCase, sessions: FakeSessions
    ) -> None:
        issued = await register.execute(email="ada@example.com", password=PASSWORD)
        assert issued.access_token and issued.refresh_token
        assert issued.access_expires_at < issued.refresh_expires_at
        assert len(sessions.live()) == 1


class TestLogin:
    async def test_correct_credentials_issue_a_pair(
        self, register: RegisterUserUseCase, login: StartSessionUseCase
    ) -> None:
        registered = await register.execute(email="ada@example.com", password=PASSWORD)
        signed_in = await login.execute(email="ada@example.com", password=PASSWORD)
        assert signed_in.user_id == registered.user_id
        assert signed_in.refresh_token != registered.refresh_token

    async def test_the_address_is_matched_case_insensitively(
        self, register: RegisterUserUseCase, login: StartSessionUseCase
    ) -> None:
        await register.execute(email="ada@example.com", password=PASSWORD)
        assert await login.execute(email="ADA@Example.com  ", password=PASSWORD)

    @pytest.mark.parametrize(
        ("email", "password"),
        [
            ("nobody@example.com", PASSWORD),  # unknown account
            ("ada@example.com", "the-wrong-password"),  # wrong password
            ("not-an-email", PASSWORD),  # malformed address
            ("", PASSWORD),  # blank
        ],
    )
    async def test_every_cause_of_failure_raises_the_same_error(
        self,
        register: RegisterUserUseCase,
        login: StartSessionUseCase,
        email: str,
        password: str,
    ) -> None:
        # FR-A5. A malformed address raising `InvariantViolation` would surface
        # as a 422 while the others surface as 401 — which tells a prober that
        # the address was at least well-formed, and by elimination which
        # addresses exist.
        await register.execute(email="ada@example.com", password=PASSWORD)
        with pytest.raises(AuthenticationFailed):
            await login.execute(email=email, password=password)

    async def test_an_unknown_address_still_pays_for_a_verification(
        self, login: StartSessionUseCase, hasher: FakeHasher
    ) -> None:
        # The timing oracle. Returning early for an unknown address makes that
        # request measurably faster, and uniform error messages do not hide a
        # 60 ms difference.
        hasher.verified.clear()
        with pytest.raises(AuthenticationFailed):
            await login.execute(email="nobody@example.com", password=PASSWORD)
        assert len(hasher.verified) == 1

    async def test_a_failed_login_records_no_session(
        self, register: RegisterUserUseCase, login: StartSessionUseCase, sessions: FakeSessions
    ) -> None:
        await register.execute(email="ada@example.com", password=PASSWORD)
        with pytest.raises(AuthenticationFailed):
            await login.execute(email="ada@example.com", password="the-wrong-password")
        assert len(sessions.live()) == 1  # only the one registration issued

    async def test_a_hash_from_weaker_parameters_is_upgraded_on_success(
        self, accounts: FakeAccounts, sessions: FakeSessions, tokens: FakeTokens
    ) -> None:
        old = FakeHasher(cost="v1")
        await accounts.create(email="ada@example.com", password_hash=old.hash(PASSWORD))
        current = FakeHasher(cost="v2")
        use_case = StartSessionUseCase(
            accounts=accounts,  # type: ignore[arg-type]
            sessions=sessions,  # type: ignore[arg-type]
            hasher=current,  # type: ignore[arg-type]
            tokens=tokens,  # type: ignore[arg-type]
        )

        await use_case.execute(email="ada@example.com", password=PASSWORD)

        assert next(iter(accounts.rows.values())).password_hash == f"v2:{PASSWORD}"

    async def test_a_current_hash_is_left_alone(
        self, register: RegisterUserUseCase, login: StartSessionUseCase, accounts: FakeAccounts
    ) -> None:
        # A rehash on every login would double the cost of the hottest
        # authenticated path for nothing.
        await register.execute(email="ada@example.com", password=PASSWORD)
        before = next(iter(accounts.rows.values())).password_hash
        await login.execute(email="ada@example.com", password=PASSWORD)
        assert next(iter(accounts.rows.values())).password_hash == before

    async def test_each_login_starts_its_own_family(
        self, register: RegisterUserUseCase, login: StartSessionUseCase, sessions: FakeSessions
    ) -> None:
        # Otherwise revoking a stolen chain on one device logs out every device.
        await register.execute(email="ada@example.com", password=PASSWORD)
        await login.execute(email="ada@example.com", password=PASSWORD)
        assert len({row.family_id for row in sessions.live()}) == 2


class TestRefreshRotation:
    async def test_a_refresh_yields_a_new_pair_in_the_same_family(
        self, register: RegisterUserUseCase, rotate: RefreshSessionUseCase, sessions: FakeSessions
    ) -> None:
        first = await register.execute(email="ada@example.com", password=PASSWORD)
        family = sessions.live()[0].family_id

        second = await rotate.execute(refresh_token=first.refresh_token)

        assert second.refresh_token != first.refresh_token
        assert second.user_id == first.user_id
        assert [row.family_id for row in sessions.live()] == [family]

    async def test_the_presented_token_dies_as_its_successor_is_issued(
        self, register: RegisterUserUseCase, rotate: RefreshSessionUseCase
    ) -> None:
        # Single use. A refresh token that survives rotation is a long-lived
        # credential that no logout can reliably kill.
        first = await register.execute(email="ada@example.com", password=PASSWORD)
        await rotate.execute(refresh_token=first.refresh_token)
        with pytest.raises(AuthenticationFailed):
            await rotate.execute(refresh_token=first.refresh_token)

    async def test_reuse_revokes_the_whole_chain_including_the_live_token(
        self, register: RegisterUserUseCase, rotate: RefreshSessionUseCase, sessions: FakeSessions
    ) -> None:
        # The theft-detection property (D84). Two parties hold tokens from one
        # chain; only one can be legitimate, and nothing here can tell which. So
        # both lose, and the real user re-authenticates.
        first = await register.execute(email="ada@example.com", password=PASSWORD)
        second = await rotate.execute(refresh_token=first.refresh_token)

        with pytest.raises(AuthenticationFailed):
            await rotate.execute(refresh_token=first.refresh_token)

        assert sessions.live() == []
        with pytest.raises(AuthenticationFailed):
            await rotate.execute(refresh_token=second.refresh_token)

    async def test_losing_the_revoking_race_is_treated_as_reuse(
        self, register: RegisterUserUseCase, rotate: RefreshSessionUseCase, sessions: FakeSessions
    ) -> None:
        # D87. The use case reads the row as usable and then revokes it; a second
        # request presenting the same token can slip between those two steps. The
        # UPDATE's row count is what detects it — simulated here by a repository
        # whose revoke reports that it changed nothing, which is exactly what
        # Postgres reports to the loser of that race.
        first = await register.execute(email="ada@example.com", password=PASSWORD)

        async def revoke_nothing(session_id: str, *, at: datetime) -> int:
            return 0

        sessions.revoke = revoke_nothing  # type: ignore[method-assign]

        with pytest.raises(AuthenticationFailed):
            await rotate.execute(refresh_token=first.refresh_token)
        # And it is treated as theft, not as a transient error: the family goes.
        assert sessions.live() == []

    async def test_an_unknown_token_is_refused_without_touching_anything(
        self, register: RegisterUserUseCase, rotate: RefreshSessionUseCase, sessions: FakeSessions
    ) -> None:
        await register.execute(email="ada@example.com", password=PASSWORD)
        with pytest.raises(AuthenticationFailed):
            await rotate.execute(refresh_token="not-a-token-we-issued")
        assert len(sessions.live()) == 1

    async def test_an_expired_token_is_refused_but_does_not_kill_the_family(
        self, sessions: FakeSessions, tokens: FakeTokens, rotate: RefreshSessionUseCase
    ) -> None:
        # Expiry is not evidence of theft. Treating it as reuse would revoke a
        # family every time a laptop came back from a fortnight away.
        stale = "stale-secret"
        await sessions.record(
            user_id="user-1",
            family_id="family-1",
            fingerprint=tokens.fingerprint(stale),
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        sibling = "live-secret"
        await sessions.record(
            user_id="user-1",
            family_id="family-1",
            fingerprint=tokens.fingerprint(sibling),
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )

        with pytest.raises(AuthenticationFailed):
            await rotate.execute(refresh_token=stale)

        # Both rows survive unrevoked: the expired one is simply past its date.
        assert len(sessions.live()) == 2


class TestLogout:
    async def test_logout_revokes_the_chain(
        self,
        register: RegisterUserUseCase,
        logout: EndSessionUseCase,
        rotate: RefreshSessionUseCase,
    ) -> None:
        issued = await register.execute(email="ada@example.com", password=PASSWORD)
        assert await logout.execute(refresh_token=issued.refresh_token) == 1
        with pytest.raises(AuthenticationFailed):
            await rotate.execute(refresh_token=issued.refresh_token)

    async def test_logout_takes_the_rotated_successor_with_it(
        self,
        register: RegisterUserUseCase,
        rotate: RefreshSessionUseCase,
        logout: EndSessionUseCase,
    ) -> None:
        # Revoking by family rather than by token: a rotation that raced the
        # logout must not leave a live credential behind.
        first = await register.execute(email="ada@example.com", password=PASSWORD)
        second = await rotate.execute(refresh_token=first.refresh_token)
        await logout.execute(refresh_token=second.refresh_token)
        with pytest.raises(AuthenticationFailed):
            await rotate.execute(refresh_token=second.refresh_token)

    async def test_logging_out_twice_is_not_an_error(
        self, register: RegisterUserUseCase, logout: EndSessionUseCase
    ) -> None:
        # A client retrying after a dropped response has already got what it
        # asked for; the second call revokes nothing and says so with 0.
        issued = await register.execute(email="ada@example.com", password=PASSWORD)
        assert await logout.execute(refresh_token=issued.refresh_token) == 1
        assert await logout.execute(refresh_token=issued.refresh_token) == 0

    async def test_an_unknown_token_is_not_an_error_either(self, logout: EndSessionUseCase) -> None:
        # Raising here would make logout an oracle for which tokens exist.
        assert await logout.execute(refresh_token="never-issued") == 0

    async def test_one_users_logout_leaves_another_users_session_alone(
        self, register: RegisterUserUseCase, logout: EndSessionUseCase, sessions: FakeSessions
    ) -> None:
        mine = await register.execute(email="ada@example.com", password=PASSWORD)
        theirs = await register.execute(email="grace@example.com", password=PASSWORD)
        await logout.execute(refresh_token=mine.refresh_token)
        assert [row.user_id for row in sessions.live()] == [theirs.user_id]
