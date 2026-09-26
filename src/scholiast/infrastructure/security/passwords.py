"""Argon2id password hashing (NFR-SEC1, D84).

Doc 7 chose `argon2-cffi` over a framework on the grounds that the security
reasoning is the interesting part. The reasoning is here.

**Why Argon2id and not bcrypt.** Both are memory-hard enough to be respectable;
Argon2id is the current recommendation and, unlike bcrypt, does not silently
truncate at 72 bytes — a real trap, because a user whose password is longer gets
security they did not receive and no error says so. `domain.identity` still caps
length, for the server's sake rather than the algorithm's.

**Parameters are the whole decision.** The defaults below cost ~64 MB and a few
hundred milliseconds per hash. That is deliberately expensive: it is the number
that decides what an offline attacker pays per guess if the database leaks. It is
also paid on every login, which is why rate limiting the login endpoint (NFR-SEC7,
phase 8) is part of the same design rather than a separate nicety.
"""

from __future__ import annotations

from argon2 import PasswordHasher as _Argon2
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

#: RFC 9106's "second recommended option" shape, and `argon2-cffi`'s defaults:
#: 64 MiB, three passes, four lanes. Comfortably above OWASP's floor (19 MiB,
#: t=2). Recorded as constants rather than left implicit so `version` can report
#: them and a future change is visible in a diff.
DEFAULT_TIME_COST = 3
DEFAULT_MEMORY_COST_KIB = 65536
DEFAULT_PARALLELISM = 4


class Argon2idHasher:
    """`PasswordHasher` over argon2-cffi."""

    def __init__(
        self,
        *,
        time_cost: int = DEFAULT_TIME_COST,
        memory_cost_kib: int = DEFAULT_MEMORY_COST_KIB,
        parallelism: int = DEFAULT_PARALLELISM,
    ) -> None:
        self._time_cost = time_cost
        self._memory_cost_kib = memory_cost_kib
        self._parallelism = parallelism
        self._hasher = _Argon2(
            time_cost=time_cost,
            memory_cost=memory_cost_kib,
            parallelism=parallelism,
        )

    @property
    def version(self) -> str:
        return f"argon2id/t={self._time_cost},m={self._memory_cost_kib},p={self._parallelism}/1.0.0"

    def hash(self, password: str) -> str:
        """Hash with a fresh random salt, embedded in the encoded output."""
        return self._hasher.hash(password)

    def verify(self, password: str, encoded: str) -> bool:
        """True when the password matches; False for every failure.

        **Returns rather than raises**, including for a malformed hash. The
        caller is a login path, where "wrong password" and "this row's hash is
        corrupt" must be indistinguishable from outside — and where an exception
        escaping would turn a failed login into a 500 that says so.
        """
        try:
            return bool(self._hasher.verify(encoded, password))
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False

    def needs_rehash(self, encoded: str) -> bool:
        """True when the stored hash used weaker parameters than current ones.

        A malformed hash counts as needing one: whatever produced it, it is not a
        hash this configuration would accept as current.
        """
        try:
            return bool(self._hasher.check_needs_rehash(encoded))
        except InvalidHashError:
            return True
