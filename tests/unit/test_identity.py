"""Email normalisation and password policy (FR-A1, D84).

Small rules, and the reason they are in the domain rather than in a validator is
that each one closes a hole rather than tidying input. The tests are named after
the holes.
"""

from __future__ import annotations

import pytest

from scholiast.domain.errors import InvariantViolation
from scholiast.domain.identity import (
    MAX_EMAIL_LENGTH,
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    normalise_email,
    validate_password,
)


class TestEmailNormalisation:
    def test_case_differences_cannot_become_two_accounts(self) -> None:
        # The hole: register as Ada@example.com, then again as ada@example.com,
        # and the second account silently owns none of the first's papers.
        assert normalise_email("Ada@Example.COM") == normalise_email("ada@example.com")

    def test_surrounding_whitespace_cannot_become_two_accounts(self) -> None:
        # Autofill and copy-paste both add it, and the user cannot see it.
        assert normalise_email("  ada@example.com\n") == "ada@example.com"

    def test_a_compatibility_equivalent_address_is_the_same_address(self) -> None:
        # NFKC: fullwidth characters render as a different string for the same
        # address. Without normalisation this is the case-collision hole again,
        # in a form no human reading the users table would spot.
        # Built from code points rather than pasted: a literal fullwidth "a"
        # is indistinguishable from a typo when reading the source.
        fullwidth = "".join(chr(c) for c in (0xFF41, 0xFF44, 0xFF41)) + "@example.com"
        assert normalise_email(fullwidth) == "ada@example.com"

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "no-at-sign",
            "@example.com",
            "ada@",
            "ada@localhost",  # no dot in the domain
            "two@at@example.com",
            "ada space@example.com",
            "ada@exam ple.com",
        ],
    )
    def test_input_that_cannot_be_an_address_is_refused(self, raw: str) -> None:
        with pytest.raises(InvariantViolation):
            normalise_email(raw)

    def test_an_address_over_the_column_width_is_refused_here_not_by_postgres(self) -> None:
        # Otherwise it surfaces as a database error, which is a 500 rather than a
        # 4xx and leaks the schema into the response.
        oversized = "a" * (MAX_EMAIL_LENGTH - len("@example.com") + 1) + "@example.com"
        assert len(oversized) > MAX_EMAIL_LENGTH
        with pytest.raises(InvariantViolation):
            normalise_email(oversized)

    def test_an_address_exactly_at_the_limit_is_accepted(self) -> None:
        # The boundary in the other direction: an off-by-one here would reject
        # legitimate addresses, which is a bug nobody reports.
        at_limit = "a" * (MAX_EMAIL_LENGTH - len("@example.com")) + "@example.com"
        assert len(at_limit) == MAX_EMAIL_LENGTH
        assert normalise_email(at_limit) == at_limit


class TestPasswordPolicy:
    def test_a_password_below_the_floor_is_refused(self) -> None:
        # The hole: Argon2id's cost protects against *offline* cracking. It does
        # nothing for a password an online guesser reaches in seconds.
        with pytest.raises(InvariantViolation):
            validate_password("a" * (MIN_PASSWORD_LENGTH - 1))

    def test_a_password_at_the_floor_is_accepted(self) -> None:
        exact = "a" * MIN_PASSWORD_LENGTH
        assert validate_password(exact) == exact

    def test_an_unbounded_password_is_refused(self) -> None:
        # Argon2id reads the whole input, so an arbitrarily long password is an
        # arbitrarily expensive login attempt that costs the attacker nothing.
        with pytest.raises(InvariantViolation):
            validate_password("a" * (MAX_PASSWORD_LENGTH + 1))

    def test_the_password_is_returned_exactly_as_typed(self) -> None:
        # Deliberately NOT normalised: NFKC here would mean a password that
        # worked yesterday fails tomorrow if the client's input method changes.
        awkward = "  " + "".join(chr(c) for c in (0xFF50, 0xFF41, 0xFF53, 0xFF53)) + "word  "
        assert validate_password(awkward) == awkward

    def test_no_composition_rule_is_imposed(self) -> None:
        # NIST SP 800-63B dropped these: they push users toward Password1! and
        # buy nothing measurable.
        assert validate_password("all lowercase and long enough")
