"""Identity rules: what an email is, and what a password must be (FR-A1, D84).

Pure policy, stdlib only. It lives in the domain because it is the part of
authentication that is *decided* rather than *implemented*: the hashing algorithm
and the token format are adapter concerns that M3 could swap, while "an address
is compared case-insensitively" and "a password has a floor" are rules the system
holds regardless of which library performs them.

**Both rules exist to close a real hole rather than to satisfy a checklist.**
Without normalisation, `Ada@example.com` and `ada@example.com` register as two
accounts and the second one silently owns nothing the first uploaded. Without a
length floor, the Argon2id cost that makes offline cracking expensive is spent
protecting a password that an online guesser finds in seconds.
"""

from __future__ import annotations

import re
import unicodedata

from scholiast.domain.errors import InvariantViolation

#: Deliberately permissive: one `@`, something either side, no whitespace, and a
#: dot in the domain. Validating addresses by regex beyond this is a known
#: fool's errand — RFC 5322 admits addresses no provider accepts — and the only
#: real proof that an address exists is sending mail to it, which M3 does not do.
#: The narrow job here is rejecting input that cannot be an address at all.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

#: 320 = 64 local part + 1 `@` + 255 domain, the practical maximum, and the width
#: `users.email` is declared with. Enforced here too so the failure is a domain
#: error rather than a database error surfacing as a 500.
MAX_EMAIL_LENGTH = 320

#: NIST SP 800-63B's floor for a user-chosen secret, and the one control it keeps
#: after dropping composition rules: length, not "one uppercase and a symbol".
MIN_PASSWORD_LENGTH = 12

#: Argon2id reads the whole password, so an unbounded one is an unbounded amount
#: of work per login attempt — a denial-of-service vector that costs the attacker
#: nothing. (bcrypt has the opposite problem, silently truncating at 72 bytes.)
MAX_PASSWORD_LENGTH = 1024


def normalise_email(raw: str) -> str:
    """Casefold and strip an address so one human means one account.

    NFKC first: `ADA@example.com` typed with fullwidth characters is the same
    address, and comparing unnormalised strings would let it become a second
    account owning none of the first's data.

    Only the whole address is casefolded, not the local part specifically. The
    local part is technically case-sensitive per RFC 5321, and essentially no
    provider treats it that way; matching what providers actually do prevents the
    duplicate-account hole, and the cost is an edge case nobody deploys.
    """
    collapsed = unicodedata.normalize("NFKC", raw).strip().casefold()
    if not collapsed:
        raise InvariantViolation("email cannot be blank")
    if len(collapsed) > MAX_EMAIL_LENGTH:
        raise InvariantViolation(
            f"email is {len(collapsed)} characters, over the {MAX_EMAIL_LENGTH} limit"
        )
    if not _EMAIL.match(collapsed):
        raise InvariantViolation(f"{raw!r} is not a usable email address")
    return collapsed


def validate_password(password: str) -> str:
    """Check a proposed password against policy and return it unchanged.

    Returns rather than mutates: a password is bytes the user chose, and
    normalising it would mean the password that worked yesterday fails tomorrow
    under a different Unicode form. The hash is computed over exactly what was
    typed.

    Length is counted in characters, and the maximum exists for the server's sake
    rather than the user's — see `MAX_PASSWORD_LENGTH`.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise InvariantViolation(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters, got {len(password)}"
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise InvariantViolation(
            f"password must be at most {MAX_PASSWORD_LENGTH} characters, got {len(password)}"
        )
    return password
