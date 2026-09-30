"""Validated configuration (doc 5 §5.3 `config/`).

Settings are a typed object, not scattered `os.environ` reads, for one reason
that matters more than tidiness: a missing or malformed value fails at startup
with the field name, rather than at 3am inside an adapter with an AttributeError.

Nothing here reaches into `domain` or `application` — configuration is an
infrastructure concern, and the layers that hold business rules must stay
unaware of where a connection string comes from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Everything the process needs from its environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Env vars win over .env: containers and CI set the environment
        # directly, and a stale .env silently overriding them is a debugging
        # afternoon nobody gets back.
        env_nested_delimiter="__",
    )

    # -- backing services --------------------------------------------------
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+asyncpg://scholiast:scholiast@localhost:5432/scholiast"),
    )
    qdrant_url: str = "http://localhost:6333"
    #: One collection for every tenant, filtered by payload (AD-4). Tenant count
    #: is the dimension expected to grow, and per-tenant indexes are the classic
    #: multi-tenancy scaling trap.
    qdrant_collection: str = "scholiast_chunks"
    redis_url: str = "redis://localhost:6379/0"
    grobid_url: str = "http://localhost:8070"

    # -- storage -----------------------------------------------------------
    #: Holds the original PDF and the retained TEI (D43). Local FS in dev,
    #: S3-compatible in prod, both behind one ObjectStore port.
    object_store_path: Path = Path("./var/objects")

    # -- local inference ---------------------------------------------------
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cuda"
    #: fp16 is specified, not preferred (D18) — fp32 reranking is 3.3x slower
    #: and consumes ~93% of NFR-P2's whole-stage budget by itself.
    embedding_precision: str = "fp16"
    #: Bounded ingestion batches (§5.7). M2 tightens this to the ~50-100 ms the
    #: priority semaphore needs, once a reranker exists to contend for the GPU.
    embedding_batch_size: int = 32
    embedding_dimensions: int = 768

    # -- ingestion worker --------------------------------------------------
    #: The cost of not having a broker (D47): with jobs in Postgres there is
    #: nothing to push, so the loop asks. Imperceptible against NFR-P6's 90 s.
    worker_poll_seconds: float = 1.0
    #: Visibility timeout. Must exceed the slowest stage by a wide margin so only
    #: a genuinely dead worker loses its claim; a live one refreshes it at every
    #: stage transition.
    worker_lease_seconds: float = 300.0
    worker_batch_size: int = 1
    #: D63: the composition root runs the worker alongside the API, so
    #: `uvicorn scholiast.main:app` is the single command M1's acceptance
    #: criterion asks for. Not `scholiast.api.app` — that module exposes only a
    #: factory, and deliberately knows nothing about the worker, because the
    #: layers contract makes `api` and `workers` independent siblings. M3 turns
    #: this off when Celery arrives: a configuration change, not a restructuring.
    worker_in_process: bool = True

    # -- maintenance (§5.6.4 step 4, D100) ---------------------------------
    #: How often the reconciliation sweep runs. Five minutes is a judgement
    #: call, not a measurement, and the two things it trades off are named so
    #: the number can be argued with: a cascade whose dispatch was lost waits at
    #: most this long before `list_deleting` finds it, and every pass scans the
    #: whole index and object store (D102), so the cost is paid whether or not
    #: there is residue. Shorter means faster convergence and more scanning.
    reconcile_interval_seconds: float = 300.0

    # -- provider circuit breaker (§5.11, D104) ----------------------------
    #: Consecutive failures before a provider is skipped without being called.
    #: Three, matching D48's retry budget for the same reason: it survives a
    #: restart of the thing being called without sitting through an outage.
    breaker_failure_threshold: int = 3
    #: How long to skip it when the provider said nothing about when to return.
    #: A 429's `Retry-After` overrides this, because the provider has stated the
    #: answer and this number can only guess at it (D103).
    breaker_cooldown_seconds: float = 30.0

    # -- generation --------------------------------------------------------
    #: Local carries sustained load; hosted quota is a deliberate spend (D26).
    generation_provider: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    #: ON whenever a provider that may train on submitted content is in the
    #: chain (D30). A user's unpublished manuscript must not reach a tier that
    #: trains on it or shows it to human reviewers.
    private_papers_local_only: bool = True

    # -- queue transport (M3 phase 3, D95, D97) ----------------------------
    #: D95. False puts ingestion in a Celery worker and makes `worker_in_process`
    #: false the sensible companion; true keeps M1's single-process topology,
    #: where a broker round trip would buy nothing because the same process is
    #: already polling the same database.
    celery_enabled: bool = False

    # -- authentication (M3 phase 1, D84) ----------------------------------
    #: HMAC key for access tokens. **The default is a development key and the
    #: adapter refuses anything under 32 characters.** A deployment that forgets
    #: to set this gets a well-known secret, which is why M6's checklist treats
    #: it as a release blocker rather than a warning.
    jwt_secret: str = "dev-only-insecure-jwt-secret-change-me"  # noqa: S105
    #: Goes into every token's `iss`, and is required on the way back in, so a
    #: token minted by another service sharing the secret is still rejected.
    jwt_issuer: str = "scholiast"
    #: Short, because nothing can revoke an access token before it expires
    #: (D84) — this number IS the revocation latency.
    access_token_minutes: int = 15
    #: Long, because the refresh token is revocable: rotation on every use and
    #: family revocation on reuse make a long life safe to offer.
    refresh_token_days: int = 14

    @field_validator("embedding_precision")
    @classmethod
    def _precision_is_known(cls, value: str) -> str:
        if value not in ("fp16", "fp32"):
            raise ValueError(f"embedding_precision must be fp16 or fp32, got {value!r}")
        return value

    @property
    def alembic_url(self) -> str:
        """The same database, spelled the way Alembic's async engine wants it."""
        return str(self.database_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. FastAPI dependencies and adapters share one instance."""
    return Settings()
