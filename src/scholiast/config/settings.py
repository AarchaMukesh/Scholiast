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

    # -- rate limiting and the ops surface (NFR-SEC7, D112-D114) -----------
    #: Off switch for the limiter. Present so a load test can measure the system
    #: rather than the limiter, and so a deployment that puts a real limiter in
    #: front can turn this one off instead of stacking two.
    rate_limit_enabled: bool = True
    #: Login, register and refresh, keyed by socket peer (D113). Ten a minute
    #: leaves room for typos and a password manager retrying, and makes
    #: credential stuffing pay a rate the Argon2id cost already makes expensive.
    rate_limit_auth_per_minute: int = 10
    #: The QA path, keyed by user. The most expensive endpoint in the system:
    #: retrieval plus generation, and NFR-P2's budget is per request. Twenty a
    #: minute is well above a human reading answers and well below monopolising
    #: the GPU.
    rate_limit_qa_per_minute: int = 20
    #: Uploads, keyed by user. Bounded because each one costs a GROBID parse and
    #: an embedding pass, not because the HTTP request is expensive.
    rate_limit_upload_per_minute: int = 30
    #: Discovery searches, keyed by user (D122). Each costs an OpenAlex search
    #: from an allowance of about 1,000 a day shared by EVERY user, and an arXiv
    #: request at arXiv's pace; one user at this limit would spend the OpenAlex
    #: allowance in under two hours. A guess, with no traffic yet to fit it to.
    rate_limit_discovery_per_minute: int = 10
    #: When set, `/metrics` and `/health/ready` require it as a bearer token
    #: (D114). Empty leaves them open, which is the development default;
    #: `/health/live` is never gated, because an orchestrator's restart decision
    #: must not need a credential.
    metrics_token: str | None = None

    # -- observability (§5.12, FR-O1) --------------------------------------
    #: JSON because the consumer is a collector, not a terminal. `false` gives
    #: structlog's console renderer for local work — a format choice, not a
    #: different logging system.
    log_json: bool = True
    log_level: str = "INFO"

    # -- provider circuit breaker (§5.11, D104) ----------------------------
    #: Consecutive failures before a provider is skipped without being called.
    #: Three, matching D48's retry budget for the same reason: it survives a
    #: restart of the thing being called without sitting through an outage.
    breaker_failure_threshold: int = 3
    #: How long to skip it when the provider said nothing about when to return.
    #: A 429's `Retry-After` overrides this, because the provider has stated the
    #: answer and this number can only guess at it (D103).
    breaker_cooldown_seconds: float = 30.0

    # -- discovery (M4 phase 3, D122-D125) -----------------------------------
    #: OpenAlex requires a key beyond a test allowance (D116). Without one,
    #: OpenAlex is absent rather than broken, as a provider with no credential is.
    openalex_api_key: str | None = None
    #: The sources' share of NFR-P1's 3.0 s; the rest is for dedup, ranking and the
    #: response (D122). Every source is cut off at the same moment.
    discovery_deadline_seconds: float = Field(default=2.0, gt=0.0)
    #: Measured, not chosen: at 25 results addendum A2 found p95 1.58 s against the
    #: 2.0 s deadline, with the same anchors as 50 (D125).
    discovery_openalex_page_size: int = Field(default=25, ge=1, le=200)
    discovery_arxiv_page_size: int = Field(default=50, ge=1, le=2000)
    #: arXiv's API terms: one request every three seconds. The floor is the terms',
    #: so a configuration cannot set a pace arXiv has asked us not to use (D123).
    discovery_arxiv_min_gap_seconds: float = Field(default=3.0, ge=3.0)
    #: The cache's own Redis (D128): capped and evicting, so it can never crowd out
    #: the Celery broker on `redis_url`. Down or unreachable, discovery runs uncached.
    discovery_cache_url: str = "redis://localhost:6380/0"
    #: D127: a day, so repeat demos are fast and identical while results stay fresh
    #: enough for a corpus that adds papers daily and a ranking that changed in three.
    discovery_cache_ttl_seconds: int = Field(default=86_400, ge=1)
    #: Bounds one Redis call. A healthy local Redis answers in well under a
    #: millisecond; this only caps what a sick one can cost a search.
    discovery_cache_timeout_seconds: float = Field(default=0.25, gt=0.0)

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
