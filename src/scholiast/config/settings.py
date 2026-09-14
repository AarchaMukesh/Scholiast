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
