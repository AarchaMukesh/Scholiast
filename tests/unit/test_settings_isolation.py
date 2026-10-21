"""A developer's `.env` cannot reach the test suite (D173, D174).

The failure this prevents is quiet: a test's settings that picked up a local
deployment's `.env` would depend on one machine's file rather than on the test,
and with that file naming a collection, a test's mirrored writes would land in
the developer's real one. Nothing would fail - the dev collection would just
gain test chunks that no embedded row accounts for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scholiast.config.settings import Settings, get_settings
from scholiast.infrastructure.vector.factory import build_vector_indexes
from scholiast.infrastructure.vector.mirrored import MirroredVectorIndex

#: A name only the `.env` below uses, so reading it shows exactly where it came from.
FROM_DOTENV = "from_the_dotenv"


@pytest.fixture
def deployment_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory with a deployment `.env` that names its own collections."""
    (tmp_path / ".env").write_text(
        f"QDRANT_COLLECTION={FROM_DOTENV}\nQDRANT_HYBRID_COLLECTION={FROM_DOTENV}_hybrid\n"
        "RETRIEVAL_MODE=dense\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for name in ("QDRANT_COLLECTION", "QDRANT_HYBRID_COLLECTION", "RETRIEVAL_MODE"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    return tmp_path


class TestTheSuiteIgnoresADeploymentEnvFile:
    def test_a_dotenv_in_the_working_directory_is_not_read(self, deployment_env: Path) -> None:
        settings = Settings()
        assert settings.qdrant_collection == "scholiast_chunks"
        assert settings.qdrant_hybrid_collection == "scholiast_chunks_hybrid"
        assert settings.retrieval_mode == "hybrid"

    def test_so_a_tests_writes_reach_only_its_own_collections(self, deployment_env: Path) -> None:
        # The concrete harm: mirrored writes into a collection the test did not name.
        indexes = build_vector_indexes(Settings(qdrant_collection="test_only"))
        assert isinstance(indexes.write, MirroredVectorIndex)
        assert indexes.write.primary.collection == "test_only"
        assert indexes.write.mirror.collection == "test_only_hybrid"

    def test_the_cached_accessor_is_isolated_too(self, deployment_env: Path) -> None:
        assert get_settings().qdrant_collection == "scholiast_chunks"
        get_settings.cache_clear()

    def test_real_environment_variables_still_apply(
        self, deployment_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only the file is off: CI configures the suite through the environment.
        monkeypatch.setenv("QDRANT_HYBRID_COLLECTION", "from_the_environment")
        assert Settings().qdrant_hybrid_collection == "from_the_environment"
