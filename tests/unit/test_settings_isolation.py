"""A developer's `.env` cannot reach the test suite (D173).

The failure this prevents is quiet: with the repository root's `.env` naming the
hybrid collection, a test's settings would mirror its writes into the real
`scholiast_chunks_hybrid`, and nothing would fail - the dev collection would just
gain test chunks that no embedded row accounts for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scholiast.config.settings import Settings, get_settings
from scholiast.infrastructure.vector.factory import build_vector_indexes
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex


@pytest.fixture
def deployment_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory with a deployment `.env` like the repository root's."""
    (tmp_path / ".env").write_text(
        "QDRANT_HYBRID_COLLECTION=scholiast_chunks_hybrid\nRETRIEVAL_MODE=hybrid\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QDRANT_HYBRID_COLLECTION", raising=False)
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    get_settings.cache_clear()
    return tmp_path


class TestTheSuiteIgnoresADeploymentEnvFile:
    def test_a_dotenv_in_the_working_directory_is_not_read(self, deployment_env: Path) -> None:
        settings = Settings()
        assert settings.qdrant_hybrid_collection is None
        assert settings.retrieval_mode == "dense"

    def test_so_a_tests_writes_reach_only_its_own_collection(self, deployment_env: Path) -> None:
        # The concrete harm: mirrored writes into the developer's hybrid collection.
        indexes = build_vector_indexes(Settings(qdrant_collection="test_only"))
        assert isinstance(indexes.write, QdrantVectorIndex)
        assert indexes.write.collection == "test_only"

    def test_the_cached_accessor_is_isolated_too(self, deployment_env: Path) -> None:
        assert get_settings().qdrant_hybrid_collection is None
        get_settings.cache_clear()

    def test_real_environment_variables_still_apply(
        self, deployment_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only the file is off: CI configures the suite through the environment.
        monkeypatch.setenv("QDRANT_HYBRID_COLLECTION", "from_the_environment")
        assert Settings().qdrant_hybrid_collection == "from_the_environment"
