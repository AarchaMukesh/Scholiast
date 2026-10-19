"""The API warms its encoder before it accepts a request (D156).

Loaded on first use, the encoder cost the first search after a start 16.2 s in
D155's walkthrough: 3.9-5.9 s importing torch, 7.7-11.1 s asking the model hub
about updates (D157), and a slow first encode. A test that never builds the real
lifespan would not notice the warm-up being removed, so these run `api_lifespan`
itself, with the encoder and the vector index replaced: no model, no Qdrant, and
an engine that never connects.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from scholiast.api import app as app_module
from scholiast.api.app import api_lifespan, create_app
from scholiast.config.settings import Settings
from scholiast.infrastructure.vector import factory as factory_module


class _Encoder:
    instances: ClassVar[list[_Encoder]] = []
    fail_with: ClassVar[BaseException | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.warmed = 0
        _Encoder.instances.append(self)

    async def warm(self) -> None:
        if _Encoder.fail_with is not None:
            raise _Encoder.fail_with
        self.warmed += 1


class _Index:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def ensure_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.fixture
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> type[_Encoder]:
    _Encoder.instances = []
    _Encoder.fail_with = None
    monkeypatch.setattr(app_module, "LocalEmbedder", _Encoder)
    # The lifespan builds its indexes through the factory (D172), so that is where
    # the stand-in goes.
    monkeypatch.setattr(factory_module, "QdrantVectorIndex", _Index)
    return _Encoder


class TestTheEncoderIsWarmBeforeTheFirstRequest:
    async def test_the_lifespan_warms_the_encoder_before_it_yields(
        self, stand_ins: type[_Encoder]
    ) -> None:
        app = create_app(Settings(celery_enabled=False))

        async with api_lifespan(app):
            # Inside the context is where uvicorn starts accepting connections.
            assert [e.warmed for e in stand_ins.instances] == [1]

    async def test_tests_can_build_the_real_lifespan_without_loading_a_model(
        self, stand_ins: type[_Encoder]
    ) -> None:
        # NFR-T3: suites that swap in a stand-in embedder after startup must not
        # load a 419 MB model they never use, nor download one in CI.
        app = create_app(Settings(celery_enabled=False, embedding_warm_at_startup=False))

        async with api_lifespan(app):
            assert [e.warmed for e in stand_ins.instances] == [0]

    async def test_a_model_that_cannot_load_fails_the_boot(self, stand_ins: type[_Encoder]) -> None:
        # An API that cannot embed can neither answer nor rank by dense relevance;
        # better refused at startup than discovered at the first request (D84).
        stand_ins.fail_with = OSError("model not found")
        app = create_app(Settings(celery_enabled=False))

        with pytest.raises(OSError, match="model not found"):
            async with api_lifespan(app):
                pytest.fail("the API started without its encoder")


def test_warming_is_on_unless_a_deployment_turns_it_off() -> None:
    # The default is the production behaviour; forgetting the setting in a
    # deployment must not bring the 16.2 s first search back.
    assert Settings.model_fields["embedding_warm_at_startup"].default is True
