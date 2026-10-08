"""The §4.4.1 failure taxonomy, end to end (FR-I8, FR-I9, FR-I10, D48, D98).

M3's acceptance criterion for this phase is "each failure class produces the
correct state, retry behavior, and message", and the three halves are tested
separately because they are decided in three places and can drift apart:

  the class     `FailureClass.retryable` and `terminal_state_for` — the domain;
  the behaviour `PostgresQueue.fail` — requeue with backoff, or dead-letter;
  the message   `FailureClass.user_message` — what the user is told.

The table below is transcribed from §4.4.1 rather than derived from the code, so
that a change to the code fails a test instead of silently redefining the spec.
"""

from __future__ import annotations

import pytest

from scholiast.application.ports.ml import GpuBudgetExceeded
from scholiast.domain.errors import FailureClass
from scholiast.domain.workspace import PaperState, terminal_state_for
from scholiast.infrastructure.ml.embedder import _is_out_of_memory

#: §4.4.1's table: class -> (retryable, terminal state).
#: `RESOURCE_EXHAUSTED` is "Yes, with reduced batch size" — retryable, and the
#: reduction is D98's ladder, tested separately below.
SPEC: dict[FailureClass, tuple[bool, PaperState]] = {
    FailureClass.SOURCE_UNAVAILABLE: (False, PaperState.FAILED),
    FailureClass.TRANSIENT: (True, PaperState.FAILED),
    FailureClass.UNPARSEABLE: (False, PaperState.UNSUPPORTED),
    FailureClass.SCANNED_IMAGE: (False, PaperState.UNSUPPORTED),
    FailureClass.TOO_LARGE: (False, PaperState.UNSUPPORTED),
    FailureClass.RESOURCE_EXHAUSTED: (True, PaperState.FAILED),
}


class TestEveryClassMatchesTheSpec:
    def test_the_taxonomy_has_exactly_the_six_documented_classes(self) -> None:
        # A seventh class added without updating §4.4.1 would be a taxonomy the
        # product spec does not describe; a missing one is a documented failure
        # mode that cannot occur.
        assert set(FailureClass) == set(SPEC)

    @pytest.mark.parametrize("failure_class", list(SPEC), ids=lambda c: c.value)
    def test_retryability_matches(self, failure_class: FailureClass) -> None:
        expected, _ = SPEC[failure_class]
        assert failure_class.retryable is expected

    @pytest.mark.parametrize("failure_class", list(SPEC), ids=lambda c: c.value)
    def test_the_terminal_state_matches(self, failure_class: FailureClass) -> None:
        # UNSUPPORTED means "no false hope of retry" (§4.4.1); FAILED offers a
        # manual retry. Getting this backwards tells the user the opposite of
        # the truth about their paper.
        _, expected = SPEC[failure_class]
        assert terminal_state_for(failure_class) is expected

    @pytest.mark.parametrize("failure_class", list(SPEC), ids=lambda c: c.value)
    def test_every_class_says_something_actionable(self, failure_class: FailureClass) -> None:
        # NFR-U2: the message says what the user can DO. Asserting non-emptiness
        # only would pass on "Error." — so the check is that it is a sentence
        # and never leaks the class name, which is a log token, not English.
        message = failure_class.user_message
        assert message and message[0].isupper()
        # The enum's own token must not appear: "scanned_image" is a log value,
        # not English. Checked literally — an earlier version normalised spaces
        # to underscores first, which flagged the perfectly good message "This
        # file is too large" for containing the words behind `too_large`.
        assert failure_class.value not in message.lower()

    def test_nothing_retryable_ever_lands_on_unsupported(self) -> None:
        """UNSUPPORTED implies not-retryable — but NOT the converse.

        The first version of this test asserted the two were equivalent and was
        wrong about the product: `SOURCE_UNAVAILABLE` is not retryable and still
        lands on FAILED, because the paper is not structurally unusable — the
        *source* was missing, and §4.4.1's remedy is "upload the PDF to
        continue". UNSUPPORTED is reserved for papers that will never work, so
        it must never carry something the system intends to try again.
        """
        for failure_class in FailureClass:
            if terminal_state_for(failure_class) is PaperState.UNSUPPORTED:
                assert not failure_class.retryable, failure_class

    def test_unsupported_is_reserved_for_the_paper_itself_being_unusable(self) -> None:
        # The complement, stated so the asymmetry above is deliberate rather
        # than an accident of the mapping: exactly these three are about the
        # bytes the user supplied.
        unsupported = {c for c in FailureClass if terminal_state_for(c) is PaperState.UNSUPPORTED}
        assert unsupported == {
            FailureClass.UNPARSEABLE,
            FailureClass.SCANNED_IMAGE,
            FailureClass.TOO_LARGE,
        }


class TestOutOfMemoryDetection:
    """D98: reactive detection, and it has to recognise both shapes."""

    def test_a_plain_runtime_error_carrying_the_message_counts(self) -> None:
        # Older torch and some allocator paths raise this rather than the typed
        # error. It is also what makes the ladder testable without a GPU.
        assert _is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))

    def test_the_match_is_case_insensitive(self) -> None:
        assert _is_out_of_memory(RuntimeError("CUDA OUT OF MEMORY"))

    def test_an_unrelated_runtime_error_does_not(self) -> None:
        # The important negative: treating every RuntimeError as an OOM would
        # silently halve the batch on a real bug and then report capacity
        # exhaustion for something that had nothing to do with memory.
        assert not _is_out_of_memory(RuntimeError("shape mismatch in forward pass"))

    def test_an_unrelated_exception_type_does_not(self) -> None:
        assert not _is_out_of_memory(ValueError("out of memory"))


class FakeModel:
    """A sentence-transformers stand-in that OOMs above a batch threshold."""

    def __init__(self, *, fits_at: int) -> None:
        self.fits_at = fits_at
        self.attempts: list[int] = []

    def encode(self, texts: list[str], *, batch_size: int, **_: object) -> list[list[float]]:
        self.attempts.append(batch_size)
        if batch_size > self.fits_at:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [[0.1, 0.2] for _ in texts]


def _embedder_with(model: FakeModel):  # type: ignore[no-untyped-def]
    from scholiast.infrastructure.ml.embedder import LocalEmbedder

    embedder = LocalEmbedder("fake-model", device="cpu")
    embedder._model = model
    return embedder


class TestTheHalvingLadder:
    """FR-I10 / D98: halve in-adapter, escalate only at the floor."""

    def test_a_too_large_batch_is_halved_until_it_fits(self) -> None:
        model = FakeModel(fits_at=8)
        vectors = _embedder_with(model)._encode(["a", "b"], 32)

        assert model.attempts == [32, 16, 8]
        assert len(vectors) == 2

    def test_the_job_never_learns_about_a_batch_that_eventually_fits(self) -> None:
        # The point of halving in the adapter: a momentary spike costs
        # milliseconds, not a redelivery that re-parses and re-chunks the paper.
        model = FakeModel(fits_at=1)
        assert _embedder_with(model)._encode(["a"], 16)

    def test_a_single_item_that_will_not_fit_escalates(self) -> None:
        # Capacity, not sizing. Nothing smaller to try, so it becomes the
        # taxonomy's problem rather than the adapter's.
        model = FakeModel(fits_at=0)
        with pytest.raises(GpuBudgetExceeded):
            _embedder_with(model)._encode(["a"], 8)

        # It reached the floor before giving up, rather than escalating early.
        assert model.attempts == [8, 4, 2, 1]

    def test_a_non_memory_error_is_not_retried_at_all(self) -> None:
        # Halving a shape mismatch would turn one clear failure into five
        # confusing ones and then report the wrong failure class.
        class Broken(FakeModel):
            def encode(
                self, texts: list[str], *, batch_size: int, **_: object
            ) -> list[list[float]]:
                self.attempts.append(batch_size)
                raise RuntimeError("shape mismatch in forward pass")

        model = Broken(fits_at=999)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            _embedder_with(model)._encode(["a"], 8)
        assert model.attempts == [8]


class TestTheEscalationIsClassifiedCorrectly:
    def test_gpu_budget_exceeded_is_retryable_capacity_not_a_broken_paper(self) -> None:
        """The class a `GpuBudgetExceeded` must become.

        Marking it UNSUPPORTED would tell the user their paper is unusable when
        the truth is that the GPU was busy — and would offer them no retry.
        """
        assert FailureClass.RESOURCE_EXHAUSTED.retryable
        assert terminal_state_for(FailureClass.RESOURCE_EXHAUSTED) is PaperState.FAILED

    def test_the_exception_carries_what_torch_reported(self) -> None:
        # Diagnostics, not a budget: D24 says these numbers cannot see other
        # processes on WDDM, which is exactly why they are carried rather than
        # used to decide anything.
        exc = GpuBudgetExceeded(requested_gb=5.5, available_gb=0.2)
        assert exc.requested_gb == 5.5
        assert exc.available_gb == 0.2
        assert "5.50" in str(exc) and "0.20" in str(exc)


class _Session:
    """Enough of an AsyncSession for `_abandon` to construct its repositories."""

    def __init__(self) -> None:
        self.info: dict[str, object] = {}
        self.committed = False

    async def commit(self) -> None:
        self.committed = True

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class TestAClassificationSurvivesAPoisonedSession:
    """The M1 debt this phase closes.

    A DB-level failure earlier in the pipeline leaves the job's session inside a
    failed transaction, so the statement recording WHY the paper failed raises
    too. The crash handler then ran with only the write's exception in hand and
    re-classified everything as TRANSIENT — so the most informative failures
    were the ones most likely to be rewritten, and the user was shown a rollback
    error instead of "this PDF has no extractable text".
    """

    def _worker(self):  # type: ignore[no-untyped-def]
        from scholiast.workers.ingestion import IngestionWorker, WorkerConfig

        return IngestionWorker(
            sessions=lambda: _Session(),  # type: ignore[arg-type]
            store=None,  # type: ignore[arg-type]
            parser=None,  # type: ignore[arg-type]
            embedder=None,  # type: ignore[arg-type]
            index=None,  # type: ignore[arg-type]
            config=WorkerConfig(),
        )

    async def test_the_original_class_is_recorded_not_the_write_error(self) -> None:
        from scholiast.application.ports.queue import QueuedJob
        from scholiast.application.ports.tenancy import TenantContext
        from scholiast.workers.ingestion import (
            IngestionFailure,
            IngestionOutcome,
            _FailureRecordingFailed,
        )

        classified = IngestionOutcome(
            paper_id="paper-1",
            failure=IngestionFailure(
                failure_class=FailureClass.UNPARSEABLE,
                detail="no extractable text",
            ),
        )
        worker = self._worker()
        recorded: list[IngestionOutcome] = []

        async def capture(job, outcome, *, queue, papers):  # type: ignore[no-untyped-def]
            recorded.append(outcome)

        worker._record_failure = capture  # type: ignore[assignment,method-assign]

        await worker._abandon(
            QueuedJob(
                id="job-1",
                workspace_id="ws-1",
                paper_id="paper-1",
                state=PaperState.EMBEDDING,
                attempts=1,
            ),
            TenantContext(user_id="user-1"),
            _FailureRecordingFailed(classified, RuntimeError("current transaction is aborted")),
        )

        assert len(recorded) == 1
        failure = recorded[0].failure
        assert failure is not None
        assert failure.failure_class is FailureClass.UNPARSEABLE
        assert failure.detail == "no extractable text"

    async def test_an_ordinary_crash_is_still_transient(self) -> None:
        # The other half: a genuine unexpected crash has no classification to
        # preserve, and TRANSIENT with bounded retries remains the right default
        # so one bug cannot mark a corpus permanently unsupported.
        from scholiast.application.ports.queue import QueuedJob
        from scholiast.application.ports.tenancy import TenantContext
        from scholiast.workers.ingestion import IngestionOutcome

        worker = self._worker()
        recorded: list[IngestionOutcome] = []

        async def capture(job, outcome, *, queue, papers):  # type: ignore[no-untyped-def]
            recorded.append(outcome)

        worker._record_failure = capture  # type: ignore[assignment,method-assign]

        await worker._abandon(
            QueuedJob(
                id="job-1",
                workspace_id="ws-1",
                paper_id="paper-1",
                state=PaperState.PARSING,
                attempts=0,
            ),
            TenantContext(user_id="user-1"),
            ZeroDivisionError("a real bug"),
        )

        failure = recorded[0].failure
        assert failure is not None
        assert failure.failure_class is FailureClass.TRANSIENT
        assert "ZeroDivisionError" in failure.detail


class TestARetryableClassSpeaksAfterRetryingHasStopped:
    """D143: a membership carries a class only once it is terminal.

    A retry clears it, and a paper being retried shows QUEUED with no message, so
    a retryable class's message is read on a FAILED paper whose attempts ran out.
    "Retrying automatically" there was false every time it was shown.
    """

    @pytest.mark.parametrize(
        "failure_class", [c for c in FailureClass if c.retryable], ids=lambda c: c.value
    )
    def test_it_never_claims_to_be_retrying(self, failure_class: FailureClass) -> None:
        message = failure_class.user_message.lower()
        assert "retrying automatically" not in message
        assert "retrying with" not in message
        # There is no retry route: what the user can do is add it again.
        assert "add it again" in message
