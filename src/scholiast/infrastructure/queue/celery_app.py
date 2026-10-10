"""The Celery application (§5.7.4, doc 7 §7.3, D95, D96, D97).

Celery is the **transport**. The job lives in Postgres and always did; what
arrives here is a wake-up carrying a job id, and everything durable about it —
the state machine, the lease, the attempt count, the tenant predicate — stays in
the row (D95). That division is what lets this module be configured for delivery
semantics alone and stay silent about the domain.

**The four settings below are the ones that matter**, and each answers a line of
§5.7.4:

  `task_acks_late`                a task is acknowledged when it FINISHES, not
                                  when it is received. A worker killed mid-paper
                                  therefore leaves the message unacknowledged and
                                  the broker redelivers it.

  `task_reject_on_worker_lost`    without it, `acks_late` still loses work when
                                  the process dies hard (SIGKILL, OOM) rather
                                  than raising — precisely the case M3's
                                  acceptance criterion names.

  `visibility_timeout`            Redis has no server-side ack, so the broker
                                  re-queues a message no worker has finished
                                  within this window. It is deliberately set
                                  from the SAME number as the database lease:
                                  two redelivery clocks that disagree produce a
                                  job running twice or a job nobody retries.

  `worker_prefetch_multiplier=1`  a long task must not sit in a dead worker's
                                  local buffer. With late acks, prefetching is
                                  how "at least once" quietly becomes "at most
                                  once, eventually".

**Two queues, by latency profile rather than by caller** (§5.7.4's "interactive
vs bulk backfill"). `ingest` carries a user's upload, which somebody is watching;
`backfill` carries discovered papers (D140). **Which job runs is not decided
here:** a wake-up claims whatever is due, and the claim takes interactive work
first (D139). The queues are how an operator tells the two kinds of work apart,
and where a deployment can give backfill its own worker. Every queue a worker
must consume is in `WORKER_QUEUES`, and a test holds every route to it.

**Beat runs two sweeps** (D100, D141): reconciliation, and an ingestion wake-up
every `ingest_sweep_seconds`. The second is what wakes a worker for a retry or a
crash redelivery, which no enqueue announces.

Idempotency is not configured here because it is not configurable: at-least-once
delivery is only safe because the consumer is idempotent (FR-I7), which the
claim-from-Postgres path already is — a redelivered job whose row is INDEXED
finds nothing to claim.
"""

from __future__ import annotations

from celery import Celery

from scholiast.config.settings import Settings

#: The wire contract between two deployables. The API addresses this string; the
#: worker registers it. It is a constant rather than a literal at either end
#: because renaming it silently is how a dispatch starts going nowhere (D96).
INGEST_TASK = "scholiast.ingest_paper"

#: A user is waiting on this one.
INGEST_QUEUE = "ingest"
#: Bulk work that may wait behind it (M4's discovery backfill).
BACKFILL_QUEUE = "backfill"

#: The workspace cascade (§5.6.4 step 2, D99) and the reconciliation sweep
#: (step 4, D100). Both address a workspace or nothing at all, never a job id.
DELETE_TASK = "scholiast.delete_workspace"
RECONCILE_TASK = "scholiast.reconcile"

#: Deletion and reconciliation. **Its own queue because `worker_prefetch_
#: multiplier=1` makes queue choice a scheduling decision** (D100): a worker
#: holds one message at a time, so a cascade sweeping three stores would occupy
#: that slot for its whole duration — on `ingest`, a user's upload would wait
#: behind somebody else's delete. Declared now for the same reason the others
#: were: a queue added after the workers are deployed is a queue nothing
#: consumes until every worker restarts.
MAINTENANCE_QUEUE = "maintenance"

#: Every queue a worker must consume, in the order the documented command lists
#: them. A route or a publish to anything outside it is a message nothing reads,
#: which is what `backfill` would have become in M4 without this (D140).
WORKER_QUEUES: tuple[str, ...] = (INGEST_QUEUE, BACKFILL_QUEUE, MAINTENANCE_QUEUE)

#: Where a task goes when it dies for a reason the pipeline never saw — an
#: unhandled crash in the task wrapper rather than a classified ingestion
#: failure. §4.4.1's taxonomy handles the latter and writes a terminal state with
#: a reason; this queue exists so the former is inspectable instead of vanishing
#: into a log line. Phase 4 owns the taxonomy; this is only its parking space.
DEAD_LETTER_QUEUE = "dead-letter"


def build_celery_app(settings: Settings) -> Celery:
    """Construct the app both the API and the worker configure themselves from.

    The API never calls a task through this object — it addresses `INGEST_TASK`
    by name (D96) — but it does need the broker URL and the queue names, and
    having one place that derives them from `Settings` is what keeps the two
    processes from drifting into different queues.
    """
    app = Celery("scholiast", broker=settings.redis_url, backend=None)
    app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_default_queue=INGEST_QUEUE,
        task_routes={
            INGEST_TASK: {"queue": INGEST_QUEUE},
            DELETE_TASK: {"queue": MAINTENANCE_QUEUE},
            RECONCILE_TASK: {"queue": MAINTENANCE_QUEUE},
        },
        # §5.6.4 step 4's "periodic" made real (D100). A sweep an operator has
        # to remember to run is an intention, not a schedule — and this one is
        # the safety net every other deletion step is allowed to be
        # interruptible *because* of. Run by exactly one beat process: two of
        # them double every entry here.
        beat_schedule={
            "reconcile": {
                "task": RECONCILE_TASK,
                "schedule": float(settings.reconcile_interval_seconds),
                "options": {"queue": MAINTENANCE_QUEUE},
            },
            # D141. The sweep every docstring here assumed and nothing ran: a
            # retried job and a crash redelivery are due with no enqueue to
            # announce them. No argument, because a wake-up names no job.
            "ingest-sweep": {
                "task": INGEST_TASK,
                "schedule": float(settings.ingest_sweep_seconds),
                "options": {"queue": INGEST_QUEUE},
            },
        },
        # Redis-specific: how long before an unfinished message is re-queued.
        # Tied to the database lease on purpose — see the module docstring.
        # No `queue_order_strategy`: kombu's "priority" ordered the queues by a
        # string hash, differently in each process (D139), and the claim, not
        # the broker, is what puts an upload first. Round-robin, the default,
        # keeps a stream of wake-ups from starving maintenance work orders.
        broker_transport_options={
            "visibility_timeout": int(settings.worker_lease_seconds),
        },
        # No result backend. Nothing waits on a task's return value: progress is
        # read from `ingestion_jobs` via GET /v1/jobs/{id} (FR-I3), and a second
        # store of the same fact would be a second thing to keep true.
        result_backend=None,
        task_ignore_result=True,
        # Serialise as JSON, never pickle. A broker that accepts pickle accepts
        # code execution from anything that can write to the queue.
        task_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )
    return app
