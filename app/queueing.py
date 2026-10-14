"""Queue wiring. Today: Redis broker via Celery (zero extra infrastructure). At scale: swap `publish` + the worker
entrypoint for SQS; nothing else changes because jobs carry only an id and all state lives in Postgres/S3."""

from celery import Celery

from app.config import get_settings

_s = get_settings()

celery_app = Celery("evalplatform")
celery_app.conf.update(
    broker_url=_s.redis_url,
    task_default_queue="evaluations",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_ignore_result=True,
    worker_prefetch_multiplier=1,  # long jobs: never hoard messages whose visibility clock is already running
    broker_transport_options={"visibility_timeout": _s.celery_visibility_timeout_s},
    broker_connection_retry_on_startup=True,
    task_time_limit=_s.job_time_limit_s + 60,
    task_soft_time_limit=_s.job_time_limit_s,
    imports=("app.worker",),
)

MEMORY_QUEUE: list[tuple[str, dict]] = []  # queue_backend=memory (tests)

TASK_BY_EVENT = {"evaluate_job": "evaluate_job", "notify_job": "deliver_webhook"}


def publish(event_type: str, payload: dict) -> None:
    if get_settings().queue_backend == "memory":
        MEMORY_QUEUE.append((event_type, payload))
        return
    task = TASK_BY_EVENT[event_type]
    celery_app.send_task(task, args=[payload["aggregate_id"]], queue="evaluations")
