"""Celery entrypoint:  celery -A app.queueing:celery_app worker -Q evaluations -c 2"""

from app.logging_setup import setup_logging
from app.notify import deliver_webhook
from app.pipeline.runner import run_job
from app.queueing import celery_app

setup_logging()


@celery_app.task(name="evaluate_job", acks_late=True)
def evaluate_job(job_id: str) -> str:
    return run_job(job_id)


@celery_app.task(name="deliver_webhook", acks_late=True)
def deliver_webhook_task(job_id: str) -> bool:
    return deliver_webhook(job_id)
