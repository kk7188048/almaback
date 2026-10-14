"""Transactional outbox. Domain rows and the 'please run this job' message are written in ONE database transaction;
this relay then publishes committed rows to the queue. Delivery is at-least-once, so every consumer is idempotent."""

import logging
from collections.abc import Callable
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import session_scope, utcnow
from app.models import Outbox
from app.queueing import publish

log = logging.getLogger(__name__)


def enqueue_event(db: Session, event_type: str, aggregate_id: str, payload: dict | None = None, available_at: datetime | None = None) -> Outbox:
    row = Outbox(event_type=event_type, aggregate_id=aggregate_id, payload=payload or {}, available_at=available_at or utcnow())
    db.add(row)
    return row


def relay_once(batch: int = 50, publisher: Callable[[str, dict], None] = publish) -> int:
    published = 0
    with session_scope() as db:
        rows = db.scalars(
            select(Outbox)
            .where(Outbox.published_at.is_(None), Outbox.available_at <= utcnow())
            .order_by(Outbox.id)
            .limit(batch)
            .with_for_update(skip_locked=True)
        ).all()
        for row in rows:
            try:
                publisher(row.event_type, {**row.payload, "aggregate_id": row.aggregate_id, "event_id": row.id})
            except Exception:
                log.exception("outbox publish failed, will retry", extra={"ctx": {"outbox_id": row.id}})
                break
            row.published_at = utcnow()
            published += 1
    return published
