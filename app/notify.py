"""Signed webhooks. The payload carries status only (no scores): the receiver fetches details with its own auth."""

import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable

from app.config import get_settings
from app.db import session_scope
from app.models import EvaluationJob, Submission
from app.ssrf import SSRFError, safe_request

log = logging.getLogger(__name__)


def sign(secret: str, timestamp: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def deliver_webhook(job_id: str, *, transport=None, sleep: Callable[[float], None] = time.sleep, resolver=None) -> bool:
    s = get_settings()
    with session_scope() as db:
        job = db.get(EvaluationJob, job_id)
        if job is None:
            return False
        sub = db.get(Submission, job.submission_id)
        if sub is None or not sub.webhook_url:
            return False
        url, status, submission_id = sub.webhook_url, job.status, sub.id
    body = json.dumps({
        "event_id": f"evt_{job_id}_{status}",  # stable: receivers dedupe on this
        "event": "evaluation.updated",
        "job_id": job_id,
        "submission_id": submission_id,
        "status": status,
        "result_url": f"{s.public_base_url}/api/v1/submissions/{submission_id}/result",
    }).encode()
    ts = str(int(time.time()))
    headers = {"Content-Type": "application/json", "X-Evalplatform-Timestamp": ts, "X-Evalplatform-Signature": sign(s.webhook_signing_secret, ts, body)}
    for attempt in range(3):
        try:
            resp = safe_request("POST", url, headers=headers, content=body, max_bytes=64 * 1024, timeout=10, max_redirects=0, transport=transport, resolver=resolver)
            if 200 <= resp.status < 300:
                return True
            log.warning("webhook returned HTTP %s", resp.status)
        except SSRFError as exc:
            if "could not connect" not in str(exc):
                log.warning("webhook destination rejected: %s", exc)
                return False
        if attempt < 2:
            sleep(2 ** (attempt + 1))
    return False
