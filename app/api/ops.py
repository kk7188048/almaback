"""Health checks, metrics and the local-dev upload endpoint."""

import tempfile
from datetime import datetime, time as dtime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db, utcnow
from app.deps import require_roles
from app.models import EvaluationAttempt, EvaluationJob, Outbox, User
from app.redis_client import get_redis
from app.storage import LocalStorage, get_storage

router = APIRouter(tags=["ops"])


@router.get("/healthz")
def liveness() -> dict:
    """Process is up. Never touches dependencies, so a DB blip does not get containers killed."""
    return {"status": "ok"}


@router.get("/readyz")
def readiness(response: Response, db: Session = Depends(get_db)) -> dict:
    """Dependencies reachable. Gates load-balancer traffic and deploy promotion."""
    checks: dict[str, str] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "fail"
    try:
        get_redis().ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "fail"
    try:
        get_storage().exists("healthcheck/none")
        checks["storage"] = "ok"
    except Exception:
        checks["storage"] = "fail"
    s = get_settings()
    checks["llm"] = "configured" if (s.llm_provider in ("mock", "bedrock") or s.llm_api_key) else "missing_credentials"
    healthy = all(v in ("ok", "configured") for v in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if healthy else "degraded", "checks": checks}


@router.get("/api/v1/admin/metrics")
def metrics(_admin: User = Depends(require_roles("admin")), db: Session = Depends(get_db)) -> dict:
    now = utcnow()
    day_start = datetime.combine(now.date(), dtime.min, tzinfo=timezone.utc)
    by_status = dict(db.execute(select(EvaluationJob.status, func.count()).group_by(EvaluationJob.status)).all())
    oldest_queued = db.scalar(select(func.min(EvaluationJob.created_at)).where(EvaluationJob.status == "QUEUED"))
    tokens = db.execute(
        select(func.coalesce(func.sum(EvaluationAttempt.tokens_in), 0), func.coalesce(func.sum(EvaluationAttempt.tokens_out), 0), func.coalesce(func.sum(EvaluationAttempt.cost_usd), 0.0), func.count())
        .where(EvaluationAttempt.started_at >= day_start, EvaluationAttempt.status.in_(["SUCCEEDED", "NEEDS_REVIEW"]))
    ).one()
    pending_outbox = db.scalar(select(func.count()).select_from(Outbox).where(Outbox.published_at.is_(None), Outbox.available_at <= now)) or 0
    oldest_outbox = db.scalar(select(func.min(Outbox.available_at)).where(Outbox.published_at.is_(None), Outbox.available_at <= now))
    return {
        "jobs_by_status": by_status,
        "oldest_queued_age_s": int((now - oldest_queued).total_seconds()) if oldest_queued else 0,
        "dead_jobs": by_status.get("FAILED", 0),  # jobs that exhausted their attempts: our DLQ equivalent
        "needs_review": by_status.get("NEEDS_REVIEW", 0),
        "outbox_pending": pending_outbox,
        "outbox_oldest_age_s": int((now - oldest_outbox).total_seconds()) if oldest_outbox else 0,
        "today": {
            "evaluations": tokens[3], "tokens_in": int(tokens[0]), "tokens_out": int(tokens[1]), "cost_usd": round(float(tokens[2]), 4),
            "avg_tokens_per_evaluation": int((tokens[0] + tokens[1]) / tokens[3]) if tokens[3] else 0,
            "daily_token_budget": get_settings().global_daily_token_budget,
        },
    }


@router.put("/api/v1/_local/upload", status_code=status.HTTP_204_NO_CONTENT, include_in_schema=False)
async def local_upload(request: Request, key: str, exp: int, max: int, sig: str) -> Response:  # noqa: A002
    """Dev/test stand-in for an S3 presigned URL. Not mounted semantics for production (S3 is used there)."""
    storage = get_storage()
    if not isinstance(storage, LocalStorage):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    if not storage.verify(key, exp, max, sig):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid or expired upload URL")
    target = storage._path(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            async for chunk in request.stream():
                total += len(chunk)
                if total > max:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "upload exceeds the allowed size")
                tmp.write(chunk)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(target)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
