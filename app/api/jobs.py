import asyncio
import json

from fastapi import APIRouter, Depends, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app import services
from app.db import get_db, session_scope
from app.deps import get_current_user
from app.models import EvaluationJob, User
from app.scoping import get_job

router = APIRouter(prefix="/api/v1", tags=["jobs"])
TERMINAL = {"SUCCEEDED", "FAILED", "NEEDS_REVIEW"}


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str, response: Response, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    job, _sub = get_job(db, user, job_id)
    if job.status not in TERMINAL:
        response.headers["Retry-After"] = "5"
    return services.job_out(db, job).model_dump(mode="json")


def _snapshot(job_id: str) -> dict | None:
    with session_scope() as db:
        job = db.get(EvaluationJob, job_id)
        return services.job_out(db, job).model_dump(mode="json") if job else None


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Server-sent events. Clients that cannot hold a stream just poll GET /jobs/{id}."""
    get_job(db, user, job_id)  # authorization, once
    db.close()

    async def stream():
        last = None
        for _ in range(900):  # ~15 min cap; the client reconnects
            if await request.is_disconnected():
                return
            snap = await run_in_threadpool(_snapshot, job_id)
            if snap is None:
                return
            payload = json.dumps({k: snap[k] for k in ("status", "current_stage", "progress", "attempt", "queue_position")})
            if payload != last:
                yield f"event: status\ndata: {payload}\n\n"
                last = payload
            if snap["status"] in TERMINAL:
                return
            await asyncio.sleep(1.0)
        yield "event: timeout\ndata: {}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
