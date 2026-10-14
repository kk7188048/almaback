from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import services
from app.db import get_db
from app.deps import get_current_user, idempotency_key, require_roles
from app.models import EvaluationJob, Submission, User
from app.schemas import CompleteUploadsIn, CreateSubmissionIn, JobOut, SubmissionCreatedOut, UploadInstructionOut
from app.scoping import get_submission, visible_submissions

router = APIRouter(prefix="/api/v1", tags=["submissions"])


@router.post("/submissions", response_model=SubmissionCreatedOut, status_code=status.HTTP_202_ACCEPTED)
def create_submission(
    body: CreateSubmissionIn, response: Response, key: str = Depends(idempotency_key),
    user: User = Depends(require_roles("student")), db: Session = Depends(get_db),
) -> SubmissionCreatedOut:
    out = services.create_submission(db, user, body, key)
    if out.idempotent_replay:
        response.headers["Idempotent-Replay"] = "true"
        response.status_code = status.HTTP_200_OK
    return out


@router.get("/submissions")
def list_submissions(
    limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), assignment_id: str | None = None,
    user: User = Depends(get_current_user), db: Session = Depends(get_db),
) -> dict:
    q = visible_submissions(user)
    if assignment_id:
        q = q.where(Submission.assignment_id == assignment_id)
    rows = db.scalars(q.order_by(Submission.created_at.desc()).limit(limit).offset(offset)).all()
    items = []
    for sub in rows:
        job = db.scalar(select(EvaluationJob).where(EvaluationJob.submission_id == sub.id).order_by(EvaluationJob.created_at.desc()))
        items.append({
            "submission_id": sub.id, "assignment_id": sub.assignment_id, "student_id": sub.student_id,
            "created_at": sub.created_at.isoformat(), "latest_job_id": job.id if job else None, "status": job.status if job else None,
            "sources": [{"type": s["type"], **({"url": s["url"]} if "url" in s else {"filename": s.get("filename")})} for s in sub.sources],
        })
    return {"items": items, "limit": limit, "offset": offset}


@router.get("/submissions/{submission_id}")
def get_one(submission_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    sub = get_submission(db, user, submission_id)
    jobs = db.scalars(select(EvaluationJob).where(EvaluationJob.submission_id == sub.id).order_by(EvaluationJob.created_at.desc())).all()
    return {
        "submission_id": sub.id, "assignment_id": sub.assignment_id, "student_id": sub.student_id, "created_at": sub.created_at.isoformat(),
        "sources": [{k: v for k, v in s.items() if k not in ("storage_key", "multipart_upload_id")} for s in sub.sources],
        "jobs": [{"job_id": j.id, "status": j.status, "created_at": j.created_at.isoformat()} for j in jobs],
    }


@router.post("/submissions/{submission_id}/uploads", response_model=list[UploadInstructionOut])
def refresh_uploads(submission_id: str, user: User = Depends(require_roles("student")), db: Session = Depends(get_db)) -> list[UploadInstructionOut]:
    sub = get_submission(db, user, submission_id)
    return services.issue_uploads(db, sub)


@router.post("/submissions/{submission_id}/uploads/complete", response_model=JobOut)
def complete_uploads(submission_id: str, body: CompleteUploadsIn, user: User = Depends(require_roles("student")), db: Session = Depends(get_db)) -> JobOut:
    sub = get_submission(db, user, submission_id)
    return services.complete_uploads(db, user, sub, body)


@router.post("/submissions/{submission_id}/evaluate", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def evaluate(
    submission_id: str, response: Response, key: str = Depends(idempotency_key),
    user: User = Depends(get_current_user), db: Session = Depends(get_db),
) -> JobOut:
    sub = get_submission(db, user, submission_id)
    job, replay = services.request_evaluation(db, user, sub, key)
    if replay:
        response.headers["Idempotent-Replay"] = "true"
        response.status_code = status.HTTP_200_OK
    return services.job_out(db, job)


@router.get("/submissions/{submission_id}/result")
def result(submission_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    sub = get_submission(db, user, submission_id)
    return services.build_result(db, user, sub)
