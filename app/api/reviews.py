from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import services
from app.db import get_db
from app.deps import require_roles
from app.models import Assignment, EvaluationAttempt, EvaluationJob, Submission, User
from app.schemas import ReviewIn
from app.scoping import evaluator_course_ids, get_attempt_for_review

router = APIRouter(prefix="/api/v1", tags=["reviews"])


@router.get("/reviews/queue")
def review_queue(
    limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
    user: User = Depends(require_roles("evaluator", "admin")), db: Session = Depends(get_db),
) -> dict:
    q = (
        select(EvaluationJob, Submission)
        .join(Submission, Submission.id == EvaluationJob.submission_id)
        .where(EvaluationJob.status == "NEEDS_REVIEW")
    )
    if user.role == "evaluator":
        q = q.join(Assignment, Assignment.id == Submission.assignment_id).where(Assignment.course_id.in_(evaluator_course_ids(user)))
    items = []
    for job, sub in db.execute(q.order_by(EvaluationJob.updated_at).limit(limit).offset(offset)).all():
        attempt = db.scalar(
            select(EvaluationAttempt).where(EvaluationAttempt.job_id == job.id, EvaluationAttempt.status == "NEEDS_REVIEW").order_by(EvaluationAttempt.attempt_no.desc())
        )
        items.append({
            "job_id": job.id, "submission_id": sub.id, "assignment_id": sub.assignment_id, "attempt_id": attempt.id if attempt else None,
            "reasons": attempt.review_reasons if attempt else [], "ai_overall_score": attempt.overall_score if attempt else None,
            "waiting_since": job.updated_at.isoformat(),
        })
    return {"items": items, "limit": limit, "offset": offset}


@router.post("/evaluations/{attempt_id}/review", status_code=status.HTTP_201_CREATED)
def review(attempt_id: str, body: ReviewIn, user: User = Depends(require_roles("evaluator", "admin")), db: Session = Depends(get_db)) -> dict:
    attempt, job, _sub = get_attempt_for_review(db, user, attempt_id)
    return services.submit_review(db, user, attempt, job, body)
