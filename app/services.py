"""Business logic behind the submission / evaluation endpoints."""

import hashlib
import ipaddress
import json
import logging
from datetime import datetime, time as dtime, timedelta, timezone
from statistics import median

from fastapi import HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import state
from app.config import get_settings
from app.db import new_id, utcnow
from app.models import (
    Assignment, EvaluationAttempt, EvaluationJob, EvaluationResult, HumanReview, Rubric, Submission, User,
)
from app.outbox import enqueue_event
from app.pipeline.aggregate import aggregate
from app.pipeline.ingest.filetypes import validate_upload
from app.schemas import CompleteUploadsIn, CreateSubmissionIn, JobOut, ReviewIn, SubmissionCreatedOut, UploadInstructionOut
from app.scoping import is_course_member
from app.ssrf import SSRFError, validate_url
from app.storage import Storage, StorageError, get_storage

log = logging.getLogger(__name__)
UPLOAD_TYPES = {"zip", "pdf", "docs", "video"}
CONTENT_TYPES = {"zip": "application/zip", "pdf": "application/pdf", "docs": "application/octet-stream", "video": "application/octet-stream"}
SIZE_ATTR = {"zip": "max_zip_bytes", "pdf": "max_pdf_bytes", "docs": "max_docs_bytes", "video": "max_video_bytes"}


def http(code: int, detail) -> HTTPException:
    return HTTPException(code, detail)


def _day_start() -> datetime:
    return datetime.combine(utcnow().date(), dtime.min, tzinfo=timezone.utc)


def request_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _check_public_url_syntax(url: str, what: str) -> None:
    """Cheap, DNS-free screening at the API edge. The authoritative check (resolve + pin + re-check) happens at fetch time."""
    try:
        parsed = validate_url(url)
        try:
            ip = ipaddress.ip_address(parsed.host)
            if not ip.is_global:
                raise SSRFError("private address")
        except ValueError as exc:
            if isinstance(exc, SSRFError):
                raise
    except SSRFError as exc:
        raise http(422, f"{what} is not an acceptable public http(s) URL") from exc


def _validate_sources(sources: list[dict]) -> None:
    from app.pipeline.ingest import drive, github
    from app.pipeline.errors import SourceUnavailable

    for src in sources:
        try:
            if src["type"] == "github":
                github.parse_repo_url(src["url"])
            elif src["type"] == "drive":
                drive.parse_folder_id(src["url"])
            elif src["type"] == "live_url":
                _check_public_url_syntax(src["url"], "live_url")
        except SourceUnavailable as exc:
            raise http(422, f"{src['type']}: {exc.message}") from exc


def _active_rubric(db: Session, assignment_id: str) -> Rubric:
    rubric = db.scalar(select(Rubric).where(Rubric.assignment_id == assignment_id, Rubric.active.is_(True)))
    if rubric is None:
        raise http(status.HTTP_409_CONFLICT, "this assignment has no active rubric yet")
    return rubric


def _existing_by_key(db: Session, user_id: str, key: str) -> EvaluationJob | None:
    return db.scalar(select(EvaluationJob).where(EvaluationJob.created_by == user_id, EvaluationJob.idempotency_key == key))


def _latest_attempt_no(db: Session, job_id: str) -> int:
    return db.scalar(select(func.max(EvaluationAttempt.attempt_no)).where(EvaluationAttempt.job_id == job_id)) or 0


def issue_uploads(db: Session, sub: Submission, storage: Storage | None = None) -> list[UploadInstructionOut]:
    """(Re)issue presigned upload instructions for every upload source whose object is not stored yet."""
    storage = storage or get_storage()
    s = get_settings()
    out: list[UploadInstructionOut] = []
    sources = [dict(x) for x in sub.sources]
    for src in sources:
        if src["type"] not in UPLOAD_TYPES or storage.exists(src["storage_key"]):
            continue
        inst = storage.presign_upload(src["storage_key"], src["size_bytes"], getattr(s, SIZE_ATTR[src["type"]]), CONTENT_TYPES[src["type"]])
        mp = inst.get("multipart")
        src["multipart_upload_id"] = mp["upload_id"] if mp else None
        out.append(UploadInstructionOut(filename=src["filename"], **inst))
    sub.sources = sources
    db.commit()
    return out


def _created_out(db: Session, sub: Submission, job: EvaluationJob, replay: bool, uploads: list[UploadInstructionOut] | None = None) -> SubmissionCreatedOut:
    get_settings()
    return SubmissionCreatedOut(
        submission_id=sub.id, job_id=job.id, status=job.status, status_url=f"/api/v1/jobs/{job.id}",
        uploads=uploads or [],
        complete_url=f"/api/v1/submissions/{sub.id}/uploads/complete" if job.status == state.AWAITING_UPLOAD else None,
        idempotent_replay=replay,
    )


def create_submission(db: Session, user: User, body: CreateSubmissionIn, key: str) -> SubmissionCreatedOut:
    s = get_settings()
    payload = body.model_dump(mode="json")
    rhash = request_hash(payload)

    existing = _existing_by_key(db, user.id, key)
    if existing is not None:
        if existing.request_hash != rhash:
            raise http(422, "this Idempotency-Key was already used with a different request")
        sub = db.get(Submission, existing.submission_id)
        assert sub is not None
        uploads = issue_uploads(db, sub) if existing.status == state.AWAITING_UPLOAD else []
        return _created_out(db, sub, existing, True, uploads)

    assignment = db.get(Assignment, body.assignment_id)
    if assignment is None or not is_course_member(db, user, assignment.course_id, "student"):
        raise http(status.HTTP_404_NOT_FOUND, "not found")
    now = utcnow()
    if (assignment.opens_at and now < assignment.opens_at) or (assignment.closes_at and now > assignment.closes_at):
        raise http(status.HTTP_409_CONFLICT, "this assignment is not open for submissions")
    rubric = _active_rubric(db, assignment.id)
    today = db.scalar(select(func.count()).select_from(Submission).where(Submission.student_id == user.id, Submission.created_at >= _day_start())) or 0
    if today >= s.daily_submission_cap:
        raise http(status.HTTP_429_TOO_MANY_REQUESTS, "daily submission limit reached")
    if body.webhook_url:
        _check_public_url_syntax(body.webhook_url, "webhook_url")

    sub_id = new_id("sub")
    sources = []
    for idx, src in enumerate(body.sources):
        d = src.model_dump(mode="json")
        if d["type"] in UPLOAD_TYPES:
            d["storage_key"] = f"uploads/{sub_id}/{idx}_{d['filename']}"
        sources.append(d)
    _validate_sources(sources)
    needs_upload = any(d["type"] in UPLOAD_TYPES for d in sources)

    sub = Submission(id=sub_id, student_id=user.id, assignment_id=assignment.id, sources=sources, storage_prefix=f"uploads/{sub_id}/", webhook_url=body.webhook_url)
    job = EvaluationJob(
        submission_id=sub_id, rubric_id=rubric.id, status=state.AWAITING_UPLOAD if needs_upload else state.QUEUED,
        idempotency_key=key, request_hash=rhash, created_by=user.id,
    )
    db.add(sub)
    db.flush()  # the job's FK needs the submission row to exist first (no ORM relationship orders these)
    db.add(job)
    db.flush()
    if not needs_upload:
        enqueue_event(db, "evaluate_job", job.id)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raced = _existing_by_key(db, user.id, key)
        if raced is None:
            raise
        if raced.request_hash != rhash:
            raise http(422, "this Idempotency-Key was already used with a different request") from None
        racing_sub = db.get(Submission, raced.submission_id)
        assert racing_sub is not None
        return _created_out(db, racing_sub, raced, True)
    uploads = []
    if needs_upload:
        try:
            uploads = issue_uploads(db, sub)
        except Exception:
            log.exception("could not issue upload URLs")
            raise http(status.HTTP_503_SERVICE_UNAVAILABLE, "storage is unavailable; retry POST .../uploads to get upload URLs") from None
    return _created_out(db, sub, job, False, uploads)


def complete_uploads(db: Session, user: User, sub: Submission, body: CompleteUploadsIn) -> JobOut:
    storage = get_storage()
    s = get_settings()
    job = db.scalar(select(EvaluationJob).where(EvaluationJob.submission_id == sub.id).order_by(EvaluationJob.created_at.desc()))
    assert job is not None
    if job.status != state.AWAITING_UPLOAD:
        return job_out(db, job)
    parts_by_file = {u.filename: u.parts for u in body.uploads}
    sources = [dict(x) for x in sub.sources]
    for src in sources:
        if src["type"] not in UPLOAD_TYPES:
            continue
        key = src["storage_key"]
        upload_id = src.get("multipart_upload_id")
        if upload_id and not storage.exists(key):
            parts = parts_by_file.get(src["filename"])
            if not parts:
                raise http(422, f"{src['filename']}: multipart completion needs the uploaded parts and ETags")
            try:
                storage.complete_multipart(key, upload_id, [p.model_dump() for p in parts])
            except Exception as exc:
                raise http(422, f"{src['filename']}: could not complete the multipart upload") from exc
        try:
            size = storage.size(key)
        except StorageError:
            raise http(422, f"{src['filename']} has not been uploaded") from None
        limit = getattr(s, SIZE_ATTR[src["type"]])
        err = f"{src['filename']} exceeds the {limit // (1024 * 1024)} MB limit" if size > limit else validate_upload(src["type"], storage.read_head(key))
        if err:
            storage.delete(key)
            raise http(422, err if err.startswith(src["filename"]) else f"{src['filename']}: {err}")
    now = utcnow()
    moved = db.execute(
        update(EvaluationJob).where(EvaluationJob.id == job.id, EvaluationJob.status == state.AWAITING_UPLOAD).values(status=state.QUEUED, updated_at=now)
    ).rowcount
    if moved == 1:
        enqueue_event(db, "evaluate_job", job.id)
    db.commit()
    db.refresh(job)
    return job_out(db, job)


def request_evaluation(db: Session, user: User, sub: Submission, key: str) -> tuple[EvaluationJob, bool]:
    """POST /submissions/{id}/evaluate - safe to call any number of times with the same Idempotency-Key."""
    s = get_settings()
    rhash = request_hash({"submission_id": sub.id})
    existing = _existing_by_key(db, user.id, key)
    if existing is not None:
        if existing.request_hash != rhash or existing.submission_id != sub.id:
            raise http(422, "this Idempotency-Key was already used with a different request")
        return existing, True
    active = db.scalar(select(EvaluationJob).where(EvaluationJob.submission_id == sub.id, EvaluationJob.status.in_(state.ACTIVE_STATUSES)))
    if active is not None:
        raise http(status.HTTP_409_CONFLICT, {"message": "an evaluation is already in progress for this submission", "job_id": active.id})
    recent = db.scalar(select(func.count()).select_from(EvaluationJob).where(EvaluationJob.submission_id == sub.id, EvaluationJob.created_at >= utcnow() - timedelta(days=1))) or 0
    if recent >= s.max_evaluations_per_submission_per_day:
        raise http(status.HTTP_429_TOO_MANY_REQUESTS, "evaluation limit for this submission reached; try again later")
    storage = get_storage()
    for src in sub.sources:
        if src["type"] in UPLOAD_TYPES and not storage.exists(src["storage_key"]):
            raise http(status.HTTP_409_CONFLICT, f"{src['filename']} is no longer stored; submit again")
    assignment = db.get(Assignment, sub.assignment_id)
    assert assignment is not None
    rubric = _active_rubric(db, assignment.id)
    job = EvaluationJob(submission_id=sub.id, rubric_id=rubric.id, status=state.QUEUED, idempotency_key=key, request_hash=rhash, created_by=user.id)
    db.add(job)
    try:
        db.flush()
        enqueue_event(db, "evaluate_job", job.id)
        db.commit()
    except IntegrityError:
        db.rollback()
        raced = _existing_by_key(db, user.id, key)
        if raced is not None and raced.submission_id == sub.id:
            return raced, True
        active = db.scalar(select(EvaluationJob).where(EvaluationJob.submission_id == sub.id, EvaluationJob.status.in_(state.ACTIVE_STATUSES)))
        raise http(status.HTTP_409_CONFLICT, {"message": "an evaluation is already in progress for this submission", "job_id": active.id if active else None}) from None
    return job, False


def _avg_duration_s(db: Session) -> float:
    rows = db.execute(
        select(EvaluationAttempt.started_at, EvaluationAttempt.finished_at)
        .where(EvaluationAttempt.status.in_(["SUCCEEDED", "NEEDS_REVIEW"]), EvaluationAttempt.finished_at.is_not(None))
        .order_by(EvaluationAttempt.finished_at.desc()).limit(20)
    ).all()
    durations = [(f - st).total_seconds() for st, f in rows]
    return median(durations) if durations else 300.0


def job_out(db: Session, job: EvaluationJob) -> JobOut:
    s = get_settings()
    position = eta = None
    if job.status == state.QUEUED:
        position = db.scalar(select(func.count()).select_from(EvaluationJob).where(EvaluationJob.status == state.QUEUED, EvaluationJob.created_at < job.created_at)) or 0
        eta = utcnow() + timedelta(seconds=_avg_duration_s(db) * (position + 1))
    elif job.status == state.RUNNING:
        eta = job.updated_at + timedelta(seconds=_avg_duration_s(db))
    error = {"code": job.error_code, "message": job.error_message} if job.status == state.FAILED and job.error_code else None
    return JobOut(
        job_id=job.id, submission_id=job.submission_id, status=job.status, current_stage=job.current_stage, progress=job.progress,
        attempt=_latest_attempt_no(db, job.id), queue_position=position, estimated_completion=eta, error=error,
        result_url=f"{s.public_base_url}/api/v1/submissions/{job.submission_id}/result" if job.status in (state.SUCCEEDED, state.NEEDS_REVIEW) else None,
    )


def latest_review(db: Session, attempt_id: str) -> HumanReview | None:
    return db.scalar(select(HumanReview).where(HumanReview.attempt_id == attempt_id).order_by(HumanReview.created_at.desc(), HumanReview.id.desc()))


def build_result(db: Session, user: User, sub: Submission) -> dict:
    s = get_settings()
    is_staff = user.role in ("evaluator", "admin")
    job = db.scalar(
        select(EvaluationJob).where(EvaluationJob.submission_id == sub.id, EvaluationJob.status.in_([state.SUCCEEDED, state.NEEDS_REVIEW])).order_by(EvaluationJob.created_at.desc())
    )
    if job is None:
        latest = db.scalar(select(EvaluationJob).where(EvaluationJob.submission_id == sub.id).order_by(EvaluationJob.created_at.desc()))
        raise http(status.HTTP_409_CONFLICT, {"message": "no evaluation result is available yet", "status": latest.status if latest else None, "job_id": latest.id if latest else None})
    if job.status == state.NEEDS_REVIEW and not is_staff:
        raise http(status.HTTP_409_CONFLICT, {"message": "your evaluation is awaiting review by an evaluator", "status": "UNDER_REVIEW", "job_id": job.id})
    attempt = db.scalar(select(EvaluationAttempt).where(EvaluationAttempt.job_id == job.id, EvaluationAttempt.status.in_(["SUCCEEDED", "NEEDS_REVIEW"])).order_by(EvaluationAttempt.attempt_no.desc()))
    assert attempt is not None
    rubric = db.get(Rubric, job.rubric_id)
    assert rubric is not None
    crit_by_id = {c.id: c for c in rubric.criteria}
    results = db.scalars(select(EvaluationResult).where(EvaluationResult.attempt_id == attempt.id)).all()
    review = latest_review(db, attempt.id)
    adjusted = (review.adjusted_scores if review else {}) or {}
    criteria = []
    for r in sorted(results, key=lambda r: crit_by_id[r.criterion_id].position):
        c = crit_by_id[r.criterion_id]
        effective = adjusted.get(c.id, r.score)
        item = {
            "criterion_id": c.id, "criterion": c.name, "weight": c.weight, "max_score": c.max_score, "score": effective,
            "justification": r.detail.get("justification", ""), "evidence": r.detail.get("evidence", []),
        }
        if is_staff:
            item.update(ai_score=r.score, confidence=r.detail.get("confidence"), agreement=r.detail.get("agreement"), samples=r.detail.get("samples"), status=r.detail.get("status"))
        elif s.student_sees_confidence:
            item["confidence"] = r.detail.get("confidence")
        criteria.append(item)
    out = {
        "submission_id": sub.id, "job_id": job.id, "attempt_no": attempt.attempt_no, "rubric_id": rubric.id, "rubric_version": rubric.version,
        "status": job.status,
        "overall_score": review.final_score if review else attempt.overall_score, "max_score": 100,
        "criteria": criteria, "warnings": attempt.warnings,
        "human_review": {"reviewed": review is not None, "decision": review.decision if review else None, "comment": review.comment if review else None,
                          "reviewed_at": review.created_at.isoformat() if review else None},
    }
    if is_staff:
        out.update(ai_overall_score=attempt.overall_score, confidence=attempt.confidence, review_required=job.status == state.NEEDS_REVIEW,
                   review_reasons=attempt.review_reasons, attempt_id=attempt.id, model_info=attempt.model_info)
    elif s.student_sees_confidence:
        out["confidence"] = attempt.confidence
    return out


def submit_review(db: Session, user: User, attempt: EvaluationAttempt, job: EvaluationJob, body: ReviewIn) -> dict:
    if attempt.status not in ("SUCCEEDED", "NEEDS_REVIEW") or job.status not in (state.NEEDS_REVIEW, state.SUCCEEDED):
        raise http(status.HTTP_409_CONFLICT, "this evaluation cannot be reviewed in its current state")
    latest = db.scalar(select(func.max(EvaluationAttempt.attempt_no)).where(EvaluationAttempt.job_id == job.id, EvaluationAttempt.status.in_(["SUCCEEDED", "NEEDS_REVIEW"])))
    if attempt.attempt_no != latest:
        raise http(status.HTTP_409_CONFLICT, "only the most recent attempt can be reviewed")
    rubric = db.get(Rubric, job.rubric_id)
    assert rubric is not None
    by_key = {}
    for c in rubric.criteria:
        by_key[c.id] = c
        by_key[c.name.lower()] = c
    results = {r.criterion_id: r for r in db.scalars(select(EvaluationResult).where(EvaluationResult.attempt_id == attempt.id)).all()}
    adjusted: dict[str, float] = {}
    for k, v in body.adjusted_scores.items():
        c = by_key.get(k) or by_key.get(k.lower())
        if c is None:
            raise http(422, f"unknown criterion: {k}")
        if not 0 <= v <= c.max_score:
            raise http(422, f"{c.name}: score must be between 0 and {c.max_score}")
        adjusted[c.id] = v
    rows = []
    for c in rubric.criteria:
        r = results[c.id]
        score = adjusted.get(c.id, r.score)
        if score is None:
            raise http(422, f"{c.name} has no AI score; provide one in adjusted_scores")
        rows.append({"score": score, "max_score": c.max_score, "weight": c.weight, "confidence": 1.0})
    final, _ = aggregate(rows)
    assert final is not None
    review = HumanReview(attempt_id=attempt.id, reviewer_id=user.id, decision=body.decision, adjusted_scores=adjusted, final_score=final, comment=body.comment.strip())
    db.add(review)
    if job.status == state.NEEDS_REVIEW:
        moved = db.execute(update(EvaluationJob).where(EvaluationJob.id == job.id, EvaluationJob.status == state.NEEDS_REVIEW).values(status=state.SUCCEEDED, updated_at=utcnow())).rowcount
        if moved == 1:
            enqueue_event(db, "notify_job", job.id)
    db.commit()
    return {"review_id": review.id, "attempt_id": attempt.id, "decision": review.decision, "final_score": final, "job_status": state.SUCCEEDED}
