"""Resource-level authorization. Every read of a tenant-owned row goes through here, so no endpoint can forget it.
Rules: students see only their own submissions; evaluators only those in courses they are assigned to; admins all.
A resource you may not see answers 404 (not 403) so ids cannot be probed."""

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Assignment, CourseMember, EvaluationAttempt, EvaluationJob, Submission, User


def not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "not found")


def evaluator_course_ids(user: User):
    return select(CourseMember.course_id).where(CourseMember.user_id == user.id, CourseMember.role == "evaluator")


def is_course_member(db: Session, user: User, course_id: str, role: str | None = None) -> bool:
    q = select(CourseMember.user_id).where(CourseMember.user_id == user.id, CourseMember.course_id == course_id)
    if role:
        q = q.where(CourseMember.role == role)
    return db.scalar(q) is not None


def can_manage_course(db: Session, user: User, course_id: str) -> bool:
    return user.role == "admin" or (user.role == "evaluator" and is_course_member(db, user, course_id, "evaluator"))


def can_view_course(db: Session, user: User, course_id: str) -> bool:
    return user.role == "admin" or is_course_member(db, user, course_id)


def can_view_submission(db: Session, user: User, sub: Submission) -> bool:
    if user.role == "admin":
        return True
    if user.role == "student":
        return sub.student_id == user.id
    assignment = db.get(Assignment, sub.assignment_id)
    return assignment is not None and is_course_member(db, user, assignment.course_id, "evaluator")


def get_submission(db: Session, user: User, submission_id: str) -> Submission:
    sub = db.get(Submission, submission_id)
    if sub is None or not can_view_submission(db, user, sub):
        raise not_found()
    return sub


def get_job(db: Session, user: User, job_id: str) -> tuple[EvaluationJob, Submission]:
    job = db.get(EvaluationJob, job_id)
    if job is None:
        raise not_found()
    return job, get_submission(db, user, job.submission_id)


def get_attempt_for_review(db: Session, user: User, attempt_id: str) -> tuple[EvaluationAttempt, EvaluationJob, Submission]:
    attempt = db.get(EvaluationAttempt, attempt_id)
    if attempt is None or user.role == "student":
        raise not_found()
    job, sub = get_job(db, user, attempt.job_id)
    return attempt, job, sub


def visible_submissions(user: User):
    q = select(Submission)
    if user.role == "student":
        return q.where(Submission.student_id == user.id)
    if user.role == "evaluator":
        return q.join(Assignment, Assignment.id == Submission.assignment_id).where(Assignment.course_id.in_(evaluator_course_ids(user)))
    return q
