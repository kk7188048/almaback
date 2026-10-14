from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user, require_roles
from app.models import Assignment, Course, CourseMember, Rubric, RubricCriterion, User
from app.schemas import AssignmentIn, CourseIn, MemberIn, RubricIn, UserCreateIn, UserOut
from app.scoping import can_manage_course, can_view_course, not_found
from app.security import hash_password

router = APIRouter(prefix="/api/v1", tags=["courses"])


def _assignment_out(a: Assignment) -> dict:
    return {"id": a.id, "course_id": a.course_id, "title": a.title, "description": a.description,
            "opens_at": a.opens_at.isoformat() if a.opens_at else None, "closes_at": a.closes_at.isoformat() if a.closes_at else None}


def _rubric_out(r: Rubric) -> dict:
    return {
        "id": r.id, "assignment_id": r.assignment_id, "version": r.version, "active": r.active,
        "criteria": [{"id": c.id, "name": c.name, "description": c.description, "weight": c.weight, "max_score": c.max_score, "scale_anchors": c.scale_anchors} for c in r.criteria],
    }


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(body: UserCreateIn, _admin: User = Depends(require_roles("admin")), db: Session = Depends(get_db)) -> User:
    user = User(email=body.email.lower(), name=body.name, password_hash=hash_password(body.password), role=body.role)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "a user with this email already exists") from None
    return user


@router.post("/courses", status_code=status.HTTP_201_CREATED)
def create_course(body: CourseIn, _admin: User = Depends(require_roles("admin")), db: Session = Depends(get_db)) -> dict:
    course = Course(title=body.title, description=body.description)
    db.add(course)
    db.commit()
    return {"id": course.id, "title": course.title, "description": course.description}


@router.get("/courses")
def list_courses(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    q = select(Course).order_by(Course.created_at.desc())
    if user.role != "admin":
        q = q.join(CourseMember, CourseMember.course_id == Course.id).where(CourseMember.user_id == user.id)
    return [{"id": c.id, "title": c.title, "description": c.description} for c in db.scalars(q.limit(200))]


@router.post("/courses/{course_id}/members", status_code=status.HTTP_201_CREATED)
def add_member(course_id: str, body: MemberIn, _admin: User = Depends(require_roles("admin")), db: Session = Depends(get_db)) -> dict:
    user = db.get(User, body.user_id)
    if db.get(Course, course_id) is None or user is None:
        raise not_found()
    allowed = (body.role == "student" and user.role == "student") or (body.role == "evaluator" and user.role in ("evaluator", "admin"))
    if not allowed:
        raise HTTPException(422, "member role must match the user's account role")
    existing = db.get(CourseMember, (body.user_id, course_id))
    if existing is None:
        db.add(CourseMember(user_id=body.user_id, course_id=course_id, role=body.role))
        db.commit()
    return {"course_id": course_id, "user_id": body.user_id, "role": body.role}


@router.post("/courses/{course_id}/assignments", status_code=status.HTTP_201_CREATED)
def create_assignment(course_id: str, body: AssignmentIn, user: User = Depends(require_roles("admin", "evaluator")), db: Session = Depends(get_db)) -> dict:
    if db.get(Course, course_id) is None or not can_manage_course(db, user, course_id):
        raise not_found()
    a = Assignment(course_id=course_id, title=body.title, description=body.description, opens_at=body.opens_at, closes_at=body.closes_at)
    db.add(a)
    db.commit()
    return _assignment_out(a)


@router.get("/courses/{course_id}/assignments")
def list_assignments(course_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[dict]:
    if not can_view_course(db, user, course_id):
        raise not_found()
    return [_assignment_out(a) for a in db.scalars(select(Assignment).where(Assignment.course_id == course_id).order_by(Assignment.created_at.desc()).limit(200))]


@router.get("/assignments/{assignment_id}")
def get_assignment(assignment_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    a = db.get(Assignment, assignment_id)
    if a is None or not can_view_course(db, user, a.course_id):
        raise not_found()
    return _assignment_out(a)


@router.post("/rubrics", status_code=status.HTTP_201_CREATED)
def create_rubric(body: RubricIn, user: User = Depends(require_roles("admin", "evaluator")), db: Session = Depends(get_db)) -> dict:
    a = db.get(Assignment, body.assignment_id)
    if a is None or not can_manage_course(db, user, a.course_id):
        raise not_found()
    version = (db.scalar(select(func.max(Rubric.version)).where(Rubric.assignment_id == a.id)) or 0) + 1
    db.query(Rubric).filter(Rubric.assignment_id == a.id, Rubric.active.is_(True)).update({"active": False})
    rubric = Rubric(assignment_id=a.id, version=version, active=True, created_by=user.id)
    rubric.criteria = [
        RubricCriterion(name=c.name.strip(), description=c.description, weight=c.weight, max_score=c.max_score, scale_anchors=c.scale_anchors, position=i)
        for i, c in enumerate(body.criteria)
    ]
    db.add(rubric)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "another rubric change happened at the same time; retry") from None
    return _rubric_out(rubric)


@router.get("/assignments/{assignment_id}/rubric")
def get_active_rubric(assignment_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    a = db.get(Assignment, assignment_id)
    if a is None or not can_view_course(db, user, a.course_id):
        raise not_found()
    rubric = db.scalar(select(Rubric).where(Rubric.assignment_id == a.id, Rubric.active.is_(True)))
    if rubric is None:
        raise not_found()
    return _rubric_out(rubric)
