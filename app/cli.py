"""Operational CLI:  python -m app.cli <command>

  create-user EMAIL PASSWORD ROLE [NAME]   create a student / evaluator / admin
  seed-demo                                demo course, users and the assignment's example rubric
  s3-configure                             apply lifecycle (abort stale multipart uploads, expiry) + CORS to the bucket
  create-tables                            dev only: create tables without Alembic
"""

import sys

from sqlalchemy import select

from app.db import Base, get_engine, session_scope
from app.models import Assignment, Course, CourseMember, Rubric, RubricCriterion, User
from app.security import hash_password

EXAMPLE_RUBRIC = [
    ("Functionality", 0.25, "Does the project work and implement the required features?"),
    ("Code Quality", 0.20, "Readability, structure, error handling, tests, lint cleanliness."),
    ("Architecture", 0.20, "Separation of concerns, data model, API and module design."),
    ("Problem Solving", 0.15, "Approach to the problem, handling of edge cases, trade-offs."),
    ("Documentation", 0.10, "README, setup instructions, API/DB docs."),
    ("Innovation / AI Usage", 0.10, "Originality and effective use of AI where relevant."),
]


def create_user(email: str, password: str, role: str, name: str = "") -> User:
    with session_scope() as db:
        existing = db.scalar(select(User).where(User.email == email.lower()))
        if existing:
            return existing
        user = User(email=email.lower(), name=name, password_hash=hash_password(password), role=role)
        db.add(user)
        db.flush()
        return user


def seed_demo() -> dict:
    admin = create_user("admin@example.com", "change-me-please-1", "admin", "Admin")
    evaluator = create_user("evaluator@example.com", "change-me-please-2", "evaluator", "Evaluator")
    student = create_user("student@example.com", "change-me-please-3", "student", "Student")
    with session_scope() as db:
        course = db.scalar(select(Course).where(Course.title == "Data Science Capstone"))
        if course is None:
            course = Course(title="Data Science Capstone", description="Demo course")
            db.add(course)
            db.flush()
            db.add_all([CourseMember(user_id=evaluator.id, course_id=course.id, role="evaluator"), CourseMember(user_id=student.id, course_id=course.id, role="student")])
            assignment = Assignment(course_id=course.id, title="Capstone project", description="Build and document an end-to-end project.")
            db.add(assignment)
            db.flush()
            rubric = Rubric(assignment_id=assignment.id, version=1, active=True, created_by=admin.id)
            rubric.criteria = [RubricCriterion(name=n, description=d, weight=w, max_score=100 * w, position=i) for i, (n, w, d) in enumerate(EXAMPLE_RUBRIC)]
            db.add(rubric)
            db.flush()
        assignment = db.scalar(select(Assignment).where(Assignment.course_id == course.id))
        return {"course_id": course.id, "assignment_id": assignment.id, "users": ["admin@example.com", "evaluator@example.com", "student@example.com"]}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    cmd, args = argv[0], argv[1:]
    if cmd == "create-user" and len(args) >= 3:
        u = create_user(args[0], args[1], args[2], args[3] if len(args) > 3 else "")
        print(u.id)
    elif cmd == "seed-demo":
        print(seed_demo())
    elif cmd == "s3-configure":
        from app.storage import S3Storage, get_storage

        storage = get_storage()
        if not isinstance(storage, S3Storage):
            print("STORAGE_BACKEND is not s3")
            return 1
        print(storage.configure_bucket())
    elif cmd == "create-tables":
        Base.metadata.create_all(get_engine())
        print("tables created")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
