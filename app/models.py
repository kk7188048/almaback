from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db import Base, UTCDateTime, new_id, utcnow

JSONType = JSON().with_variant(JSONB, "postgresql")

_settings = get_settings()
if _settings.use_pgvector:
    from pgvector.sqlalchemy import Vector

    VectorType = Vector(_settings.embedding_dim)
else:
    VectorType = JSON

_ACTIVE_WHERE = text("status IN ('AWAITING_UPLOAD','QUEUED','RUNNING')")


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("usr"))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    __table_args__ = (CheckConstraint("role IN ('student','evaluator','admin')", name="ck_users_role"),)


class Course(Base):
    __tablename__ = "courses"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("crs"))
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class CourseMember(Base):
    __tablename__ = "course_members"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str] = mapped_column(String(20))
    __table_args__ = (
        CheckConstraint("role IN ('student','evaluator')", name="ck_members_role"),
        Index("ix_course_members_course", "course_id"),
    )


class Assignment(Base):
    __tablename__ = "assignments"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("asg"))
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    opens_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    closes_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Rubric(Base):
    __tablename__ = "rubrics"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("rub"))
    assignment_id: Mapped[str] = mapped_column(ForeignKey("assignments.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    criteria: Mapped[list["RubricCriterion"]] = relationship(
        order_by="RubricCriterion.position", cascade="all, delete-orphan", lazy="selectin"
    )
    __table_args__ = (
        UniqueConstraint("assignment_id", "version", name="uq_rubric_version"),
        Index(
            "uq_rubric_one_active",
            "assignment_id",
            unique=True,
            postgresql_where=text("active"),
            sqlite_where=text("active"),
        ),
    )


class RubricCriterion(Base):
    __tablename__ = "rubric_criteria"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("crit"))
    rubric_id: Mapped[str] = mapped_column(ForeignKey("rubrics.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    weight: Mapped[float] = mapped_column(Float)
    max_score: Mapped[float] = mapped_column(Float)
    scale_anchors: Mapped[dict] = mapped_column(JSONType, default=dict)
    position: Mapped[int] = mapped_column(Integer, default=0)
    __table_args__ = (
        UniqueConstraint("rubric_id", "name", name="uq_criterion_name"),
        CheckConstraint("weight > 0 AND weight <= 1", name="ck_criterion_weight"),
        CheckConstraint("max_score > 0", name="ck_criterion_max"),
    )


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("sub"))
    student_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    assignment_id: Mapped[str] = mapped_column(ForeignKey("assignments.id"), index=True)
    sources: Mapped[list] = mapped_column(JSONType)
    storage_prefix: Mapped[str] = mapped_column(String(255))
    webhook_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class EvaluationJob(Base):
    __tablename__ = "evaluation_jobs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("job"))
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)
    rubric_id: Mapped[str] = mapped_column(ForeignKey("rubrics.id"))
    status: Mapped[str] = mapped_column(String(20))
    current_stage: Mapped[str | None] = mapped_column(String(30), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    idempotency_key: Mapped[str] = mapped_column(String(100))
    request_hash: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    locked_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    __table_args__ = (
        CheckConstraint(
            "status IN ('AWAITING_UPLOAD','QUEUED','RUNNING','SUCCEEDED','FAILED','NEEDS_REVIEW')",
            name="ck_job_status",
        ),
        UniqueConstraint("created_by", "idempotency_key", name="uq_job_idempotency"),
        Index("ix_jobs_status_created", "status", "created_at"),
        Index(
            "uq_job_one_active_per_submission",
            "submission_id",
            unique=True,
            postgresql_where=_ACTIVE_WHERE,
            sqlite_where=_ACTIVE_WHERE,
        ),
    )


class EvaluationAttempt(Base):
    __tablename__ = "evaluation_attempts"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("att"))
    job_id: Mapped[str] = mapped_column(ForeignKey("evaluation_jobs.id", ondelete="CASCADE"))
    attempt_no: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_info: Mapped[dict] = mapped_column(JSONType, default=dict)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    review_reasons: Mapped[list] = mapped_column(JSONType, default=list)
    warnings: Mapped[list] = mapped_column(JSONType, default=list)
    context_pack_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tokens_in: Mapped[int] = mapped_column(BigInteger, default=0)
    tokens_out: Mapped[int] = mapped_column(BigInteger, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    __table_args__ = (
        UniqueConstraint("job_id", "attempt_no", name="uq_attempt_no"),
        CheckConstraint("status IN ('RUNNING','SUCCEEDED','FAILED','NEEDS_REVIEW')", name="ck_attempt_status"),
        Index("ix_attempts_started", "started_at"),
    )


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("res"))
    attempt_id: Mapped[str] = mapped_column(ForeignKey("evaluation_attempts.id", ondelete="CASCADE"))
    criterion_id: Mapped[str] = mapped_column(ForeignKey("rubric_criteria.id"))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_score: Mapped[float] = mapped_column(Float)
    weight: Mapped[float] = mapped_column(Float)
    detail: Mapped[dict] = mapped_column(JSONType, default=dict)
    __table_args__ = (UniqueConstraint("attempt_id", "criterion_id", name="uq_result_attempt_criterion"),)


class HumanReview(Base):
    __tablename__ = "human_reviews"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("rev"))
    attempt_id: Mapped[str] = mapped_column(ForeignKey("evaluation_attempts.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    decision: Mapped[str] = mapped_column(String(20))
    adjusted_scores: Mapped[dict] = mapped_column(JSONType, default=dict)
    final_score: Mapped[float] = mapped_column(Float)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    __table_args__ = (CheckConstraint("decision IN ('approve','override')", name="ck_review_decision"),)


class Outbox(Base):
    __tablename__ = "outbox"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(40))
    aggregate_id: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    __table_args__ = (
        Index(
            "ix_outbox_unpublished",
            "available_at",
            postgresql_where=text("published_at IS NULL"),
            sqlite_where=text("published_at IS NULL"),
        ),
    )


class Embedding(Base):
    __tablename__ = "embeddings"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(String(40))
    chunk_ref: Mapped[str] = mapped_column(String(400))
    text: Mapped[str] = mapped_column(Text)
    vector: Mapped[list] = mapped_column(VectorType)
