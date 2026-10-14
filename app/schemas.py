import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from app.config import get_settings

_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ ()\-]{0,99}$")
_EXT = {
    "zip": {".zip"},
    "pdf": {".pdf"},
    "docs": {".md", ".markdown", ".txt", ".rst", ".pdf", ".docx", ".adoc"},
    "video": {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"},
}
_SIZE_ATTR = {"zip": "max_zip_bytes", "pdf": "max_pdf_bytes", "docs": "max_docs_bytes", "video": "max_video_bytes"}


class GithubSource(BaseModel):
    type: Literal["github"]
    url: str = Field(max_length=300)


class DriveSource(BaseModel):
    type: Literal["drive"]
    url: str = Field(max_length=300)


class LiveUrlSource(BaseModel):
    type: Literal["live_url"]
    url: str = Field(max_length=500)


class UploadSource(BaseModel):
    type: Literal["zip", "pdf", "docs", "video"]
    filename: str
    size_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def check(self):
        if not _FILENAME.match(self.filename) or ".." in self.filename:
            raise ValueError("filename may contain letters, digits, spaces and . _ - ( ) only")
        ext = "." + self.filename.rsplit(".", 1)[-1].lower() if "." in self.filename else ""
        if ext not in _EXT[self.type]:
            raise ValueError(f"a {self.type} upload must have one of these extensions: {sorted(_EXT[self.type])}")
        limit = getattr(get_settings(), _SIZE_ATTR[self.type])
        if self.size_bytes > limit:
            raise ValueError(f"{self.type} uploads are limited to {limit // (1024 * 1024)} MB")
        return self


SourceIn = Annotated[GithubSource | DriveSource | LiveUrlSource | UploadSource, Field(discriminator="type")]


class CreateSubmissionIn(BaseModel):
    assignment_id: str = Field(max_length=40)
    sources: list[SourceIn] = Field(min_length=1, max_length=8)
    webhook_url: str | None = Field(default=None, max_length=1000)

    @field_validator("sources")
    @classmethod
    def limit_sources(cls, v):
        if len(v) > get_settings().max_sources:
            raise ValueError(f"at most {get_settings().max_sources} sources are allowed")
        return v


class UploadInstructionOut(BaseModel):
    filename: str
    method: str
    url: str | None
    fields: dict[str, str] = {}
    headers: dict[str, str] = {}
    expires_in: int
    multipart: dict | None = None


class SubmissionCreatedOut(BaseModel):
    submission_id: str
    job_id: str
    status: str
    status_url: str
    uploads: list[UploadInstructionOut] = []
    complete_url: str | None = None
    idempotent_replay: bool = False


class PartIn(BaseModel):
    part_number: int = Field(ge=1, le=10000)
    etag: str = Field(max_length=200)


class UploadDoneIn(BaseModel):
    filename: str
    parts: list[PartIn] = []


class CompleteUploadsIn(BaseModel):
    uploads: list[UploadDoneIn] = []


class JobOut(BaseModel):
    job_id: str
    submission_id: str
    status: str
    current_stage: str | None
    progress: float
    attempt: int
    queue_position: int | None = None
    estimated_completion: datetime | None = None
    error: dict | None = None
    result_url: str | None = None


class UserCreateIn(BaseModel):
    email: EmailStr
    name: str = Field(default="", max_length=255)
    password: str = Field(min_length=10, max_length=200)
    role: Literal["student", "evaluator", "admin"]


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: str
    name: str
    role: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=200)


class RefreshIn(BaseModel):
    refresh_token: str = Field(max_length=200)


class CourseIn(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=5000)


class MemberIn(BaseModel):
    user_id: str = Field(max_length=40)
    role: Literal["student", "evaluator"]


class AssignmentIn(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=10000)
    opens_at: datetime | None = None
    closes_at: datetime | None = None

    @model_validator(mode="after")
    def order(self):
        if self.opens_at and self.closes_at and self.closes_at <= self.opens_at:
            raise ValueError("closes_at must be after opens_at")
        return self


class CriterionIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=3000)
    weight: float = Field(gt=0, le=1)
    max_score: float = Field(gt=0, le=1000)
    scale_anchors: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def anchors_ok(self):
        for k, v in self.scale_anchors.items():
            try:
                score = float(k)
            except ValueError as exc:
                raise ValueError("scale_anchors keys must be numbers") from exc
            if not 0 <= score <= self.max_score:
                raise ValueError("scale_anchors keys must be between 0 and max_score")
            if len(v) > 600:
                raise ValueError("scale anchor text too long")
        return self


class RubricIn(BaseModel):
    assignment_id: str = Field(max_length=40)
    criteria: list[CriterionIn] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def weights_sum(self):
        if abs(sum(c.weight for c in self.criteria) - 1.0) > 1e-6:
            raise ValueError("criterion weights must sum to 1.0")
        names = [c.name.strip().lower() for c in self.criteria]
        if len(set(names)) != len(names):
            raise ValueError("criterion names must be unique")
        return self


class ReviewIn(BaseModel):
    decision: Literal["approve", "override"]
    adjusted_scores: dict[str, float] = Field(default_factory=dict)  # criterion id or name -> score
    comment: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def need_comment(self):
        if self.decision == "override" and not (self.adjusted_scores and self.comment.strip()):
            raise ValueError("an override needs adjusted_scores and a comment")
        return self
