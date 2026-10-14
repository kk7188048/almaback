"""Job execution: atomic claim, lease with heartbeat, append-only attempts, stage checkpoints, and a guarded
state transition for every outcome (success, review, retry, deferral, failure).
"""

import logging
import os
import shutil
import socket
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app import state
from app.config import Settings, get_settings
from app.db import session_scope, utcnow
from app.logging_setup import job_id_var
from app.models import EvaluationAttempt, EvaluationJob, EvaluationResult, Rubric, Submission
from app.outbox import enqueue_event
from app.pipeline.aggregate import aggregate, review_reasons
from app.pipeline.checkpoints import Checkpoints
from app.pipeline.context.pack import build_context_pack
from app.pipeline.errors import DeferJob, LeaseLost, PipelineError
from app.pipeline.evaluate import Evaluator
from app.pipeline.ingest.run import ingest_submission
from app.pipeline.llm import prompts
from app.pipeline.llm.base import (
    BudgetExceeded, CircuitBreaker, LLMAuthError, LLMClient, LLMError, Provider, RequestPacer, UsageTracker,
)
from app.pipeline.llm.providers import build_provider
from app.pipeline.retrieval import retrieve, store_chunks
from app.storage import Storage, get_storage

log = logging.getLogger(__name__)

DEFERRED = "deferred_budget"
_breaker: CircuitBreaker | None = None
_semaphore: threading.BoundedSemaphore | None = None
_pacer: RequestPacer | None = None
_singleton_lock = threading.Lock()


def _shared_llm_controls(s: Settings) -> tuple[CircuitBreaker, threading.BoundedSemaphore, RequestPacer]:
    global _breaker, _semaphore, _pacer
    with _singleton_lock:
        if _breaker is None:
            _breaker = CircuitBreaker(s.llm_breaker_threshold, s.llm_breaker_cooldown_s)
            _semaphore = threading.BoundedSemaphore(s.llm_max_concurrency)
            _pacer = RequestPacer(s.llm_requests_per_minute)
        assert _semaphore is not None and _pacer is not None
        return _breaker, _semaphore, _pacer


@dataclass
class RunnerDeps:
    storage: Storage | None = None
    provider: Provider | None = None
    transport: object | None = None  # injected httpx transport for ingest HTTP (tests)
    llm_kwargs: dict = field(default_factory=dict)  # e.g. a no-op sleep in tests


@dataclass
class Claim:
    job_id: str
    attempt_id: str
    attempt_no: int
    submission_id: str
    sources: list[dict]
    webhook_url: str | None
    rubric_id: str
    criteria: list[dict]
    worker_id: str


@dataclass
class ExecResult:
    pack: dict
    results: list[dict]
    overall: float | None
    confidence: float | None
    reasons: list[str]


class Heartbeat:
    def __init__(self, job_id: str, worker_id: str, s: Settings):
        self.job_id, self.worker_id, self.s = job_id, worker_id, s
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"hb-{job_id}")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def check(self) -> None:
        if self._lost.is_set():
            raise LeaseLost(self.job_id)

    def _run(self) -> None:
        while not self._stop.wait(self.s.heartbeat_interval_s):
            try:
                with session_scope() as db:
                    n = db.execute(
                        update(EvaluationJob)
                        .where(EvaluationJob.id == self.job_id, EvaluationJob.locked_by == self.worker_id, EvaluationJob.status == state.RUNNING)
                        .values(lease_expires_at=utcnow() + timedelta(seconds=self.s.lease_ttl_s))
                    ).rowcount
                if n != 1:
                    self._lost.set()
                    return
            except Exception:
                log.exception("heartbeat failed")


def count_real_failures(db: Session, job_id: str) -> int:
    return db.scalar(
        select(func.count()).select_from(EvaluationAttempt).where(
            EvaluationAttempt.job_id == job_id, EvaluationAttempt.status == "FAILED", EvaluationAttempt.error_code != DEFERRED
        )
    ) or 0


def recover_stale_job(db: Session, job_id: str, s: Settings, *, republish: bool) -> str | None:
    """A RUNNING job whose lease expired lost its worker. Close its attempt as failed and requeue (or fail) it.
    Returns the new job status, or None if the job was not stale."""
    now = utcnow()
    job = db.get(EvaluationJob, job_id)
    if job is None or job.status != state.RUNNING or job.lease_expires_at is None or job.lease_expires_at >= now:
        return None
    db.execute(
        update(EvaluationAttempt)
        .where(EvaluationAttempt.job_id == job_id, EvaluationAttempt.status == "RUNNING")
        .values(status="FAILED", error_code="worker_lost", error_message="the worker stopped responding", finished_at=now)
    )
    failures = count_real_failures(db, job_id)
    if failures >= s.max_attempts:
        new_status, extra = state.FAILED, {"error_code": "max_attempts_exceeded", "error_message": "the evaluation failed repeatedly; please contact support", "finished_at": now}
    else:
        new_status, extra = state.QUEUED, {}
    n = db.execute(
        update(EvaluationJob)
        .where(EvaluationJob.id == job_id, EvaluationJob.status == state.RUNNING, EvaluationJob.lease_expires_at < now)
        .values(status=new_status, locked_by=None, lease_expires_at=None, current_stage=None, updated_at=now, **extra)
    ).rowcount
    if n != 1:
        return None
    if new_status == state.QUEUED and republish:
        enqueue_event(db, "evaluate_job", job_id)
    if new_status == state.FAILED:
        enqueue_event(db, "notify_job", job_id)
    return new_status


def recover_all_stale_jobs(s: Settings | None = None) -> int:
    s = s or get_settings()
    n = 0
    with session_scope() as db:
        ids = db.scalars(select(EvaluationJob.id).where(EvaluationJob.status == state.RUNNING, EvaluationJob.lease_expires_at < utcnow())).all()
        for job_id in ids:
            if recover_stale_job(db, job_id, s, republish=True):
                n += 1
    return n


def claim_job(job_id: str, worker_id: str, s: Settings) -> Claim | None:
    with session_scope() as db:
        job = db.get(EvaluationJob, job_id)
        if job is None:
            return None
        if job.status == state.RUNNING:
            if recover_stale_job(db, job_id, s, republish=False) is None:
                return None  # genuinely in flight on another worker: this delivery is a duplicate
            db.flush()
            db.refresh(job)
        if job.status != state.QUEUED:
            return None
        now = utcnow()
        claimed = db.execute(
            update(EvaluationJob)
            .where(EvaluationJob.id == job_id, EvaluationJob.status == state.QUEUED)
            .values(status=state.RUNNING, locked_by=worker_id, lease_expires_at=now + timedelta(seconds=s.lease_ttl_s), current_stage="ingesting", progress=0.0, updated_at=now)
        ).rowcount
        if claimed != 1:
            return None
        attempt_no = (db.scalar(select(func.max(EvaluationAttempt.attempt_no)).where(EvaluationAttempt.job_id == job_id)) or 0) + 1
        attempt = EvaluationAttempt(job_id=job_id, attempt_no=attempt_no, status="RUNNING", worker_id=worker_id)
        db.add(attempt)
        sub = db.get(Submission, job.submission_id)
        rubric = db.get(Rubric, job.rubric_id)
        assert sub is not None and rubric is not None
        criteria = [
            {"id": c.id, "name": c.name, "description": c.description, "weight": c.weight, "max_score": c.max_score, "scale_anchors": c.scale_anchors or {}, "position": c.position}
            for c in rubric.criteria
        ]
        db.flush()
        return Claim(job_id, attempt.id, attempt_no, sub.id, list(sub.sources), sub.webhook_url, rubric.id, criteria, worker_id)


def _set_stage(job_id: str, worker_id: str, stage: str, progress: float) -> None:
    with session_scope() as db:
        n = db.execute(
            update(EvaluationJob)
            .where(EvaluationJob.id == job_id, EvaluationJob.locked_by == worker_id, EvaluationJob.status == state.RUNNING)
            .values(current_stage=stage, progress=progress, updated_at=utcnow())
        ).rowcount
    if n != 1:
        raise LeaseLost(job_id)


def _next_midnight_utc() -> datetime:
    tomorrow = (utcnow() + timedelta(days=1)).date()
    return datetime.combine(tomorrow, dtime.min, tzinfo=timezone.utc)


def _guard_daily_budget(s: Settings) -> None:
    start = datetime.combine(utcnow().date(), dtime.min, tzinfo=timezone.utc)
    with session_scope() as db:
        used = db.scalar(select(func.coalesce(func.sum(EvaluationAttempt.tokens_in + EvaluationAttempt.tokens_out), 0)).where(EvaluationAttempt.started_at >= start)) or 0
    if used >= s.global_daily_token_budget:
        raise DeferJob("global daily LLM token budget reached", _next_midnight_utc())


def _execute(claim: Claim, deps: RunnerDeps, hb: Heartbeat, usage: UsageTracker, s: Settings) -> ExecResult:
    storage = deps.storage or get_storage()
    ckpt = Checkpoints(storage, claim.job_id)
    provider = deps.provider or build_provider(s)
    breaker, semaphore, pacer = _shared_llm_controls(s)
    llm = LLMClient(provider, s, usage, breaker=breaker, semaphore=semaphore, pacer=pacer, **deps.llm_kwargs)
    boundary = prompts.job_boundary(claim.job_id, s.jwt_secret)
    _guard_daily_budget(s)

    pack = ckpt.get("context_pack")
    if pack is None:
        _set_stage(claim.job_id, claim.worker_id, "ingesting", 0.05)
        workdir = Path(tempfile.mkdtemp(prefix=f"job-{claim.job_id}-"))
        try:
            ingest = ingest_submission(claim.sources, workdir, storage, s, transport=deps.transport)
            hb.check()
            _set_stage(claim.job_id, claim.worker_id, "analyzing", 0.25)
            pack, chunks = build_context_pack(ingest, s, llm, boundary)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        with session_scope() as db:
            store_chunks(db, claim.submission_id, claim.job_id, chunks)
        ckpt.put("context_pack", pack)
    hb.check()
    _set_stage(claim.job_id, claim.worker_id, "llm_evaluation", 0.45)

    with session_scope() as db:
        retrieved = {c["id"]: retrieve(db, claim.submission_id, f"{c['name']} {c['description']}", s) for c in claim.criteria}
    evaluator = Evaluator(llm, s, ckpt, boundary, pack, claim.criteria, retrieved, hb.check)
    understanding = evaluator.understanding()
    analyses = evaluator.analyses(understanding)
    results = evaluator.score_criteria(understanding, analyses)

    _set_stage(claim.job_id, claim.worker_id, "aggregating", 0.92)
    overall, confidence = aggregate(results)
    return ExecResult(pack, results, overall, confidence, review_reasons(results, overall, confidence, pack, s))


def _persist_success(claim: Claim, res: ExecResult, usage: UsageTracker, provider_name: str, ckpt_key: str) -> str:
    status = state.NEEDS_REVIEW if res.reasons else state.SUCCEEDED
    now = utcnow()
    snap = usage.snapshot()
    with session_scope() as db:
        _set_stage_in_tx(db, claim, "persisting", 0.97)
        att = db.get(EvaluationAttempt, claim.attempt_id)
        assert att is not None
        att.status = status
        att.overall_score, att.confidence = res.overall, res.confidence
        att.review_reasons, att.warnings = res.reasons, res.pack["warnings"]
        att.model_info = {"provider": provider_name, **snap}
        att.tokens_in, att.tokens_out, att.cost_usd = snap["input_tokens"], snap["output_tokens"], snap["cost_usd"]
        att.context_pack_key = ckpt_key
        att.finished_at = now
        for r in res.results:
            db.add(EvaluationResult(
                attempt_id=claim.attempt_id, criterion_id=r["criterion_id"], score=r["score"], max_score=r["max_score"], weight=r["weight"],
                detail={k: r.get(k) for k in ("name", "status", "justification", "evidence", "confidence", "agreement", "samples", "injection_suspected", "error")},
            ))
        n = db.execute(
            update(EvaluationJob)
            .where(EvaluationJob.id == claim.job_id, EvaluationJob.locked_by == claim.worker_id, EvaluationJob.status == state.RUNNING)
            .values(status=status, current_stage=None, progress=1.0, locked_by=None, lease_expires_at=None, error_code=None, error_message=None, finished_at=now, updated_at=now)
        ).rowcount
        if n != 1:
            raise LeaseLost(claim.job_id)
        enqueue_event(db, "notify_job", claim.job_id)
    return status


def _set_stage_in_tx(db: Session, claim: Claim, stage: str, progress: float) -> None:
    n = db.execute(
        update(EvaluationJob)
        .where(EvaluationJob.id == claim.job_id, EvaluationJob.locked_by == claim.worker_id, EvaluationJob.status == state.RUNNING)
        .values(current_stage=stage, progress=progress)
    ).rowcount
    if n != 1:
        raise LeaseLost(claim.job_id)


def _persist_failure(claim: Claim, s: Settings, usage: UsageTracker, *, code: str, message: str, retryable: bool) -> str:
    now = utcnow()
    snap = usage.snapshot()
    with session_scope() as db:
        att = db.get(EvaluationAttempt, claim.attempt_id)
        assert att is not None
        att.status, att.error_code, att.error_message, att.finished_at = "FAILED", code, message, now
        att.tokens_in, att.tokens_out, att.cost_usd, att.model_info = snap["input_tokens"], snap["output_tokens"], snap["cost_usd"], snap
        db.flush()
        failures = count_real_failures(db, claim.job_id)
        guard = (EvaluationJob.id == claim.job_id, EvaluationJob.locked_by == claim.worker_id, EvaluationJob.status == state.RUNNING)
        if retryable and failures < s.max_attempts:
            delay = s.retry_backoff_base_s * 2 ** (failures - 1)
            n = db.execute(update(EvaluationJob).where(*guard).values(status=state.QUEUED, locked_by=None, lease_expires_at=None, current_stage=None, error_code=code, error_message=message, updated_at=now)).rowcount
            if n == 1:
                enqueue_event(db, "evaluate_job", claim.job_id, available_at=now + timedelta(seconds=delay))
            final = state.QUEUED
        else:
            n = db.execute(update(EvaluationJob).where(*guard).values(status=state.FAILED, locked_by=None, lease_expires_at=None, current_stage=None, error_code=code, error_message=message, finished_at=now, updated_at=now)).rowcount
            if n == 1:
                enqueue_event(db, "notify_job", claim.job_id)
            final = state.FAILED
        if n != 1:
            raise LeaseLost(claim.job_id)
    return final


def _persist_deferred(claim: Claim, d: DeferJob) -> str:
    now = utcnow()
    with session_scope() as db:
        att = db.get(EvaluationAttempt, claim.attempt_id)
        assert att is not None
        att.status, att.error_code, att.error_message, att.finished_at = "FAILED", DEFERRED, d.reason, now
        n = db.execute(
            update(EvaluationJob)
            .where(EvaluationJob.id == claim.job_id, EvaluationJob.locked_by == claim.worker_id, EvaluationJob.status == state.RUNNING)
            .values(status=state.QUEUED, locked_by=None, lease_expires_at=None, current_stage=None, error_code=DEFERRED, error_message=d.reason, updated_at=now)
        ).rowcount
        if n != 1:
            raise LeaseLost(claim.job_id)
        enqueue_event(db, "evaluate_job", claim.job_id, available_at=d.until)
    return state.QUEUED


def run_job(job_id: str, deps: RunnerDeps | None = None) -> str:
    """Process one job delivery. Returns: skipped | lease_lost | SUCCEEDED | NEEDS_REVIEW | QUEUED | FAILED."""
    token = job_id_var.set(job_id)
    try:
        return _run_job(job_id, deps or RunnerDeps())
    finally:
        job_id_var.reset(token)


def _run_job(job_id: str, deps: RunnerDeps) -> str:
    s = get_settings()
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
    claim = claim_job(job_id, worker_id, s)
    if claim is None:
        log.info("delivery ignored (job terminal, in flight elsewhere, or unknown)")
        return "skipped"
    log.info("job claimed", extra={"ctx": {"attempt_no": claim.attempt_no, "worker": worker_id}})
    hb = Heartbeat(job_id, worker_id, s)
    hb.start()
    usage = UsageTracker(s, s.attempt_token_budget)
    provider_name = (deps.provider.name if deps.provider else s.llm_provider)
    try:
        res = _execute(claim, deps, hb, usage, s)
        hb.check()
        outcome = _persist_success(claim, res, usage, provider_name, Checkpoints(deps.storage or get_storage(), job_id).key("context_pack"))
        log.info("job finished", extra={"ctx": {"status": outcome, **usage.snapshot()}})
        return outcome
    except LeaseLost:
        log.warning("lease lost, abandoning this delivery without touching state")
        return "lease_lost"
    except DeferJob as deferred:
        return _safe_persist(_persist_deferred, claim, deferred)
    except PipelineError as err:
        log.info("pipeline error", extra={"ctx": {"error_code": err.code, "retryable": err.retryable}})
        return _safe_persist(_persist_failure, claim, s, usage, code=err.code, message=err.message, retryable=err.retryable)
    except BudgetExceeded:
        return _safe_persist(_persist_failure, claim, s, usage, code="token_budget_exceeded", message="this project is too large to evaluate within the token budget", retryable=False)
    except LLMAuthError:
        log.error("LLM credentials rejected")
        return _safe_persist(_persist_failure, claim, s, usage, code="llm_misconfigured", message="the evaluation service is temporarily unavailable", retryable=True)
    except LLMError as llm_err:
        log.warning("LLM failure: %s: %s", type(llm_err).__name__, llm_err)
        return _safe_persist(_persist_failure, claim, s, usage, code=f"llm_{type(llm_err).__name__.lower()}", message="the AI service was unavailable; the evaluation will be retried", retryable=True)
    except Exception:
        log.exception("unexpected error while processing job")
        return _safe_persist(_persist_failure, claim, s, usage, code="internal_error", message="an internal error occurred while evaluating your submission", retryable=True)
    finally:
        hb.stop()


def _safe_persist(fn, *args, **kwargs) -> str:
    try:
        return fn(*args, **kwargs)
    except LeaseLost:
        return "lease_lost"
