"""Failure handling: every scenario from the design doc's reliability section, exercised for real."""

import io
import threading
import zipfile
from datetime import timedelta

from sqlalchemy import select, update

from app.config import get_settings
from app.db import session_scope, utcnow
from app.models import EvaluationAttempt, EvaluationJob, Outbox
from app.pipeline.llm.base import LLMRateLimited, LLMRequest, LLMTokenLimit, LLMUnavailable
from app.pipeline.llm.providers import MockProvider
from app.pipeline.runner import RunnerDeps, claim_job, recover_all_stale_jobs, run_job
from tests.conftest import drain, make_zip, no_sleep, sample_zip


class ScriptedProvider(MockProvider):
    """MockProvider whose behaviour a test can intercept; records every call."""

    name = "scripted"

    def __init__(self, behavior=None):
        super().__init__(get_settings())
        self.behavior = behavior
        self.calls: list[LLMRequest] = []
        self.lock = threading.Lock()

    def complete(self, req):
        with self.lock:
            self.calls.append(req)
            n = sum(1 for c in self.calls if c.schema_name == req.schema_name)
        if self.behavior:
            out = self.behavior(req, n)
            if out is not None:
                return out
        return super().complete(req)

    def count(self, schema_name: str) -> int:
        return sum(1 for c in self.calls if c.schema_name == schema_name)


def deps_with(provider, sleeps=None):
    return RunnerDeps(provider=provider, llm_kwargs={"sleep": (sleeps.append if sleeps is not None else no_sleep)})


def make_due():
    with session_scope() as db:
        db.execute(update(Outbox).where(Outbox.published_at.is_(None)).values(available_at=utcnow() - timedelta(seconds=1)))


def job_row(job_id):
    with session_scope() as db:
        j = db.get(EvaluationJob, job_id)
        attempts = db.scalars(select(EvaluationAttempt).where(EvaluationAttempt.job_id == job_id).order_by(EvaluationAttempt.attempt_no)).all()
        return j, attempts


# ---------------------------------------------------------------- LLM failures
def test_llm_rate_limit_is_retried_with_backoff_and_succeeds(world):
    created = world.submit_zip()

    def behavior(req, n):
        if req.schema_name == "file_summaries" and n <= 3:
            raise LLMRateLimited(retry_after=7)

    sleeps: list[float] = []
    provider = ScriptedProvider(behavior)
    assert drain(deps_with(provider, sleeps)) == [(created["job_id"], "SUCCEEDED")]
    assert len(sleeps) == 3 and all(d >= 7 for d in sleeps)  # honours Retry-After
    j, attempts = job_row(created["job_id"])
    assert len(attempts) == 1  # absorbed inside the call, no new attempt needed


def test_llm_outage_retries_attempts_with_backoff_then_fails_cleanly(client, world, settings):
    settings.max_attempts = 3
    created = world.submit_zip()
    provider = ScriptedProvider(lambda req, n: (_ for _ in ()).throw(LLMUnavailable("down")))
    deps = deps_with(provider)

    assert drain(deps) == [(created["job_id"], "QUEUED")]  # attempt 1 failed -> requeued, delayed
    j, attempts = job_row(created["job_id"])
    assert j.status == "QUEUED" and attempts[0].status == "FAILED" and attempts[0].error_code.startswith("llm_")
    with session_scope() as db:  # the retry is parked in the outbox with a future available_at
        pending = db.scalars(select(Outbox).where(Outbox.published_at.is_(None))).all()
        assert len(pending) == 1 and pending[0].available_at > utcnow()
    assert drain(deps) == []  # not due yet: nothing is published

    make_due()
    assert drain(deps) == [(created["job_id"], "QUEUED")]
    make_due()
    assert drain(deps) == [(created["job_id"], "FAILED")]  # third failure exhausts max_attempts
    j, attempts = job_row(created["job_id"])
    assert [a.status for a in attempts] == ["FAILED"] * 3 and [a.attempt_no for a in attempts] == [1, 2, 3]
    assert j.status == "FAILED" and j.error_message and "Traceback" not in j.error_message
    status = client.get(f"/api/v1/jobs/{created['job_id']}", headers=world.h(world.student)).json()
    assert status["status"] == "FAILED" and status["error"]["code"].startswith("llm_")
    assert client.get("/api/v1/admin/metrics", headers=world.h(world.admin)).json()["dead_jobs"] == 1


def test_malformed_output_is_repaired_by_reprompting(world):
    created = world.submit_zip()

    def behavior(req, n):
        if req.schema_name == "criterion_score" and "rejected" not in req.task and req.meta["criterion"] == "Functionality":
            base = MockProvider(get_settings()).complete(req)
            base.data["score"] = 9999  # out of range
            return base

    provider = ScriptedProvider(behavior)
    assert drain(deps_with(provider)) == [(created["job_id"], "SUCCEEDED")]
    repairs = [c for c in provider.calls if c.schema_name == "criterion_score" and "rejected" in c.task]
    assert len(repairs) == 1 and "between 0 and" in repairs[0].task


def test_permanently_invalid_criterion_goes_to_human_review(client, world):
    created = world.submit_zip()

    def behavior(req, n):
        if req.schema_name == "criterion_score" and req.meta["criterion"] == "Documentation":
            base = MockProvider(get_settings()).complete(req)
            base.data["score"] = -5
            return base

    assert drain(deps_with(ScriptedProvider(behavior))) == [(created["job_id"], "NEEDS_REVIEW")]
    staff = client.get(f"/api/v1/submissions/{created['submission_id']}/result", headers=world.h(world.evaluator)).json()
    assert staff["ai_overall_score"] is None and "criterion_failed:Documentation" in staff["review_reasons"]
    # approving an incomplete evaluation is refused; the reviewer must supply the missing score
    r = client.post(f"/api/v1/evaluations/{staff['attempt_id']}/review", json={"decision": "approve"}, headers=world.h(world.evaluator))
    assert r.status_code == 422 and "Documentation" in r.json()["detail"]
    ok = client.post(f"/api/v1/evaluations/{staff['attempt_id']}/review", json={"decision": "override", "adjusted_scores": {"Documentation": 6}, "comment": "scored manually"}, headers=world.h(world.evaluator))
    assert ok.status_code == 201 and ok.json()["final_score"] > 0


def test_token_limit_shrinks_context_and_retries(world):
    created = world.submit_zip()

    def behavior(req, n):
        if req.schema_name == "criterion_score" and n <= 2:
            raise LLMTokenLimit("too big")

    provider = ScriptedProvider(behavior)
    assert drain(deps_with(provider)) == [(created["job_id"], "SUCCEEDED")]
    assert provider.count("criterion_score") >= 8  # 6 criteria + the 2 rejected calls


# ---------------------------------------------------------------- worker crashes / duplicate delivery
class SimulatedCrash(BaseException):
    """Stands in for kill -9: not an Exception, so no handler in the runner can run."""


def test_worker_crash_resumes_from_checkpoints_without_repeating_llm_work(client, world):
    created = world.submit_zip()
    crash_once = {"armed": True}

    def behavior(req, n):
        if req.schema_name == "criterion_score" and n == 2 and crash_once["armed"]:
            crash_once["armed"] = False
            raise SimulatedCrash()

    provider = ScriptedProvider(behavior)
    deps = deps_with(provider)
    from tests.conftest import relay_once

    relay_once()
    import pytest

    with pytest.raises(SimulatedCrash):
        run_job(created["job_id"], deps)
    j, attempts = job_row(created["job_id"])
    assert j.status == "RUNNING" and attempts[0].status == "RUNNING"  # nothing cleaned up: the worker "died"
    calls_before = {k: provider.count(k) for k in ("file_summaries", "understanding", "analysis")}

    # a live lease protects the job from being stolen
    assert recover_all_stale_jobs() == 0
    assert run_job(created["job_id"], deps) == "skipped"

    with session_scope() as db:  # time passes: lease expires
        db.execute(update(EvaluationJob).where(EvaluationJob.id == created["job_id"]).values(lease_expires_at=utcnow() - timedelta(seconds=5)))
    assert recover_all_stale_jobs() == 1
    j, attempts = job_row(created["job_id"])
    assert j.status == "QUEUED" and attempts[0].status == "FAILED" and attempts[0].error_code == "worker_lost"

    # the original delivery is still in the queue (at-least-once) next to the recovery event: one runs, one is a no-op
    outcomes = [o for _, o in drain(deps)]
    assert outcomes.count("SUCCEEDED") == 1 and set(outcomes) <= {"SUCCEEDED", "skipped"}
    j, attempts = job_row(created["job_id"])
    assert [(a.attempt_no, a.status) for a in attempts] == [(1, "FAILED"), (2, "SUCCEEDED")]
    # the expensive stages were NOT repeated
    for k, v in calls_before.items():
        assert provider.count(k) == v, k
    assert client.get(f"/api/v1/jobs/{created['job_id']}", headers=world.h(world.student)).json()["attempt"] == 2


def test_stale_lease_is_also_taken_over_directly_by_claim(world):
    created = world.submit_zip()
    from tests.conftest import relay_once

    relay_once()
    assert claim_job(created["job_id"], "worker-a", get_settings()) is not None
    assert claim_job(created["job_id"], "worker-b", get_settings()) is None  # duplicate delivery while in flight
    with session_scope() as db:
        db.execute(update(EvaluationJob).where(EvaluationJob.id == created["job_id"]).values(lease_expires_at=utcnow() - timedelta(seconds=1)))
    taken = claim_job(created["job_id"], "worker-b", get_settings())
    assert taken is not None and taken.attempt_no == 2
    _, attempts = job_row(created["job_id"])
    assert [a.status for a in attempts] == ["FAILED", "RUNNING"]


def test_concurrent_deliveries_claim_exactly_once(world):
    created = world.submit_zip()
    results: list = []
    barrier = threading.Barrier(4)

    def attempt(i):
        barrier.wait()
        results.append(claim_job(created["job_id"], f"w{i}", get_settings()))

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(r is not None for r in results) == 1
    _, attempts = job_row(created["job_id"])
    assert len(attempts) == 1


def test_terminal_job_redelivered_is_a_noop(world):
    created = world.submit_zip()
    assert drain() == [(created["job_id"], "SUCCEEDED")]
    provider = ScriptedProvider()
    assert run_job(created["job_id"], deps_with(provider)) == "skipped"
    assert provider.calls == []  # no LLM money spent on the duplicate


def test_lost_lease_abandons_without_clobbering_the_new_owner(world):
    created = world.submit_zip()

    def behavior(req, n):
        if req.schema_name == "file_summaries" and n == 1:
            with session_scope() as db:  # another worker took over while we were busy
                db.execute(update(EvaluationJob).where(EvaluationJob.id == created["job_id"]).values(locked_by="someone-else"))

    from tests.conftest import relay_once

    relay_once()
    assert run_job(created["job_id"], deps_with(ScriptedProvider(behavior))) == "lease_lost"
    j, _ = job_row(created["job_id"])
    assert j.status == "RUNNING" and j.locked_by == "someone-else"  # untouched


def test_heartbeat_extends_the_lease_of_a_slow_job(world):
    created = world.submit_zip()
    seen = {}

    def behavior(req, n):
        if req.schema_name == "understanding":
            import time

            with session_scope() as db:
                seen["before"] = db.get(EvaluationJob, created["job_id"]).lease_expires_at
            time.sleep(2.3)  # heartbeat interval is 1s in tests
            with session_scope() as db:
                seen["after"] = db.get(EvaluationJob, created["job_id"]).lease_expires_at

    assert drain(deps_with(ScriptedProvider(behavior))) == [(created["job_id"], "SUCCEEDED")]
    assert seen["after"] > seen["before"]


# ---------------------------------------------------------------- hostile / broken inputs
def final_state(job_id):
    j, attempts = job_row(job_id)
    return j.status, j.error_code, len(attempts)


def test_corrupt_zip_fails_fast_with_a_user_facing_error_and_no_retries(client, world):
    created = world.submit({"project.zip": b"PK\x03\x04" + b"this is not really a zip archive" * 20})
    assert drain() == [(created["job_id"], "FAILED")]
    assert final_state(created["job_id"]) == ("FAILED", "corrupt_archive", 1)  # non-retryable: one attempt
    err = client.get(f"/api/v1/jobs/{created['job_id']}", headers=world.h(world.student)).json()["error"]
    assert "corrupt" in err["message"]


def test_zip_slip_is_rejected(world):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("proj/README.md", "ok")
        zf.writestr("../../evil.txt", "pwned")
    created = world.submit({"project.zip": buf.getvalue()})
    assert drain() == [(created["job_id"], "FAILED")]
    assert final_state(created["job_id"])[:2] == ("FAILED", "unsafe_archive")
    from pathlib import Path
    import tempfile

    assert not (Path(tempfile.gettempdir()) / "evil.txt").exists()


def test_zip_bomb_is_rejected_by_compression_ratio(settings, world):
    settings.zip_max_ratio = 50
    bomb = make_zip({"data.txt": b"\0" * (6 * 1024 * 1024)})  # 6 MB of zeros -> a few KB compressed
    created = world.submit({"project.zip": bomb})
    assert drain() == [(created["job_id"], "FAILED")]
    assert final_state(created["job_id"])[:2] == ("FAILED", "unsafe_archive")


def test_zip_total_size_cap(settings, world):
    settings.zip_max_uncompressed_bytes = 1024
    created = world.submit({"project.zip": sample_zip({"big.txt": "x" * 5000})})
    assert drain() == [(created["job_id"], "FAILED")]
    assert final_state(created["job_id"])[:2] == ("FAILED", "unsafe_archive")


def test_executable_disguised_as_zip_is_rejected_at_upload_completion(client, world):
    body = {"assignment_id": world.assignment_id, "sources": [{"type": "zip", "filename": "project.zip", "size_bytes": 64}]}
    created = client.post("/api/v1/submissions", json=body, headers={**world.h(world.student), "Idempotency-Key": "disguise-key-1"}).json()
    client.put(created["uploads"][0]["url"], content=b"MZ" + b"\0" * 62)
    r = client.post(created["complete_url"], json={"uploads": []}, headers=world.h(world.student))
    assert r.status_code == 422 and "executable" in r.json()["detail"]
    assert client.get(f"/api/v1/jobs/{created['job_id']}", headers=world.h(world.student)).json()["status"] == "AWAITING_UPLOAD"


def test_completing_without_uploading_is_rejected_and_nothing_is_queued(client, world):
    body = {"assignment_id": world.assignment_id, "sources": [{"type": "zip", "filename": "project.zip", "size_bytes": 64}]}
    created = client.post("/api/v1/submissions", json=body, headers={**world.h(world.student), "Idempotency-Key": "noupload-key-1"}).json()
    r = client.post(created["complete_url"], json={"uploads": []}, headers=world.h(world.student))
    assert r.status_code == 422
    assert drain() == []


def test_one_bad_source_does_not_block_the_rest_but_flags_review(client, world):
    garbage_pdf = b"%PDF-1.4\nthis is not a parseable pdf\n"
    created = world.submit({"project.zip": sample_zip(), "report.pdf": garbage_pdf})
    assert drain() == [(created["job_id"], "NEEDS_REVIEW")]
    staff = client.get(f"/api/v1/submissions/{created['submission_id']}/result", headers=world.h(world.evaluator)).json()
    assert "partial_submission" in staff["review_reasons"]
    assert staff["ai_overall_score"] is not None  # the zip was still evaluated


def test_live_url_resolving_to_internal_address_is_blocked_not_fetched(client, world):
    created = world.submit_zip(extra_sources=[{"type": "live_url", "url": "http://localhost:8080/admin"}])
    assert drain() == [(created["job_id"], "SUCCEEDED")]
    staff = client.get(f"/api/v1/submissions/{created['submission_id']}/result", headers=world.h(world.evaluator)).json()
    assert any("URL not allowed" in w for w in staff["warnings"])


def test_secrets_in_student_code_never_reach_the_llm(world):
    leaked = "AWS_KEY = 'AKIA" + "IOSFODNN7EXAMPLE'\nTOKEN = 'ghp_" + "a" * 36 + "'\n"
    created = world.submit_zip(zip_bytes=sample_zip({"app/config.py": leaked}))
    provider = ScriptedProvider()
    assert drain(deps_with(provider)) == [(created["job_id"], "SUCCEEDED")]
    blob = "".join(c.system + c.context + c.task for c in provider.calls)
    assert "AKIA" + "IOSFODNN7EXAMPLE" not in blob and "a" * 36 not in blob


def test_student_text_is_wrapped_in_a_per_job_secret_data_marker(world):
    created = world.submit_zip()
    provider = ScriptedProvider()
    drain(deps_with(provider))
    from app.pipeline.llm.prompts import job_boundary

    marker = job_boundary(created["job_id"], get_settings().jwt_secret)
    scoring = [c for c in provider.calls if c.schema_name == "criterion_score"]
    assert scoring and all(f"<<<{marker}_BEGIN>>>" in c.context and marker in c.system for c in scoring)


# ---------------------------------------------------------------- budgets
def test_daily_budget_exhaustion_defers_without_burning_an_attempt(settings, world):
    created = world.submit_zip()
    settings.global_daily_token_budget = 0
    provider = ScriptedProvider()
    assert drain(deps_with(provider)) == [(created["job_id"], "QUEUED")]
    j, attempts = job_row(created["job_id"])
    assert j.status == "QUEUED" and attempts[0].error_code == "deferred_budget" and provider.calls == []
    with session_scope() as db:
        pending = db.scalars(select(Outbox).where(Outbox.published_at.is_(None))).all()
        assert pending[0].available_at > utcnow() + timedelta(minutes=1)  # parked until tomorrow
    settings.global_daily_token_budget = 10**12
    make_due()
    assert drain(deps_with(ScriptedProvider())) == [(created["job_id"], "SUCCEEDED")]
    _, attempts = job_row(created["job_id"])
    assert [a.error_code for a in attempts] == ["deferred_budget", None]


def test_per_attempt_token_budget_stops_runaway_cost(settings, world):
    created = world.submit_zip()
    settings.attempt_token_budget = 50
    assert drain(deps_with(ScriptedProvider())) == [(created["job_id"], "FAILED")]
    assert final_state(created["job_id"]) == ("FAILED", "token_budget_exceeded", 1)
