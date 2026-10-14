"""API contract: auth, RBAC / tenant isolation, validation rules, rubric management, rate limiting, ops endpoints."""

import pytest

from tests.conftest import drain, sample_zip


def idem(world, who=None, key="api-key-000001"):
    return {**world.h(who or world.student), "Idempotency-Key": key}


# ---------------------------------------------------------------- auth
def test_login_refresh_rotation_and_logout(client, world):
    assert client.post("/api/v1/auth/login", json={"email": "student@example.com", "password": "wrong-password"}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "whatever-123"}).status_code == 401
    refresh = world.student["_refresh"]
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": refresh}).status_code == 401  # single use
    new = r.json()
    assert client.get("/api/v1/me", headers={"Authorization": f"Bearer {new['access_token']}"}).json()["email"] == "student@example.com"
    assert client.post("/api/v1/auth/logout", json={"refresh_token": new["refresh_token"]}, headers={"Authorization": f"Bearer {new['access_token']}"}).status_code == 204
    assert client.post("/api/v1/auth/refresh", json={"refresh_token": new["refresh_token"]}).status_code == 401  # revoked


def test_unauthenticated_and_tampered_tokens_rejected(client, world):
    assert client.get("/api/v1/me").status_code == 401
    assert client.get("/api/v1/me", headers={"Authorization": "Bearer abc.def.ghi"}).status_code == 401
    token = world.student["Authorization"].split()[1]
    assert client.get("/api/v1/me", headers={"Authorization": f"Bearer {token[:-3]}xyz"}).status_code == 401


def test_expired_token_rejected(client, world, settings):
    settings.access_token_ttl_s = -10
    from app.security import create_access_token

    token = create_access_token("usr_x", "admin")
    assert client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_login_is_rate_limited(client, world, settings):
    settings.login_rate_capacity = 3
    settings.login_rate_per_s = 0.0001
    codes = [client.post("/api/v1/auth/login", json={"email": "student@example.com", "password": "bad-password-1"}).status_code for _ in range(5)]
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]
    assert "retry-after" in client.post("/api/v1/auth/login", json={"email": "student@example.com", "password": "bad-password-1"}).headers


def test_api_is_rate_limited_per_user(client, world, settings):
    settings.rate_limit_capacity = 5
    settings.rate_limit_per_s = 0.0001
    codes = [client.get("/api/v1/me", headers=world.h(world.student)).status_code for _ in range(8)]
    assert codes.count(200) == 5 and codes[-1] == 429
    assert client.get("/api/v1/me", headers=world.h(world.evaluator)).status_code == 200  # separate bucket


# ---------------------------------------------------------------- RBAC and tenant isolation
def test_only_admin_can_manage_users_and_courses(client, world):
    for who in (world.student, world.evaluator):
        assert client.post("/api/v1/courses", json={"title": "x"}, headers=world.h(who)).status_code == 403
        assert client.post("/api/v1/users", json={"email": "a@b.co", "password": "long-enough-pw", "role": "admin"}, headers=world.h(who)).status_code == 403
    assert client.post("/api/v1/courses", json={"title": "New course"}, headers=world.h(world.admin)).status_code == 201


def test_students_cannot_see_each_others_submissions_jobs_or_results(client, world):
    created = world.submit_zip()
    drain()
    sid, jid = created["submission_id"], created["job_id"]
    for who in (world.student2, world.evaluator2):  # other student, evaluator of another course
        h = world.h(who)
        assert client.get(f"/api/v1/submissions/{sid}", headers=h).status_code == 404
        assert client.get(f"/api/v1/jobs/{jid}", headers=h).status_code == 404
        assert client.get(f"/api/v1/jobs/{jid}/events", headers=h).status_code == 404
        assert client.get(f"/api/v1/submissions/{sid}/result", headers=h).status_code == 404
        assert client.post(f"/api/v1/submissions/{sid}/evaluate", headers=idem(world, who, "iso-key-000001")).status_code == 404
    assert client.get("/api/v1/submissions", headers=world.h(world.student2)).json()["items"] == []
    assert len(client.get("/api/v1/submissions", headers=world.h(world.student)).json()["items"]) == 1
    assert len(client.get("/api/v1/submissions", headers=world.h(world.evaluator)).json()["items"]) == 1
    assert client.get("/api/v1/submissions", headers=world.h(world.evaluator2)).json()["items"] == []
    for who in (world.student, world.evaluator, world.admin):
        assert client.get(f"/api/v1/jobs/{jid}", headers=world.h(who)).status_code == 200


def test_unenrolled_student_cannot_submit(client, world):
    body = {"assignment_id": world.assignment_id, "sources": [{"type": "live_url", "url": "https://example.com"}]}
    assert client.post("/api/v1/submissions", json=body, headers=idem(world, world.student2)).status_code == 404
    assert client.post("/api/v1/submissions", json=body, headers=idem(world, world.evaluator)).status_code == 403


def test_evaluator_of_other_course_cannot_manage_rubric_or_assignments(client, world):
    crit = [{"name": "All", "weight": 1.0, "max_score": 10}]
    assert client.post("/api/v1/rubrics", json={"assignment_id": world.assignment_id, "criteria": crit}, headers=world.h(world.evaluator2)).status_code == 404
    assert client.post(f"/api/v1/courses/{world.course_id}/assignments", json={"title": "hax"}, headers=world.h(world.evaluator2)).status_code == 404
    assert client.get(f"/api/v1/assignments/{world.assignment_id}/rubric", headers=world.h(world.student2)).status_code == 404
    assert client.get(f"/api/v1/assignments/{world.assignment_id}/rubric", headers=world.h(world.student)).status_code == 200


def test_metrics_admin_only(client, world):
    assert client.get("/api/v1/admin/metrics", headers=world.h(world.evaluator)).status_code == 403
    m = client.get("/api/v1/admin/metrics", headers=world.h(world.admin)).json()
    assert {"jobs_by_status", "oldest_queued_age_s", "dead_jobs", "outbox_pending", "today"} <= set(m)


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("source,fragment", [
    ({"type": "zip", "filename": "../../etc/passwd.zip", "size_bytes": 10}, "filename"),
    ({"type": "zip", "filename": "project.exe", "size_bytes": 10}, "extension"),
    ({"type": "zip", "filename": "a.zip", "size_bytes": 0}, "greater than 0"),
    ({"type": "zip", "filename": "a.zip", "size_bytes": 10**12}, "limited to"),
    ({"type": "pdf", "filename": "a.zip", "size_bytes": 10}, "extension"),
    ({"type": "video", "filename": "a.mp4", "size_bytes": 10**11}, "limited to"),
    ({"type": "ftp", "url": "ftp://x"}, "type"),
])
def test_source_validation(client, world, source, fragment):
    body = {"assignment_id": world.assignment_id, "sources": [source]}
    r = client.post("/api/v1/submissions", json=body, headers=idem(world))
    assert r.status_code == 422 and fragment in r.text


@pytest.mark.parametrize("source", [
    {"type": "github", "url": "https://gitlab.com/o/r"},
    {"type": "github", "url": "https://github.com.evil.com/o/r"},
    {"type": "github", "url": "http://github.com/o/r"},
    {"type": "github", "url": "https://github.com/onlyowner"},
    {"type": "drive", "url": "https://evil.com/drive/folders/1AbCdEfGhIjKlMn"},
    {"type": "live_url", "url": "http://127.0.0.1:8000"},
    {"type": "live_url", "url": "http://169.254.169.254/latest/meta-data"},
    {"type": "live_url", "url": "file:///etc/passwd"},
    {"type": "live_url", "url": "http://user:pass@example.com"},
])
def test_url_sources_validated_at_the_edge(client, world, source):
    body = {"assignment_id": world.assignment_id, "sources": [source]}
    assert client.post("/api/v1/submissions", json=body, headers=idem(world)).status_code == 422


def test_empty_and_oversized_source_lists_rejected(client, world):
    assert client.post("/api/v1/submissions", json={"assignment_id": world.assignment_id, "sources": []}, headers=idem(world)).status_code == 422
    many = [{"type": "live_url", "url": f"https://example.com/{i}"} for i in range(9)]
    assert client.post("/api/v1/submissions", json={"assignment_id": world.assignment_id, "sources": many}, headers=idem(world)).status_code == 422


def test_valid_url_sources_accepted_without_upload_step(client, world):
    body = {"assignment_id": world.assignment_id, "sources": [
        {"type": "github", "url": "https://github.com/octocat/Hello-World"},
        {"type": "live_url", "url": "https://hello.example.com"},
        {"type": "drive", "url": "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQ"},
    ]}
    r = client.post("/api/v1/submissions", json=body, headers=idem(world))
    assert r.status_code == 202 and r.json()["status"] == "QUEUED" and r.json()["uploads"] == [] and r.json()["complete_url"] is None


def test_rubric_validation_and_versioning(client, world):
    h = world.h(world.evaluator)
    base = {"assignment_id": world.assignment_id}
    bad_sum = {**base, "criteria": [{"name": "A", "weight": 0.5, "max_score": 10}, {"name": "B", "weight": 0.4, "max_score": 10}]}
    assert client.post("/api/v1/rubrics", json=bad_sum, headers=h).status_code == 422
    dup = {**base, "criteria": [{"name": "A", "weight": 0.5, "max_score": 10}, {"name": "a", "weight": 0.5, "max_score": 10}]}
    assert client.post("/api/v1/rubrics", json=dup, headers=h).status_code == 422
    bad_anchor = {**base, "criteria": [{"name": "A", "weight": 1.0, "max_score": 10, "scale_anchors": {"11": "too high"}}]}
    assert client.post("/api/v1/rubrics", json=bad_anchor, headers=h).status_code == 422
    good = {**base, "criteria": [{"name": "Correctness", "weight": 0.6, "max_score": 60, "scale_anchors": {"0": "broken", "60": "flawless"}}, {"name": "Style", "weight": 0.4, "max_score": 40}]}
    r = client.post("/api/v1/rubrics", json=good, headers=h)
    assert r.status_code == 201 and r.json()["version"] == 2 and r.json()["active"]
    active = client.get(f"/api/v1/assignments/{world.assignment_id}/rubric", headers=h).json()
    assert active["version"] == 2 and [c["name"] for c in active["criteria"]] == ["Correctness", "Style"]


def test_new_rubric_is_used_without_engine_changes_and_old_results_keep_their_rubric(client, world):
    first = world.submit_zip(key="rub-key-0000001")
    drain()
    r1 = client.get(f"/api/v1/submissions/{first['submission_id']}/result", headers=world.h(world.student)).json()
    assert len(r1["criteria"]) == 6 and r1["rubric_version"] == 1

    new = {"assignment_id": world.assignment_id, "criteria": [{"name": "Correctness", "weight": 0.7, "max_score": 70}, {"name": "Clarity", "weight": 0.3, "max_score": 30}]}
    assert client.post("/api/v1/rubrics", json=new, headers=world.h(world.evaluator)).status_code == 201
    ev = client.post(f"/api/v1/submissions/{first['submission_id']}/evaluate", headers=idem(world, key="rub-key-0000002"))
    assert ev.status_code == 202
    drain()
    r2 = client.get(f"/api/v1/submissions/{first['submission_id']}/result", headers=world.h(world.student)).json()
    assert r2["rubric_version"] == 2 and [c["criterion"] for c in r2["criteria"]] == ["Correctness", "Clarity"]
    assert 0 <= r2["overall_score"] <= 100


def test_assignment_window_and_missing_rubric(client, world):
    from datetime import timedelta

    from app.db import session_scope, utcnow
    from app.models import Assignment

    with session_scope() as db:
        db.get(Assignment, world.assignment_id).closes_at = utcnow() - timedelta(hours=1)
    body = {"assignment_id": world.assignment_id, "sources": [{"type": "live_url", "url": "https://example.com"}]}
    assert client.post("/api/v1/submissions", json=body, headers=idem(world)).status_code == 409

    new = client.post(f"/api/v1/courses/{world.course_id}/assignments", json={"title": "No rubric yet"}, headers=world.h(world.evaluator)).json()
    body["assignment_id"] = new["id"]
    r = client.post("/api/v1/submissions", json=body, headers=idem(world, key="api-key-000002"))
    assert r.status_code == 409 and "rubric" in r.text


def test_daily_submission_cap(client, world, settings):
    settings.daily_submission_cap = 2
    for i in range(3):
        body = {"assignment_id": world.assignment_id, "sources": [{"type": "live_url", "url": f"https://example.com/{i}"}]}
        r = client.post("/api/v1/submissions", json=body, headers=idem(world, key=f"cap-key-00000{i}"))
        assert r.status_code == (202 if i < 2 else 429)


def test_result_not_ready_returns_conflict_with_status(client, world):
    created = world.submit_zip()
    r = client.get(f"/api/v1/submissions/{created['submission_id']}/result", headers=world.h(world.student))
    assert r.status_code == 409 and r.json()["detail"]["status"] == "QUEUED"
    status = client.get(f"/api/v1/jobs/{created['job_id']}", headers=world.h(world.student))
    assert status.headers["retry-after"] == "5" and status.json()["queue_position"] == 0 and status.json()["estimated_completion"]


def test_sse_stream_reports_status_and_ends_on_terminal(client, world):
    created = world.submit_zip()
    drain()
    with client.stream("GET", f"/api/v1/jobs/{created['job_id']}/events", headers=world.h(world.student)) as r:
        text = "".join(r.iter_text())
    assert "event: status" in text and '"status": "SUCCEEDED"' in text


def test_refresh_upload_urls_and_complete_is_idempotent(client, world):
    body = {"assignment_id": world.assignment_id, "sources": [{"type": "zip", "filename": "p.zip", "size_bytes": 100}]}
    created = client.post("/api/v1/submissions", json=body, headers=idem(world)).json()
    again = client.post("/api/v1/submissions", json=body, headers=idem(world))  # client lost the first response
    assert again.status_code == 200 and again.json()["uploads"] and again.json()["job_id"] == created["job_id"]
    fresh = client.post(f"/api/v1/submissions/{created['submission_id']}/uploads", headers=world.h(world.student)).json()
    assert fresh and client.put(fresh[0]["url"], content=sample_zip()).status_code == 204
    one = client.post(created["complete_url"], json={}, headers=world.h(world.student))
    two = client.post(created["complete_url"], json={}, headers=world.h(world.student))
    assert one.json()["status"] == two.json()["status"] == "QUEUED"
    from sqlalchemy import select

    from app.db import session_scope
    from app.models import Outbox

    with session_scope() as db:
        assert len(db.scalars(select(Outbox)).all()) == 1  # completing twice enqueues once


def test_local_upload_url_is_signed_and_size_capped(client, world):
    body = {"assignment_id": world.assignment_id, "sources": [{"type": "zip", "filename": "p.zip", "size_bytes": 100}]}
    up = client.post("/api/v1/submissions", json=body, headers=idem(world)).json()["uploads"][0]
    assert client.put(up["url"].replace("sig=", "sig=0"), content=b"PK\x03\x04").status_code == 403
    assert client.put(up["url"].replace("key=", "key=x"), content=b"PK\x03\x04").status_code == 403  # key is bound to the signature
    assert client.put(up["url"], content=b"PK\x03\x04").status_code == 204


def test_local_upload_enforces_the_signed_size_ceiling(client, world):
    from app.storage import get_storage

    inst = get_storage().presign_upload("uploads/x/cap.zip", 10, 100, "application/zip")
    assert client.put(inst["url"], content=b"A" * 100).status_code == 204
    assert client.put(inst["url"], content=b"A" * 101).status_code == 413
    assert not get_storage().exists("uploads/x/cap.zip") or get_storage().size("uploads/x/cap.zip") == 100  # the oversized body never replaced it


def test_health_endpoints(client, world):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"database": "ok", "redis": "ok", "storage": "ok", "llm": "configured"}
    from app.redis_client import set_redis

    class Dead:
        def ping(self):
            raise ConnectionError("down")

    set_redis(Dead())
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json()["checks"]["redis"] == "fail"


def test_error_responses_do_not_leak_internals(client, world):
    r = client.get("/api/v1/submissions/does-not-exist", headers=world.h(world.student))
    assert r.status_code == 404 and r.json() == {"detail": "not found"}
    assert client.get("/api/v1/me", headers=world.h(world.student)).headers["x-content-type-options"] == "nosniff"


def test_production_refuses_default_secrets(settings):
    from app.config import assert_production_ready

    with pytest.raises(RuntimeError) as info:
        assert_production_ready(settings.model_copy(update={"jwt_secret": "dev-only-change-me-dev-only-change-me", "llm_provider": "mock"}))
    assert "JWT_SECRET" in str(info.value) and "LLM_PROVIDER=mock" in str(info.value)
    good = settings.model_copy(update={"jwt_secret": "a" * 40, "webhook_signing_secret": "b" * 40, "upload_signing_secret": "c" * 40, "llm_provider": "bedrock"})
    assert_production_ready(good)


def test_cors_only_allows_configured_origins(settings):
    from fastapi.testclient import TestClient

    settings.cors_origins = "https://portal.example.com"
    from app.main import create_app

    c = TestClient(create_app())
    ok = c.options("/api/v1/submissions", headers={"Origin": "https://portal.example.com", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "idempotency-key"})
    assert ok.headers.get("access-control-allow-origin") == "https://portal.example.com"
    bad = c.options("/api/v1/submissions", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
