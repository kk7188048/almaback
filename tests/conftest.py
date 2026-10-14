import io
import os
import shutil
import tempfile
import zipfile

_TMP = tempfile.mkdtemp(prefix="evaltests-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("USE_PGVECTOR", "false")
os.environ["STORAGE_BACKEND"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = f"{_TMP}/storage"
os.environ["QUEUE_BACKEND"] = "memory"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["JWT_SECRET"] = "test-secret-test-secret-test-secret"
os.environ["RATE_LIMIT_CAPACITY"] = "100000"
os.environ["LOGIN_RATE_CAPACITY"] = "100000"
os.environ["HEARTBEAT_INTERVAL_S"] = "1"
os.environ["LEASE_TTL_S"] = "30"
os.environ["RETRY_BACKOFF_BASE_S"] = "1"

import fakeredis  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import queueing  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import Base, get_engine  # noqa: E402
from app.main import app  # noqa: E402
from app.outbox import relay_once  # noqa: E402
from app.pipeline.runner import RunnerDeps, run_job  # noqa: E402
from app.redis_client import set_redis  # noqa: E402
from app.storage import set_storage  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    set_redis(fakeredis.FakeRedis(decode_responses=True))
    set_storage(None)
    shutil.rmtree(os.environ["LOCAL_STORAGE_DIR"], ignore_errors=True)
    queueing.MEMORY_QUEUE.clear()
    import app.pipeline.runner as runner

    runner._breaker = None
    runner._semaphore = None
    yield


@pytest.fixture
def settings():
    s = get_settings()
    saved = s.model_dump()
    yield s
    for k, v in saved.items():
        setattr(s, k, v)


@pytest.fixture
def client():
    return TestClient(app)


def no_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def deps():
    return RunnerDeps(llm_kwargs={"sleep": no_sleep})


def drain(deps: RunnerDeps | None = None, max_rounds: int = 20) -> list[tuple[str, str]]:
    """Run the outbox relay and process every queued event inline (what scheduler + celery worker do in production)."""
    deps = deps or RunnerDeps(llm_kwargs={"sleep": no_sleep})
    seen: list[tuple[str, str]] = []
    for _ in range(max_rounds):
        relay_once()
        if not queueing.MEMORY_QUEUE:
            break
        batch, queueing.MEMORY_QUEUE[:] = list(queueing.MEMORY_QUEUE), []
        for event_type, payload in batch:
            if event_type == "evaluate_job":
                seen.append((payload["aggregate_id"], run_job(payload["aggregate_id"], deps)))
    return seen


def make_zip(files: dict[str, str | bytes], top: str | None = "proj") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(f"{top}/{name}" if top else name, content)
    return buf.getvalue()


SAMPLE_PROJECT = {
    "README.md": "# Todo API\n\nA small FastAPI todo service.\n\n## Setup\n\n```\npip install -r requirements.txt\nuvicorn app.main:app\n```\n",
    "requirements.txt": "fastapi==0.110.0\nuvicorn\npytest\n",
    "app/__init__.py": "",
    "app/main.py": "from fastapi import FastAPI\nfrom app.service import add_todo\n\napp = FastAPI()\n\n\n@app.post('/todos')\ndef create(title: str):\n    return add_todo(title)\n",
    "app/service.py": "TODOS = []\n\n\ndef add_todo(title: str) -> dict:\n    item = {'id': len(TODOS) + 1, 'title': title}\n    TODOS.append(item)\n    return item\n",
    "tests/test_service.py": "from app.service import add_todo\n\n\ndef test_add():\n    assert add_todo('x')['id'] == 1\n",
    "docs/design.md": "# Design\n\nIn-memory list, no database. Chosen for simplicity.\n",
    ".gitignore": "__pycache__/\n*.pyc\n",
}


def sample_zip(extra: dict[str, str | bytes] | None = None) -> bytes:
    return make_zip({**SAMPLE_PROJECT, **(extra or {})})


class World:
    """Users, a course, an assignment with the example rubric - created through the real CLI helpers."""

    def __init__(self, client: TestClient):
        from app import cli
        from sqlalchemy import select

        from app.db import session_scope
        from app.models import Assignment

        self.client = client
        self.info = cli.seed_demo()
        self.assignment_id = self.info["assignment_id"]
        self.course_id = self.info["course_id"]
        self.student = self.login("student@example.com", "change-me-please-3")
        self.evaluator = self.login("evaluator@example.com", "change-me-please-2")
        self.admin = self.login("admin@example.com", "change-me-please-1")
        # second, unrelated tenant for isolation checks
        cli.create_user("student2@example.com", "change-me-please-4", "student")
        cli.create_user("evaluator2@example.com", "change-me-please-5", "evaluator")
        self.student2 = self.login("student2@example.com", "change-me-please-4")
        self.evaluator2 = self.login("evaluator2@example.com", "change-me-please-5")
        with session_scope() as db:
            self.assignment = db.scalar(select(Assignment).where(Assignment.id == self.assignment_id))

    def login(self, email: str, password: str) -> dict:
        r = self.client.post("/api/v1/auth/login", json={"email": email, "password": password})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}", "_refresh": r.json()["refresh_token"]}

    @staticmethod
    def h(who: dict) -> dict:
        return {k: v for k, v in who.items() if not k.startswith("_")}

    def submit(self, files: dict[str, bytes] | None = None, who: dict | None = None, key: str = "key-0000-0001", extra_sources: list | None = None, **extra) -> dict:
        """Create a submission, upload every file through its presigned URL, and complete the upload."""
        who = who or self.student
        files = files if files is not None else {"project.zip": sample_zip()}
        kinds = {".zip": "zip", ".pdf": "pdf", ".md": "docs", ".txt": "docs", ".mp4": "video"}
        sources = [{"type": kinds["." + name.rsplit(".", 1)[1]], "filename": name, "size_bytes": len(data)} for name, data in files.items()]
        body = {"assignment_id": self.assignment_id, "sources": [*sources, *(extra_sources or [])], **extra}
        r = self.client.post("/api/v1/submissions", json=body, headers={**self.h(who), "Idempotency-Key": key})
        assert r.status_code == 202, r.text
        created = r.json()
        for up in created["uploads"]:
            put = self.client.put(up["url"], content=files[up["filename"]], headers=up["headers"])
            assert put.status_code == 204, put.text
        if created["complete_url"]:
            done = self.client.post(created["complete_url"], json={"uploads": []}, headers=self.h(who))
            assert done.status_code == 200, done.text
            assert done.json()["status"] == "QUEUED"
        return created

    def submit_zip(self, who: dict | None = None, zip_bytes: bytes | None = None, key: str = "key-0000-0001", extra_sources: list | None = None, **extra) -> dict:
        data = zip_bytes if zip_bytes is not None else sample_zip()
        return self.submit({"project.zip": data}, who, key, extra_sources, **extra)


@pytest.fixture
def world(client):
    return World(client)
