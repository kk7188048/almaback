"""Multi-process smoke test: real uvicorn API + real scheduler + real Celery worker, talking over the Redis protocol.

    python scripts/smoke.py            # uses a throw-away SQLite DB, local storage and a fake Redis TCP server

It exercises exactly the production process topology (api / scheduler / worker) without needing Docker.
Pass DATABASE_URL=postgresql+psycopg://... to run the same flow against Postgres.
"""

import ast
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PORT_API, PORT_REDIS = 18080, 16399


def zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("proj/README.md", "# Notes API\n\nSmall REST service. Run: `uvicorn app:app`.\n")
        zf.writestr("proj/app.py", "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get('/ping')\ndef ping():\n    return {'ok': True}\n")
        zf.writestr("proj/test_app.py", "def test_ok():\n    assert True\n")
        zf.writestr("proj/requirements.txt", "fastapi\n")
    return buf.getvalue()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smoke-"))
    env = {
        **os.environ,
        "DATABASE_URL": os.environ.get("DATABASE_URL", f"sqlite:///{tmp}/smoke.db"),
        "USE_PGVECTOR": "false",
        "REDIS_URL": f"redis://127.0.0.1:{PORT_REDIS}/0",
        "STORAGE_BACKEND": "local",
        "LOCAL_STORAGE_DIR": str(tmp / "storage"),
        "PUBLIC_BASE_URL": f"http://127.0.0.1:{PORT_API}",
        "LLM_PROVIDER": "mock",
        "JWT_SECRET": "smoke-secret-smoke-secret-smoke-secret",
        "RELAY_POLL_INTERVAL_S": "0.5",
        "PYTHONPATH": str(ROOT),
    }
    procs: list[subprocess.Popen] = []

    def spawn(name: str, *cmd: str) -> None:
        log = open(tmp / f"{name}.log", "wb")
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))

    try:
        spawn("redis", sys.executable, "-c", f"from fakeredis import TcpFakeServer; s = TcpFakeServer(('127.0.0.1', {PORT_REDIS})); s.serve_forever()")
        time.sleep(1.5)
        subprocess.run([sys.executable, "-m", "app.cli", "create-tables"], cwd=ROOT, env=env, check=True, capture_output=True)
        seeded = subprocess.run([sys.executable, "-m", "app.cli", "seed-demo"], cwd=ROOT, env=env, check=True, capture_output=True, text=True).stdout
        assignment_id = ast.literal_eval(seeded.strip())["assignment_id"]
        spawn("api", sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT_API))
        spawn("scheduler", sys.executable, "-m", "app.scheduler")
        spawn("worker", sys.executable, "-m", "celery", "-A", "app.queueing:celery_app", "worker", "-Q", "evaluations", "-c", "1", "--loglevel=INFO", "--pool=solo", "--without-gossip", "--without-mingle")

        base = f"http://127.0.0.1:{PORT_API}"
        client = httpx.Client(base_url=base, timeout=30)
        for _ in range(60):
            try:
                if client.get("/healthz").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.5)
        else:
            raise RuntimeError("api did not start")

        token = client.post("/api/v1/auth/login", json={"email": "student@example.com", "password": "change-me-please-3"}).json()["access_token"]
        auth = {"Authorization": f"Bearer {token}"}
        data = zip_bytes()
        body = {"assignment_id": assignment_id, "sources": [{"type": "zip", "filename": "project.zip", "size_bytes": len(data)}]}
        created = client.post("/api/v1/submissions", json=body, headers={**auth, "Idempotency-Key": "smoke-key-0001"}).json()
        print("created:", created["job_id"], created["status"])
        up = created["uploads"][0]
        assert client.put(up["url"], content=data).status_code == 204
        print("complete:", client.post(created["complete_url"], json={}, headers=auth).json()["status"])

        seen = []
        deadline = time.time() + 120
        while time.time() < deadline:
            job = client.get(f"/api/v1/jobs/{created['job_id']}", headers=auth).json()
            line = (job["status"], job["current_stage"], round(job["progress"], 2))
            if not seen or seen[-1] != line:
                seen.append(line)
                print("  job:", line)
            if job["status"] in ("SUCCEEDED", "FAILED", "NEEDS_REVIEW"):
                break
            time.sleep(0.5)
        assert job["status"] == "SUCCEEDED", job
        result = client.get(f"/api/v1/submissions/{created['submission_id']}/result", headers=auth).json()
        print("overall:", result["overall_score"], "/ 100")
        for c in result["criteria"]:
            print(f"  {c['criterion']:<24}{c['score']:>6} / {c['max_score']:<5} {c['evidence'][:1]}")
        print("SMOKE TEST PASSED")
        return 0
    except Exception:
        for name in ("api", "scheduler", "worker"):
            p = tmp / f"{name}.log"
            if p.exists():
                print(f"--- {name}.log (tail) ---\n" + p.read_text()[-2500:])
        raise
    finally:
        for p in procs:
            p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
