"""Run the whole stack locally with no Docker:  python scripts/dev.py
API + scheduler + Celery worker + a Redis-protocol server (fakeredis), SQLite, local storage, mock LLM.
Swagger UI: http://127.0.0.1:8000/docs   |   stop with Ctrl-C.  Data and logs live in ./.devrun
Set LLM_PROVIDER / LLM_API_KEY / LLM_MODEL_* in the environment to use a real model (e.g. Gemini free tier)."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / ".devrun"
PORT, REDIS_PORT = int(os.environ.get("PORT", "8000")), 16379


def main() -> int:
    RUN.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "DATABASE_URL": os.environ.get("DATABASE_URL", f"sqlite:///{RUN}/dev.db"),
        "USE_PGVECTOR": "false",
        "REDIS_URL": f"redis://127.0.0.1:{REDIS_PORT}/0",
        "STORAGE_BACKEND": "local",
        "LOCAL_STORAGE_DIR": str(RUN / "storage"),
        "PUBLIC_BASE_URL": f"http://127.0.0.1:{PORT}",
        "LLM_PROVIDER": os.environ.get("LLM_PROVIDER", "mock"),
        "PYTHONPATH": str(ROOT),
    }
    procs: list[subprocess.Popen] = []

    def spawn(name: str, *cmd: str) -> None:
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(RUN / f"{name}.log", "wb"), stderr=subprocess.STDOUT))

    def stop(*_):
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    spawn("redis", sys.executable, "-c", f"from fakeredis import TcpFakeServer; TcpFakeServer(('127.0.0.1', {REDIS_PORT})).serve_forever()")
    time.sleep(1.2)
    subprocess.run([sys.executable, "-m", "app.cli", "create-tables"], cwd=ROOT, env=env, check=True, capture_output=True)
    ids = subprocess.run([sys.executable, "-m", "app.cli", "seed-demo"], cwd=ROOT, env=env, check=True, capture_output=True, text=True).stdout.strip()
    (RUN / "seed.txt").write_text(ids)
    spawn("api", sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT))
    spawn("scheduler", sys.executable, "-m", "app.scheduler")
    spawn("worker", sys.executable, "-m", "celery", "-A", "app.queueing:celery_app", "worker", "-Q", "evaluations", "-c", "1", "--pool=solo", "--loglevel=INFO", "--without-gossip", "--without-mingle")
    print(f"""
  API + Swagger UI : http://127.0.0.1:{PORT}/docs
  Demo logins      : student@example.com / change-me-please-3
                     evaluator@example.com / change-me-please-2
                     admin@example.com / change-me-please-1
  Seed ids         : {ids}
  LLM provider     : {env['LLM_PROVIDER']}      logs: {RUN}/*.log      Ctrl-C to stop
""", flush=True)
    while True:
        time.sleep(1)
        dead = [p for p in procs if p.poll() is not None]
        if dead:
            print("a process exited; see .devrun/*.log", flush=True)
            stop()


if __name__ == "__main__":
    main()
