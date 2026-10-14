"""Submit a project to a running local stack and print the evaluation.

    python scripts/demo_submit.py                          # built-in sample project (ZIP)
    python scripts/demo_submit.py --zip my_project.zip
    python scripts/demo_submit.py --github https://github.com/owner/repo
    python scripts/demo_submit.py --live-url https://example.com
Options can be combined. Defaults to the Docker stack on http://localhost; use --base for scripts/dev.py (http://127.0.0.1:8000)."""

import argparse
import io
import sys
import time
import uuid
import zipfile
from pathlib import Path

import httpx

SAMPLE = {
    "README.md": "# Notes API\n\nA small FastAPI service for notes.\n\n## Run\n`pip install -r requirements.txt && uvicorn app.main:app`\n",
    "requirements.txt": "fastapi\nuvicorn\n",
    "app/main.py": "from fastapi import FastAPI\nfrom app.store import add_note\n\napp = FastAPI()\n\n\n@app.post('/notes')\ndef create(text: str):\n    return add_note(text)\n",
    "app/store.py": "NOTES: list[dict] = []\n\n\ndef add_note(text: str) -> dict:\n    note = {'id': len(NOTES) + 1, 'text': text}\n    NOTES.append(note)\n    return note\n",
    "tests/test_store.py": "from app.store import add_note\n\n\ndef test_add():\n    assert add_note('x')['id'] == 1\n",
}


def sample_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in SAMPLE.items():
            zf.writestr(f"notes-api/{name}", content)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost")
    ap.add_argument("--zip")
    ap.add_argument("--github")
    ap.add_argument("--live-url")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.base, timeout=60)
    tok = c.post("/api/v1/auth/login", json={"email": "student@example.com", "password": "change-me-please-3"}).json()["access_token"]
    auth = {"Authorization": f"Bearer {tok}"}
    course = c.get("/api/v1/courses", headers=auth).json()[0]
    assignment_id = c.get(f"/api/v1/courses/{course['id']}/assignments", headers=auth).json()[0]["id"]

    sources, uploads = [], {}
    if args.zip:
        data = Path(args.zip).read_bytes()
        sources.append({"type": "zip", "filename": Path(args.zip).name, "size_bytes": len(data)})
        uploads[Path(args.zip).name] = data
    if args.github:
        sources.append({"type": "github", "url": args.github})
    if args.live_url:
        sources.append({"type": "live_url", "url": args.live_url})
    if not sources:
        data = sample_zip()
        sources.append({"type": "zip", "filename": "notes-api.zip", "size_bytes": len(data)})
        uploads["notes-api.zip"] = data

    r = c.post("/api/v1/submissions", json={"assignment_id": assignment_id, "sources": sources}, headers={**auth, "Idempotency-Key": uuid.uuid4().hex})
    if r.status_code != 202:
        print("submit failed:", r.status_code, r.text)
        return 1
    created = r.json()
    print(f"submitted  job={created['job_id']}  status={created['status']}")
    for up in created["uploads"]:
        assert c.put(up["url"], content=uploads[up["filename"]], headers=up["headers"]).status_code == 204
        print(f"uploaded   {up['filename']}")
    if created["complete_url"]:
        print("completed ", c.post(created["complete_url"], json={}, headers=auth).json()["status"])

    last = None
    for _ in range(600):
        job = c.get(f"/api/v1/jobs/{created['job_id']}", headers=auth).json()
        cur = (job["status"], job["current_stage"], round(job["progress"], 2))
        if cur != last:
            print("  ...", cur)
            last = cur
        if job["status"] in ("SUCCEEDED", "FAILED", "NEEDS_REVIEW"):
            break
        time.sleep(1)
    if job["status"] == "FAILED":
        print("FAILED:", job["error"])
        return 1
    res = c.get(f"/api/v1/submissions/{created['submission_id']}/result", headers=auth)
    if res.status_code == 409:
        print("Evaluation is awaiting human review (flagged by the AI pipeline). Review it as evaluator@example.com via GET /api/v1/reviews/queue.")
        return 0
    out = res.json()
    print(f"\nOVERALL: {out['overall_score']} / 100   (rubric v{out['rubric_version']})\n")
    for k in out["criteria"]:
        print(f"{k['criterion']:<24}{k['score']:>7} / {k['max_score']:<6}  {k['justification'][:110]}")
    for w in out["warnings"]:
        print("warning:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
