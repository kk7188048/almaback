# AI Project Evaluation Platform

Backend for AlmaBetter-style project evaluation: a student submits a project (GitHub repo, ZIP, PDF, Google Drive
folder, video, documentation, live URL); the platform analyses it asynchronously with an LLM pipeline against a
**configurable rubric** and returns a structured, auditable score, with human review for anything the AI is unsure about.

This is the implementation of [`docs/DESIGN.md`](docs/DESIGN.md) (architecture, trade-offs, research citations).
Stack: **Python 3.12 · FastAPI · PostgreSQL (+pgvector) · Redis/Celery · S3 · tree-sitter · Docker + NGINX**; the LLM
layer is provider-agnostic (offline mock, Gemini / Groq free tiers, AWS Bedrock).

```
 client ─HTTPS─► NGINX ─► FastAPI (auth · RBAC · validation · presign)        browser ──direct──► S3
                             │  one DB transaction: submission + job + OUTBOX row
                             ▼
                        PostgreSQL ◄────────── scheduler: outbox relay (1s) + stale-lease reaper (30s)
                             ▲                           │ publish
                             │ checkpoints/results       ▼
                   Celery worker(s) ◄──────────────── Redis queue
                     ingest → context pack → LLM prompt chain → deterministic aggregation → persist
                     (SSRF-safe fetch · hardened ZIP · ClamAV · tree-sitter · static analysis)
                             │
                             ▼
                     LLM provider (mock | gemini | openai_compat | bedrock)
```

## What is implemented

| Area | Where |
|---|---|
| Submission API with idempotency, presigned (multipart) uploads, `AWAITING_UPLOAD → QUEUED` flow | `app/api/submissions.py`, `app/services.py`, `app/storage.py` |
| Transactional outbox + relay, Celery queue | `app/outbox.py`, `app/queueing.py`, `app/scheduler.py` |
| Job runner: atomic claim, lease + heartbeat, append-only attempts, checkpoint/resume, retry/backoff, budget deferral | `app/pipeline/runner.py` |
| Ingest: GitHub (shallow hardened clone, GitHub App tokens), ZIP, PDF/docs, Drive folder, live URL probe, video transcription | `app/pipeline/ingest/` |
| Context pack: filtering, tree-sitter repo map + PageRank, static analysis (ruff, radon, optional semgrep/pip-audit), secret redaction, injection scan, map-reduce summaries, embeddings | `app/pipeline/context/` |
| LLM layer: schemas, spotlighted prompts, retry/circuit-breaker/repair client, 4 providers | `app/pipeline/llm/` |
| Evaluation: understanding → 3 analyses (parallel) → per-criterion scoring with self-consistency → **code-side** weighted aggregation → review routing | `app/pipeline/evaluate.py`, `aggregate.py` |
| Human review (append-only), review queue, student/evaluator result views | `app/api/reviews.py`, `app/services.py` |
| Auth (JWT + revocable rotating refresh tokens, argon2), RBAC + tenant isolation, Redis token-bucket rate limits | `app/security.py`, `app/deps.py`, `app/scoping.py`, `app/ratelimit.py` |
| Signed webhooks, SSE, polling | `app/notify.py`, `app/api/jobs.py` |
| Health/readiness, admin metrics, JSON logs with request/job ids | `app/api/ops.py`, `app/logging_setup.py` |
| Alembic migration, Dockerfile, compose, NGINX, CI + deploy workflows | `migrations/`, `Dockerfile`, `docker-compose.yml`, `nginx/`, `.github/workflows/` |

## Run it

### Tests (no services needed)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                      # 176 tests; SQLite + fakeredis + local storage + mock LLM
python scripts/smoke.py     # real uvicorn + scheduler + Celery worker over the Redis protocol (fake Redis server)
DATABASE_URL=postgresql+psycopg://user@localhost/db USE_PGVECTOR=false pytest   # same suite on real Postgres
```

### Free-tier deployment (one EC2 `t3.micro`, or any Docker host)
```bash
cp .env.example .env            # set POSTGRES_PASSWORD and the three secrets (openssl rand -hex 32)
docker compose up -d --build    # db · redis · migrate · api · worker · scheduler · nginx
docker compose exec api python -m app.cli seed-demo      # demo users, course, the example rubric
```
* **LLM**: `LLM_PROVIDER=mock` works offline. For real scoring on a free tier set `gemini` (Google AI Studio key) or
  `openai_compat` with Groq's base URL; see `.env.example`. Production target is `bedrock` (IAM role, no key).
* **Storage**: `local` (docker volume) by default; `STORAGE_BACKEND=s3` with a bucket (free tier: 5 GB) gives direct
  browser→S3 uploads; run `python -m app.cli s3-configure` once (aborts stale multipart uploads, expiry, CORS).
* Optional: `--profile clamav` (malware scanning, ~1.5 GB RAM), `--profile s3` (MinIO), `INSTALL_FFMPEG=true` build arg
  + `TRANSCRIPTION_*` for video transcripts, `GITHUB_APP_*` for private repos, `GOOGLE_API_KEY` for Drive folders.
* `ENV=prod` makes the app refuse to start with default secrets or the mock LLM.

## API (all under `/api/v1`, Bearer JWT unless noted)

| | |
|---|---|
| `POST /auth/login` · `/auth/refresh` · `/auth/logout` · `GET /me` | rotating single-use refresh tokens, revocable |
| `POST /users` · `/courses` · `/courses/{id}/members` · `/courses/{id}/assignments` | admin (assignments: course evaluator too) |
| `POST /rubrics` · `GET /assignments/{id}/rubric` | versioned; weights must sum to 1; new version becomes active, old results keep theirs |
| `POST /submissions` (`Idempotency-Key` required) | returns `202` + `job_id` + presigned upload instructions; replays return `200` |
| `POST /submissions/{id}/uploads` · `/uploads/complete` | re-issue URLs · verify objects (size, magic bytes) and enqueue |
| `POST /submissions/{id}/evaluate` (`Idempotency-Key`) | re-run; safe to call repeatedly |
| `GET /jobs/{id}` · `/jobs/{id}/events` (SSE) | status, stage, progress, queue position, ETA, error |
| `GET /submissions/{id}/result` | students: final results only; staff: AI draft + confidence + review reasons |
| `GET /reviews/queue` · `POST /evaluations/{attempt_id}/review` | evaluator/admin, course-scoped; reviews are appended, never overwrite the AI attempt |
| `GET /admin/metrics` · `GET /healthz` · `GET /readyz` | queue age, dead jobs, outbox lag, tokens/cost today · liveness · readiness |

## Reliability model (each point has a test in `tests/test_reliability.py`)

* **At-least-once everywhere, idempotent consumers.** Duplicate deliveries, outbox republish and five identical `POST`s
  collapse to one effect (unique constraints + conditional `UPDATE` claims + a partial unique index allowing one active job per submission).
* **Worker crash**: lease expires → reaper/claim marks the attempt `worker_lost`, requeues; the new attempt **resumes from checkpoints**
  (context pack, understanding, analyses, each criterion), so completed LLM calls are not repeated or re-billed.
* **Lost lease** (another worker took over): the old worker abandons without touching state.
* **LLM**: backoff + jitter honouring `Retry-After`, circuit breaker, concurrency cap, JSON-schema + semantic validation with bounded
  repair prompts, automatic context shrinking on token-limit errors, per-attempt token budget, global daily budget (jobs are *deferred*, not failed).
* **Bad input** fails fast with a user-readable reason and no retries (corrupt/unsafe archives, missing repo); transient errors retry with backoff.
  One bad source never blocks the rest: the job is evaluated and routed to human review (`partial_submission`).

## Security model (tests in `tests/test_security.py`, `tests/test_api.py`)

Untrusted code is **never executed**: it is parsed (tree-sitter, `ast`, ruff) in the worker with a minimal environment.
ZIP: zip-slip, symlinks, encrypted/nested archives, entry-count / size / ratio caps measured on real streamed bytes. Uploads: magic-byte
validation, presigned POST with a server-enforced size ceiling (S3), optional ClamAV. URLs: scheme/host allow-lists, resolve-once + IP-pinned
connections, private/link-local/multicast blocked, redirect re-validation, response caps. Secrets in student code are redacted before any
prompt. Prompt injection: per-job random data markers, tool-less grader, heuristic scan → human review, code-side scoring.
Tenant isolation: every read goes through `app/scoping.py` and answers `404`, not `403`.

## Deliberate differences from `docs/DESIGN.md`

1. **`AWAITING_UPLOAD` state + `/uploads/complete`.** The design queued the job at `POST`; with direct-to-S3 uploads that lets a worker start before the bytes exist. Jobs with no uploaded files (GitHub/Drive/URL only) still enqueue immediately.
2. **Presigned POST (not PUT) for single S3 uploads**, because only a POST policy lets S3 itself enforce `content-length-range`. Multipart (>100 MB) uses presigned part URLs and the size is verified at completion.
3. **Idempotency is enforced in Postgres** (`UNIQUE (created_by, idempotency_key)`), not Redis: it is the durable source of truth and avoids a second system that can disagree. Redis holds refresh tokens, rate-limit buckets and the Celery queue.
4. **Job status is read from Postgres** (indexed primary-key lookup) rather than cached in Redis; at this scale a cache adds invalidation risk for no measurable gain.
5. **Checkpoints are keyed by job, not attempt**, so a *new* attempt can resume the previous one's work.
6. **Embeddings use a local feature-hashing embedder** (free, offline, deterministic, lexical). The `HashEmbedder` interface is where a hosted embedding model plugs in. No ANN index: retrieval is scoped to one submission's chunks.
7. **The outbox is relayed by a polling scheduler (≈1 s)**; there is no inline fast-path publish.
8. **Dead-letter queue = `FAILED` jobs after `MAX_ATTEMPTS`** (surfaced as `dead_jobs` in `/admin/metrics`); Redis/Celery has no native DLQ. SQS + a real DLQ is the Phase-5 swap.

## Not built (by design or yet)

SQS queue backend, per-stage worker pools / autoscaling, priority queues, Terraform, OpenTelemetry traces, Email/Slack notifications,
microVM sandbox for *executing* student tests (analysis is static-only), video keyframe analysis (audio transcript only),
batch-API / provisioned-throughput integrations. These are the roadmap's scale-phase items.

## Verification status — read this before relying on it

* **Verified here**: 176 automated tests pass on SQLite *and* on real PostgreSQL 16; Alembic upgrade → `alembic check` (no model drift) → downgrade → upgrade;
  a multi-process smoke test (uvicorn + scheduler + real Celery worker) passes on both databases; crash-resume, concurrent claim, heartbeat, lease-loss,
  zip-slip/bomb, SSRF (incl. DNS-pinning, redirects, multicast), idempotency and RBAC isolation are exercised for real.
* **Verified only at the HTTP/SDK boundary (mocked transports/stubs)**: Gemini, OpenAI-compatible (Groq), Bedrock Converse, GitHub API/App tokens, Google Drive, S3
  presigning (signing is real, the network call is not). Run each once against the real service before trusting it.
* **Not executed in this environment**: `docker build`/`compose` (the Docker daemon was not running), NGINX config, the GitHub Actions workflows, ClamAV, ffmpeg,
  and the **pgvector** column/query path (no pgvector extension locally). CI runs the Postgres+pgvector job to cover the last one.
* Mock-LLM scores are deterministic placeholders; scoring *quality* with a real model has not been evaluated and should be calibrated against human grades.
