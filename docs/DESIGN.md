# AI-Powered Project Evaluation Platform — Backend Design

**Author:** Krishna Kumar
**Assignment:** Design an AI-Powered Project Evaluation Platform (AlmaBetter)
**Date:** 2026-10-14

---

## 0. TL;DR

A student submits a project (GitHub repo / ZIP / PDF / Drive / video / live URL). The platform stores it, runs an **asynchronous, multi-stage LLM evaluation** against a **configurable rubric**, and returns a structured, auditable score with human-review fallback.

**Guiding principles**
1. **Don't over-build for today.** 30 submissions/day ≈ 4.25 worker-hours/day — one worker is 80% idle. The "today" build runs on a **single free-tier box**. I only introduce a component when a concrete problem demands it.
2. **The queue is the backbone.** It converts spiky, deadline-clustered arrivals into a smooth drain rate. We scale workers to meet an SLA, not to match arrivals.
3. **LLM tokens dominate cost** at every scale. Every optimization that matters is an LLM lever (model-tiering, prompt caching, batching), not a hardware lever.
4. **Treat every submission as hostile.** Students upload untrusted code and attacker-controlled URLs. Nothing runs or fetches on a trusted host with credentials.
5. **The rubric is data, not code.** Scoring is deterministic where it can be; the LLM interprets, it does not arbitrate the final number.

**Recommended stack:** Python (FastAPI) · PostgreSQL + pgvector · Redis (Celery) · S3 · tree-sitter · Bedrock/Claude (tiered) · Docker + NGINX on EC2 → ECS Fargate at scale.

> **Free-tier note (per the "build it today for free" constraint):** the entire *initial* system fits inside always-free / 12-month-free tiers — one EC2 `t3.micro`, RDS `db.t3.micro` (or Neon free Postgres), S3 5 GB, SQS 1 M requests/mo, Redis in a container, and a **free-tier LLM API** (e.g. Google Gemini / Groq free tier) for the PoC. The *only* unavoidable metered cost at production scale is LLM tokens. Paid/managed services (ECS Fargate, RDS Proxy, ElastiCache, Bedrock Provisioned Throughput) appear **only in the "at scale" column** and are explicitly deferred.

---

## 1. System Architecture

### 1.1 High-level diagram (today → scale)

```
                                   ┌──────────────────────────────────────────┐
                                   │                CLIENTS                     │
                                   │  Student portal · Evaluator dashboard ·    │
                                   │  Admin · (browser → direct S3 upload)      │
                                   └───────────────────────┬────────────────────┘
                                                           │ HTTPS
                                                           ▼
                                         ┌──────────────────────────────┐
                                         │   EDGE: NGINX (TLS) / ALB     │
                                         │   + WAF + rate limiting       │
                                         └───────────────┬───────────────┘
                                                         ▼
          ┌────────────────────────────── API LAYER (FastAPI, stateless) ──────────────────────────────┐
          │  AuthN/AuthZ (JWT + refresh)  ·  Validation (Pydantic)  ·  Submission svc  ·  Rubric svc     │
          │  Issues S3 presigned URLs · writes DB row + outbox in ONE txn · returns 202 + status URL     │
          └───┬───────────────────────────┬──────────────────────────────┬───────────────────────┬──────┘
              │                           │                              │                       │
              ▼                           ▼                              ▼                       ▼
     ┌─────────────────┐        ┌───────────────────┐          ┌──────────────────┐     ┌──────────────┐
     │  PostgreSQL      │        │  Object Storage    │          │  Redis            │     │  Outbox      │
     │  (RDS / Neon)    │        │  S3 (presigned,    │          │  cache · status · │     │  relay       │
     │  + pgvector      │        │  multipart)        │          │  rate-limit ·     │     │  (polls DB → │
     │  typed + JSONB   │        │  100MB–1GB files   │          │  idempotency keys │     │   enqueue)   │
     └─────────────────┘        └───────────────────┘          └──────────────────┘     └──────┬───────┘
                                                                                                │
                                                                                                ▼
                                                                                   ┌─────────────────────────┐
                                                                                   │   QUEUE                  │
                                                                                   │   Redis/Celery (today)   │
                                                                                   │   → AWS SQS + DLQ (scale)│
                                                                                   │   per-stage + priority   │
                                                                                   └────────────┬─────────────┘
                                                                                                │ at-least-once
                 ┌──────────────────────────────────────────────────────────────────────────────┤
                 ▼ (stage pools scale independently)                                              │
   ┌──────────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────┐
   │  INGEST / EXTRACT WORKER       │   │  ANALYSIS WORKER              │   │  LLM EVALUATION WORKER        │
   │  (CPU/disk-bound)              │   │  (CPU-bound)                  │   │  (I/O-bound, rate-limit gated)│
   │  · fetch repo/zip/pdf/drive    │──▶│  · tree-sitter repo map       │──▶│  · per-criterion LLM calls    │
   │  · zip-bomb/zip-slip guards    │   │  · static analysis (Semgrep,  │   │  · structured JSON outputs    │
   │  · SSRF-safe fetch             │   │    linters, complexity,       │   │  · self-consistency on low-   │
   │  · malware scan (ClamAV)       │   │    dep-audit)                 │   │    confidence criteria        │
   │  · build "context pack"        │   │  · map-reduce summaries       │   │  · deterministic aggregation  │
   │  (optional: microVM sandbox    │   │  · embeddings → pgvector      │   │                               │
   │   if code must be executed)    │   │                               │   │        │  (concurrency capped by      │
   └──────────────┬─────────────────┘   └───────────────┬───────────────┘   │        ▼   provider rate limit)  │
                  │   checkpoint each stage → DB/S3 (keyed by attempt_id)    │   ┌──────────────────────────┐  │
                  └──────────────────────────────────────────────────────── │──▶│  AI / LLM LAYER           │  │
                                                                             │   │  Bedrock / Claude (tiered │  │
                                                                             │   │  Haiku→Sonnet→Opus) +     │  │
                                                                             │   │  prompt cache + Batch API │  │
                                                                             │   └──────────────────────────┘  │
                                                                             └───────────────┬────────────────┘
                                                                                             ▼
                                                                                   persist results → notify
                                                                                   (SSE / webhook / poll)
                                                                                             │
                                                                 ┌───────────────────────────┤
                                                                 ▼                           ▼
                                                      ┌────────────────────┐    ┌─────────────────────────┐
                                                      │  Human review queue │    │  MONITORING / LOGGING    │
                                                      │  (needs_review)     │    │  CloudWatch + OTel traces│
                                                      └────────────────────┘    │  queue age · DLQ · 429 · │
                                                                                 │  cost/eval               │
                                                                                 └─────────────────────────┘

  EXTERNAL INTEGRATIONS: GitHub App (repo clone) · Google Drive API · Email/Slack (notify) · LLM provider
```

### 1.2 Why each component (and why not more)

| Component | Choice | Why | Why not the alternative |
|---|---|---|---|
| **API** | FastAPI (Python) | Async, Pydantic validation doubles as the API schema, auto-OpenAPI. Shares a codebase/libs with the AI workers. | Node/NestJS is fine if the team is JS-first, but the AI ecosystem (tree-sitter, LLM SDKs, Whisper) is Python-native. |
| **Auth** | JWT (short-lived) + revocable refresh token in Redis | Stateless request path; refresh token is the revocation point. | Pure stateless JWT can't be revoked before expiry. |
| **DB** | PostgreSQL + pgvector + JSONB | Relational domain with FK integrity + ACID; JSONB for flexible per-criterion LLM output; pgvector removes a whole separate system. | MongoDB — domain isn't structurally heterogeneous and we need transactions/FKs. |
| **Object storage** | S3, presigned direct upload | API never proxies 100 MB–1 GB bytes; stays stateless and cheap. | Storing blobs in the DB; proxying through the API. |
| **Queue** | Redis/Celery today → SQS at scale | Redis already present → zero new infra today; SQS removes broker ops when workers autoscale. | Kafka (streaming log, overkill for discrete long jobs); RabbitMQ (you run the broker). |
| **Cache/Redis** | job status, rate-limit buckets, idempotency keys | Cheap reads off the DB; one instance covers several needs. | — |
| **Workers** | Python, split per stage | Stages have different resource profiles and scale independently. | One monolithic worker couples a slow LLM stage to fast extraction. |
| **AI layer** | Bedrock/Claude, tiered | Data stays in-account; cross-region throughput + Provisioned Throughput at peak; structured outputs. | All-frontier/no-cache (10–20× cost); single mega-call (unreliable, unauditable). |
| **Monitoring** | CloudWatch + OpenTelemetry | Trace a submission API→queue→worker→LLM; alert on the metrics that predict pain. | — |

---

## 2. Submission Processing Pipeline

### 2.1 Lifecycle

```
POST /submissions ─┐  (SYNC — fast, <300ms)
                   ├─▶ 1. AuthN/AuthZ + validate (Pydantic, size/type caps)
                   ├─▶ 2. Create submission + evaluation_job rows
                   ├─▶ 3a. Sources with no upload (GitHub/Drive/URL): write OUTBOX row in the SAME txn → job is QUEUED
                   ├─▶ 3b. Sources with uploads (ZIP/PDF/video): job starts as AWAITING_UPLOAD
                   └─▶ 4. Return 202 Accepted + { job_id, status_url, presigned upload instructions (if any) }

  [client uploads files DIRECTLY to S3 via presigned URL — bypasses the API entirely]

POST /submissions/{id}/uploads/complete ─▶ verify objects landed (size + magic bytes) → write OUTBOX → QUEUED

Outbox relay ─────▶ 5. Publish job to QUEUE (at-least-once)        (ASYNC from here down)
                         │
Worker (pull) ───────────┼─▶ 6. INGEST: fetch repo/zip/pdf/drive (SSRF-safe), scan, extract
                         │       guards: zip-bomb, zip-slip, malware, size/time caps
                         ├─▶ 7. ANALYSE: tree-sitter repo map, static analysis, summaries, embeddings
                         │       → build "context pack" (persisted artifact)
                         ├─▶ 8. LLM EVALUATE: per-criterion scoring calls (structured JSON)
                         │       self-consistency on low-confidence; prompt-cache the context pack
                         ├─▶ 9. AGGREGATE: deterministic weighted sum (code, not LLM)
                         ├─▶ 10. PERSIST results (new attempt row, append-only)
                         └─▶ 11. NOTIFY: SSE/webhook/poll; route low-confidence → human review

  Each stage checkpoints its output to DB/S3 keyed by job_id → a new retry attempt inherits prior checkpoints and resumes, never re-runs completed LLM calls.
```

### 2.2 Sync vs async

- **Synchronous (request path):** auth, validation, create records, write outbox, issue presigned URL, return `202`. Target < 300 ms. Nothing slow or failure-prone happens inline.
- **Asynchronous (everything else):** fetch, extract, analyse, LLM calls, aggregate, persist, notify. These are minutes-long, failure-prone, and cost money — they belong behind the queue.

### 2.3 Answers to the required questions

- **Where are queues required?** Between the API and the long-running work (ingest→analyse→evaluate). At scale, **per-stage queues** so a slow LLM stage doesn't block fast extraction, plus a **priority queue** for instructor re-runs.
- **How do workers communicate?** They don't talk to each other directly. They coordinate through the queue (hand-off messages) and shared state in Postgres/S3. A stage completes, checkpoints its output, and enqueues the next stage.
- **How is status tracked?** A state machine on `evaluation_job`: `QUEUED → RUNNING → {SUCCEEDED | FAILED | NEEDS_REVIEW}` plus a `current_stage` sub-field. Clients **poll** `GET /jobs/:id` with backoff; UI gets **SSE**; server consumers get **signed webhooks**.
- **Worker crashes mid-job?** Queues are at-least-once, so the message is redelivered after the visibility timeout / lock expiry. Because each stage is checkpointed, the retry resumes from the last completed stage — it does **not** repeat expensive LLM calls.
- **LLM API fails?** Retry with exponential backoff honoring `Retry-After` (see §6); on repeated failure the attempt is marked failed and the job moves to the DLQ / `NEEDS_REVIEW`.
- **Same submission processed twice?** Idempotency key + unique DB constraint make duplicates a no-op (see §6.3).

---

## 3. API Design

All endpoints require a Bearer JWT unless noted. All list endpoints are paginated and **scoped to the caller's tenant** (student sees own; evaluator sees assigned courses; admin sees all).

### 3.1 Create submission

```
POST /api/v1/submissions
Headers: Authorization: Bearer <jwt>
         Idempotency-Key: <uuid>            ← required; dedupes retries
Body:
{
  "assignment_id": "asg_123",
  "sources": [
    { "type": "github", "url": "https://github.com/user/repo" },
    { "type": "zip",    "filename": "project.zip", "size_bytes": 10485760 },
    { "type": "live_url","url": "https://myapp.example.com" }
  ]
}

202 Response:
{
  "submission_id": "sub_456",
  "job_id": "job_789",
  "status": "AWAITING_UPLOAD",          ← QUEUED immediately if no uploads needed (GitHub/Drive/URL only)
  "status_url": "/api/v1/jobs/job_789",
  "uploads": [
    { "filename": "project.zip",
      "method": "POST",                 ← presigned POST with content-length-range policy (S3 enforces the size cap)
      "url": "https://s3...",
      "fields": { ... },                ← signed form fields
      "expires_in": 900 }
  ],
  "complete_url": "/api/v1/submissions/sub_456/uploads/complete"
}
```
**Upload strategy:** single-file uploads use **presigned POST** with a `content-length-range` condition so S3 itself rejects oversized payloads (a presigned PUT cannot enforce this). Files >100 MB use **presigned multipart part URLs**; the size is verified server-side at completion. Cloudflare R2 and other S3-compatibles that lack POST policies fall back to presigned PUT with a signed `Content-Length` header.

After the client uploads, it calls `POST /submissions/{id}/uploads/complete` — the server verifies each object's size and magic bytes, then writes the outbox row and transitions the job to `QUEUED`.

**Validation:** assignment exists and is open; caller enrolled in the course; ≤ N sources; per-type size caps; URL scheme/host allowlist for `github`/`live_url` (SSRF). **Auth:** student role, enrolled.

### 3.2 Get job status (poll)

```
GET /api/v1/jobs/{job_id}
Response:
{
  "job_id": "job_789",
  "status": "RUNNING",
  "current_stage": "llm_evaluation",
  "progress": 0.6,
  "attempt": 1,
  "estimated_completion": "2026-10-14T10:32:00Z",
  "result_url": null
}
Headers: Retry-After: 5
```
**Auth:** owner student, assigned evaluator, or admin.

### 3.3 Get evaluation result

```
GET /api/v1/submissions/{submission_id}/result
Response:
{
  "submission_id": "sub_456",
  "rubric_id": "rub_v3",
  "overall_score": 78.5,
  "max_score": 100,
  "status": "SUCCEEDED",
  "confidence": 0.86,
  "criteria": [
    { "criterion": "Functionality", "weight": 0.25, "score": 20, "max": 25,
      "justification": "...", "evidence": ["src/app.py:L40-80", "README.md"],
      "confidence": 0.9 },
    { "criterion": "Code Quality", "weight": 0.20, "score": 15, "max": 20, "...": "..." }
  ],
  "human_review": { "required": false, "reviewed_by": null, "adjusted_score": null }
}
```
**Auth:** owner/evaluator/admin. Students may be shown a redacted view (no internal confidence) per course policy.

### 3.4 Rubric management (configurable, not hard-coded)

```
POST /api/v1/rubrics                      ← create rubric version (evaluator/admin)
Body: { "assignment_id": "...", "criteria": [
          { "name":"Functionality","weight":0.25,"max_score":25,
            "scale_anchors":{ "0":"...", "13":"...", "25":"..." } }, ... ] }
Validation: weights sum to 1.0 (±ε); max_scores > 0; name unique per rubric.
GET  /api/v1/assignments/{id}/rubric      ← active rubric used by the engine
```

### 3.5 Human review

```
POST /api/v1/evaluations/{attempt_id}/review     ← evaluator only, course-scoped
Body: { "adjusted_scores": { "Functionality": 22 }, "comment": "...", "decision": "approve|override" }
Effect: appends a human_review row; never mutates the AI attempt (audit trail preserved).
```

### 3.6 Auth

```
POST /api/v1/auth/login     → { access_token (15m), refresh_token (opaque, revocable) }
POST /api/v1/auth/refresh   → new access_token
POST /api/v1/auth/logout    → revokes refresh_token in Redis
```

---

## 4. Database Design

### 4.1 SQL vs NoSQL — **PostgreSQL**

The domain is strongly relational with hard integrity rules: a rubric's weights must sum correctly, a result must reference a real criterion, a job must reference a real submission. Postgres gives FK enforcement, multi-table joins, and **ACID transactions** to prevent inconsistent states. Flexible per-criterion LLM output lives in a **JSONB** column — so we get relational integrity *and* schema flexibility without adding MongoDB. 30→10k/day is trivial for one primary + read replicas.

### 4.2 ER diagram

```
┌──────────┐        ┌──────────┐        ┌──────────────┐        ┌──────────┐
│  users   │        │ courses  │        │ assignments  │        │ rubrics  │
│──────────│        │──────────│        │──────────────│        │──────────│
│ id (PK)  │   ┌───<│ id (PK)  │>──┐    │ id (PK)      │>──┐    │ id (PK)  │
│ email    │   │    │ title    │   └───<│ course_id FK │   └──<│ assign_id│
│ role     │   │    │ ...      │        │ title        │       │ version  │
│ ...      │   │    └──────────┘        │ opens/closes │       │ active   │
└────┬─────┘   │                        └──────┬───────┘       └────┬─────┘
     │         │                               │                    │
     │ (enroll/assign many-to-many)            │                    │
     │   ┌──────────────────┐                  │             ┌──────┴────────────┐
     └──<│ course_members   │>─────────────────┘             │ rubric_criteria   │
         │ user_id, course_id, role            │             │───────────────────│
         └──────────────────┘                  │             │ id (PK)           │
                                                │             │ rubric_id FK      │
┌──────────────┐                                │             │ name              │
│ submissions  │                                │             │ weight            │
│──────────────│                                │             │ max_score         │
│ id (PK)      │>───────────────────────────────┘             │ scale_anchors JSON│
│ student_id FK│                                               └──────┬────────────┘
│ assign_id FK │                                                      │
│ sources JSONB│         ┌────────────────────┐                       │
│ s3_prefix    │        <│ evaluation_jobs     │                       │
│ status       │────┐    │────────────────────│                       │
└──────────────┘    └───<│ id (PK)             │                       │
                         │ submission_id FK     │                       │
                         │ rubric_id FK ────────┼───────────────────────┘
                         │ status (state mach.) │
                         │ current_stage        │
                         │ idempotency_key UNQ  │
                         └──────────┬───────────┘
                                    │ 1..N (append-only history)
                         ┌──────────┴────────────┐
                         │ evaluation_attempts    │
                         │────────────────────────│
                         │ id (PK)                 │
                         │ job_id FK               │
                         │ attempt_no              │
                         │ status, model_versions  │
                         │ overall_score           │
                         │ context_pack_s3         │
                         │ started/finished_at     │
                         └──────────┬──────────────┘
                                    │ 1..N
                      ┌─────────────┴──────────────┐          ┌───────────────────┐
                      │ evaluation_results          │          │ human_reviews      │
                      │─────────────────────────────│          │────────────────────│
                      │ id (PK)                      │          │ id (PK)            │
                      │ attempt_id FK                │          │ attempt_id FK      │
                      │ criterion_id FK              │          │ reviewer_id FK     │
                      │ score, max, weight           │          │ adjusted_scores JSB│
                      │ detail JSONB (justification, │          │ comment, decision  │
                      │   evidence[], tokens, conf.) │          │ created_at         │
                      └──────────────────────────────┘          └───────────────────┘

      outbox(id, aggregate_id, payload JSONB, published_at NULL)   ← dual-write safety
      embeddings(id, submission_id, chunk_ref, vector pgvector)    ← per-criterion retrieval
```

### 4.3 Indexes

- `evaluation_jobs (status, created_at)` — worker/dashboard scans.
- **Unique partial** `evaluation_jobs (submission_id) WHERE status IN ('QUEUED','RUNNING')` — blocks duplicate active jobs.
- `evaluation_jobs (idempotency_key)` UNIQUE — dedup.
- `evaluation_attempts (job_id, attempt_no)` — history lookup.
- `submissions (student_id)`, `submissions (assignment_id)` — scoping.
- `outbox (published_at) WHERE published_at IS NULL` — relay poll.
- GIN on `evaluation_results.detail` (JSONB) and `submissions.sources`.
- IVFFlat/HNSW on `embeddings.vector`.

### 4.4 History & consistency

- **History:** retries create a **new `evaluation_attempt`**; existing attempts are never mutated → immutable audit trail. Human reviews are appended, never overwriting the AI attempt.
- **Consistency:** state transitions + outbox insert happen in **one transaction**; the status machine rejects illegal transitions; `SELECT ... FOR UPDATE` (or a version column) guards concurrent job updates; the unique partial index makes a double-create physically impossible.

---

## 5. AI / LLM Architecture

> This is the weighted section. The core idea: **the project is a filesystem, not a prompt.** We never send raw files to a grader.

### 5.1 Project understanding — build a "context pack"

```
raw submission (≤1GB)
   │
   ├─ 1. EXTRACT + FILTER (deterministic, no LLM)
   │     exclude: .git, node_modules, vendor, dist/build, lockfiles, minified JS,
   │              binaries, images (except thumbs), .gitignore matches → ~MBs of signal
   │     detect: languages, LOC, project type (package.json / requirements.txt / Dockerfile)
   │
   ├─ 2. REPO MAP (tree-sitter)  — parse symbols (defs/refs), rank by personalized PageRank,
   │     render signatures (bodies elided) to a fixed token budget → architectural shape, cheap
   │
   ├─ 3. MAP-REDUCE SUMMARIES    — summarize per-file → per-directory → subsystem tree
   │     (handles "50 docs" without blowing context)
   │
   ├─ 4. MULTI-MODAL             — images → vision captions; video → transcript (Whisper) + keyframes
   │
   ├─ 5. DETERMINISTIC SIGNALS   — linters, Semgrep, complexity, dependency audit, test results
   │     (facts the LLM interprets, not computes)
   │
   └─▶ CONTEXT PACK (persisted JSON/markdown): metadata · repo map · summaries · README verbatim ·
       signals · embeddings refs.  Cacheable, reproducible, cheap to re-read across all scoring calls.
```

**Context-window safety:** the pack is budgeted to a fixed token size; raw files are never the grader's input. For a specific criterion, a "deep-dive" step retrieves the most relevant 5–10 chunks via **pgvector**.

### 5.2 RAG vs long-context

For **holistic grading**, a **curated context pack beats RAG** — grading needs breadth (architecture, docs-vs-code consistency) that retrieval fragments. Embeddings are used **selectively**: dedupe near-identical files, per-criterion chunk retrieval, doc semantic search. The **tree-sitter repo map is the primary code-context mechanism**; vectors are a secondary lookup.

### 5.3 Evaluation — prompt chaining, not a free-roaming agent

```
context pack
   │
   ├─▶ Project Understanding (1 call, summary of what the project is)
   │
   ├─▶ Specialized analysis (parallel):  Architecture  |  Code Quality  |  Documentation
   │
   ├─▶ Rubric Evaluation — ONE scoring call per criterion (rubric loaded from DB):
   │       inject criterion description + 0..N scale anchors + 1–2 few-shot exemplars
   │       → structured JSON { score, max, justification, evidence[] }
   │       self-consistency: sample 3–5× on low-confidence/borderline, take median, emit agreement
   │
   ├─▶ Score Aggregation — DETERMINISTIC CODE: weighted sum from DB weights (never the LLM)
   │
   └─▶ Final Evaluation — assemble report; confidence low OR extreme score → NEEDS_REVIEW (human)
```

**Why this shape:** single mega-call is unreliable and unauditable; a free-roaming agent is unnecessary risk. Prompt chaining trades a little latency for accuracy and gives a per-criterion audit trail. **The rubric is data** — adding a course's custom rubric requires zero engine changes.

### 5.4 Structured outputs & validation

Schema-constrained JSON (tool-use / structured outputs; GA on Bedrock since Feb 2026) for every scoring call. **Schema ≠ semantics** — still validate score-in-range, evidence refs exist, justification non-empty; on failure do a bounded **repair-retry** then fall back.

### 5.5 Reliability of the judgement

- **Deterministic signals feed the LLM** (don't ask it to compute lint counts or coverage).
- **LLM-as-judge bias mitigations:** low temperature; self-consistency on borderline; mandatory evidence citations; optional cross-family ensemble for high-stakes; periodic calibration against human scores (position/verbosity/self-preference bias are documented pitfalls).
- **Human-in-the-loop:** low confidence, extreme scores, or detected injection → `NEEDS_REVIEW`.

### 5.6 Cost

Model-tiering (**Haiku** for extraction/summaries → **Sonnet** for heavy analysis → **Opus** only for appeals) + **prompt-cache** the context pack across the ~8–12 per-criterion calls + **Batch API** (−50%) for non-urgent runs → **≈ $0.15–$0.60 per submission**. Naive all-frontier/no-cache is 10–20× that. *(Free-tier PoC: swap in Gemini/Groq free-tier models behind the same provider abstraction.)*

---

## 6. Reliability & Failure Handling

### 6.1 LLM failures

| Failure | Handling |
|---|---|
| Timeout | Per-call timeout; retry w/ exponential backoff + jitter; cap attempts. |
| Rate limit (429) | Honor `Retry-After`; token-bucket/semaphore caps concurrency to the provider quota so we don't self-DOS; spill non-urgent to Batch API. |
| Unavailable | Circuit breaker; fail the attempt → DLQ; optional provider failover via the abstraction layer. |
| Malformed response | Schema validation → bounded repair-retry → fall back to `NEEDS_REVIEW`. |
| Token limit exceeded | Context pack is pre-budgeted; if still over, degrade to summaries / fewer chunks for that call. |

### 6.2 Processing failures

| Failure | Handling |
|---|---|
| Corrupt ZIP | Validated on extract; attempt fails with a clear user-facing error, not a crash. |
| Invalid GitHub repo | Clone pre-check (exists, size, reachable); fail fast with reason. |
| Large repo (1 GB+) | Depth-1 clone, byte cap, timeout; extraction filter discards non-signal; reject over hard cap. |
| Unsupported file | Skip with a logged warning; never block the whole submission. |
| Worker crash | At-least-once redelivery + **stage checkpointing** (keyed by job_id, so a new attempt inherits completed stages from the prior attempt) → resume, don't re-run LLM. |
| Network failure | Retries with backoff; idempotent fetch steps. |

### 6.3 Idempotency (POST /evaluate called 5×)

```
Client sends Idempotency-Key (one per logical action, reused on every retry).
Server: INSERT ... ON CONFLICT DO NOTHING on (idempotency_key).
   • row created  → new job
   • conflict, job in-flight   → 409 (or return existing job_id)
   • conflict, job completed   → replay the stored response
Plus a UNIQUE partial index on (submission_id) WHERE status active → concurrent retries can't both create.
```
Queue redelivery and outbox republish are made safe the same way: every consumer is idempotent on a stable event ID.

### 6.4 Dual-write safety (transactional outbox)

Writing the DB row and enqueuing are two systems. We write the queue message into an **`outbox` table in the same transaction** as the job row; a relay publishes committed rows to the queue and marks them sent. Guarantees at-least-once (not exactly-once) → consumers stay idempotent.

---

## 7. Security

Students upload **untrusted code and attacker-controlled URLs.** Core rule: *nothing runs or fetches on a trusted host with ambient network access or credentials.*

| Risk | Mitigation |
|---|---|
| **AuthN** | Short-lived JWT + revocable server-side refresh tokens. |
| **RBAC / isolation** | Roles student/evaluator/admin; **resource-level authz** scoped by token-derived IDs (prevents IDOR / OWASP A01); evaluators scoped to assigned courses; scoping in a shared data-access layer so no endpoint can forget it. |
| **File upload** | Validate by **magic bytes** (not extension); hard size caps at the edge; **ClamAV** scan; private SSE-KMS bucket; **never execute uploads on app hosts**. |
| **ZIP bombs** | Cap per-entry + cumulative uncompressed size, compression ratio, entry count, nesting depth; count actual decompressed bytes while streaming. |
| **Path traversal (zip-slip)** | Canonicalize each entry's resolved path, verify it stays inside the extraction root; reject `../`/absolute/symlink entries (stdlib does NOT check this). |
| **Running untrusted code** | **Default: don't.** Static analysis covers most of the rubric. If tests/live-probe are required: **gVisor/Firecracker microVM**, non-root, read-only FS, egress off/proxied, cgroup limits, ephemeral, isolated fleet with no platform creds. |
| **SSRF** (GitHub/Drive/**live-app URL**) | Resolve-then-pin-IP + re-check at connect (defeats DNS rebinding); block loopback/private/link-local ranges; reject non-HTTP schemes; **enforce IMDSv2**; fetchers behind a default-deny egress proxy. |
| **GitHub access** | `--depth 1` + size cap + timeout; **GitHub App per-installation tokens** (short-lived, scoped, revocable) over student PATs. |
| **Secrets/API keys** | AWS Secrets Manager/SSM via IAM role; never baked into images; scan uploads for leaked secrets (gitleaks/trufflehog). |
| **LLM prompt injection** | Data-marking/spotlighting of submission content; the grader is **tool-less and output-only** so injection can't act; code (not the model) writes the authoritative score; flag suspected injection → human review. |
| **Rate limiting / cost abuse** | Per-user submission caps; token-bucket API limits; **LLM spend budgets + global daily circuit-breaker + idempotency** so a submission can't be re-run/re-charged. |

---

## 8. Scaling: 30/day → 10,000/day

**Thesis: don't add Kubernetes everywhere — each layer evolves for a specific, measured reason. The queue absorbs the deadline spike (500–2,000/hour); we scale workers to drain within an SLA.**

| Layer | Today (30/day) | At scale (10k/day) | Trigger to evolve |
|---|---|---|---|
| **API** | 1–2 stateless instances | 4–8 behind ALB, target-track on request count | Request latency / CPU. |
| **Workers** | 1–2 containers, all stages | **Separate pools per stage**, autoscale on **backlog-per-task** (`msgs_visible ÷ running_tasks`), **scale-in protection** so a 10-min job isn't killed at min 9 | Queue age exceeds SLA. |
| **LLM concurrency** | small semaphore | semaphore keyed to provider **RPM/TPM**; more workers past the ceiling just yield 429s | Throttle rate rises. |
| **Queue** | Redis/Celery | **SQS** standard + DLQ; per-stage + priority queues | Broker ops / worker autoscaling. |
| **DB** | single instance | ① **connection pooling** (RDS Proxy/PgBouncer — workers exhaust connections before CPU) → ② read replicas for dashboards → ③ partition history tables by month | In that order, each when metered. |
| **Object storage** | S3 (5 GB free tier) | S3 + lifecycle to IA/Glacier; abort-incomplete-multipart rule | Storage cost (1 TB/day at scale). |
| **LLM throughput** | one key | **prompt caching** (~−90% on shared context), **Batch API** (−50%, outside rate limits), **Bedrock cross-region + Provisioned Throughput** | Throttling / cost. |
| **Backpressure** | n/a | queue is the buffer; cap at rate-limit ceiling; honest ETAs from depth÷drain; spill to Batch; shed non-urgent | Backlog > SLA. |
| **Caching** | Redis | + result cache by content hash (identical submission → prior result) | Duplicate load. |
| **Monitoring** | logs | **oldest-message age** (true latency), backlog-per-task, per-stage latency, 429 rate, **cost/eval**, DLQ size | — |

---

## 9. Deployment Architecture

### 9.1 Topology

```
                         Internet
                            │
                      ┌─────▼─────┐
                      │ ALB + WAF │  (NGINX reverse-proxy/TLS on the EC2 tier today)
                      └─────┬─────┘
                ┌───────────▼───────────┐
                │  API containers (ECS)  │
                └───┬──────┬──────┬──────┘
          ┌─────────┘      │      └──────────┐
          ▼                ▼                 ▼
   RDS Postgres     ElastiCache Redis       S3 (presigned)
   (Multi-AZ,         (status/cache/          │
    +replica,          rate-limit)            │
    via RDS Proxy)                            │
          ▲                                   │
          │                 ┌─────────────────┘
          │                 ▼
          │            SQS (per-stage + DLQ)
          │                 │
          │   ┌─────────────┼──────────────┐
          │   ▼             ▼              ▼
          │ Worker pool  Worker pool   Worker pool    (ECS Fargate + Spot, autoscaled on backlog)
          │ (ingest)     (analysis)    (LLM eval)
          │                                 │
          └──────── results ◀───────────────┤
                                            ▼
                                   Bedrock / Claude (concurrency-capped)
```

### 9.2 EC2+Docker+NGINX (today) vs ECS Fargate (scale)

- **Today / free-tier:** Dockerized API + worker + NGINX (TLS reverse proxy) via `docker compose` on **one EC2 `t3.micro`** (free tier). Redis + Postgres can run as containers or RDS free tier. Simplest possible, matches the JD's EC2/Docker/NGINX expectation.
- **At scale:** **ECS Fargate + Fargate Spot** for workers — bursty sporadic jobs where idle costs nothing, and Spot interruptions are free because the queue retries. Reach for ECS-on-EC2 + Savings Plans only once utilization is steadily high. **Avoid Kubernetes** unless org-wide need justifies it.

### 9.3 CI/CD, environments, ops

- **CI/CD (GitHub Actions, free tier):** build → unit/integration tests → image scan (Trivy) → push ECR → render task def → deploy. **OIDC + scoped IAM** (no long-lived keys). Manual approval gate before prod.
- **Images:** multi-stage, distroless/alpine, non-root, pinned versions, **commit-SHA tags**, immutable ECR.
- **Zero-downtime & rollback:** ECS native **blue/green** (or CodeDeploy) with canary/linear traffic shift + post-deploy smoke test; **deployment circuit breaker** + ALB health checks for auto-revert; previous task-def ARN kept for instant manual rollback.
- **Health checks:** separate **liveness** (process up) and **readiness** (DB/Redis/queue/LLM reachable) — readiness gates traffic and promotion.
- **Environments:** dev/staging/prod via **Terraform**, parity enforced.
- **Secrets:** Secrets Manager/SSM injected at runtime, IAM-scoped, rotated.
- **Observability:** structured JSON logs → CloudWatch; **OpenTelemetry** traces following a submission API→queue→worker→LLM; alerts on queue age, DLQ growth, 429 rate, cost/eval.

---

## 10. Technology Decisions + Alternatives (with sources)

**Decision 1 — PostgreSQL over MongoDB.** Relational domain with FK integrity (rubric→criteria→results) and ACID state transitions; JSONB absorbs the flexible LLM output. Mongo fits structurally-heterogeneous, shard-day-one data — not this. → [Postgres vs Mongo](https://betterstack.com/community/guides/databases/postgresql-vs-mongodb/), [JSONB deep dive](https://dev.to/sparsh9/mongodb-vs-postgresql-jsonb-a-deep-dive-into-performance-and-use-cases-5bge)

**Decision 2 — Redis/Celery today, SQS at scale (not Kafka/RabbitMQ).** Kafka is a partitioned streaming log, overkill for discrete multi-minute jobs; RabbitMQ means running a broker. SQS removes broker ops when autoscaling; the **visibility-timeout heartbeat** pattern handles 2–15-min jobs. → [Kafka vs RabbitMQ vs SQS](https://blog.algomaster.io/p/kafka-vs-rabbitmq-vs-sqs), [SQS visibility timeout](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/AboutVT.html), [heartbeat pattern](https://www.tecracer.com/blog/2023/03/the-beating-heart-of-sqs-of-heartbeats-and-watchdogs.html)

**Decision 3 — Prompt-chaining pipeline + deterministic aggregation (not one mega-call, not an agent).** Each call is easier → more accurate and auditable; weighted sum is code, not LLM. → [Anthropic, Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

**Decision 4 — Curated context pack + tree-sitter repo map (RAG only selectively).** Holistic grading needs breadth retrieval fragments; a repo map gives architectural shape cheaply. → [How Aider indexes a codebase (tree-sitter + PageRank)](https://kitemetric.com/blogs/how-aider-indexes-your-codebase-for-ai-powered-code-assistance)

**Decision 5 — pgvector now, Qdrant later (not Pinecone day-one).** pgvector is sufficient to ~5–10M vectors and co-located/transactional with Postgres; avoid a separate vector DB + lock-in prematurely. → [Vector DB comparison 2026](https://blog.codercops.com/blog/vector-database-comparison-pgvector-pinecone-weaviate-qdrant-2026)

**Decision 6 — Transactional outbox for the dual-write.** DB row + enqueue aren't one transaction; outbox makes them atomic, consumers stay idempotent. → [AWS transactional outbox](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.md)

**Decision 7 — ECS Fargate (+Spot) over EC2-always-on / Kubernetes at scale.** Pay-per-second fits bursty deadline spikes; Spot interruptions are free behind a queue. → [EC2 vs Fargate](https://www.usage.ai/blogs/ec2-vs-fargate/)

**Decision 8 — Autoscale workers on backlog-per-task, not CPU.** AWS warns raw queue depth over-provisions; scale-in protection prevents killing long jobs. → [ECS SQS autoscaling](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-autoscaling-queue.html)

**Decision 9 — Model tiering + prompt caching + Batch API.** LLM tokens dominate cost; these levers cut it ~10–20×. → [Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), [Batch processing](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing)

**Decision 10 — SSRF defense-in-depth for untrusted URLs.** Live-app/GitHub/Drive URLs are the sharpest edge. → [OWASP SSRF cheat sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html), [AWS IMDS/SSRF defense](https://aws.amazon.com/blogs/security/defense-in-depth-open-firewalls-reverse-proxies-ssrf-vulnerabilities-ec2-instance-metadata-service/)

---

## 11. Implementation Roadmap

A pragmatic build order. Everything in Phases 0–3 fits **free tiers**; paid/managed services appear only in Phase 5.

### Phase 0 — Foundations (Day 1)
- Repo, Docker Compose (API + worker + Postgres + Redis), FastAPI skeleton, Alembic migrations.
- Core schema (users, courses, assignments, rubrics, criteria, submissions, jobs, attempts, results, human_reviews, outbox, embeddings).
- JWT auth + RBAC middleware; resource-scoping data-access layer.

### Phase 1 — Submission + storage (Day 1–2)
- `POST /submissions` with Idempotency-Key, Pydantic validation, presigned S3 (multipart) upload.
- Outbox write in the create transaction; outbox relay → queue.
- Status endpoint (`GET /jobs/:id`) + polling contract.

### Phase 2 — Ingest + context pack (Day 2–3)
- Ingest worker: SSRF-safe fetch, GitHub depth-1 clone, zip-bomb/zip-slip guards, ClamAV scan, size/time caps.
- Extraction filter + tree-sitter repo map + map-reduce summaries + deterministic signals (lint/Semgrep/complexity/dep-audit).
- Persist context pack to S3; stage checkpointing.

### Phase 3 — LLM evaluation engine (Day 3–4)
- Provider abstraction (free-tier Gemini/Groq for PoC; Bedrock/Claude for prod).
- Rubric loader → per-criterion structured-output scoring calls; self-consistency on low-confidence; schema validation + repair-retry.
- Deterministic weighted aggregation; confidence → `NEEDS_REVIEW` routing.
- Prompt-cache the context pack; model tiering.

### Phase 4 — Human review + results + notify (Day 4)
- Result endpoints (student redacted vs evaluator full); review endpoint (append-only).
- Notifications: SSE for UI, signed webhooks for consumers.

### Phase 5 — Hardening & scale (post-PoC / production)
- Swap Redis/Celery → SQS + DLQ; per-stage queues; backlog-per-task autoscaling + scale-in protection.
- RDS Proxy / read replicas / history partitioning (only when metered).
- ECS Fargate + Spot workers; blue/green deploys; Terraform IaC.
- CI/CD (GitHub Actions + OIDC), Trivy scan, Secrets Manager.
- Observability: OTel traces, CloudWatch alarms (queue age, DLQ, 429, cost/eval), Batch API + Bedrock Provisioned Throughput at peak.
- Optional: microVM sandbox for executing student tests; cross-family judge ensemble; monthly calibration vs human scores.

### Milestones
| Milestone | Deliverable |
|---|---|
| M1 (end Day 2) | Submit → store → queued, status polling works end-to-end. |
| M2 (end Day 3) | Context pack produced for a real GitHub repo + ZIP. |
| M3 (end Day 4) | Full rubric evaluation with structured scores + human-review path. |
| M4 (production) | Autoscaling, observability, CI/CD, failure recovery proven. |

---

## 12. AI Tool Usage Disclosure

Per the assignment's requirement to disclose where AI tools were used:

- **Research:** I used an AI assistant (Claude) to run parallel web research across four domains (AI/LLM evaluation, async infra & data model, security for untrusted uploads, scaling & deployment). All claims were tied to primary/official sources, which are cited inline in §10 and throughout.
- **Drafting:** AI assisted in structuring and drafting this document. The architectural decisions, trade-off reasoning, and the free-tier/no-overkill constraints reflect my own engineering judgement and are what I'll defend in the interview.
- **Not used for:** fabricating benchmarks. Where third-party sources disagreed (exact Bedrock/Anthropic rate-limit tiers), I deliberately did **not** cite specific numbers — those should be read live from the Service Quotas console / `anthropic-ratelimit-*` headers.

---

## Appendix — Key Sources

- Anthropic — [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) · [Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) · [Batch processing](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) · [Rate limits](https://docs.anthropic.com/en/api/rate-limits)
- AWS — [SQS visibility timeout](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/AboutVT.html) · [ECS SQS autoscaling](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-autoscaling-queue.html) · [Transactional outbox](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.md) · [IMDS/SSRF defense-in-depth](https://aws.amazon.com/blogs/security/defense-in-depth-open-firewalls-reverse-proxies-ssrf-vulnerabilities-ec2-instance-metadata-service/) · [Bedrock structured outputs](https://aws.amazon.com/about-aws/whats-new/2026/02/structured-outputs-available-amazon-bedrock) · [RDS Proxy vs PgBouncer](https://aws.amazon.com/blogs/database/understanding-the-differences-between-pgbouncer-and-amazon-rds-proxy-for-postgresql-connection-pooling)
- OWASP — [SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html) · [Top 10 (A01, A10)](https://owasp.org/Top10/)
- Security — [Zip path traversal](https://developer.android.com/privacy-and-security/risks/zip-path-traversal) · [OpenSSF archive extraction](https://best.openssf.org/Secure-Coding-Guide-for-Python/04_neutralization/pyscg-0012/)
- Data/infra — [Postgres vs Mongo](https://betterstack.com/community/guides/databases/postgresql-vs-mongodb/) · [Kafka vs RabbitMQ vs SQS](https://blog.algomaster.io/p/kafka-vs-rabbitmq-vs-sqs) · [SQS heartbeat pattern](https://www.tecracer.com/blog/2023/03/the-beating-heart-of-sqs-of-heartbeats-and-watchdogs.html)
- AI/infra — [Aider repo map / tree-sitter](https://kitemetric.com/blogs/how-aider-indexes-your-codebase-for-ai-powered-code-assistance) · [Vector DB comparison 2026](https://blog.codercops.com/blog/vector-database-comparison-pgvector-pinecone-weaviate-qdrant-2026) · [EC2 vs Fargate](https://www.usage.ai/blogs/ec2-vs-fargate/)
- LLM-judge reliability — [Self-consistency of LLM evaluators](https://arxiv.org/pdf/2412.00543) · [Self-preference bias](https://arxiv.org/pdf/2410.21819)
