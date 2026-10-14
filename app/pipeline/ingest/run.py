"""Fetch every submitted source into a work directory, applying the safety checks for each type.

Failure policy:
  * retryable error (network, rate limit)  -> raised, the attempt is retried (checkpoints keep LLM work)
  * non-retryable error on one source      -> that source is marked failed; the others still get evaluated
  * non-retryable errors on ALL sources    -> the job fails with the first error (student must fix the submission)
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.config import Settings
from app.pipeline.errors import IngestError, PipelineError, SourceUnavailable, TransientIngestError
from app.pipeline.ingest import documents, drive, github, live_url, video
from app.pipeline.ingest.filetypes import validate_upload
from app.pipeline.ingest.malware import scan_file
from app.pipeline.ingest.safe_zip import ZipLimits, project_root, safe_extract
from app.ssrf import SSRFError
from app.storage import Storage, StorageError

log = logging.getLogger(__name__)

SIZE_LIMIT_ATTR = {"zip": "max_zip_bytes", "pdf": "max_pdf_bytes", "docs": "max_docs_bytes", "video": "max_video_bytes"}


@dataclass
class SourceOutcome:
    index: int
    type: str
    prefix: str
    status: str  # ok | failed
    warnings: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("index", "type", "prefix", "status", "warnings", "extras", "error_code", "error_message")}


@dataclass
class IngestResult:
    root: Path
    outcomes: list[SourceOutcome]

    @property
    def warnings(self) -> list[str]:
        return [f"{o.prefix}: {w}" for o in self.outcomes for w in o.warnings]


def _upload_to_local(src: dict, idx: int, kind: str, workdir: Path, storage: Storage, settings: Settings) -> Path:
    key = src.get("storage_key")
    if not key:
        raise SourceUnavailable("upload is missing", code="upload_missing")
    try:
        size = storage.size(key)
        limit = getattr(settings, SIZE_LIMIT_ATTR[kind])
        if size > limit:
            raise SourceUnavailable(f"upload exceeds the {limit // (1024 * 1024)} MB limit", code="upload_too_large")
        err = validate_upload(kind, storage.read_head(key))
        if err:
            raise IngestError(err, code="invalid_file_type")
        local = workdir / "uploads" / f"{idx}_{Path(src['filename']).name}"
        storage.download_to(key, local)
    except StorageError as exc:
        raise SourceUnavailable("upload could not be found", code="upload_missing") from exc
    scan_file(local, settings)
    return local


def _ingest_one(idx: int, src: dict, project: Path, workdir: Path, storage: Storage, settings: Settings, transport) -> SourceOutcome:
    kind = src["type"]
    prefix = f"src{idx}_{kind}"
    dest = project / prefix
    outcome = SourceOutcome(idx, kind, prefix, "ok")

    if kind == "github":
        owner, repo = github.parse_repo_url(src["url"])
        token = github.installation_token(owner, repo, settings)
        github.check_repo(owner, repo, token, settings, transport=transport)
        outcome.extras["git"] = github.clone_repo(owner, repo, dest, token, settings)
    elif kind == "zip":
        local = _upload_to_local(src, idx, kind, workdir, storage, settings)
        extract_to = workdir / "extract" / prefix
        report = safe_extract(
            local, extract_to,
            ZipLimits(settings.zip_max_uncompressed_bytes, settings.zip_max_entry_bytes, settings.zip_max_ratio, settings.zip_max_entries),
        )
        shutil.move(str(project_root(extract_to)), str(dest))
        shutil.rmtree(extract_to, ignore_errors=True)
        local.unlink(missing_ok=True)
        outcome.warnings += [f"skipped {s}" for s in report.skipped]
        outcome.extras["extracted"] = {"files": report.files, "bytes": report.bytes_written}
    elif kind in ("pdf", "docs"):
        local = _upload_to_local(src, idx, kind, workdir, storage, settings)
        outcome.warnings += documents.convert_document(local, dest, src["filename"])
        local.unlink(missing_ok=True)
    elif kind == "video":
        local = _upload_to_local(src, idx, kind, workdir, storage, settings)
        outcome.extras["video"] = video.process_video(local, dest, settings, transport=transport if isinstance(transport, httpx.BaseTransport) else None)
        if not outcome.extras["video"].get("transcribed"):
            outcome.warnings.append("video was not transcribed: " + outcome.extras["video"].get("note", "unknown reason"))
        local.unlink(missing_ok=True)
    elif kind == "drive":
        info = drive.fetch_folder(src["url"], dest, settings, transport=transport)
        outcome.extras["drive"] = {"files": info["files"], "bytes": info["bytes"]}
        outcome.warnings += info["warnings"]
    elif kind == "live_url":
        probe = live_url.probe(src["url"], settings, transport=transport)
        outcome.extras["live_probe"] = probe
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
        if not probe.get("reachable"):
            outcome.warnings.append("live application URL could not be reached" + (" (URL not allowed)" if probe.get("blocked") else ""))
    else:
        raise IngestError(f"unknown source type {kind}", code="unknown_source_type")
    return outcome


def ingest_submission(sources: list[dict], workdir: Path, storage: Storage, settings: Settings, transport=None) -> IngestResult:
    project = workdir / "project"
    project.mkdir(parents=True, exist_ok=True)
    outcomes: list[SourceOutcome] = []
    first_error: PipelineError | None = None
    for idx, src in enumerate(sources):
        try:
            outcomes.append(_ingest_one(idx, src, project, workdir, storage, settings, transport))
        except SSRFError as exc:
            if "could not connect" in str(exc):
                raise TransientIngestError(f"network error fetching source {idx}") from exc
            err = SourceUnavailable(str(exc), code="source_not_allowed")
            first_error = first_error or err
            outcomes.append(SourceOutcome(idx, src["type"], f"src{idx}_{src['type']}", "failed", error_code=err.code, error_message=err.message))
        except PipelineError as exc:
            if exc.retryable:
                raise
            first_error = first_error or exc
            log.info("source %s failed: %s", idx, exc.message, extra={"ctx": {"error_code": exc.code}})
            shutil.rmtree(project / f"src{idx}_{src['type']}", ignore_errors=True)
            outcomes.append(SourceOutcome(idx, src["type"], f"src{idx}_{src['type']}", "failed", error_code=exc.code, error_message=exc.message))
    if not any(o.status == "ok" for o in outcomes):
        assert first_error is not None
        raise first_error
    return IngestResult(root=project, outcomes=outcomes)
