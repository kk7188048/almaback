"""Build the persisted 'context pack': a compact, structured, redacted digest of the whole submission.
Every later LLM call reads this - never raw files - so a 1 GB repository becomes a few tens of thousands of tokens."""

import hashlib
import json
import logging
import posixpath
from collections import defaultdict

from app.config import Settings
from app.pipeline.context import secrets
from app.pipeline.context.embeddings import HashEmbedder, chunk_text
from app.pipeline.context.filters import FileInfo, read_text, walk_project
from app.pipeline.context.injection import find_injection
from app.pipeline.context.repomap import build_repo_map
from app.pipeline.context.signals import compute_signals
from app.pipeline.ingest.run import IngestResult
from app.pipeline.llm import prompts
from app.pipeline.llm.base import LLMClient, LLMMalformedResponse
from app.pipeline.llm.schemas import DirSummaryOut, FileSummariesOut

log = logging.getLogger(__name__)
SUMMARY_FILE_CHARS = 3500


def _file_tree(files: list[FileInfo], max_lines: int = 160) -> str:
    dirs: dict[str, list[str]] = defaultdict(list)
    for f in files:
        dirs[posixpath.dirname(f.path)].append(posixpath.basename(f.path))
    lines = []
    for d in sorted(dirs):
        if d.count("/") > 3:
            continue
        names = sorted(dirs[d])
        shown = ", ".join(names[:8]) + (f", ... (+{len(names) - 8})" if len(names) > 8 else "")
        lines.append(f"{d or '.'}/  [{len(names)} files]  {shown}")
        if len(lines) >= max_lines:
            lines.append("... (tree truncated)")
            break
    return "\n".join(lines)


def _summarise_files(llm: LLMClient, boundary: str, targets: list[FileInfo], settings: Settings) -> dict[str, str]:
    """Map step, batched to keep the number of LLM calls low."""
    summaries: dict[str, str] = {}
    batch: list[dict] = []
    size = 0

    def flush():
        nonlocal batch, size
        if not batch:
            return
        req = prompts.file_summaries_request(boundary, batch)
        asked = {b["path"] for b in batch}
        try:
            out = llm.call(req, FileSummariesOut)
            for item in out.items:
                if item.path in asked and item.summary.strip():
                    summaries[item.path] = item.summary.strip()[:600]
        except LLMMalformedResponse:
            log.warning("file summary batch failed validation, using first-line fallback")
        for b in batch:
            summaries.setdefault(b["path"], next((ln.strip() for ln in b["text"].splitlines() if ln.strip()), "")[:200])
        batch, size = [], 0

    for f in targets:
        text = secrets.redact(read_text(f, SUMMARY_FILE_CHARS))
        if size + len(text) > settings.summarize_batch_chars:
            flush()
        batch.append({"path": f.path, "text": text})
        size += len(text)
    flush()
    return summaries


def _reduce_by_directory(llm: LLMClient, boundary: str, file_summaries: dict[str, str]) -> dict[str, str]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, summary in sorted(file_summaries.items()):
        groups[posixpath.dirname(path)].append((path, summary))
    out: dict[str, str] = {}
    for directory, items in groups.items():
        joined = " ".join(s for _p, s in items)
        if len(items) >= 3 and len(joined) > 1200:
            try:
                out[directory] = llm.call(prompts.dir_summary_request(boundary, directory, items), DirSummaryOut).summary.strip()
                continue
            except LLMMalformedResponse:
                pass
        out[directory] = " | ".join(f"{posixpath.basename(p)}: {s}" for p, s in items)[:1500]
    return out


def build_context_pack(ingest: IngestResult, settings: Settings, llm: LLMClient, boundary: str) -> tuple[dict, list[tuple[str, str, list[float]]]]:
    walk = walk_project(ingest.root, settings.max_text_file_bytes, settings.max_walk_files)
    analysable = walk.analyzable
    warnings = list(ingest.warnings)
    if walk.truncated:
        warnings.append(f"project has more than {settings.max_walk_files} files; the rest were ignored")

    code_inputs: list[tuple[str, str, bytes]] = []
    for f in analysable:
        if f.kind in ("code", "test") and f.lang and f.size <= settings.max_parse_file_bytes and len(code_inputs) < 3000:
            code_inputs.append((f.path, f.lang, f.abs.read_bytes()))
    repo_map = build_repo_map(code_inputs, settings.repo_map_tokens)
    ranking = [p for p, _s in repo_map.ranking]

    git_info = [{"source": o.prefix, **o.extras["git"]} for o in ingest.outcomes if o.status == "ok" and "git" in o.extras]
    signals = compute_signals(ingest.root, analysable, walk.excluded, settings, git_info)

    readme_file = min((f for f in analysable if f.abs.name.lower().startswith("readme")), key=lambda f: len(f.path), default=None)
    readme = {"path": readme_file.path, "text": secrets.redact(read_text(readme_file, settings.readme_tokens * 4))} if readme_file else None

    injection_flags: list[dict] = []
    for f in analysable:
        if f.kind == "data" or f.size > settings.max_text_file_bytes:
            continue
        snippet = find_injection(read_text(f, 200_000))
        if snippet:
            injection_flags.append({"path": f.path, "snippet": secrets.redact(snippet)})
            if len(injection_flags) >= 10:
                break
    for outcome in ingest.outcomes:
        probe = outcome.extras.get("live_probe")
        if probe and (snippet := find_injection(probe.get("text_excerpt", ""))):
            injection_flags.append({"path": outcome.prefix + "/live_url", "snippet": secrets.redact(snippet)})

    top_code = [f for f in analysable if f.kind == "code" and f.path in set(ranking[: settings.summarize_top_code_files])]
    docs = [f for f in analysable if f.kind == "doc"][:60]
    file_summaries = _summarise_files(llm, boundary, docs + top_code, settings)
    dir_summaries = _reduce_by_directory(llm, boundary, file_summaries)

    embedder = HashEmbedder(settings.embedding_dim)
    order = {p: i for i, p in enumerate(ranking)}
    embed_files = sorted(
        [f for f in analysable if f.kind in ("doc", "code", "test", "config")],
        key=lambda f: (0 if f.kind == "doc" else 1, order.get(f.path, 10**6), f.path),
    )
    chunks: list[tuple[str, str, list[float]]] = []
    for f in embed_files:
        if len(chunks) >= settings.max_embedding_chunks:
            break
        text = secrets.redact(read_text(f, settings.max_text_file_bytes))
        for ch in chunk_text(f.path, text):
            chunks.append((ch.ref, ch.text, embedder.embed(ch.text)))
            if len(chunks) >= settings.max_embedding_chunks:
                break

    extras = {
        "live_probes": [o.extras["live_probe"] for o in ingest.outcomes if "live_probe" in o.extras],
        "videos": [{"source": o.prefix, **o.extras["video"]} for o in ingest.outcomes if "video" in o.extras],
    }
    code_loc = signals["total_code_loc"]
    pack = {
        "version": 1,
        "sources": [o.to_dict() for o in ingest.outcomes],
        "warnings": warnings,
        "metadata": {
            "file_count": len(analysable),
            "code_loc": code_loc,
            "languages": signals["loc_by_language"],
            "frameworks": signals["frameworks"],
            "excluded": dict(walk.excluded),
            "walk_truncated": walk.truncated,
            "parsed_files": repo_map.parsed_files,
            "symbols": repo_map.symbol_count,
        },
        "file_tree": _file_tree(analysable),
        "repo_map": repo_map.text,
        "readme": readme,
        "file_summaries": file_summaries,
        "dir_summaries": dir_summaries,
        "signals": signals,
        "extras": extras,
        "injection_flags": injection_flags,
        "ranked_files": ranking[:50],
        "manifest": sorted(f.path for f in analysable),
        "assets": sorted(f.path for f in walk.files if f.kind == "asset")[:200],
        "chunks_indexed": len(chunks),
    }
    return pack, chunks


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 20)] + "\n[truncated]"


def render_context(pack: dict, budget_tokens: int) -> str:
    """Render the pack into the shared prompt prefix, shrinking the least important sections first."""
    budget = budget_tokens * 4
    sig = pack["signals"]
    compact_signals = {k: sig[k] for k in ("counts", "loc_by_language", "structure", "frameworks", "lint", "complexity", "semgrep", "dependency_audit", "todo_fixme_count", "git", "tests_executed", "note") if k in sig}
    compact_signals["secrets_found_count"] = len(sig.get("secrets_found", []))
    compact_signals["secrets_found_files"] = sorted({s["path"] for s in sig.get("secrets_found", [])})[:10]
    sections: list[tuple[str, str, int]] = [  # (title, body, max chars; later = shrinks first)
        ("PROJECT OVERVIEW", json.dumps({**pack["metadata"], "sources": [{k: s[k] for k in ("prefix", "type", "status", "error_message") if s.get(k)} for s in pack["sources"]]}), 4000),
        ("MEASURED SIGNALS (static analysis; code was NOT executed)", json.dumps(compact_signals), 7000),
        ("README", (pack["readme"] or {}).get("text", "(no README found)"), 24000),
        ("LIVE APPLICATION / VIDEO", json.dumps(pack["extras"])[:3500], 3500),
        ("FILE TREE", pack["file_tree"], 8000),
        ("DOCUMENT & FILE SUMMARIES (by directory)", "\n".join(f"[{d or '.'}] {s}" for d, s in sorted(pack["dir_summaries"].items())), 30000),
        ("REPO MAP (ranked symbol signatures)", pack["repo_map"], 60000),
        ("PIPELINE WARNINGS", "\n".join(pack["warnings"]) or "(none)", 2000),
    ]
    sizes = [min(len(b), cap) for _t, b, cap in sections]
    shrink_order = [6, 5, 4, 2]  # repo map, summaries, file tree, readme
    overhead = sum(len(t) + 8 for t, _b, _c in sections)
    for idx in shrink_order:
        over = sum(sizes) + overhead - budget
        if over <= 0:
            break
        sizes[idx] = max(500, sizes[idx] - over)
    parts = [f"## {t}\n{_clip(secrets.redact(b), n)}" for (t, b, _c), n in zip(sections, sizes)]
    return "\n\n".join(parts)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
