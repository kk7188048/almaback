"""Deterministic signals: facts measured by tools. The LLM interprets these; it never computes them.
Code is analysed statically (parsed, never executed)."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from pathlib import Path

from app.config import Settings
from app.pipeline.context import secrets
from app.pipeline.context.filters import FileInfo, read_text

log = logging.getLogger(__name__)

FRAMEWORK_HINTS = {
    "fastapi": "FastAPI", "flask": "Flask", "django": "Django", "express": "Express", "react": "React",
    "next": "Next.js", "vue": "Vue", "svelte": "Svelte", "@angular/core": "Angular", "spring-boot": "Spring Boot",
    "sqlalchemy": "SQLAlchemy", "prisma": "Prisma", "mongoose": "Mongoose", "pytorch": "PyTorch", "torch": "PyTorch",
    "tensorflow": "TensorFlow", "scikit-learn": "scikit-learn", "langchain": "LangChain", "openai": "OpenAI SDK",
    "anthropic": "Anthropic SDK", "pytest": "pytest", "jest": "Jest", "streamlit": "Streamlit", "pandas": "pandas",
}
MAX_LINT_FILES = 1500
MAX_RADON_FILES = 300


def _tool_env() -> dict[str, str]:
    """Analysis tools get a minimal environment: no API keys, no cloud credentials."""
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tempfile.gettempdir()}


def _run(cmd: list[str], timeout: int, cwd: Path | None = None) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=_tool_env(), cwd=cwd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.info("tool failed: %s (%s)", cmd[0], type(exc).__name__)
        return None


def _which(name: str) -> str | None:
    return shutil.which(name) or (str(Path(sys.executable).parent / name) if (Path(sys.executable).parent / name).exists() else None)


def parse_dependencies(files: list[FileInfo]) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {}
    for f in files:
        name = f.abs.name
        try:
            if name == "package.json":
                data = json.loads(read_text(f, 400_000))
                deps[f.path] = sorted({*data.get("dependencies", {}), *data.get("devDependencies", {})})
            elif re.fullmatch(r"requirements[\w.-]*\.txt", name):
                found = []
                for line in read_text(f, 200_000).splitlines():
                    line = line.split("#")[0].strip()
                    if line and not line.startswith("-"):
                        found.append(re.split(r"[=<>!~\[; ]", line, maxsplit=1)[0].lower())
                deps[f.path] = sorted(set(found))
            elif name == "pyproject.toml":
                data = tomllib.loads(read_text(f, 400_000))
                found = [re.split(r"[=<>!~\[; ]", d, maxsplit=1)[0].lower() for d in data.get("project", {}).get("dependencies", [])]
                found += [k.lower() for k in data.get("tool", {}).get("poetry", {}).get("dependencies", {}) if k.lower() != "python"]
                deps[f.path] = sorted(set(found))
        except (ValueError, tomllib.TOMLDecodeError, AttributeError):
            deps[f.path] = []
    return deps


def detect_frameworks(deps: dict[str, list[str]]) -> list[str]:
    names = {d for lst in deps.values() for d in lst}
    return sorted({label for hint, label in FRAMEWORK_HINTS.items() if hint in names})


def _ruff_json(ruff: str, files: list[Path], select: str, ignore: str = "") -> list[dict]:
    findings: list[dict] = []
    for i in range(0, min(len(files), MAX_LINT_FILES), 500):
        batch = [str(p) for p in files[i:i + 500]]
        cmd = [ruff, "check", "--isolated", "--no-cache", "--exit-zero", "--output-format", "json", "--select", select]
        if ignore:
            cmd += ["--ignore", ignore]
        proc = _run([*cmd, *batch], 180)
        if proc is None or not proc.stdout.strip():
            continue
        try:
            findings.extend(json.loads(proc.stdout))
        except ValueError:
            continue
    return findings


def lint_python(files: list[Path], test_files: set[Path] | None = None) -> dict:
    """Style/bug rules on every file; security (bandit-style) rules only on non-test code, because `assert`,
    hard-coded fixtures and subprocess calls in tests are normal. E501 is ignored: line length is a project convention
    (ruff's default of 88 would flag most real-world code)."""
    ruff = _which("ruff")
    if ruff is None or not files:
        return {"available": ruff is not None, "files": 0}
    test_files = test_files or set()
    findings = _ruff_json(ruff, files, "E,F,W,B,C90", ignore="E501")
    findings += _ruff_json(ruff, [f for f in files if f not in test_files], "S")
    codes = Counter(f.get("code") or "syntax" for f in findings)
    return {
        "available": True,
        "tool": "ruff",
        "files": min(len(files), MAX_LINT_FILES),
        "total_issues": len(findings),
        "by_category": dict(Counter((c or "syntax")[:1] for c in codes.elements())),
        "top_rules": codes.most_common(10),
        "security_findings": sum(n for c, n in codes.items() if str(c).startswith("S")),
        "syntax_errors": codes.get("syntax", 0) + codes.get("E999", 0),
        "notes": "E501 ignored; security rules skipped for test files",
    }


def complexity_python(files: list[Path], max_bytes: int) -> dict:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit

    complexities: list[int] = []
    mi_scores: list[float] = []
    syntax_errors = 0
    worst: list[tuple[int, str, str]] = []
    for path in files[:MAX_RADON_FILES]:
        try:
            if path.stat().st_size > max_bytes:
                continue
            src = path.read_text(encoding="utf-8", errors="replace")
            blocks = cc_visit(src)
            mi_scores.append(mi_visit(src, multi=True))
        except SyntaxError:
            syntax_errors += 1
            continue
        except (RecursionError, ValueError, MemoryError, OSError):
            continue
        for b in blocks:
            complexities.append(b.complexity)
            worst.append((b.complexity, path.name, b.name))
    worst.sort(reverse=True)
    return {
        "functions_analysed": len(complexities),
        "avg_complexity": round(sum(complexities) / len(complexities), 2) if complexities else None,
        "max_complexity": max(complexities) if complexities else None,
        "functions_over_10": sum(1 for c in complexities if c > 10),
        "avg_maintainability_index": round(sum(mi_scores) / len(mi_scores), 1) if mi_scores else None,
        "syntax_errors": syntax_errors,
        "most_complex": [{"complexity": c, "file": f, "function": n} for c, f, n in worst[:5]],
    }


def run_semgrep(root: Path, settings: Settings) -> dict:
    exe = _which("semgrep")
    if not settings.semgrep_enabled or exe is None:
        return {"available": False}
    proc = _run([exe, "scan", "--config", settings.semgrep_config, "--json", "--metrics", "off", "--quiet", "--timeout", "20", str(root)], 900)
    if proc is None or not proc.stdout.strip():
        return {"available": True, "error": "semgrep did not produce output"}
    try:
        results = json.loads(proc.stdout).get("results", [])
    except ValueError:
        return {"available": True, "error": "unparseable semgrep output"}
    return {
        "available": True,
        "total": len(results),
        "by_severity": dict(Counter(r.get("extra", {}).get("severity", "UNKNOWN") for r in results)),
        "top_rules": Counter(r.get("check_id", "?") for r in results).most_common(8),
    }


def run_pip_audit(requirements: Path, settings: Settings) -> dict:
    exe = _which("pip-audit")
    if not settings.pip_audit_enabled or exe is None:
        return {"available": False}
    proc = _run([exe, "-r", str(requirements), "-f", "json", "--progress-spinner", "off"], 300)
    if proc is None or not proc.stdout.strip():
        return {"available": True, "error": "pip-audit did not produce output"}
    try:
        deps = json.loads(proc.stdout).get("dependencies", [])
    except ValueError:
        return {"available": True, "error": "unparseable pip-audit output"}
    return {"available": True, "vulnerable_packages": sum(1 for d in deps if d.get("vulns")), "total_packages": len(deps)}


def compute_signals(root: Path, files: list[FileInfo], excluded: Counter, settings: Settings, git_info: list[dict]) -> dict:
    code = [f for f in files if f.kind in ("code", "test")]
    by_lang: Counter = Counter()
    for f in code:
        by_lang[f.lang or "other"] += f.lines
    names_lower = {f.abs.name.lower() for f in files}
    paths_lower = [f.path.lower() for f in files]
    deps = parse_dependencies(files)
    py_files = [f.abs for f in code if f.lang == "python"]
    py_tests = {f.abs for f in code if f.lang == "python" and f.kind == "test"}
    readme = next((f for f in files if f.abs.name.lower().startswith("readme")), None)

    secret_findings: list[dict] = []
    todo = 0
    for f in files:
        if f.kind in ("asset", "data") or f.size > settings.max_text_file_bytes:
            continue
        text = read_text(f, settings.max_text_file_bytes)
        secret_findings += secrets.scan_text(f.path, text, limit=50 - len(secret_findings)) if len(secret_findings) < 50 else []
        todo += len(re.findall(r"\b(?:TODO|FIXME|HACK|XXX)\b", text))

    requirements = next((f.abs for f in files if re.fullmatch(r"requirements[\w.-]*\.txt", f.abs.name)), None)
    return {
        "counts": {
            "analysed_files": len(files),
            "code_files": sum(1 for f in files if f.kind == "code"),
            "test_files": sum(1 for f in files if f.kind == "test"),
            "doc_files": sum(1 for f in files if f.kind == "doc"),
            "config_files": sum(1 for f in files if f.kind == "config"),
            "assets": sum(1 for f in files if f.kind == "asset"),
            "excluded": dict(excluded),
        },
        "loc_by_language": dict(by_lang.most_common()),
        "total_code_loc": sum(by_lang.values()),
        "structure": {
            "has_readme": readme is not None,
            "readme_words": len(read_text(readme, 200_000).split()) if readme else 0,
            "has_tests": any(f.kind == "test" for f in files),
            "has_dockerfile": "dockerfile" in names_lower,
            "has_ci": any(".github/workflows/" in p or p.endswith(("gitlab-ci.yml", "jenkinsfile")) for p in paths_lower),
            "has_license": any(n.startswith("license") for n in names_lower),
            "has_gitignore": ".gitignore" in names_lower,
            "has_env_example": ".env.example" in names_lower,
            "has_api_docs": any(("openapi" in p or "swagger" in p or "/docs/" in p) for p in paths_lower),
            "has_schema_files": any(p.endswith(".sql") or "schema" in p or "migration" in p for p in paths_lower),
        },
        "dependencies": {k: v[:60] for k, v in deps.items()},
        "frameworks": detect_frameworks(deps),
        "todo_fixme_count": todo,
        "lint": lint_python(py_files, py_tests),
        "complexity": complexity_python(py_files, settings.max_parse_file_bytes) if py_files else {},
        "semgrep": run_semgrep(root, settings),
        "dependency_audit": run_pip_audit(requirements, settings) if requirements else {"available": False},
        "secrets_found": [{"path": s["path"], "line": s["line"], "rule": s["rule"]} for s in secret_findings],
        "git": git_info,
        "tests_executed": False,
        "note": "Code was analysed statically; nothing was executed.",
    }
