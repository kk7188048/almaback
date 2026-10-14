"""Deterministic extraction + filtering: decide which files carry signal. No LLM involved."""

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "vendor", "bower_components", "dist", "build", "out", "target",
    ".next", ".nuxt", "__pycache__", ".venv", "venv", "env", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".idea", ".vscode", "coverage", ".gradle", ".dart_tool", "Pods", "__MACOSX",
    ".terraform", "site-packages", ".cache", ".parcel-cache", ".svelte-kit", ".angular", "htmlcov",
}
LOCKFILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock", "composer.lock",
    "Gemfile.lock", "Cargo.lock", "go.sum", "gradle.lockfile", "bun.lockb", "uv.lock", "pubspec.lock",
}
ASSET_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tiff", ".mp3", ".wav", ".ogg",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".ttf", ".otf", ".woff", ".woff2", ".eot",
}
BINARY_EXT = {
    ".pyc", ".pyo", ".class", ".jar", ".war", ".dll", ".so", ".dylib", ".exe", ".o", ".a", ".bin", ".dat",
    ".db", ".sqlite", ".sqlite3", ".pkl", ".pickle", ".npy", ".npz", ".h5", ".onnx", ".pt", ".pth", ".zip",
    ".gz", ".tar", ".tgz", ".7z", ".rar", ".pdf", ".psd", ".ai", ".docx", ".xlsx", ".pptx", ".apk", ".ipa",
}
LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".cs": "csharp", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".kt": "kotlin", ".swift": "swift", ".dart": "dart", ".scala": "scala", ".sh": "shell", ".sql": "sql",
    ".html": "html", ".css": "css", ".scss": "scss", ".vue": "vue", ".svelte": "svelte", ".r": "r",
    ".ipynb": "notebook", ".lua": "lua", ".pl": "perl",
}
DOC_EXT = {".md", ".markdown", ".rst", ".txt", ".adoc"}
CONFIG_EXT = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".xml", ".gradle", ".properties", ".tf"}
CONFIG_NAMES = {"dockerfile", "makefile", "procfile", "docker-compose.yml", "docker-compose.yaml", ".gitignore", ".dockerignore", ".env.example"}
DATA_EXT = {".csv", ".tsv", ".jsonl", ".parquet", ".xls"}
TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs", "e2e", "cypress"}


@dataclass
class FileInfo:
    path: str  # posix, relative to the project root (includes the srcN_type prefix)
    abs: Path
    size: int
    lang: str | None
    kind: str  # code | doc | config | test | data | asset
    lines: int = 0

    def manifest(self) -> dict:
        return {"path": self.path, "size": self.size, "lang": self.lang, "kind": self.kind, "lines": self.lines}


@dataclass
class WalkResult:
    files: list[FileInfo] = field(default_factory=list)
    excluded: Counter = field(default_factory=Counter)
    truncated: bool = False

    @property
    def analyzable(self) -> list[FileInfo]:
        return [f for f in self.files if f.kind != "asset"]


def classify(rel: str, name: str, ext: str) -> tuple[str, str | None]:
    lower = name.lower()
    parts = {p.lower() for p in rel.split("/")[:-1]}
    lang = LANG_BY_EXT.get(ext)
    is_test = (
        bool(parts & TEST_DIR_NAMES)
        or lower.startswith("test_")
        or lower.endswith(("_test.py", "_test.go", ".test.js", ".test.ts", ".test.tsx", ".test.jsx", ".spec.js", ".spec.ts", "tests.py"))
    )
    if ext in ASSET_EXT:
        return "asset", None
    if lang and is_test:
        return "test", lang
    if lang:
        return "code", lang
    if ext in DOC_EXT:
        return "doc", None
    if lower in CONFIG_NAMES or ext in CONFIG_EXT or rel.startswith(".github/") or "/.github/" in rel:
        return "config", None
    if ext in DATA_EXT:
        return "data", None
    return "data", None


def _looks_minified(head: bytes, size: int, ext: str) -> bool:
    if ext not in (".js", ".css", ".mjs") or size < 10_000:
        return False
    lines = head.count(b"\n") + 1
    return len(head) / lines > 300


def _gitignore(root: Path) -> pathspec.PathSpec | None:
    gi = root / ".gitignore"
    if not gi.is_file():
        return None
    try:
        lines = gi.read_text(encoding="utf-8", errors="replace").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    except Exception:
        return None


def walk_project(project: Path, max_text_bytes: int, max_files: int) -> WalkResult:
    result = WalkResult()
    specs: dict[str, pathspec.PathSpec | None] = {}
    for source_dir in sorted(p for p in project.iterdir() if p.is_dir()):
        spec = _gitignore(source_dir)
        specs[source_dir.name] = spec
        for dirpath, dirnames, filenames in os.walk(source_dir, followlinks=False):
            rel_dir = Path(dirpath).relative_to(source_dir).as_posix()
            kept = []
            for d in sorted(dirnames):
                rel_d = d if rel_dir == "." else f"{rel_dir}/{d}"
                if d in EXCLUDE_DIRS or d.startswith(".") and d not in (".github",):
                    result.excluded["ignored_directory"] += 1
                elif spec and spec.match_file(rel_d + "/"):
                    result.excluded["gitignored"] += 1
                else:
                    kept.append(d)
            dirnames[:] = kept
            for name in sorted(filenames):
                if len(result.files) >= max_files:
                    result.truncated = True
                    return result
                abs_path = Path(dirpath) / name
                rel_in_src = name if rel_dir == "." else f"{rel_dir}/{name}"
                rel = f"{source_dir.name}/{rel_in_src}"
                ext = Path(name).suffix.lower()
                if abs_path.is_symlink():
                    result.excluded["symlink"] += 1
                    continue
                if name in LOCKFILES:
                    result.excluded["lockfile"] += 1
                    continue
                if spec and spec.match_file(rel_in_src):
                    result.excluded["gitignored"] += 1
                    continue
                try:
                    size = abs_path.stat().st_size
                except OSError:
                    continue
                kind, lang = classify(rel, name, ext)
                if kind == "asset":
                    result.files.append(FileInfo(rel, abs_path, size, None, "asset"))
                    continue
                if ext in BINARY_EXT:
                    result.excluded["binary"] += 1
                    continue
                if size > max_text_bytes:
                    result.excluded["oversized"] += 1
                    continue
                try:
                    with abs_path.open("rb") as f:
                        data = f.read(max_text_bytes)
                except OSError:
                    continue
                if b"\x00" in data[:4096]:
                    result.excluded["binary"] += 1
                    continue
                if _looks_minified(data[:20_000], size, ext):
                    result.excluded["minified"] += 1
                    continue
                result.files.append(FileInfo(rel, abs_path, size, lang, kind, lines=data.count(b"\n") + (1 if data else 0)))
    return result


def read_text(info: FileInfo, limit: int) -> str:
    with info.abs.open("rb") as f:
        return f.read(limit).decode("utf-8", "replace")
