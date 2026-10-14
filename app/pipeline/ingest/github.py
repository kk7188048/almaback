"""GitHub repositories: SSRF-pinned metadata check, hardened shallow clone, optional GitHub App tokens."""

import base64
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import jwt

from app.config import Settings
from app.pipeline.errors import IngestError, SourceUnavailable, TransientIngestError
from app.ssrf import SSRFError, safe_get, safe_request, validate_url

log = logging.getLogger(__name__)

GITHUB_HOSTS = {"github.com", "www.github.com"}
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_API_HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


def parse_repo_url(url: str) -> tuple[str, str]:
    try:
        validate_url(url, allowed_hosts=GITHUB_HOSTS, schemes=("https",))
    except SSRFError as exc:
        raise SourceUnavailable(f"not a valid GitHub repository URL: {exc}", code="invalid_repo_url") from exc
    parts = [p for p in urlsplit(url).path.split("/") if p]
    if len(parts) < 2:
        raise SourceUnavailable("GitHub URL must look like https://github.com/<owner>/<repo>", code="invalid_repo_url")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if not (_NAME.match(owner) and _NAME.match(repo)) or owner.strip(".") == "" or repo.strip(".") == "":
        raise SourceUnavailable("GitHub URL contains invalid characters", code="invalid_repo_url")
    return owner, repo


def installation_token(owner: str, repo: str, settings: Settings) -> str | None:
    """Short-lived, repo-scoped read token from the GitHub App, or None if the app isn't configured/installed."""
    if not (settings.github_app_id and settings.github_app_private_key):
        return None
    now = int(time.time())
    app_jwt = jwt.encode(
        {"iat": now - 60, "exp": now + 540, "iss": settings.github_app_id},
        settings.github_app_private_key.replace("\\n", "\n"),
        algorithm="RS256",
    )
    headers = {**_API_HEADERS, "Authorization": f"Bearer {app_jwt}"}
    resp = safe_get(f"https://api.github.com/repos/{owner}/{repo}/installation", headers=headers, allowed_hosts={"api.github.com"}, timeout=settings.fetch_timeout_s)
    if resp.status == 404:
        return None
    if resp.status != 200:
        raise TransientIngestError(f"GitHub App lookup failed with HTTP {resp.status}")
    import json

    installation_id = json.loads(resp.body)["id"]
    tok = safe_request(
        "POST",
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers=headers,
        json_body={"repositories": [repo], "permissions": {"contents": "read", "metadata": "read"}},
        allowed_hosts={"api.github.com"},
        timeout=settings.fetch_timeout_s,
    )
    if tok.status != 201:
        raise TransientIngestError(f"GitHub App token request failed with HTTP {tok.status}")
    return json.loads(tok.body)["token"]


def check_repo(owner: str, repo: str, token: str | None, settings: Settings, transport=None) -> int:
    """Returns the repo size in bytes as reported by GitHub; raises if it is missing, private or too large."""
    import json

    headers = dict(_API_HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = safe_get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers, allowed_hosts={"api.github.com"}, timeout=settings.fetch_timeout_s, transport=transport)
    if resp.status == 404:
        raise SourceUnavailable(
            "repository not found, or it is private (make it public or install the evaluation GitHub App)",
            code="repo_not_found",
        )
    if resp.status in (403, 429):
        raise TransientIngestError("GitHub rate limit reached, will retry")
    if resp.status != 200:
        raise TransientIngestError(f"GitHub API returned HTTP {resp.status}")
    size_bytes = int(json.loads(resp.body).get("size", 0)) * 1024
    if size_bytes > settings.github_max_repo_bytes:
        raise SourceUnavailable(
            f"repository is too large ({size_bytes // (1024 * 1024)} MB > {settings.github_max_repo_bytes // (1024 * 1024)} MB limit)",
            code="repo_too_large",
        )
    return size_bytes


def _clone_command(clone_url: str, dest: Path) -> list[str]:
    return [
        "git",
        "-c", "core.symlinks=false",
        "-c", "core.hooksPath=/dev/null",
        "-c", "protocol.allow=never",
        "-c", "protocol.https.allow=always",
        "clone", "--depth", "1", "--single-branch", "--no-tags", "--quiet",
        "--", clone_url, str(dest),
    ]


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def clone_repo(owner: str, repo: str, dest: Path, token: str | None, settings: Settings) -> dict:
    if shutil.which("git") is None:
        raise IngestError("git is not installed on this worker", code="git_missing")
    home = Path(tempfile.mkdtemp(prefix="gitenv-"))
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.extraHeader", GIT_CONFIG_VALUE_0=f"Authorization: Basic {basic}")
    cmd = _clone_command(f"https://github.com/{owner}/{repo}.git", dest)
    started = time.monotonic()
    cap = int(settings.github_max_repo_bytes * 1.15)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    last_size_check = 0.0
    try:
        while proc.poll() is None:
            now = time.monotonic()
            if now - started > settings.github_clone_timeout_s:
                proc.kill()
                raise TransientIngestError("cloning the repository timed out")
            if now - last_size_check > 2.0 and dest.exists():
                last_size_check = now
                if _dir_size(dest) > cap:
                    proc.kill()
                    raise SourceUnavailable("repository is too large to clone", code="repo_too_large")
            time.sleep(0.2)
        if proc.returncode != 0:
            stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")[-400:]
            if "not found" in stderr.lower() or "authentication" in stderr.lower():
                raise SourceUnavailable("repository not found or not accessible", code="repo_not_found")
            raise TransientIngestError(f"git clone failed: {stderr.strip()}")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        shutil.rmtree(home, ignore_errors=True)

    if _dir_size(dest) > settings.github_max_repo_bytes:  # fast clones can finish between watchdog polls
        shutil.rmtree(dest, ignore_errors=True)
        raise SourceUnavailable("repository is too large to evaluate", code="repo_too_large")

    info: dict = {"owner": owner, "repo": repo}
    for key, fmt in (("commit", "%H"), ("committed_at", "%cI")):
        out = subprocess.run(
            ["git", "-C", str(dest), "log", "-1", f"--format={fmt}"],
            capture_output=True, text=True, timeout=30, env={"PATH": env["PATH"], "GIT_CONFIG_NOSYSTEM": "1"},
        )
        info[key] = out.stdout.strip() or None
    shutil.rmtree(dest / ".git", ignore_errors=True)
    return info
