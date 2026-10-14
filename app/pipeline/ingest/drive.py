"""Public Google Drive folders via the Drive v3 REST API (API key; folder must be shared 'anyone with the link')."""

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from app.config import Settings
from app.pipeline.errors import SourceUnavailable, TransientIngestError
from app.ssrf import SSRFError, safe_get, validate_url

API = "https://www.googleapis.com/drive/v3"
_ID = re.compile(r"^[A-Za-z0-9_-]{10,100}$")
_FOLDER = "application/vnd.google-apps.folder"
_EXPORT = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
}
_TRANSIENT_REASONS = {"rateLimitExceeded", "userRateLimitExceeded", "backendError", "internalError"}
MAX_DEPTH = 4


def parse_folder_id(url: str) -> str:
    try:
        validate_url(url, allowed_hosts={"drive.google.com"}, schemes=("https",))
    except SSRFError as exc:
        raise SourceUnavailable(f"not a valid Google Drive URL: {exc}", code="invalid_drive_url") from exc
    parts = urlsplit(url)
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", parts.path)
    folder_id = m.group(1) if m else (parse_qs(parts.query).get("id") or [""])[0]
    if not _ID.match(folder_id):
        raise SourceUnavailable("could not find a folder id in the Google Drive URL", code="invalid_drive_url")
    return folder_id


def _safe_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^\w.\- ]", "_", name.replace("/", "_").replace("\\", "_")).strip(" .")
    return (cleaned or fallback)[:120]


def _api_get(path: str, params: dict[str, str], settings: Settings, *, max_bytes: int, transport=None):
    query = "&".join(f"{k}={quote(v, safe='')}" for k, v in {**params, "key": settings.google_api_key}.items())
    resp = safe_get(f"{API}{path}?{query}", allowed_hosts={"www.googleapis.com"}, max_bytes=max_bytes, overflow="error", timeout=settings.fetch_timeout_s * 3, transport=transport)
    if resp.status == 200:
        return resp
    reason = ""
    try:
        reason = json.loads(resp.body)["error"]["errors"][0]["reason"]
    except Exception:
        pass
    if resp.status >= 500 or reason in _TRANSIENT_REASONS or resp.status == 429:
        raise TransientIngestError(f"Google Drive API error ({resp.status} {reason})")
    raise SourceUnavailable("Drive folder not found or not shared with 'anyone with the link'", code="drive_not_accessible")


def fetch_folder(url: str, dest: Path, settings: Settings, transport=None) -> dict:
    if not settings.google_api_key:
        raise SourceUnavailable("Google Drive import is not enabled on this deployment", code="drive_not_configured")
    root_id = parse_folder_id(url)
    dest.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    files = 0
    total = 0
    stack: list[tuple[str, Path, int]] = [(root_id, dest, 0)]
    while stack:
        folder_id, folder_path, depth = stack.pop()
        page_token = ""
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed=false",
                "fields": "nextPageToken,files(id,name,mimeType,size)",
                "pageSize": "1000",
            }
            if page_token:
                params["pageToken"] = page_token
            listing = json.loads(_api_get("/files", params, settings, max_bytes=5 * 1024 * 1024, transport=transport).body)
            for item in listing.get("files", []):
                name = _safe_name(item.get("name", ""), item["id"])
                mime = item.get("mimeType", "")
                if mime == _FOLDER:
                    if depth + 1 > MAX_DEPTH:
                        warnings.append(f"skipped folder {name} (too deep)")
                    else:
                        sub = folder_path / name
                        sub.mkdir(parents=True, exist_ok=True)
                        stack.append((item["id"], sub, depth + 1))
                    continue
                if files >= settings.drive_max_files:
                    warnings.append("file limit reached, remaining Drive files skipped")
                    stack.clear()
                    break
                remaining = settings.drive_max_total_bytes - total
                if mime in _EXPORT:
                    export_mime, ext = _EXPORT[mime]
                    path = f"/files/{item['id']}/export"
                    params2 = {"mimeType": export_mime}
                    name = name if name.endswith(ext) else name + ext
                elif mime.startswith("application/vnd.google-apps."):
                    warnings.append(f"skipped {name} (unsupported Google file type)")
                    continue
                else:
                    if int(item.get("size", 0)) > remaining:
                        warnings.append(f"skipped {name} (size limit)")
                        continue
                    path, params2 = f"/files/{item['id']}", {"alt": "media"}
                try:
                    body = _api_get(path, params2, settings, max_bytes=max(remaining, 1), transport=transport).body
                except SSRFError:
                    warnings.append(f"skipped {name} (too large)")
                    continue
                (folder_path / name).write_bytes(body)
                files += 1
                total += len(body)
            page_token = listing.get("nextPageToken", "")
            if not page_token:
                break
    if files == 0:
        raise SourceUnavailable("the Drive folder is empty or contains no readable files", code="drive_empty")
    return {"files": files, "bytes": total, "warnings": warnings}
