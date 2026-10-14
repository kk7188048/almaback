"""Static probe of a live application URL. We fetch the page (SSRF-safe); we never execute anything."""

import re
import time
from html.parser import HTMLParser

from app.config import Settings
from app.ssrf import SSRFError, safe_get

SECURITY_HEADERS = (
    "strict-transport-security",
    "content-security-policy",
    "x-content-type-options",
    "x-frame-options",
    "referrer-policy",
)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self.links = 0
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            self.links += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip and data.strip():
            self.parts.append(data.strip())


def probe(url: str, settings: Settings, transport=None, resolver=None) -> dict:
    started = time.monotonic()
    try:
        resp = safe_get(
            url, max_bytes=settings.live_url_max_bytes, timeout=settings.fetch_timeout_s,
            transport=transport, resolver=resolver,
        )
    except SSRFError as exc:
        msg = str(exc)
        blocked = "routable" in msg or "not allowed" in msg or "malformed" in msg or "credentials" in msg
        return {"url": url, "reachable": False, "blocked": blocked, "error": "URL is not allowed" if blocked else "could not connect"}
    elapsed_ms = int((time.monotonic() - started) * 1000)
    ctype = resp.headers.get("content-type", "")
    result = {
        "url": url,
        "final_url": resp.final_url,
        "reachable": True,
        "blocked": False,
        "status": resp.status,
        "content_type": ctype,
        "response_ms": elapsed_ms,
        "security_headers_present": [h for h in SECURITY_HEADERS if h in resp.headers],
        "truncated": resp.truncated,
    }
    if "html" in ctype.lower() or resp.body[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
        parser = _TextExtractor()
        parser.feed(resp.body.decode("utf-8", "replace"))
        text = re.sub(r"\s+", " ", " ".join(parser.parts))
        result.update(title=re.sub(r"\s+", " ", parser.title).strip()[:200], link_count=parser.links, text_excerpt=text[:3000])
    else:
        result["text_excerpt"] = resp.body[:1000].decode("utf-8", "replace")
    return result
