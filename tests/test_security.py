import hashlib
import hmac
import io
import json
import os
import zipfile
from pathlib import Path

import httpx
import pytest

from app.notify import deliver_webhook, sign
from app.pipeline.errors import CorruptArchive, UnsafeArchive
from app.pipeline.ingest.filetypes import validate_upload
from app.pipeline.ingest.safe_zip import ZipLimits, project_root, safe_extract
from app.ssrf import SSRFError, resolve_public_ips, safe_get, validate_url

LIMITS = ZipLimits(max_total_bytes=10 * 1024 * 1024, max_entry_bytes=5 * 1024 * 1024, max_ratio=100, max_entries=100)


def zip_with(entries: list[tuple[str, bytes]], path: Path) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return path


# ---------------------------------------------------------------- zip hardening
def test_safe_extract_normal_archive(tmp_path):
    z = zip_with([("proj/a.py", b"print(1)"), ("proj/sub/b.txt", b"hi")], tmp_path / "ok.zip")
    report = safe_extract(z, tmp_path / "out", LIMITS)
    assert report.files == 2 and (tmp_path / "out/proj/sub/b.txt").read_text() == "hi"
    assert project_root(tmp_path / "out").name == "proj"


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/etc/passwd", "C:/Windows/evil.txt", "..\\evil.txt", "ok/..\\..\\evil.txt"])
def test_zip_slip_variants_rejected(tmp_path, name):
    z = zip_with([("fine.txt", b"x"), (name, b"pwn")], tmp_path / "slip.zip")
    with pytest.raises(UnsafeArchive):
        safe_extract(z, tmp_path / "out", LIMITS)
    assert not (tmp_path / "evil.txt").exists() and not Path("/etc/passwd.pwn").exists()


def test_symlink_entries_are_skipped_not_followed(tmp_path):
    path = tmp_path / "sym.zip"
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "/etc/passwd")
        zf.writestr("real.txt", "ok")
    report = safe_extract(path, tmp_path / "out", LIMITS)
    assert not (tmp_path / "out/link").exists() and (tmp_path / "out/real.txt").exists()
    assert any("symlink" in s for s in report.skipped)


def test_zip_bomb_by_ratio(tmp_path):
    z = zip_with([("zeros.bin", b"\0" * (8 * 1024 * 1024))], tmp_path / "bomb.zip")
    with pytest.raises(UnsafeArchive, match="compression ratio"):
        safe_extract(z, tmp_path / "out", ZipLimits(100 * 1024 * 1024, 100 * 1024 * 1024, 50, 100))


def test_zip_bomb_by_total_and_entry_size_and_count(tmp_path):
    z = zip_with([(f"f{i}.txt", b"x" * 2000) for i in range(5)], tmp_path / "a.zip")
    with pytest.raises(UnsafeArchive, match="expand"):
        safe_extract(z, tmp_path / "o1", ZipLimits(5000, 5000, 1000, 100))
    with pytest.raises(UnsafeArchive, match="too many entries"):
        safe_extract(z, tmp_path / "o2", ZipLimits(10**9, 10**9, 1000, 3))
    big = zip_with([("big.txt", os.urandom(300_000))], tmp_path / "b.zip")
    with pytest.raises(UnsafeArchive, match="per-file"):
        safe_extract(big, tmp_path / "o3", ZipLimits(10**9, 100_000, 1000, 100))


def test_declared_size_lies_are_caught_by_counting_real_bytes(tmp_path):
    """Header says 10 bytes, payload is bigger: only real streamed bytes are trusted."""
    z = zip_with([("liar.txt", b"y" * 400_000)], tmp_path / "liar.zip")
    raw = bytearray(z.read_bytes())
    # patch the declared uncompressed size (local header + central directory) down to 10
    size = (400_000).to_bytes(4, "little")
    small = (10).to_bytes(4, "little")
    z.write_bytes(bytes(raw).replace(size, small))
    with pytest.raises((UnsafeArchive, CorruptArchive)):
        safe_extract(z, tmp_path / "out", ZipLimits(10**9, 1000, 10**6, 100))


def test_nested_archives_are_not_extracted_and_encrypted_rejected(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("x.txt", "x")
    z = zip_with([("proj/inner.zip", inner.getvalue()), ("proj/a.txt", b"a")], tmp_path / "n.zip")
    report = safe_extract(z, tmp_path / "out", LIMITS)
    assert not (tmp_path / "out/proj/inner.zip").exists() and any("nested archive" in s for s in report.skipped)

    enc = zip_with([("s.txt", b"secret")], tmp_path / "e.zip")
    raw = bytearray(enc.read_bytes())
    for i in range(len(raw) - 4):  # set the "encrypted" general-purpose flag on local + central headers
        if raw[i:i + 4] == b"PK\x03\x04":
            raw[i + 6] |= 1
        if raw[i:i + 4] == b"PK\x01\x02":
            raw[i + 8] |= 1
    enc.write_bytes(bytes(raw))
    with pytest.raises(UnsafeArchive, match="encrypted"):
        safe_extract(enc, tmp_path / "out2", LIMITS)


def test_garbage_is_a_corrupt_archive(tmp_path):
    p = tmp_path / "bad.zip"
    p.write_bytes(b"PK\x03\x04" + b"junk" * 50)
    with pytest.raises(CorruptArchive):
        safe_extract(p, tmp_path / "out", LIMITS)


def test_magic_byte_validation():
    assert validate_upload("zip", b"PK\x03\x04rest") is None
    assert "not a ZIP" in validate_upload("zip", b"%PDF-1.4")
    assert "executable" in validate_upload("zip", b"MZ\x90\x00")
    assert "executable" in validate_upload("docs", b"\x7fELF\x02")
    assert validate_upload("pdf", b"%PDF-1.7\n") is None and "not a PDF" in validate_upload("pdf", b"hello")
    assert validate_upload("video", b"\x00\x00\x00\x18ftypmp42") is None and validate_upload("video", b"PK\x03\x04") is not None
    assert validate_upload("docs", b"# readme\n") is None and validate_upload("docs", b"\x00\x01\x02\x03binary") is not None


# ---------------------------------------------------------------- SSRF
def fake_resolver(mapping):
    def resolver(host, port, type=None):
        ips = mapping[host]
        return [(2, 1, 6, "", (ip, port)) for ip in ips]

    return resolver


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.1.2.3", "172.16.0.5", "192.168.1.1", "169.254.169.254", "0.0.0.0", "100.64.0.1",
    "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "::ffff:169.254.169.254", "224.0.0.1",
])
def test_non_public_addresses_blocked_as_literals(ip):
    with pytest.raises(SSRFError):
        resolve_public_ips(ip, 80)


def test_hostname_resolving_to_internal_ip_is_blocked_including_mixed_answers():
    with pytest.raises(SSRFError):
        resolve_public_ips("evil.example", 80, fake_resolver({"evil.example": ["127.0.0.1"]}))
    with pytest.raises(SSRFError):  # one good + one internal answer must still be rejected
        resolve_public_ips("mixed.example", 80, fake_resolver({"mixed.example": ["93.184.216.34", "10.0.0.5"]}))
    assert resolve_public_ips("ok.example", 80, fake_resolver({"ok.example": ["93.184.216.34"]})) == ["93.184.216.34"]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/1", "ftp://x/f", "http://user:pw@example.com/", "http:///nohost", "http://example.com:99999/", "javascript:alert(1)"])
def test_bad_urls_rejected(url):
    with pytest.raises(SSRFError):
        validate_url(url)


def test_allowed_hosts_enforced():
    with pytest.raises(SSRFError):
        validate_url("https://github.com.evil.com/x/y", allowed_hosts={"github.com"})
    assert validate_url("https://GitHub.com/x/y", allowed_hosts={"github.com"}).host == "github.com"


def test_connection_is_pinned_to_the_validated_ip_and_keeps_host_header():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"], seen["host"], seen["sni"] = str(request.url), request.headers["host"], request.extensions.get("sni_hostname")
        return httpx.Response(200, text="hello")

    r = safe_get("https://app.example/path?q=1", resolver=fake_resolver({"app.example": ["93.184.216.34"]}), transport=httpx.MockTransport(handler))
    assert r.status == 200 and r.body == b"hello"
    assert seen["url"].startswith("https://93.184.216.34/path?q=1") and seen["host"] == "app.example" and seen["sni"] == "app.example"


def test_redirect_to_internal_address_is_blocked():
    def handler(request: httpx.Request):
        return httpx.Response(302, headers={"location": "http://metadata.internal/latest/meta-data/"})

    resolver = fake_resolver({"app.example": ["93.184.216.34"], "metadata.internal": ["169.254.169.254"]})
    with pytest.raises(SSRFError, match="routable"):
        safe_get("http://app.example/", resolver=resolver, transport=httpx.MockTransport(handler))


def test_redirect_loop_and_response_cap():
    def loop(request):
        return httpx.Response(302, headers={"location": "/again"})

    resolver = fake_resolver({"a.example": ["93.184.216.34"]})
    with pytest.raises(SSRFError, match="too many redirects"):
        safe_get("http://a.example/", resolver=resolver, transport=httpx.MockTransport(loop))
    big = httpx.MockTransport(lambda req: httpx.Response(200, content=b"x" * 10_000))
    r = safe_get("http://a.example/", resolver=resolver, transport=big, max_bytes=1000)
    assert len(r.body) == 1000 and r.truncated
    with pytest.raises(SSRFError, match="too large"):
        safe_get("http://a.example/", resolver=resolver, transport=big, max_bytes=1000, overflow="error")


# ---------------------------------------------------------------- webhooks
def test_webhook_is_signed_over_the_exact_bytes_and_dedupable(client, world):
    from tests.conftest import drain

    received = {}

    def handler(request: httpx.Request):
        received.update(body=request.content, headers=dict(request.headers))
        return httpx.Response(200)

    created = world.submit_zip(webhook_url="https://hooks.example/evaluation")
    drain()
    resolver = fake_resolver({"hooks.example": ["93.184.216.34"]})
    assert deliver_webhook(created["job_id"], transport=httpx.MockTransport(handler), resolver=resolver) is True
    ts = received["headers"]["x-evalplatform-timestamp"]
    expected = "sha256=" + hmac.new(b"dev-webhook-secret", ts.encode() + b"." + received["body"], hashlib.sha256).hexdigest()
    assert received["headers"]["x-evalplatform-signature"] == expected == sign("dev-webhook-secret", ts, received["body"])
    payload = json.loads(received["body"])
    assert payload["event_id"] == f"evt_{created['job_id']}_SUCCEEDED" and "score" not in received["body"].decode()


def test_webhook_to_internal_destination_is_refused_and_retries_on_5xx(client, world):
    from tests.conftest import drain

    created = world.submit_zip(webhook_url="https://hooks.example/x")
    drain()
    calls = []

    def flaky(request):
        calls.append(1)
        return httpx.Response(503)

    sleeps = []
    ok = deliver_webhook(created["job_id"], transport=httpx.MockTransport(flaky), sleep=sleeps.append, resolver=fake_resolver({"hooks.example": ["93.184.216.34"]}))
    assert ok is False and len(calls) == 3 and sleeps == [2, 4]
    assert deliver_webhook(created["job_id"], transport=httpx.MockTransport(flaky), resolver=fake_resolver({"hooks.example": ["10.0.0.9"]})) is False
    assert len(calls) == 3  # never connected to the internal address


def test_webhook_url_to_private_literal_rejected_at_submission(client, world):
    body = {"assignment_id": world.assignment_id, "sources": [{"type": "live_url", "url": "https://example.com"}], "webhook_url": "http://169.254.169.254/latest"}
    r = client.post("/api/v1/submissions", json=body, headers={**world.h(world.student), "Idempotency-Key": "webhook-key-01"})
    assert r.status_code == 422
