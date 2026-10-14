"""SSRF-safe outbound HTTP for every URL a student controls (live app, webhook, Drive, GitHub).
The host is resolved once and each address must be public; the connection is pinned to that IP (defeats DNS rebinding);
redirects are re-validated per hop and responses are size-capped.
"""

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

Resolver = Callable[..., list]


class SSRFError(ValueError):
    pass


@dataclass
class FetchResult:
    status: int
    headers: dict[str, str]
    body: bytes
    final_url: str
    truncated: bool = False


def _check_ip(raw: str) -> str:
    ip = ipaddress.ip_address(raw)
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_unspecified:  # is_global alone admits multicast
        raise SSRFError("destination address is not publicly routable")
    return str(ip)


def resolve_public_ips(host: str, port: int, resolver: Resolver | None = None) -> list[str]:
    try:
        return [_check_ip(host)]
    except ValueError as exc:
        if isinstance(exc, SSRFError):
            raise
    try:
        infos = (resolver or socket.getaddrinfo)(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SSRFError(f"cannot resolve host: {host}") from exc
    ips = []
    for info in infos:
        ip = _check_ip(info[4][0])
        if ip not in ips:
            ips.append(ip)
    if not ips:
        raise SSRFError(f"no addresses for host: {host}")
    return ips


@dataclass
class ParsedURL:
    scheme: str
    host: str
    port: int
    path_query: str
    explicit_port: bool


def validate_url(url: str, *, allowed_hosts: set[str] | None = None, schemes: tuple[str, ...] = ("http", "https")) -> ParsedURL:
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as exc:
        raise SSRFError("malformed URL") from exc
    if parts.scheme not in schemes:
        raise SSRFError("URL scheme not allowed")
    if not parts.hostname:
        raise SSRFError("URL has no host")
    if "@" in parts.netloc:
        raise SSRFError("credentials in URL are not allowed")
    host = parts.hostname.lower().rstrip(".")
    if allowed_hosts is not None and host not in allowed_hosts:
        raise SSRFError(f"host not allowed: {host}")
    default_port = 443 if parts.scheme == "https" else 80
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    return ParsedURL(parts.scheme, host, port or default_port, path, port is not None)


def safe_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: object | None = None,
    content: bytes | None = None,
    max_bytes: int = 2 * 1024 * 1024,
    overflow: str = "truncate",  # "truncate" | "error"
    timeout: float = 20.0,
    allowed_hosts: set[str] | None = None,
    max_redirects: int = 3,
    resolver: Resolver | None = None,
    transport: httpx.BaseTransport | None = None,
) -> FetchResult:
    current = url
    method = method.upper()
    for _hop in range(max_redirects + 1):
        parsed = validate_url(current, allowed_hosts=allowed_hosts)
        ips = resolve_public_ips(parsed.host, parsed.port, resolver)
        last_exc: Exception | None = None
        response: FetchResult | None = None
        for ip in ips:
            host_literal = f"[{ip}]" if ":" in ip else ip
            pinned = f"{parsed.scheme}://{host_literal}:{parsed.port}{parsed.path_query}"
            req_headers = {"User-Agent": "almabetter-eval/1.0", **(headers or {})}
            req_headers["Host"] = parsed.host + (f":{parsed.port}" if parsed.explicit_port else "")
            try:
                with httpx.Client(transport=transport, timeout=timeout, follow_redirects=False) as client:
                    with client.stream(
                        method,
                        pinned,
                        headers=req_headers,
                        json=json_body,
                        content=content,
                        extensions={"sni_hostname": parsed.host},
                    ) as resp:
                        body = bytearray()
                        truncated = False
                        if resp.status_code not in (301, 302, 303, 307, 308):
                            for chunk in resp.iter_bytes():
                                body.extend(chunk)
                                if len(body) > max_bytes:
                                    if overflow == "error":
                                        raise SSRFError("response too large")
                                    del body[max_bytes:]
                                    truncated = True
                                    break
                        response = FetchResult(
                            resp.status_code,
                            {k.lower(): v for k, v in resp.headers.items()},
                            bytes(body),
                            current,
                            truncated,
                        )
                break
            except (httpx.TransportError, OSError) as exc:
                last_exc = exc
                continue
        if response is None:
            raise SSRFError(f"could not connect: {last_exc!r}")
        if response.status in (301, 302, 303, 307, 308) and "location" in response.headers:
            current = urljoin(current, response.headers["location"])
            if response.status in (301, 302, 303):
                method, json_body, content = "GET", None, None
            continue
        return response
    raise SSRFError("too many redirects")


def safe_get(url: str, **kwargs) -> FetchResult:
    return safe_request("GET", url, **kwargs)
