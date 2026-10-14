"""Leaked-secret detection and redaction. Secrets must never reach the LLM, logs, or the context pack."""

import re

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_live_key", re.compile(r"\b[sr]k_live_[0-9a-zA-Z]{20,}\b")),
    ("generic_secret_assignment", re.compile(r"""(?i)\b(?:api[_-]?key|secret|passwd|password|token)\b\s*[:=]\s*['"][A-Za-z0-9_\-/+=]{16,}['"]""")),
]
_REDACTION = "[REDACTED-SECRET]"


def scan_text(path: str, text: str, limit: int = 50) -> list[dict]:
    findings: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if len(line) > 2000:
            continue
        for rule, pattern in PATTERNS:
            if pattern.search(line):
                findings.append({"path": path, "line": lineno, "rule": rule})
                if len(findings) >= limit:
                    return findings
    return findings


def redact(text: str) -> str:
    for _rule, pattern in PATTERNS:
        text = pattern.sub(_REDACTION, text)
    return text
