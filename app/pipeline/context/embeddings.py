"""Chunking, a local feature-hashing embedder (free, offline, lexical) and cosine similarity.
Swap HashEmbedder for a hosted model if semantic similarity becomes important.
"""

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        out.append(tok.lower())
        parts = [p.lower() for p in _CAMEL.split(tok) if len(p) > 1]
        if len(parts) > 1:
            out.extend(parts)
    return out


class HashEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        vec = np.sign(vec) * np.sqrt(np.abs(vec))
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm else vec.tolist()


@dataclass
class Chunk:
    ref: str  # path:start-end
    text: str


def chunk_text(path: str, text: str, max_chars: int = 1500, overlap_lines: int = 3) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        size = 0
        end = start
        while end < len(lines) and size + len(lines[end]) + 1 <= max_chars:
            size += len(lines[end]) + 1
            end += 1
        if end == start:  # a single very long line
            end = start + 1
        body = "\n".join(lines[start:end])[:max_chars]
        if body.strip():
            chunks.append(Chunk(f"{path}:{start + 1}-{end}", body))
        if end >= len(lines):
            break
        start = max(end - overlap_lines, start + 1)
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)
