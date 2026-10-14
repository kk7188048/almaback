"""Stage checkpoints in object storage, keyed by job (not attempt) so a retry attempt resumes where the last stopped.
This is what keeps a worker crash from re-paying for LLM calls that already completed."""

import json
from collections.abc import Callable
from typing import Any

from app.storage import Storage, StorageError


class Checkpoints:
    def __init__(self, storage: Storage, job_id: str):
        self.storage = storage
        self.prefix = f"jobs/{job_id}/ckpt"

    def key(self, name: str) -> str:
        return f"{self.prefix}/{name}.json"

    def get(self, name: str) -> Any | None:
        try:
            return json.loads(self.storage.get_bytes(self.key(name)))
        except (StorageError, ValueError):
            return None

    def put(self, name: str, value: Any) -> None:
        self.storage.put_bytes(self.key(name), json.dumps(value).encode())

    def get_or_compute(self, name: str, fn: Callable[[], Any]) -> Any:
        cached = self.get(name)
        if cached is not None:
            return cached
        value = fn()
        self.put(name, value)
        return value
