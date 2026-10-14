"""Single small loop: outbox relay (every second) + stale-lease reaper (every 30s).
Both are idempotent and use row-level skip-locked / guarded updates, so running two copies is safe.

    python -m app.scheduler
"""

import logging
import signal
import threading
import time

from app.config import get_settings
from app.logging_setup import setup_logging
from app.outbox import relay_once
from app.pipeline.runner import recover_all_stale_jobs

log = logging.getLogger("scheduler")


def run_forever(stop: threading.Event | None = None) -> None:
    s = get_settings()
    stop = stop or threading.Event()
    last_reap = 0.0
    while not stop.is_set():
        try:
            while relay_once() >= 50:
                pass
            if time.monotonic() - last_reap >= s.reaper_interval_s:
                last_reap = time.monotonic()
                recovered = recover_all_stale_jobs(s)
                if recovered:
                    log.warning("recovered %d stale job(s)", recovered)
        except Exception:
            log.exception("scheduler iteration failed")
        stop.wait(s.relay_poll_interval_s)


if __name__ == "__main__":
    setup_logging()
    stop_event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())
    run_forever(stop_event)
