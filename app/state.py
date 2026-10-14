"""Job status machine. The only legal status changes are listed in ALLOWED."""

AWAITING_UPLOAD = "AWAITING_UPLOAD"
QUEUED = "QUEUED"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
NEEDS_REVIEW = "NEEDS_REVIEW"

ALL_STATUSES = (AWAITING_UPLOAD, QUEUED, RUNNING, SUCCEEDED, FAILED, NEEDS_REVIEW)
ACTIVE_STATUSES = (AWAITING_UPLOAD, QUEUED, RUNNING)

ALLOWED: dict[str, set[str]] = {
    AWAITING_UPLOAD: {QUEUED, FAILED},
    QUEUED: {RUNNING, FAILED},
    RUNNING: {SUCCEEDED, FAILED, NEEDS_REVIEW, QUEUED},
    NEEDS_REVIEW: {SUCCEEDED},
    SUCCEEDED: set(),
    FAILED: set(),
}

STAGES = ("ingesting", "analyzing", "llm_evaluation", "aggregating", "persisting")


class IllegalTransition(Exception):
    pass


def assert_transition(old: str, new: str) -> None:
    if new not in ALLOWED.get(old, set()):
        raise IllegalTransition(f"{old} -> {new} is not allowed")
