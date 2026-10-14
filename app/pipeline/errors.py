from datetime import datetime


class PipelineError(Exception):
    """An expected failure with a stable code. `retryable` decides whether another attempt can help."""

    code = "pipeline_error"
    retryable = False

    def __init__(self, message: str, *, code: str | None = None, retryable: bool | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable


class IngestError(PipelineError):
    code = "ingest_error"


class CorruptArchive(IngestError):
    code = "corrupt_archive"


class UnsafeArchive(IngestError):
    code = "unsafe_archive"


class MalwareDetected(IngestError):
    code = "malware_detected"


class UnsupportedFile(IngestError):
    code = "unsupported_file"


class SourceUnavailable(IngestError):
    """Remote source (repo, drive, url) is missing/private/too big: the student must fix it."""

    code = "source_unavailable"


class TransientIngestError(IngestError):
    code = "transient_ingest_error"
    retryable = True


class DeferJob(Exception):
    """Put the job back on the queue until `until` without counting it as a failure (e.g. daily LLM budget)."""

    def __init__(self, reason: str, until: datetime):
        super().__init__(reason)
        self.reason = reason
        self.until = until


class LeaseLost(Exception):
    """Another worker took over this job; stop without touching state."""
