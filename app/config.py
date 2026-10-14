from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MB = 1024 * 1024
GB = 1024 * MB


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"
    public_base_url: str = "http://localhost:8000"
    cors_origins: str = ""  # comma-separated origins of the student portal / evaluator dashboard

    database_url: str = "sqlite:///./dev.db"
    use_pgvector: bool = True
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret: str = "dev-only-change-me-dev-only-change-me"
    access_token_ttl_s: int = 15 * 60
    refresh_token_ttl_s: int = 7 * 24 * 3600
    webhook_signing_secret: str = "dev-webhook-secret"
    upload_signing_secret: str = "dev-upload-secret"

    # storage: "local" (dev/tests) or "s3" (AWS S3 / MinIO via s3_endpoint_url)
    storage_backend: str = "local"
    local_storage_dir: str = "./.storage"
    s3_bucket: str = "evalplatform-submissions"
    s3_endpoint_url: str | None = None
    s3_region: str = "ap-south-1"  # Cloudflare R2: "auto"
    s3_presign_mode: str = "post"  # "post": AWS S3 (server-enforced size policy). "put": Cloudflare R2 / other S3-compatibles without POST policies
    presign_ttl_s: int = 900
    multipart_threshold_bytes: int = 100 * MB
    multipart_part_size: int = 64 * MB

    # queue: "celery" (Redis broker) or "memory" (tests)
    queue_backend: str = "celery"
    celery_visibility_timeout_s: int = 2 * 3600
    job_time_limit_s: int = 45 * 60

    # worker / reliability
    lease_ttl_s: int = 120
    heartbeat_interval_s: int = 30
    max_attempts: int = 3
    retry_backoff_base_s: int = 30
    relay_poll_interval_s: float = 1.0
    reaper_interval_s: int = 30

    # upload / ingest limits
    max_sources: int = 8
    max_zip_bytes: int = 2 * GB
    max_pdf_bytes: int = 100 * MB
    max_docs_bytes: int = 50 * MB
    max_video_bytes: int = 500 * MB
    zip_max_uncompressed_bytes: int = 3 * GB
    zip_max_entry_bytes: int = 512 * MB
    zip_max_ratio: int = 200
    zip_max_entries: int = 100_000
    github_max_repo_bytes: int = 1 * GB
    github_clone_timeout_s: int = 600
    drive_max_files: int = 200
    drive_max_total_bytes: int = 500 * MB
    live_url_max_bytes: int = 2 * MB
    fetch_timeout_s: float = 20.0
    clamd_host: str | None = None
    clamd_port: int = 3310
    clamav_required: bool = False

    # integrations (all optional)
    github_app_id: str | None = None
    github_app_private_key: str | None = None
    google_api_key: str | None = None
    transcription_base_url: str | None = None
    transcription_api_key: str | None = None
    transcription_model: str = "whisper-large-v3"
    semgrep_enabled: bool = False
    semgrep_config: str = "p/default"
    pip_audit_enabled: bool = False

    # context pack
    context_max_tokens: int = 60_000
    repo_map_tokens: int = 12_000
    readme_tokens: int = 6_000
    max_text_file_bytes: int = 512 * 1024
    max_parse_file_bytes: int = 300 * 1024
    max_walk_files: int = 30_000
    max_embedding_chunks: int = 1500
    embedding_dim: int = 256
    retrieval_top_k: int = 6
    summarize_top_code_files: int = 12
    summarize_batch_chars: int = 18_000

    # llm
    llm_provider: str = "mock"  # mock | gemini | openai_compat | bedrock
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model_cheap: str = "mock-cheap"
    llm_model_standard: str = "mock-standard"
    llm_model_premium: str = "mock-premium"
    llm_max_concurrency: int = 4
    llm_timeout_s: float = 90.0
    llm_max_retries: int = 4
    llm_backoff_base_s: float = 2.0
    llm_backoff_max_s: float = 60.0
    llm_max_repairs: int = 2
    llm_breaker_threshold: int = 6
    llm_breaker_cooldown_s: float = 60.0
    llm_prompt_cache: bool = False  # Bedrock cachePoint after the shared context
    price_in_per_mtok: dict[str, float] = Field(default_factory=dict)
    price_out_per_mtok: dict[str, float] = Field(default_factory=dict)

    # evaluation policy
    sc_confidence_threshold: float = 0.7
    sc_borderline_low: float = 0.45
    sc_borderline_high: float = 0.55
    sc_extra_samples: int = 2
    sc_agreement_tolerance: float = 0.10
    review_confidence_threshold: float = 0.6
    review_agreement_threshold: float = 0.67
    attempt_token_budget: int = 600_000
    global_daily_token_budget: int = 20_000_000

    # abuse controls
    rate_limit_capacity: int = 60
    rate_limit_per_s: float = 2.0
    login_rate_capacity: int = 10
    login_rate_per_s: float = 0.2
    daily_submission_cap: int = 20
    max_evaluations_per_submission_per_day: int = 5
    student_sees_confidence: bool = False


_DEV_SECRETS = {"dev-only-change-me-dev-only-change-me", "dev-webhook-secret", "dev-upload-secret"}


def assert_production_ready(s: Settings) -> None:
    """Refuse to boot a production deployment that still uses the built-in development secrets."""
    problems = []
    for name in ("jwt_secret", "webhook_signing_secret", "upload_signing_secret"):
        value = getattr(s, name)
        if value in _DEV_SECRETS or len(value) < 32:
            problems.append(f"{name.upper()} must be a unique random value of at least 32 characters")
    if s.llm_provider == "mock":
        problems.append("LLM_PROVIDER=mock is not allowed in production")
    if problems:
        raise RuntimeError("unsafe production configuration: " + "; ".join(problems))


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
