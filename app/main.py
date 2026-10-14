import logging
import time
import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, courses, jobs, ops, reviews, submissions
from app.config import assert_production_ready, get_settings
from app.deps import rate_limit
from app.logging_setup import request_id_var, setup_logging

log = logging.getLogger("api")


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()
    if settings.env == "prod":
        assert_production_ready(settings)
    app = FastAPI(title="AI Project Evaluation Platform", version="1.0.0")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"ctx": {"path": request.url.path}})
            response = JSONResponse({"detail": "internal server error"}, status_code=500)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
        if not request.url.path.startswith(("/healthz", "/readyz")):
            log.info("request", extra={"ctx": {"method": request.method, "path": request.url.path, "status": response.status_code, "ms": int((time.monotonic() - started) * 1000)}})
        return response

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
            allow_methods=["GET", "POST", "PUT"], allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
            expose_headers=["Retry-After", "Idempotent-Replay", "X-Request-ID"], max_age=600,
        )

    limited = [Depends(rate_limit)]
    app.include_router(ops.router)
    for module in (auth, courses, submissions, jobs, reviews):
        app.include_router(module.router, dependencies=limited)
    return app


app = create_app()
