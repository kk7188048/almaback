# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
# all dependencies ship manylinux wheels, so no compiler toolchain is needed in any stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
ARG INSTALL_FFMPEG=false
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates tini \
 && if [ "$INSTALL_FFMPEG" = "true" ]; then apt-get install -y --no-install-recommends ffmpeg; fi \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --create-home --uid 10001 app \
 && mkdir -p /data/storage && chown -R 10001 /data
COPY --from=builder /install /usr/local
WORKDIR /srv
COPY alembic.ini .
COPY migrations migrations
COPY app app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 10001
EXPOSE 8000
ENTRYPOINT ["tini", "--"]
# API by default; the worker and scheduler services override the command (see docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
