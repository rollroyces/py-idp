# Production Dockerfile for py-idp
#
# Build:  docker build -t py-idp:0.2.0 .
# Run:    docker run --rm -p 8080:8080 \
#           -e IDP_API_KEY=... \
#           -e IDP_STORAGE_BACKEND=sql \
#           -e IDP_STORAGE_PATH=sqlite:////data/idp.db \
#           -v idp-data:/data \
#           py-idp:0.2.0

# ---- Build stage (deps only; cached separately from source) ----
FROM python:3.12-slim AS deps

WORKDIR /app

# Install build deps (none currently required; future-proofs native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY examples ./examples

# Install with the [api] extra (FastAPI + uvicorn). --no-cache-dir keeps the
# image small; --no-compile keeps .pyc files out of the layer.
RUN pip install --no-cache-dir --no-compile -e .[api]

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

# Hardening: run as a non-root user. Bind to a fixed UID so volume mounts
# from the host work predictably across deployments.
RUN groupadd --system --gid 1000 idp \
    && useradd  --system --uid 1000 --gid idp --home-dir /app --shell /sbin/nologin idp

WORKDIR /app
COPY --from=deps --chown=idp:idp /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps --chown=idp:idp /usr/local/bin /usr/local/bin
COPY --from=deps --chown=idp:idp /app/src /app/src
COPY --from=deps --chown=idp:idp /app/examples /app/examples

# Persistent storage for SQLite + uploads. Mount a volume here in prod.
RUN mkdir -p /data/uploads && chown -R idp:idp /data
ENV IDP_UPLOAD_DIR=/data/uploads

# Real-time logs (no buffering) so container log collectors see entries
# immediately.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IDP_API_HOST=0.0.0.0 \
    IDP_API_PORT=8080 \
    IDP_LOG_LEVEL=INFO \
    IDP_LOG_FORMAT=human \
    IDP_API_KEY_REQUIRED=true \
    IDP_STORAGE_BACKEND=memory \
    IDP_RATE_LIMIT_PER_MINUTE=60

USER idp
EXPOSE 8080

# Healthcheck hits /healthz (cheap, no LLM call). 30s interval, 5s timeout,
# 3 retries before marking unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).read()" \
    || exit 1

# Run uvicorn directly. Single worker is correct for development; for prod
# use --workers $(( 2 * $(nproc) )) or rely on the orchestrator (k8s) to
# scale replicas.
CMD ["uvicorn", "idp.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--log-level", "info"]