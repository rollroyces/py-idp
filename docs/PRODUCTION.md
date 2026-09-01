# Production deployment guide

This document describes how to run py-idp in a production environment.
It assumes you're comfortable with Docker, Linux, and basic ops.

> For local development and quick demos, the README + `examples/api.py`
> are still the right starting points.

## 1. Container build

```bash
docker build -t py-idp:0.2.0 .
docker run --rm -p 8080:8080 \
  -e IDP_API_KEY="$(openssl rand -hex 32)" \
  -e IDP_STORAGE_BACKEND=sql \
  -e IDP_STORAGE_PATH=sqlite:////data/idp.db \
  -v idp-data:/data \
  py-idp:0.2.0
```

A multi-stage Dockerfile ships in this repo. It:
- uses a slim `python:3.12-slim` base
- creates a non-root `idp` user
- sets `PYTHONUNBUFFERED=1` for real-time logs
- uses `pip install --no-cache-dir` to keep the image small

## 2. Configuration

Every runtime setting reads from an `IDP_*` environment variable.
**All variables are validated at startup**; invalid values abort the
process with a clear error so misconfiguration never reaches production
traffic.

| Variable | Default | Notes |
|---|---|---|
| `IDP_API_HOST` | `0.0.0.0` | bind address |
| `IDP_API_PORT` | `8080` | 1-65535 |
| `IDP_API_WORKERS` | `1` | uvicorn workers (use 2× CPU cores) |
| `IDP_API_KEY` | — | required if `IDP_API_KEY_REQUIRED=true` |
| `IDP_API_KEY_REQUIRED` | `true` | fail-closed if `IDP_API_KEY` unset |
| `IDP_RATE_LIMIT_PER_MINUTE` | `60` | per-key; `0` = unlimited |
| `IDP_STORAGE_BACKEND` | `memory` | `memory` / `json` / `sql` |
| `IDP_STORAGE_PATH` | — | JSON path or `sqlite:///…` / `postgresql://…` |
| `IDP_BACKEND` | `mock` | default LLM backend |
| `IDP_MODEL` | — | default model |
| `IDP_LLM_TIMEOUT` | `60.0` | seconds |
| `IDP_LLM_MAX_RETRIES` | `2` | exponential backoff |
| `IDP_MAX_UPLOAD_BYTES` | `26214400` (25 MB) | enforced on stream |
| `IDP_MAX_PDF_PAGES` | `100` | parser cap |
| `IDP_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` |
| `IDP_LOG_FORMAT` | `human` | `human` or `json` (for log aggregators) |
| `IDP_METRICS_ENABLED` | `true` | expose `/metrics` |
| `IDP_CORS_ORIGINS` | — | comma-separated list of allowed origins |
| `IDP_UPLOAD_DIR` | `/tmp/idp-uploads` | persistent volume recommended |

## 3. Health & readiness

The API exposes three liveness/readiness endpoints:

* `GET /healthz` — always returns `200 OK` if the process is up.
  Wire to your k8s liveness probe (`periodSeconds: 10`).
* `GET /readyz` — returns `200 ready` after Settings have loaded;
  `503` before. Wire to readiness probe (`periodSeconds: 5`,
  `failureThreshold: 3`).
* `GET /version` — returns the package version. Useful for ops
  dashboards.

Example k8s manifest:

```yaml
livenessProbe:
  httpGet: { path: /healthz, port: 8080 }
  periodSeconds: 10
readinessProbe:
  httpGet: { path: /readyz, port: 8080 }
  periodSeconds: 5
  failureThreshold: 3
```

## 4. Metrics

`GET /metrics` returns Prometheus text-format metrics. Current series:

- `http_requests{method,path}` — request count
- `http_responses{status,path}` — response count
- `http_errors{type}` — error count by exception type
- `http_request_duration_seconds{path}` — request latency histogram
- `extractions{backend,schema}` — extraction count

Example Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: py-idp
    metrics_path: /metrics
    static_configs:
      - targets: ['py-idp:8080']
```

## 5. Logging

Set `IDP_LOG_FORMAT=json` for log aggregators (Datadog, Loki, ELK).
Each request is logged with `method`, `path`, `status`, and `duration`.

```json
{"ts": "2026-09-01T12:00:00Z", "level": "INFO", "logger": "idp.api",
 "msg": "py-idp API starting: host=0.0.0.0 port=8080 storage=sql backend=openai"}
```

## 6. Security

* **API key**: always set `IDP_API_KEY` in production. Generate with
  `openssl rand -hex 32`. Rotate every 90 days.
* **HTTPS**: terminate TLS at the reverse proxy (nginx, Caddy,
  Cloudflare). Do not expose the API on the public Internet without it.
* **CORS**: only set `IDP_CORS_ORIGINS` for known origins. Default
  (empty) blocks all cross-origin requests.
* **Upload size**: the default 25 MB cap protects the LLM backend
  from OOM. Tune via `IDP_MAX_UPLOAD_BYTES`.
* **Rate limit**: default 60 req/min per key. Tune via
  `IDP_RATE_LIMIT_PER_MINUTE`. For multi-instance deployments,
  replace the in-process limiter with a Redis-backed one.
* **Storage**: SQLite works for a single instance. Use Postgres
  (`postgresql://user:pass@host/db`) for multi-instance. Install the
  `py-idp[sql]` extra for the `psycopg` driver.

## 7. Scaling

py-idp is **CPU-light and I/O-bound on the LLM call** (per the
2026-08 perf benchmark: 100% of wall time is the LLM call).

* **Vertical scaling** is bounded by LLM provider throughput.
* **Horizontal scaling**: deploy multiple uvicorn workers behind a
  load balancer. Each worker keeps its own in-process rate limiter
  and metrics — use Redis-backed shared counters for accurate
  rate limiting across replicas.
* **Caching**: LLM response caching is NOT shipped (each doc is
  unique in practice). If you have repeated docs, cache by
  `(sha256(document_bytes), schema_name)` and reuse results.

## 8. Monitoring alerts

Recommended Prometheus alerts:

```yaml
- alert: PyIdpHighErrorRate
  expr: rate(http_errors[5m]) > 0.1
  for: 10m
- alert: PyIdpSlowRequests
  expr: histogram_quantile(0.95, http_request_duration_seconds) > 30
  for: 5m
- alert: PyIdpRateLimited
  expr: rate(http_responses{status="429"}[5m]) > 10
  for: 10m
```

## 9. Troubleshooting

**App refuses to start** with a `ConfigurationError`:
read the error, fix the env var, redeploy. Examples:
- `IDP_API_PORT must be an integer, got 'not-a-number'`
- `IDP_LOG_LEVEL='TRACE' not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')`
- `IDP_MAX_UPLOAD_BYTES=100 out of range [1024, 1073741824]`

**`429 Too Many Requests`**: rate limit hit. Increase
`IDP_RATE_LIMIT_PER_MINUTE`, or add a Redis-backed limiter for
multi-instance deployments.

**`413 Payload Too Large`**: upload exceeds cap. Increase
`IDP_MAX_UPLOAD_BYTES` or reject at the proxy.

**`503 Not Ready`**: settings are still loading. Wait a moment, or
check the startup logs for a `ConfigurationError`.

**`503 Service Unavailable`**: API key is required (`IDP_API_KEY_REQUIRED=true`)
but `IDP_API_KEY` is unset. Set the env var.

## 10. Upgrade path

* 0.1 → 0.2: schema required-fields change is a breaking change for
  callers that relied on `validate(extraction={}, schema='Invoice')`
  returning `passed=True`. Update your callers, or wrap with a custom
  Pydantic schema that has only optional fields.