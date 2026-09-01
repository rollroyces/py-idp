# Multi-region & high-availability deployment

This document covers deploying py-idp across multiple regions / zones
for resilience. It assumes you're comfortable with k8s, Postgres, and
common HA patterns.

> For single-region single-instance deployments, see `PRODUCTION.md`.

## TL;DR topology

```
                                    ┌────────────────────────────────┐
                                    │   Global load /  (Anycast IP    │
                                    │   or DNS-based geo-routing)     │
                                    └────────────────────────────────┘
                                       │              │           │
                              ┌────────┘              │           └────────┐
                              ▼                       ▼                    ▼
                     ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
                     │  Region A    │       │  Region B    │       │  Region C    │
                     │  (us-east-1) │       │  (eu-west-1) │       │  (ap-south) │
                     │              │       │              │       │              │
                     │ ┌──────────┐ │       │ ┌──────────┐ │       │ ┌──────────┐ │
                     │ │ py-idp   │ │       │ │ py-idp   │ │       │ │ py-idp   │ │
                     │ │ (k8s)    │ │       │ │ (k8s)    │ │       │ │ (k8s)    │ │
                     │ └────┬─────┘ │       │ └────┬─────┘ │       │ └────┬─────┘ │
                     │      │       │       │      │       │       │      │       │
                     │ ┌────▼─────┐ │       │ ┌────▼─────┐ │       │ ┌────▼─────┐ │
                     │ │ Postgres │ │       │ │ Postgres │ │       │ │ Postgres │ │
                     │ │ primary  │ │       │ │ replica  │ │       │ │ replica  │ │
                     │ └──────────┘ │       │ └──────────┘ │       │ └──────────┘ │
                     └──────────────┘       └──────────────┘       └──────────────┘
```

Each region runs:
- py-idp API (k8s deployment, 2-3 replicas)
- Postgres (1 writer, 2+ read replicas per region)
- Object storage for upload persistence (S3-compatible)

## 1. Stateless API tier

The py-idp API is **stateless** — every request is self-contained.
You can deploy any number of replicas behind a load balancer without
state coordination. Scale by adding pods.

### Recommended: 3 replicas per region minimum

- 1 active, 2 standby for rolling deploys
- Behind k8s `Deployment` with `replicas: 3` and `strategy: RollingUpdate`
- Anti-affinity to spread across nodes

### Health probes

```yaml
livenessProbe:
  httpGet: { path: /healthz, port: 8080 }
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 3
readinessProbe:
  httpGet: { path: /readyz, port: 8080 }
  initialDelaySeconds: 2
  periodSeconds: 5
  failureThreshold: 2
```

### Resource sizing

Based on the 2026-09-01 load test results:
- Single uvicorn worker caps at ~9 RPS with 100ms simulated LLM
- For higher throughput, scale horizontally (more replicas)
- Each worker holds one in-flight request → `CPU = RPS_target × 100ms / 1000`

For 100 RPS @ 1.5s LLM latency:
- 100 × 1.5s / 60s ≈ 2.5 replicas per region
- Add headroom: 5-8 replicas
- CPU: 500m-1000m per replica
- Memory: 512MB-1GB per replica

## 2. Storage tier (Postgres)

**SQLite is for single-region single-instance only.** Production multi-region
deployments MUST use Postgres.

### Recommended topology

- **Region A (primary)**: 1 writer + 2 replicas in same region
- **Region B, C**: 2 replicas each, async replicated from A
- **Failover**: use a managed Postgres (RDS, Cloud SQL, Supabase, Neon)
  that handles primary election. Don't write your own.

### Connection from py-idp

```yaml
env:
  - name: IDP_DB_URL
    # AWS RDS example
    value: postgresql://user:pass@my-rds.cluster-xyz.us-east-1.rds.amazonaws.com:5432/idp
  - name: IDP_SQL_POOL_SIZE
    value: "10"  # tune to ~2 × num_workers
```

### Schema migrations

The migration files in `src/idp/storage/migrations/` are applied at startup
via `SqlStorage._apply_migrations()`. For multi-instance deployment:

1. Use a tool like `Flyway`, `Liquibase`, or `Alembic` for production
   migration management. The current `SqlStorage` migration is a
   **bootstrap helper**, not a production migration tool.
2. Apply migrations as a separate `Job` (k8s `Job` resource) before
   rolling out new API versions.
3. Never let two API pods race to apply migrations — use a
   Postgres advisory lock or a separate one-shot migration job.

### Read-after-write consistency

With async replication, a write to Region A may not be visible in
Region B immediately. py-idp's `Storage` abstraction doesn't guarantee
cross-region read-after-write. For users in Region B reading their
own recently-submitted results:

- **Option A** (simple): route writes to the region closest to the user,
  but reads always go to the writer region.
- **Option B** (complex): use CRDTs or per-user stickiness to local
  writers. py-idp doesn't ship this.

## 3. Storage of upload files

The current `idp.api` writes files to `IDP_UPLOAD_DIR` (local disk).
For multi-region:

1. **Move uploads to S3 / GCS / Azure Blob** before going multi-region.
   Add a small abstraction (`idp.storage.blob`) — not yet shipped.
2. **Don't share the local `/tmp/idp-uploads`** across pods via NFS.
   It's a latency and reliability trap.
3. **Set up cleanup**: uploads accumulate. Use an S3 lifecycle rule
   to delete after 7 days, or write a cron job.

Workaround until blob storage ships:

```python
# Mount an S3-compatible FUSE filesystem (e.g. s3fs, goofys)
# to IDP_UPLOAD_DIR. Slow but works for low-volume use.
```

## 4. Rate limiting across replicas

The in-process `RateLimiter` only counts requests per pod. For accurate
global limits, replace with a Redis-backed limiter:

```python
# Pseudocode — replace in idp.api:
from idp.ratelimit import RateLimiter
from redis.asyncio import Redis

class RedisRateLimiter(RateLimiter):
    def __init__(self, redis_url: str, per_key_per_minute: int):
        self.redis = Redis.from_url(redis_url)
        self.per_key = per_key_per_minute

    async def check(self, key: str | None = None) -> None:
            bucket = f"rl:{key or '_anon'}"
            current = await self.redis.incr(bucket)
            if current == 1:
                await self.redis.expire(bucket, 60)
            if current > self.per_key:
                raise RateLimitedError(...)
```

This is a small patch (~50 lines) but ships out-of-scope for now
because not every deployment needs Redis.

## 5. LLM backend resilience

The LLM call is the long pole. Three patterns:

### A. Per-region LLM vendor (recommended)

Each region uses a LLM provider in-region. e.g.:
- Region A (US-East): OpenAI `gpt-4o`
- Region B (EU-West): Mistral `mistral-large` (EU residency)
- Region C (AP-South): Bedrock in Mumbai region

Trade-off: extraction quality may vary slightly across regions.

### B. Centralised LLM (lowest latency variance)

All regions call a single LLM provider in one location. Higher latency
for distant regions but consistent quality. Acceptable if your latency
budget tolerates cross-region LLM RTT.

### C. Active-active LLM with fallback

py-idp's `Backend` is a single interface. Wrap with retry + circuit
breaker:

```python
# Pseudocode
class ResilientBackend(Backend):
    def __init__(self, primary: str, fallback: str, ...):
        self.primary = get_backend(primary)
        self.fallback = get_backend(fallback)

    def complete(self, req):
        for attempt in [self.primary, self.fallback]:
            try:
                return attempt.complete(req)
            except BackendUnavailableError:
                continue
        raise BackendUnavailableError("both primary and fallback exhausted")
```

Add circuit breaker (e.g. `pybreaker`) to avoid hammering a down
provider.

## 6. Deploying safely

### Blue/green

1. Deploy new version as `py-idp-canary` with traffic split 5%.
2. Watch `/metrics` for `py_idp_extractions_total` and
   `py_idp_http_errors_total` for 30 min.
3. If error rate ≤ baseline: shift to 50%, then 100%.
4. If error rate spikes: roll back by setting canary to 0%.

### Database migrations

Run **before** deploying new API code:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: py-idp-migrate-001
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: migrate
        image: py-idp:0.2.0
        command: ["python", "-m", "idp.migrate_audit", "--db-url", "..."]
```

### Configuration changes

- **Non-breaking**: rolling deploy is safe. Examples: rate limit bumps,
  log level changes, new env vars.
- **Breaking**: requires a coordinated deploy:
  1. Bump config to accept both old + new (e.g. add `IDP_RATE_LIMIT_PER_MINUTE_V2`)
  2. Deploy new code that reads new env var
  3. Switch callers to new env var
  4. Remove old env var in next release

## 7. Multi-region failover

### Active-passive (cheaper)

- Region A runs at full capacity, Region B is warm standby
- DNS-based failover (Route 53, Cloudflare) on health check failure
- RTO: 5-30 min (DNS propagation + boot time)
- RPO: 0-60s (Postgres replication lag)

### Active-active (more expensive, lower RTO)

- All regions serve traffic; data replicated async
- Geo-DNS routes each user to nearest healthy region
- Writes go to user's home region (sticky session)
- RTO: <1 min (only DNS TTL to wait)
- RPO: 0-60s (replication lag)

For py-idp's stateless API tier, active-active is straightforward.
The hard part is the database (Postgres replication) and the LLM
backend (cross-region latency).

## 8. Disaster recovery

| scenario | mitigation |
|---|---|
| **Pod crash** | k8s restarts; readiness probe fails until `Settings.load()` succeeds |
| **Node failure** | pods rescheduled; no data loss (stateless API) |
| **Region failure (active-passive)** | DNS failover; ~5 min downtime |
| **Region failure (active-active)** | DNS removes region; ~30s downtime |
| **Postgres primary failure** | managed service promotes replica; ~30s downtime; potential data loss if sync replication wasn't on |
| **LLM provider outage** | fallback backend (`ResilientBackend` pattern above) |
| **Bad deploy** | blue/green rollback; new pods never see traffic until healthy |

## 9. Observability for multi-region

The `/metrics` endpoint returns aggregated metrics per pod. For
multi-region aggregation:

- Each region exposes `/metrics` to a regional Prometheus
- Use Prometheus federation or remote_write to aggregate to a central store
- Alert on:
  - Per-region error rate > baseline
  - Per-region p99 latency > SLO
  - Postgres replication lag > 10s
  - LLM call timeout rate > 1%

## 10. Cost considerations

For 1M extractions/month at 100 RPS target:
- **Compute**: 8 × 1 CPU × 730h = $50-200/mo (cloud-dependent)
- **Postgres**: managed 2 vCPU × 16 GB = $200-500/mo
- **S3**: ~$10/mo for uploads
- **LLM**: $10,000-$30,000/mo (the real cost — extraction is LLM-heavy)
- **Total infra (ex-LLM)**: ~$500-1,000/mo

The LLM cost dwarfs everything. Don't bother over-optimizing the API
tier — a 5% gain there is invisible against LLM costs.

## 11. What's NOT shipped yet

Honest list of multi-region gaps in py-idp today:

| gap | workaround |
|---|---|
| Blob storage abstraction | mount S3 via FUSE; or use NFS for low-volume |
| Redis rate limiter | use HAProxy / nginx rate limit at LB |
| Cross-region cache | don't cache (LLM responses are doc-specific) |
| Per-user sticky routing | use client-side affinity cookies |
| Distributed tracing | drop in OpenTelemetry middleware; not shipped |
| Automatic failover for Postgres | use managed service (RDS / Cloud SQL) |

These are deployment-layer concerns; py-idp is the right shape to
accommodate them but doesn't ship them out-of-the-box because every
organization has different infra preferences.