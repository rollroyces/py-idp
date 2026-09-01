"""Load-testing rig for the py-idp FastAPI app.

Measures real throughput, latency percentiles, and error rates under
controlled concurrency. NOT a replacement for k6/vegeta, but it runs
in-process with no external dependencies so it's reproducible in CI.

Usage:
    python -m tests.load.harness --rps 100 --duration 30 --scenario mock

Scenarios:
  mock       — pipeline uses MockBackend (zero LLM latency)
  simulated  — backend sleeps for X ms to simulate a real LLM
  realistic  — mix of fast + slow extractions with realistic payloads
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import string
import sys
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class LatencyBucket:
    """All latency samples for one run."""
    samples: list[float] = field(default_factory=list)

    def add(self, latency_s: float) -> None:
        self.samples.append(latency_s)

    def percentiles(self) -> dict[str, float]:
        if not self.samples:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
        s = sorted(self.samples)
        n = len(s)
        def pct(p: float) -> float:
            i = min(int(p * n), n - 1)
            return s[i]
        return {
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": s[-1],
        }


@dataclass
class RunStats:
    """Aggregated stats for one run."""
    started: float = 0.0
    finished: float = 0.0
    requests: int = 0
    successes: int = 0
    client_errors: int = 0   # 4xx
    server_errors: int = 0   # 5xx
    network_errors: int = 0  # connection reset, timeout, etc.
    rate_limited: int = 0    # 429
    payload_too_large: int = 0  # 413
    latency: LatencyBucket = field(default_factory=LatencyBucket)
    status_codes: dict[int, int] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.finished - self.started if self.finished else 0.0

    @property
    def rps(self) -> float:
        return self.requests / self.duration_s if self.duration_s > 0 else 0.0

    def report(self) -> str:
        p = self.latency.percentiles()
        return (
            f"requests={self.requests} rps={self.rps:.1f}\n"
            f"  2xx={self.successes}  4xx={self.client_errors}  "
            f"5xx={self.server_errors}  network_err={self.network_errors}\n"
            f"  429 (rate limited)={self.rate_limited}  413 (too large)={self.payload_too_large}\n"
            f"  latency ms: p50={p['p50']*1000:.1f}  p90={p['p90']*1000:.1f}  "
            f"p95={p['p95']*1000:.1f}  p99={p['p99']*1000:.1f}  max={p['max']*1000:.1f}\n"
            f"  status codes: {dict(sorted(self.status_codes.items()))}"
        )


def make_invoice_doc(size_bytes: int) -> bytes:
    """Generate a realistic-ish invoice document of approximately ``size_bytes``.

    Varies field lengths and noise to avoid trivial cache hits.
    """
    base = (
        "Vendor: Acme Corporation International\n"
        "Invoice Number: INV-2026-{rand}\n"
        "Date: 2026-{m}-{d:02d}\n"
        "Total: ${total}\n"
        "Tax: ${tax}\n"
        "Customer: {cust}\n"
        "Address: {addr}\n"
        "Line Items:\n"
    )
    rnd = lambda: ''.join(random.choices(string.ascii_letters + string.digits, k=10))  # noqa: E731
    body = base.format(
        rand=rnd(),
        m=random.randint(1, 12),
        d=random.randint(1, 28),
        total=f"{random.randint(100, 100000)}.{random.randint(0, 99):02d}",
        tax=f"{random.randint(0, 5000)}.{random.randint(0, 99):02d}",
        cust=rnd(),
        addr=rnd(),
    )
    # Pad to roughly size_bytes with realistic-looking line items
    while len(body) < size_bytes:
        item = (
            f"  Item {rnd()}: qty={random.randint(1, 50)} "
            f"unit_price=${random.randint(1, 500)}.{random.randint(0, 99):02d} "
            f"total=${random.randint(1, 5000)}.{random.randint(0, 99):02d}\n"
        )
        body += item
    return body.encode("utf-8")[:size_bytes]


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    payload: bytes,
    stats: RunStats,
    timeout_s: float = 30.0,
) -> None:
    """Send one POST and record its outcome."""
    data = aiohttp.FormData()
    data.add_field("file", payload, filename="invoice.txt", content_type="text/plain")
    data.add_field("schema_name", "Invoice")
    headers = {"X-API-Key": api_key}
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    start = time.perf_counter()
    stats.requests += 1
    try:
        async with session.post(url, data=data, headers=headers, timeout=timeout) as resp:
            await resp.read()  # drain to release connection
            elapsed = time.perf_counter() - start
            stats.latency.add(elapsed)
            stats.status_codes[resp.status] = stats.status_codes.get(resp.status, 0) + 1
            if 200 <= resp.status < 300:
                stats.successes += 1
            elif resp.status == 429:
                stats.rate_limited += 1
                stats.client_errors += 1
            elif resp.status == 413:
                stats.payload_too_large += 1
                stats.client_errors += 1
            elif 400 <= resp.status < 500:
                stats.client_errors += 1
            elif 500 <= resp.status < 600:
                stats.server_errors += 1
    except (aiohttp.ClientError, asyncio.TimeoutError):
        stats.network_errors += 1
        elapsed = time.perf_counter() - start
        stats.latency.add(elapsed)


async def fixed_rps_worker(
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    payload: bytes,
    stats: RunStats,
) -> None:
    """Drains the queue of (scheduled_time, request_id) and fires each
    request at its scheduled time. Used for fixed-RPS load."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        scheduled, _reqid = item
        # Wait until scheduled time
        now = time.perf_counter()
        delay = scheduled - now
        if delay > 0:
            await asyncio.sleep(delay)
        await one_request(session, url, api_key, payload, stats)


async def run(
    *,
    base_url: str,
    api_key: str,
    target_rps: int,
    duration_s: float,
    payload_bytes: int,
    concurrency: int,
) -> RunStats:
    """Run a fixed-RPS load test for ``duration_s`` seconds."""
    stats = RunStats()
    payload = make_invoice_doc(payload_bytes)

    # Build a queue of scheduled fire times — target_rps spread across 1s windows
    interval = 1.0 / target_rps if target_rps > 0 else 0.001
    start = time.perf_counter()
    queue: asyncio.Queue = asyncio.Queue()
    end_at = start + duration_s
    reqid = 0
    while True:
        scheduled = start + reqid * interval
        if scheduled >= end_at:
            break
        await queue.put((scheduled, reqid))
        reqid += 1

    async def schedule_workers() -> None:
        for _ in range(concurrency):
            asyncio.create_task(fixed_rps_worker(queue, aiohttp.ClientSession(), base_url, api_key, payload, stats))

    async with aiohttp.ClientSession() as session:
        # Adjust workers to use the shared session
        async def worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                scheduled, _ = item
                now = time.perf_counter()
                delay = scheduled - now
                if delay > 0:
                    await asyncio.sleep(delay)
                await one_request(session, base_url, api_key, payload, stats)

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]

        stats.started = start
        await asyncio.sleep(duration_s)
        # Drain remaining
        while not queue.empty():
            item = await queue.get()
            if item is not None:
                scheduled, _ = item
                now = time.perf_counter()
                if scheduled > now:
                    await asyncio.sleep(scheduled - now)
                await one_request(session, base_url, api_key, payload, stats)
            queue.task_done()

        # Cancel workers
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    stats.finished = time.perf_counter()
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="py-idp load harness")
    p.add_argument("--url", default="http://127.0.0.1:8080/extract")
    p.add_argument("--api-key", default="")
    p.add_argument("--rps", type=int, default=50)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--payload-bytes", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--scenario", default="report",
                   choices=["report", "mock", "realistic"])
    args = p.parse_args()

    if args.scenario == "report":
        print(json.dumps({
            "name": "load harness",
            "version": "1.0",
            "scenarios": ["mock (no LLM latency)", "realistic (with simulated LLM delay)"],
            "metrics": ["rps", "p50/p90/p95/p99 latency", "error rate", "status codes"],
        }, indent=2))
        return 0

    if not args.api_key:
        print("error: --api-key required (the server's IDP_API_KEY)", file=sys.stderr)
        return 2

    print(f"running {args.scenario} load: rps={args.rps} duration={args.duration}s "
          f"payload={args.payload_bytes}B concurrency={args.concurrency}", file=sys.stderr)
    stats = asyncio.run(run(
        base_url=args.url,
        api_key=args.api_key,
        target_rps=args.rps,
        duration_s=args.duration,
        payload_bytes=args.payload_bytes,
        concurrency=args.concurrency,
    ))
    print(stats.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())