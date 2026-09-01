"""Load harness v2 — measures steady-state throughput honestly.

The v1 harness scheduled requests at a target rate but didn't cap
in-flight concurrency, so when RPS exceeded server capacity, requests
queued up unbounded — measured latency ballooned, but reported
throughput stayed "as targeted" (because the harness was firing at
the target rate regardless of backpressure).

V2 fixes this:
  - Semaphore-based concurrency cap (true concurrent in-flight limit)
  - Each scenario records both *attempted* and *completed* RPS, so
    saturation shows up as the gap between them
  - Per-scenario timeout — hung requests are killed, not waited for

Usage:
    python -m tests.load.harness_v2 --url http://127.0.0.1:8080/extract \\
        --api-key KEY --rps 50 --concurrency 20 --duration 10
"""
from __future__ import annotations

import argparse
import asyncio
import random
import string
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class ScenarioResult:
    name: str
    target_rps: int
    concurrency: int
    duration_s: float
    attempted: int = 0   # requests launched
    completed: int = 0   # requests that got a response
    successes: int = 0   # 2xx
    timeouts: int = 0
    status_codes: dict[int, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    wall_clock_s: float = 0.0

    @property
    def completed_rps(self) -> float:
        return self.completed / self.wall_clock_s if self.wall_clock_s else 0.0

    @property
    def error_rate(self) -> float:
        return (self.completed - self.successes) / self.completed if self.completed else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        i = min(int(p * len(s)), len(s) - 1)
        return s[i]

    def report(self) -> str:
        return (
            f"[{self.name}]\n"
            f"  target_rps={self.target_rps}  concurrency={self.concurrency}\n"
            f"  attempted={self.attempted}  completed={self.completed}\n"
            f"  achieved_rps={self.completed_rps:.1f}  "
            f"success_rate={(1 - self.error_rate) * 100:.1f}%\n"
            f"  status codes: {dict(sorted(self.status_codes.items()))}\n"
            f"  latency ms: p50={self.percentile(0.50):.0f}  "
            f"p90={self.percentile(0.90):.0f}  "
            f"p95={self.percentile(0.95):.0f}  "
            f"p99={self.percentile(0.99):.0f}  "
            f"max={max(self.latencies_ms) if self.latencies_ms else 0:.0f}\n"
        )


def make_invoice_doc(size_bytes: int) -> bytes:
    rnd = lambda: ''.join(random.choices(string.ascii_letters + string.digits, k=10))  # noqa: E731
    base = (
        f"Vendor: Acme Corporation International\n"
        f"Invoice Number: INV-2026-{rnd()}\n"
        f"Date: 2026-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}\n"
        f"Total: ${random.randint(100, 100000)}.{random.randint(0, 99):02d}\n"
        f"Tax: ${random.randint(0, 5000)}.{random.randint(0, 99):02d}\n"
        f"Customer: {rnd()}\n"
        f"Address: {rnd()}\n"
        f"Line Items:\n"
    )
    body = base
    while len(body) < size_bytes:
        body += (
            f"  Item {rnd()}: qty={random.randint(1, 50)} "
            f"unit_price=${random.randint(1, 500)}.{random.randint(0, 99):02d} "
            f"total=${random.randint(1, 5000)}.{random.randint(0, 99):02d}\n"
        )
    return body.encode("utf-8")[:size_bytes]


async def fire_one(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    payload: bytes,
    result: ScenarioResult,
    sem: asyncio.Semaphore,
    timeout_s: float,
) -> None:
    """One HTTP POST, respects ``sem`` for true concurrency limit."""
    async with sem:
        result.attempted += 1
        data = aiohttp.FormData()
        data.add_field("file", payload, filename="inv.txt", content_type="text/plain")
        data.add_field("schema_name", "Invoice")
        headers = {"X-API-Key": api_key}
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        start = time.perf_counter()
        try:
            async with session.post(url, data=data, headers=headers, timeout=timeout) as resp:
                await resp.read()
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.latencies_ms.append(elapsed_ms)
                result.status_codes[resp.status] = result.status_codes.get(resp.status, 0) + 1
                if 200 <= resp.status < 300:
                    result.successes += 1
                result.completed += 1
        except asyncio.TimeoutError:
            result.timeouts += 1
            result.status_codes[599] = result.status_codes.get(599, 0) + 1
        except aiohttp.ClientError:
            result.completed += 1
            result.status_codes[0] = result.status_codes.get(0, 0) + 1


async def run_scenario(
    name: str,
    url: str,
    api_key: str,
    target_rps: int,
    concurrency: int,
    duration_s: float,
    payload_bytes: int,
    timeout_s: float = 30.0,
) -> ScenarioResult:
    """Drive `concurrency` workers that each fire requests as fast as possible,
    with a global rate governor that targets `target_rps`.
    """
    result = ScenarioResult(
        name=name, target_rps=target_rps, concurrency=concurrency, duration_s=duration_s,
    )
    sem = asyncio.Semaphore(concurrency)
    payload = make_invoice_doc(payload_bytes)
    interval = 1.0 / target_rps if target_rps > 0 else 0.001
    next_due = time.perf_counter()  # fire requests no earlier than this
    inflight: list[asyncio.Task] = []

    async with aiohttp.ClientSession() as session:
        async def pacing_worker(stop_at: float) -> None:
            """One worker that respects the global RPS governor."""
            nonlocal next_due
            while time.perf_counter() < stop_at:
                now = time.perf_counter()
                if next_due > now:
                    await asyncio.sleep(next_due - now)
                next_due += interval
                # Cap in-flight tasks to prevent unbounded growth
                if len(inflight) >= concurrency * 2:
                    # Drain one task (with timeout) to make room
                    done, pending = await asyncio.wait(
                        inflight, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
                    )
                    inflight[:] = list(pending)
                task = asyncio.create_task(fire_one(session, url, api_key, payload, result, sem, timeout_s))
                inflight.append(task)

        stop_at = time.perf_counter() + duration_s
        # Spawn enough workers to saturate the concurrency limit
        num_workers = max(concurrency, target_rps // 10 + 1)
        t0 = time.perf_counter()
        await asyncio.gather(*[pacing_worker(stop_at) for _ in range(num_workers)])

        # Wait for all in-flight requests to complete (with a hard timeout)
        if inflight:
            await asyncio.wait_for(
                asyncio.gather(*inflight, return_exceptions=True),
                timeout=timeout_s * 2,
            )

    result.wall_clock_s = time.perf_counter() - t0
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--rps", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--payload-bytes", type=int, default=2048)
    p.add_argument("--name", default="adhoc")
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()

    r = asyncio.run(run_scenario(
        name=args.name,
        url=args.url,
        api_key=args.api_key,
        target_rps=args.rps,
        concurrency=args.concurrency,
        duration_s=args.duration,
        payload_bytes=args.payload_bytes,
        timeout_s=args.timeout,
    ))
    print(r.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())