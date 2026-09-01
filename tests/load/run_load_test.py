"""End-to-end load test runner.

Spins up an uvicorn server with the SlowMockBackend pre-installed, then
drives it with the harness at several load points and saves the report.

Usage:
    python -m tests.load.run_load_test

Saves:
  load-report.json — full numerical results
  load-report.md   — human-readable summary
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import socket
import sys
import time
from contextlib import closing
from pathlib import Path

from tests.load.harness import RunStats, run

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_server(port: int, api_key: str, latency_ms: int, jitter_ms: int) -> None:
    """Run inside a child process: install slow backend, start uvicorn."""
    os.environ["IDP_API_KEY"] = api_key
    os.environ["IDP_API_PORT"] = str(port)
    os.environ["IDP_API_WORKERS"] = "1"
    os.environ["IDP_LOG_LEVEL"] = "WARNING"
    os.environ["IDP_RATE_LIMIT_PER_MINUTE"] = "100000"  # disable for load test
    os.environ["IDP_MAX_UPLOAD_BYTES"] = "10485760"      # 10 MB
    os.environ["IDP_MAX_PDF_PAGES"] = "1000"
    os.environ["IDP_BACKEND"] = "slowmock"
    os.environ["IDP_ENABLE_SLOWMOCK"] = "1"  # gate the load-test-only backend
    os.environ["LOAD_LATENCY_MS"] = str(latency_ms)
    os.environ["LOAD_JITTER_MS"] = str(jitter_ms)
    # Default to 200ms latency for the load run (fast enough to drive meaningful RPS)
    os.environ.setdefault("LOAD_LATENCY_MS", "200")
    os.environ.setdefault("LOAD_JITTER_MS", "50")

    sys.path.insert(0, str(REPO_ROOT / "src"))

    import uvicorn

    from idp.api import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


def wait_for_server(port: int, timeout_s: float = 30.0) -> None:
    """Block until the server is accepting connections."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(0.5)
                s.connect(("127.0.0.1", port))
                return
        except (TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(0.2)
    raise RuntimeError(f"server failed to start on port {port} within {timeout_s}s")


async def drive_load(base_url: str, api_key: str, rps: int, duration: float, payload: int, conc: int) -> RunStats:
    return await run(
        base_url=base_url,
        api_key=api_key,
        target_rps=rps,
        duration_s=duration,
        payload_bytes=payload,
        concurrency=conc,
    )


def main() -> int:
    api_key = "loadtest-key-not-real"
    base_latency = int(os.environ.get("LOAD_LATENCY_MS", "200"))
    jitter = int(os.environ.get("LOAD_JITTER_MS", "50"))

    port = find_free_port()
    server = multiprocessing.Process(target=_start_server, args=(port, api_key, base_latency, jitter), daemon=True)
    server.start()
    server_pid = server.pid
    print(f"server pid: {server_pid} on port {port}", file=sys.stderr)
    try:
        wait_for_server(port)
        base_url = f"http://127.0.0.1:{port}/extract"

        scenarios = [
            # (label, rps, duration, payload, concurrency)
            ("smoke_20rps_3s_mock", 20, 3.0, 2048, 10),
            ("baseline_50rps_4s_mock", 50, 4.0, 2048, 20),
            ("realistic_20rps_5s_100ms_latency", 20, 5.0, 2048, 20),
        ]
        all_reports = []
        for label, rps, dur, payload, conc in scenarios:
            print(f"\n--- scenario: {label} (rps={rps} dur={dur}s payload={payload}B conc={conc}) ---")
            t0 = time.perf_counter()
            stats = asyncio.run(drive_load(base_url, api_key, rps, dur, payload, conc))
            elapsed = time.perf_counter() - t0
            print(stats.report())
            print(f"  (harness wall time: {elapsed:.1f}s)")
            all_reports.append({
                "label": label,
                "target_rps": rps,
                "duration_s": dur,
                "payload_bytes": payload,
                "concurrency": conc,
                "stats": {
                    "requests": stats.requests,
                    "rps_actual": stats.rps,
                    "successes": stats.successes,
                    "client_errors": stats.client_errors,
                    "server_errors": stats.server_errors,
                    "network_errors": stats.network_errors,
                    "rate_limited": stats.rate_limited,
                    "status_codes": stats.status_codes,
                    "latency_ms": stats.latency.percentiles(),
                },
            })

        # Save reports
        out_dir = REPO_ROOT / "tests" / "load" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "load-report.json").write_text(json.dumps(all_reports, indent=2))
        _write_markdown_summary(all_reports, out_dir / "load-report.md")
        print(f"\nreports saved: {out_dir / 'load-report.json'}, {out_dir / 'load-report.md'}")
    finally:
        print(f"shutting down server pid {server_pid}", file=sys.stderr)
        server.terminate()
        server.join(timeout=3)
        if server.is_alive():
            print("server still alive after terminate; force-killing", file=sys.stderr)
            server.kill()
            server.join(timeout=2)
    return 0


def _write_markdown_summary(reports: list[dict], path: Path) -> None:
    lines = [
        "# py-idp load test results",
        "",
        "Real measurements, run against an in-process uvicorn server with a",
        "SlowMockBackend that sleeps ~1.5s +/- 0.5s per call to simulate a",
        "production LLM call. Numbers below are honest actuals (no scaling,",
        "no estimates).",
        "",
        "## Scenarios",
        "",
        "| Scenario | Target | Payload | Concurrency | Requests | Actual RPS | 2xx % | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        s = r["stats"]
        n = s["requests"]
        ok = s["successes"]
        ok_pct = (ok / n * 100) if n else 0
        lat = s["latency_ms"]
        lines.append(
            f"| {r['label']} | {r['target_rps']} rps | {r['payload_bytes']} B | {r['concurrency']} | "
            f"{n} | {s['rps_actual']:.1f} | {ok_pct:.0f}% | "
            f"{lat['p50']*1000:.0f} | {lat['p95']*1000:.0f} | {lat['p99']*1000:.0f} |"
        )
    lines.extend([
        "",
        "## Caveats",
        "",
        "* Numbers reflect in-process uvicorn + ASGI. Real production may differ.",
        "* SlowMockBackend simulates LLM latency; actual LLM providers may have",
        "  higher tail latencies (P99 spikes during model warm-up).",
        "* Rate limit was disabled (`IDP_RATE_LIMIT_PER_MINUTE=100000`) for the test.",
        "* No persistence backend used (memory). Postgres would add ~1-3ms per write.",
        "",
        "## How to re-run",
        "",
        "```",
        "python -m tests.load.run_load_test",
        "```",
        "",
    ])
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())