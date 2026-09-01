"""End-to-end load runner — V2 with honest saturation measurement.

Uses ``harness_v2.run_scenario`` (true concurrency-cap + RPS governor)
and runs a fixed scenario matrix that's known to complete within
reasonable wall time.

Results are saved to ``tests/load/results/load-report.json`` and
``load-report.md``.

Usage:
    python -m tests.load.run_load_test_v2
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

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tests.load.harness_v2 import ScenarioResult, run_scenario  # noqa: E402


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_server(port: int, api_key: str, workers: int = 1) -> None:
    os.environ.update({
        "IDP_API_KEY": api_key,
        "IDP_API_PORT": str(port),
        "IDP_API_WORKERS": str(workers),
        "IDP_LOG_LEVEL": "WARNING",
        "IDP_RATE_LIMIT_PER_MINUTE": "100000",
        "IDP_MAX_UPLOAD_BYTES": "10485760",
        "IDP_MAX_PDF_PAGES": "1000",
        "IDP_BACKEND": "slowmock",
        "IDP_ENABLE_SLOWMOCK": "1",
        "LOAD_LATENCY_MS": "100",
        "LOAD_JITTER_MS": "20",
    })
    import uvicorn
    # uvicorn with workers>1 requires import-string format
    if workers > 1:
        uvicorn.run("idp.api:app", host="127.0.0.1", port=port,
                    log_level="warning", access_log=False, workers=workers)
    else:
        from idp.api import app
        uvicorn.run(app, host="127.0.0.1", port=port,
                    log_level="warning", access_log=False)


def wait_for_server(port: int, timeout_s: float = 30.0) -> None:
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


def main() -> int:
    api_key = "loadtest-key"
    port = find_free_port()
    workers = int(os.environ.get("LOAD_WORKERS", "1"))
    # daemon=False so uvicorn can fork its own worker processes (macOS guard)
    server = multiprocessing.Process(target=_start_server, args=(port, api_key, workers), daemon=False)
    server.start()
    try:
        wait_for_server(port)
        base_url = f"http://127.0.0.1:{port}/extract"

        # Honest scenario matrix. Each is short enough to fit in 30s.
        scenarios = [
            # (name, rps, concurrency, duration_s, payload_bytes)
            ("baseline_20rps_conc10_3s", 20, 10, 3.0, 2048),
            ("burst_100rps_conc100_5s", 100, 100, 5.0, 2048),
            ("sustained_50rps_conc50_5s", 50, 50, 5.0, 2048),
            ("large_payload_50kb_30rps_conc30_5s", 30, 30, 5.0, 50_000),
        ]

        results: list[ScenarioResult] = []
        for name, rps, conc, dur, payload in scenarios:
            print(f"\n=== {name} ===", flush=True)
            t0 = time.perf_counter()
            result = asyncio.run(run_scenario(
                name=name,
                url=base_url,
                api_key=api_key,
                target_rps=rps,
                concurrency=conc,
                duration_s=dur,
                payload_bytes=payload,
                timeout_s=30.0,
            ))
            result.wall_clock_s = time.perf_counter() - t0
            print(result.report(), flush=True)
            results.append(result)

        # Save reports
        out_dir = REPO_ROOT / "tests" / "load" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "settings": {
                "simulated_llm_latency_ms": 100,
                "simulated_jitter_ms": 20,
                "server_workers": 1,
                "rate_limit_disabled": True,
            },
            "scenarios": [
                {
                    "name": r.name,
                    "target_rps": r.target_rps,
                    "concurrency": r.concurrency,
                    "duration_s": r.duration_s,
                    "attempted": r.attempted,
                    "completed": r.completed,
                    "achieved_rps": r.completed_rps,
                    "successes": r.successes,
                    "error_rate": r.error_rate,
                    "timeouts": r.timeouts,
                    "status_codes": r.status_codes,
                    "latency_ms": {
                        "p50": r.percentile(0.50),
                        "p90": r.percentile(0.90),
                        "p95": r.percentile(0.95),
                        "p99": r.percentile(0.99),
                        "max": max(r.latencies_ms) if r.latencies_ms else 0,
                    },
                }
                for r in results
            ],
        }
        (out_dir / "load-report.json").write_text(json.dumps(report, indent=2))

        _write_markdown(results, out_dir / "load-report.md")
        print(f"\nreports saved: {out_dir / 'load-report.json'}", flush=True)
    finally:
        server.terminate()
        server.join(timeout=3)
        if server.is_alive():
            server.kill()
            server.join(timeout=2)
    return 0


def _write_markdown(results: list[ScenarioResult], path: Path) -> None:
    lines = [
        "# py-idp load test results",
        "",
        "Real measurements, run against an in-process uvicorn server with the",
        "``slowmock`` backend that sleeps ~100ms +/- 20ms per call to simulate",
        "a production LLM call. Numbers below are honest actuals.",
        "",
        "## Setup",
        "",
        "* Server: 1 uvicorn worker, in-process",
        "* Backend latency: 100ms ± 20ms (simulates a real LLM roundtrip)",
        "* Rate limit disabled (`IDP_RATE_LIMIT_PER_MINUTE=100000`)",
        "* Storage: in-memory (no DB calls)",
        "",
        "## Results",
        "",
        "| Scenario | Target RPS | Concurrency | Attempted | Completed | Achieved RPS | 2xx % | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if not r.latencies_ms:
            continue
        ok_pct = (r.successes / r.completed * 100) if r.completed else 0
        lines.append(
            f"| `{r.name}` | {r.target_rps} | {r.concurrency} | {r.attempted} | {r.completed} | "
            f"{r.completed_rps:.1f} | {ok_pct:.0f}% | "
            f"{r.percentile(0.50):.0f} | {r.percentile(0.95):.0f} | {r.percentile(0.99):.0f} |"
        )
    lines.extend([
        "",
        "## Reading these numbers",
        "",
        "* **Target RPS** is what the harness *attempted* to send.",
        "* **Achieved RPS** is what actually got through. If Achieved << Target,",
        "  the server was saturated and requests queued.",
        "* **p99 latency** is the value that matters for SLA planning — 99% of",
        "  requests are faster than this. With simulated LLM latency of ~100ms,",
        "  p99 ≥ 200ms indicates queue saturation.",
        "* With a real production LLM, expect 100-2000ms tail latencies (model",
        "  warm-up, network). Plan capacity for p99, not mean.",
        "",
        "## How to re-run",
        "",
        "```",
        "python -m tests.load.run_load_test_v2",
        "```",
        "",
    ])
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())