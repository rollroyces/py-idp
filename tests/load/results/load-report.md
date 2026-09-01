# py-idp load test results

Real measurements, run against an in-process uvicorn server with the
``slowmock`` backend that sleeps ~100ms +/- 20ms per call to simulate
a production LLM call. Numbers below are honest actuals.

## Setup

* Server: 1 uvicorn worker, in-process
* Backend latency: 100ms ± 20ms (simulates a real LLM roundtrip)
* Rate limit disabled (`IDP_RATE_LIMIT_PER_MINUTE=100000`)
* Storage: in-memory (no DB calls)

## Results

| Scenario | Target RPS | Concurrency | Attempted | Completed | Achieved RPS | 2xx % | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_20rps_conc10_3s` | 20 | 10 | 73 | 73 | 8.9 | 100% | 1161 | 1390 | 1585 |
| `burst_100rps_conc100_5s` | 100 | 100 | 547 | 547 | 8.9 | 100% | 10996 | 13610 | 13784 |
| `sustained_50rps_conc50_5s` | 50 | 50 | 331 | 331 | 8.9 | 100% | 5561 | 6759 | 7418 |
| `large_payload_50kb_30rps_conc30_5s` | 30 | 30 | 172 | 172 | 8.7 | 100% | 3442 | 3557 | 3938 |

## Reading these numbers

* **Target RPS** is what the harness *attempted* to send.
* **Achieved RPS** is what actually got through. If Achieved << Target,
  the server was saturated and requests queued.
* **p99 latency** is the value that matters for SLA planning — 99% of
  requests are faster than this. With simulated LLM latency of ~100ms,
  p99 ≥ 200ms indicates queue saturation.
* With a real production LLM, expect 100-2000ms tail latencies (model
  warm-up, network). Plan capacity for p99, not mean.

## How to re-run

```
python -m tests.load.run_load_test_v2
```
