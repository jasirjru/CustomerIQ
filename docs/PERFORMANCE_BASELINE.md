# Local performance baseline

Recorded on 2026-09-19 using the production security configuration, a single local Uvicorn process, Python 3.14.6 on Windows, and the fixed synthetic payload in `scripts/load_test.py`. This is a developer-machine diagnostic, not a production capacity claim or a completed load/soak gate.

| Requests | Concurrency | Success | Mean | p50 | p95 | p99 | Max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1 | 100% | 23.35 ms | 27.49 ms | 35.52 ms | 38.02 ms | 60.55 ms |
| 200 | 10 | 100% | 225.84 ms | 221.35 ms | 311.64 ms | 431.30 ms | 481.12 ms |

The concurrency-10 result exceeds the provisional 250 ms p95 engineering objective. Do not approve the load/soak gate from these results. Before release, define the expected request and batch mix, concurrency, hardware, worker count, duration, warm-up, and error/latency limits; then test the immutable candidate in an authorized production-like environment with resource saturation and recovery monitoring.

Reproduction commands:

```powershell
python scripts/load_test.py --url http://127.0.0.1:8767/predict --requests 100 --concurrency 1 --api-key <local-test-key>
python scripts/load_test.py --url http://127.0.0.1:8767/predict --requests 200 --concurrency 10 --api-key <local-test-key>
```

Only synthetic inputs were sent. No repository dataset or reserved holdout was accessed.
