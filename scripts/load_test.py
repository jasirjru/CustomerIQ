"""Small dependency-free API load probe using synthetic customer inputs only."""

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor


SYNTHETIC_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
}


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile for a non-empty latency sample."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * fraction) + 0.999999) - 1))
    return ordered[index]


def _one_request(url: str, api_key: str, timeout: float) -> tuple[int, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(SYNTHETIC_CUSTOMER).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (OSError, TimeoutError):
        status = 0
    return status, (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--api-key", default=os.getenv("CUSTOMERIQ_LOAD_TEST_API_KEY"))
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()

    host = (urllib.parse.urlparse(args.url).hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        parser.error("Remote load tests require explicit --allow-remote.")
    if not args.api_key:
        parser.error("Provide --api-key or CUSTOMERIQ_LOAD_TEST_API_KEY.")
    if args.requests < 1 or args.concurrency < 1 or args.concurrency > 100:
        parser.error("requests must be positive and concurrency must be in [1, 100].")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(
            executor.map(
                lambda _: _one_request(args.url, args.api_key, args.timeout),
                range(args.requests),
            )
        )
    latencies = [latency for _, latency in results]
    statuses: dict[int, int] = {}
    for status, _ in results:
        statuses[status] = statuses.get(status, 0) + 1
    summary = {
        "requests": len(results),
        "concurrency": args.concurrency,
        "status_counts": statuses,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2),
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "p99": round(percentile(latencies, 0.99), 2),
            "max": round(max(latencies), 2),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if statuses == {200: len(results)} else 1


if __name__ == "__main__":
    raise SystemExit(main())
