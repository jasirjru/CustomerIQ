# Observability and service objectives

The authenticated `/metrics` endpoint exposes payload-free Prometheus text metrics for canonical endpoints, response status, latency, inference outcomes, score bins, and validation error types. Values are process-local and reset on restart.

Recommended initial objectives—subject to an approved traffic profile—are 99.9% successful request availability excluding client 4xx responses, p95 single-request latency below 250 ms, and no artifact-integrity startup failures. Treat these as engineering targets, not demonstrated guarantees, until load and soak tests establish baselines.

Suggested alerts:

- 5xx ratio above 1% for 5 minutes or any sustained increase from baseline.
- p95 latency above the approved SLO for 10 minutes.
- 401, 413, 422, or 429 rates increasing sharply by client and gateway identity.
- health check failure or restart loop.
- score histogram, review rate, missingness, category mix, or cohort rates drifting from a versioned baseline.
- absence of traffic/metrics when traffic is expected.

`ops/prometheus-alerts.yml` provides version-controlled starter rules for the service-error ratio, p95 latency, authentication failures, missing model readiness, and validation spikes. Validate the expressions against the production Prometheus topology, approved SLOs, and real traffic baseline before installing them. Response procedures are in `ops/runbooks/alerts.md`.

Do not use high-cardinality customer identifiers, raw categories, payloads, keys, or free-form paths as metric labels. Aggregate multiple workers through a supported Prometheus deployment pattern or replace the in-memory registry with an external telemetry backend before horizontal scaling.

Use `scripts/load_test.py` only against localhost by default. Remote use requires the explicit `--allow-remote` flag and prior authorization. The script sends a fixed synthetic customer, never repository datasets.

The current developer-machine measurements and their limitations are recorded in `docs/PERFORMANCE_BASELINE.md`. They are diagnostic evidence only and do not satisfy the production load/soak release gate.
