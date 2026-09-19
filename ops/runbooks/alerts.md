# CustomerIQ alert runbooks

These are initial engineering thresholds, not demonstrated SLOs. Confirm traffic baselines before production use. Never inspect or log customer payloads while diagnosing alerts.

## High server error rate

1. Identify affected canonical routes and request IDs in structured logs.
2. Check artifact-integrity startup events and dependency health.
3. Stop rollout if a new image or configuration correlates with the increase.
4. Roll back to the preceding immutable image if the error budget continues burning.

## High prediction latency

1. Compare request rate, batch sizes, CPU, memory, and worker saturation.
2. Separate `/predict` from `/predict-batch` behavior.
3. Apply gateway backpressure; do not silently drop or partially score batches.
4. Scale only after metrics and rate limiting are externalized across workers.

## Authentication failures

1. Check whether failures are isolated to a known client identity at the gateway.
2. Confirm recent key rotation and client configuration without exposing keys.
3. Rate-limit abusive sources and rotate a key if exposure is suspected.

## Model unavailable

1. Check health, restart count, and startup logs.
2. Verify manifest and artifact hashes against the immutable release record.
3. Do not bypass integrity checks; roll back to the last verified image.

## Validation spike

1. Group only by bounded validation error type and canonical route.
2. Compare against recent client releases and schema-version adoption.
3. Contact the client owner using request IDs; never copy payloads into tickets.
