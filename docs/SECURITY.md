# Inference API security

## Authentication

Production mode fails closed unless `CUSTOMERIQ_REQUIRE_API_KEY=true` and `CUSTOMERIQ_API_KEYS` contains one or more comma-separated keys. Keys must be unique ASCII strings of 32–256 characters. Send a key in `X-API-Key` for `/predict`, `/predict-batch`, `/model-info`, and `/metrics`. `/health`, the static browser, and the OpenAPI document remain unauthenticated.

API keys are bootstrap controls, not a complete identity system. Terminate TLS at a trusted reverse proxy or managed ingress, store keys in a secret manager, rotate them, and assign separate keys per client. Never place keys in URLs. The browser keeps the optional key only in the password field and does not persist it.

## Boundary protections

- Strict request schemas reject unknown fields, invalid categories, non-finite numbers, and inconsistent telecom states.
- Request bodies and batch size are bounded.
- In-process rate limiting limits accidental or small-scale abuse.
- Client 500 responses are generic; detailed exceptions go to server logs with a request ID.
- Metrics use canonical route labels and never include payload values or API keys.
- The container runs as an unprivileged user and verifies artifact integrity at startup.

## Production requirements

Use an external API gateway/WAF for distributed rate limits, authenticated identity, TLS, request timeouts, and audit logging. Multiple worker processes do not share the in-memory limiter or metrics registry. Export operational metrics to a managed collector before scaling horizontally.

Treat customer attributes and scores as sensitive data. Do not log request bodies. Define retention, access, deletion, incident-response, and breach-notification policies before accepting real customer traffic.

## Rotation and incident response

1. Add a new key through the secret manager and restart/roll the service.
2. Verify the new key, update the client, then remove the old key and roll again.
3. If exposure is suspected, revoke first, inspect request-ID-based logs and gateway audit records, and notify the security owner.
4. Artifact hash failure, unexpected schema changes, score-distribution shifts, or elevated 5xx rates should stop rollout and trigger rollback.
