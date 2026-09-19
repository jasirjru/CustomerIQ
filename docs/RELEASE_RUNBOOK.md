# CustomerIQ release runbook

This runbook describes gates only. It does not authorize evaluation, artifact replacement, deployment, or release.

Start with `docs/release-evidence.template.json`, attach reviewable evidence and accountable approvers, then run `python scripts/check_release_readiness.py --evidence <candidate-evidence.json>`. The command is read-only and exits non-zero while any gate remains open. A successful report is necessary but does not itself authorize release.

## 1. Define and approve

- Name the business owner, ML owner, security owner, and rollback decision-maker.
- Freeze intended use, prediction horizon, population, intervention, human review, capacity, and prohibited uses.
- Complete and approve `docs/business-costs.json`; keep illustrative 500/100 costs out of production policy.
- Define acceptance criteria for ranking, calibration, subgroup behavior, latency, availability, and business impact before independent evaluation.

## 2. Build on development data only

- Keep the historical final holdout sealed.
- Run repeated/nested CV for model and calibration choices on development partitions only.
- Evaluate threshold sensitivity under approved costs and capacity; freeze one policy before independent evaluation.
- Generate immutable dataset, code, dependency, schema, model, and preprocessor hashes.

## 3. Independent evaluation gate

- Obtain fresh external or forward-time labeled data with point-in-time feature correctness.
- Obtain explicit approval for the one-time evaluation.
- Run the frozen pipeline once and record all prespecified metrics, confidence intervals, exclusions, failures, and subgroup results.
- Do not tune from these results. A failed gate returns to development and requires a new independent dataset for the next release claim.

## 4. Operational qualification

- Build from `requirements-api.lock`; run unit, adversarial, container smoke, vulnerability, and license checks.
- Run authorized load and soak tests at expected peak traffic plus headroom.
- Treat `docs/PERFORMANCE_BASELINE.md` only as a local diagnostic; repeat against the immutable candidate in the approved production-like topology.
- Verify TLS, managed secrets, key rotation, gateway rate limits, timeouts, log redaction, metrics collection, alerts, backups, and incident contacts.
- Rehearse rollback to the preceding immutable image and policy configuration.

## 5. Controlled rollout

- Require recorded human approval and an immutable image digest. Never deploy a mutable `latest` tag as the release identifier.
- Shadow first if allowed, then canary to a small traffic fraction without automated customer action.
- Compare schema failures, latency, score distribution, review rate, and cohort diagnostics against frozen limits.
- Promote only after the observation window. Roll back on integrity failure, excessive errors/latency, material drift, or policy-limit breach.

## 6. Post-release

- Monitor technical and outcome metrics on defined cadences.
- Revalidate on temporal data, record incidents and overrides, and expire approvals at a defined date.
- Version every model, calibrator, threshold, schema, manifest, and policy change. Never silently modify a released artifact.
