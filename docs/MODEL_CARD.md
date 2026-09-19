# CustomerIQ v1.1 model card

## Release status

This is a legacy Random Forest serving artifact wrapped by a hardened v1.1 inference contract. It is **not release-approved**. Its scores are not calibrated probabilities, its `0.35` decision threshold is `legacy_unapproved`, and no current independent final-test estimate exists.

The development-only threshold `0.13503908770159245` came from illustrative false-negative/false-positive costs of 500/100. It is explicitly non-deployable and is not used by the API.

## Intended use

- Internal decision-support experiments for telecom churn review.
- Ranking or triage only after the score limitations, operational capacity, and human-review process are accepted.
- Synthetic or approved non-production data while release gates remain open.

Not intended for fully automated adverse decisions, eligibility, pricing, credit, employment, healthcare, or other high-impact decisions. The output is not a causal estimate of whether an intervention will retain a customer.

## Inputs and output

The API accepts 19 model features plus an optional correlation-only `customer_id`. Strict categorical vocabularies, numeric bounds, unknown-field rejection, and telecom cross-field constraints are defined by the API schema and recorded in `models/manifest.v1.json`.

`churn_probability` is the model's unchanged positive-class score. It is named for API compatibility but calibration has not been fitted or validated. `churn_prediction` is a policy decision using the legacy `0.35` threshold; it must not be interpreted as ground truth.

## Artifact identity and provenance

- Model: `legacy-rf-09608a080c72`, `RandomForestClassifier`.
- Preprocessor SHA-256: `56d14058ff09f1a6f59f5a460207a8443b9da51d4c24d06e3f5a51e5735c58b2`.
- Model SHA-256: `09608a080c720820305a24bf629f218ff9f114cd5f3f6baf45387f9c383b8d11`.
- Original artifact creation time, training commit, training package versions, and full source-dataset hash were not recorded.
- The manifest's package versions describe the v1.1 verification environment, not the unknown original training environment.

Runtime startup fails if hashes, estimator class, parameters, classes, feature order, request schema, or global importance metadata disagree with the manifest.

## Evaluation

Development model comparison used repeated stratified cross-validation on the training partition. Those candidate metrics do not establish performance of the unchanged serving artifact. The reserved 1,409-row historical holdout is sealed in v1.1 and has not been scored. It was exposed in older notebooks, so fresh external or forward-time labeled data is required for an independent release estimate.

Calibration utilities now support training-only nested-CV method selection, but no calibrator has been fitted into or attached to the production artifacts.

## Explanations

The manifest contains global impurity-based feature importance for the fitted forest. This describes aggregate model behavior and is not a local explanation. The API correctly returns `explanation_status: "not_available"`, no local drivers, and no retention recommendation.

## Fairness and risk

Gender and senior status are model inputs. Fairness is not established. Offline cohort diagnostics can compare review, true-positive, and false-positive rates, but those descriptive metrics are not a legal fairness determination. Any release needs approved cohort definitions, minimum sample sizes, uncertainty analysis, and accountable review.

Known risks include historical-data bias, temporal and geographic drift, label-horizon ambiguity, input-policy mismatch, uncalibrated scores, an unapproved threshold, and lack of causal evidence for retention actions.

## Required release gates

1. Approve the decision use case, prediction horizon, eligible population, and human-review process.
2. Obtain fresh, point-in-time-correct external or forward-time labeled data.
3. Freeze candidate, features, calibration, threshold policy, and acceptance criteria before the one-time independent evaluation.
4. Fit and select calibration using development data only; report Brier score, log loss, reliability bins, and uncertainty.
5. Approve cost assumptions and contact capacity in `docs/business-costs.json` with evidence and an accountable owner.
6. Complete cohort risk review, privacy/security review, load test, incident rehearsal, and rollback rehearsal.
7. Record an immutable release manifest and explicit human approval. Do not silently overwrite current artifacts.
