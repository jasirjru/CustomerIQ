# CustomerIQ prediction contract

Status: experimental retrospective contract. This contract does not authorize
prospective production use.

## Decision and population

CustomerIQ ranks telecom customer accounts by observed association with the
historical `Churn` label. One row represents one account at the dataset snapshot.
The intended decision is prioritization for human review or a separately
evaluated retention program; the prediction must not autonomously determine
pricing, service termination, eligibility, or adverse customer treatment.

## Prediction time and target

At scoring time, all submitted features must have been known before the churn
outcome occurred. The identifier `customerID` is excluded. The model returns a
score for the positive `Churn = Yes` class and a binary decision at a threshold
frozen before final-test evaluation.

The source dataset does not document a prospective observation timestamp,
label window, censoring rule, or exact definition of churn. Consequently:

- the current label is retrospective rather than a verified future-horizon label;
- `tenure`, `MonthlyCharges`, and `TotalCharges` are only admissible if their
  values were available at the chosen prediction timestamp; and
- reported results cannot establish performance on future customers or periods.

Production release is blocked until the data owner supplies the observation
timestamp, churn horizon, label construction, censoring policy, and feature
availability timestamps.

## Evaluation protocol

The deterministic allocation is:

- 60% model-training: preprocessing fit, repeated cross-validation, and model
  family selection;
- 20% validation: one fit of the selected pipeline and decision-threshold
  selection; and
- 20% final test: reserved for one evaluation after every model and threshold
  choice is frozen.

Preprocessing is part of each candidate pipeline and is refit inside every
cross-validation fold. Neither candidate selection nor threshold optimization
loads the final-test partition. Array argument names alone cannot prove provenance.
The executable development loader reads only `X_train_raw.csv` and `y_train.csv`.
A filesystem audit guard blocks historical test CSVs and the full raw CSV during
development runs. Examining final-test metrics and then changing
the system invalidates that partition for future final evaluation.

The primary cross-validation selection metric is average precision because the
positive class is the minority and ranking churners is the intended use. ROC-AUC,
F1, and balanced accuracy are secondary diagnostics.

Run the executable development protocol with:

```powershell
python scripts/run_development_experiment.py --output reports/development-evidence.json
```

The report records development-data hashes, Git state, package versions,
repeated-CV model selection, and nested-CV calibration selection. It does not
read final-test files, select a production threshold, or persist a model.

## Threshold and cost assumptions

The experimental threshold minimizes:

`validation false negatives × 500 + validation false positives × 100`

These values are illustrative units derived from the existing dashboard, not
verified financial estimates. They exclude treatment success, contact capacity,
discount cost variation, customer lifetime uncertainty, and causal uplift. A
production threshold requires approved cost provenance and sensitivity analysis.

## Out of scope for this phase

This contract does not validate probability calibration, causal retention
effects, fairness, drift, security, local explanations, or production serving.
Those controls require separate release gates.

## Historical exposure and release status

The reserved 1,409 rows are the same seed-42 partition used by the legacy
notebooks and the earlier audit. They are sealed from further access in v1.1,
but they cannot honestly be described as historically unseen. A credible future
release assessment needs a genuinely new, preferably later-in-time labeled
cohort. No final-test metrics will be published by this development workflow.

## v1.1 serving contract (specified before implementation)

The browser sends the 19 model features and optional `customer_id` to FastAPI.
The ID is for response correlation only and never enters preprocessing. Missing
IDs remain null; non-null IDs must be unique within a batch. Names stay in the
browser. Payload fields are normalized and validated before the existing saved
preprocessor and classifier execute. Neither saved artifact is replaced.

The API returns the classifier's unrounded positive-class probability. The UI
only formats that number as a percentage; it cannot add engagement heuristics.
The existing threshold 0.35 is retained as `legacy_unapproved`, not described as
approved or optimal. Binary risk bands use exactly the returned decision at that
threshold. The experimental threshold 0.135039 is never used for serving.

No calibrator is currently attached: the score is an estimated churn probability.
Invalid local attribution and causal retention offers are withdrawn. Global
impurity importance, if displayed, is explicitly a population-level model
summary without direction or causal meaning.

Categories follow the fitted vocabulary. Serving guardrails are tenure 0–120
months, MonthlyCharges 0–1,000, TotalCharges 0–120,000; these operational caps
are not asserted training ranges. Null TotalCharges becomes 0 only for tenure=0;
for positive tenure an observed non-negative value is required. tenure=0 requires
zero TotalCharges. Historical spend is never recomputed from a current discount.
Values are finite; strings are trimmed and bounded; contradictory services are
rejected. The six internet add-ons must say 'No internet service' exactly when
InternetService='No'. PhoneService and MultipleLines must agree.
