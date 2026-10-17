# CRE Default Risk — End-to-End MLOps Mini-Stack

A small, complete machine-learning system for **commercial real-estate loan default risk**: train → track → register → serve → monitor → retrain, with Docker and CI.

Built to close a specific gap: I have owned the *substance* of the model lifecycle for years (deployment, monitoring, drift, retraining), but had done it with my own tooling rather than with MLflow and Docker. This project does it with the standard stack.

---

## Why these choices

**The domain is deliberate.** Structured/tabular financial data — LTV, DSCR, occupancy, cap rate, sponsor experience — is where gradient-boosted trees earn their keep, and it mirrors how a real-estate lender actually underwrites.

**The modeling discipline is deliberate too.** An explainable logistic-regression baseline is fit *first*. A more complex model only ships if it beats that baseline, and CI fails the build if it doesn't. Complexity has to be earned.

Current result on the held-out set — five candidates, bagging and boosting tested against the baseline:

| model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| `logreg_baseline` (explainable) | 0.7399 | 0.3255 | 0.1985 |
| **`xgb_shallow`** (depth 3) — *champion* | **0.7416** | **0.3343** | 0.1693 |
| `xgb_tuned` (depth 5) | 0.7239 | 0.3097 | 0.1325 |
| `random_forest` (100 trees, depth 10) | 0.7377 | 0.3143 | 0.1322 |
| `random_forest_tuned` (400 trees, depth 25) | 0.7398 | 0.3193 | 0.0887 |

The shallow boosted model wins, beating the baseline by **+2.71% PR-AUC**. Three findings matter more than the ranking itself.

**Deeper boosting hurt.** `xgb_tuned` at depth 5 is the worst ranker of the five. On a rare-event target (~11% default rate), the extra capacity overfits. That is the argument for starting simple, stated as a measurement rather than an opinion.

**Neither random forest beat the explainable baseline on PR-AUC** — 0.3143 and 0.3193 against the baseline's 0.3255. Bagging reduces variance but does not chase residual structure the way boosting does, and here that mattered. A 400-tree forest is a far more expensive model than a logistic regression it fails to outrank, so by this repo's own shipping rule it does not ship. Note also that extra capacity *helped* the forest while it hurt the booster: independently grown trees average their errors out, sequentially fitted ones compound them.

**The metrics disagree, which is the most useful finding here.** `random_forest_tuned` has the best Brier score of all five (0.0887, against the champion's 0.1693) while ranking worse. Brier measures calibration; PR-AUC measures ranking. Averaging over 400 trees produces better-calibrated probabilities here, while the booster's weighted objective and the baseline's `class_weight="balanced"` both push scores away from true rates to handle the class imbalance. So the right champion depends on the decision: to *rank* loans for review, take the booster; to feed a probability into an expected-loss calculation, the forest's calibration may be worth more than its ranking. This registry promotes on PR-AUC, which is a choice worth making explicitly rather than by default.

---

## Explainability — and the bug it caught

Each training run now logs two kinds of explanation, and the difference between them matters.

**Built-in importance** (`plots/feature_importance.png`, `feature_importance.csv`). The method depends on the model and is now named in the plot title, the CSV and a `builtin_importance_method` tag: **gain** for XGBoost (average loss reduction from splits on a feature), **mean decrease in impurity** for the random forests, and **|coefficient|** on standardized inputs for the logistic baseline. These are useful as a quick sanity check, but they have no direction (they say a feature matters, never which way), they are global only, their units differ across models so they can't be compared, and impurity-based importance is biased toward continuous, high-cardinality features. In this repo they ranked pure-noise variables like `noi` and `cap_rate` surprisingly high.

**SHAP** (`plots/shap_beeswarm.png`, `plots/shap_dependence.png`, `explainability/shap_summary.csv`, plus `shap_mean_abs_*` metrics for the top five drivers). `TreeExplainer` gives exact contributions for XGBoost (in log-odds) and the forests (in probability), and `LinearExplainer` covers the baseline. Each prediction decomposes as *base value + sum of feature contributions = model output*, and a test checks that additivity. One-hot dummies are summed back into their parent variable, so `property_type` appears as one driver.

Champion (`xgb_shallow`) drivers, by mean |SHAP| in log-odds:

| feature | mean \|SHAP\| | direction |
|---|---|---|
| `ltv` | 0.504 | higher → more risk |
| `dscr` | 0.472 | higher → less risk |
| `sponsor_experience_years` | 0.397 | higher → less risk |
| `property_type` | 0.320 | riskiest office, safest industrial |
| `interest_rate` | 0.194 | higher → more risk |
| `occupancy_rate` | 0.184 | higher → less risk |

**The bug.** The first SHAP dependence plot for DSCR bent the wrong way: below about 1.2x coverage, predicted risk *fell* as DSCR fell. Any credit analyst knows that is backwards. The model wasn't wrong; it was faithfully learning the data, and the data generator had a sign error (`- 2.40 * max(1.20 - dscr, 0)` where it should have been `+`). Built-in importance could never have caught this, because it only said DSCR mattered, not in which direction. Fixed in `data/generate_data.py`, with a regression test (`tests/test_explainability.py`) that fails on the old sign. The DSCR dependence curve is now monotonic: risk climbs steadily as coverage falls.

Two caveats. SHAP explains the **model**, not the world: a large contribution means the model relies on a feature, not that changing it would cause a default. And because the fix changed the data, every number in this README was re-run after it.

`shap` is in `requirements-dev.txt` only, so the serving image doesn't grow. If it's missing, training logs a warning and carries on.

---

## Quick start

```bash
pip install -r requirements-dev.txt

python data/generate_data.py   # synthetic CRE portfolio (3 vintages)
python src/train.py            # train, track, register champion
python src/monitor.py          # drift / skew / concept drift / latency
pytest tests/ -v               # 20 tests
mlflow ui                      # http://localhost:5000
```

With Docker (MLflow server + API as two services):

```bash
docker compose up --build
# MLflow UI  -> http://localhost:5001
# API docs   -> http://localhost:8000/docs
curl http://localhost:8000/health
```

---

## What's in here

```
data/generate_data.py   synthetic CRE portfolio: reference + drifted vintages
src/train.py            baseline vs GBT vs RF, MLflow tracking + artifacts + SHAP + registry
src/monitor.py          PSI, KS, feature skew, concept drift, latency, retrain trigger
src/api.py              FastAPI serving the registered model by alias
tests/test_api.py       15 tests: drift math + serving contract
tests/test_explainability.py  5 tests: generator rules + SHAP additivity
Dockerfile              multi-stage, non-root, HEALTHCHECK
docker-compose.yml      MLflow tracking server + API
.github/workflows/ci.yml  test → train → monitor → quality gate → build image
```

### MLflow — the three capabilities, concretely
- **Experiment tracking** — every candidate logs params, metrics (ROC-AUC, PR-AUC, Brier), and tags.
- **Artifact storage** — ROC curves, feature-importance and SHAP plots and CSVs stored per run.
- **Model registry** — the winner is registered as `cre-default-risk` and given the `@champion` alias. The API loads `models:/cre-default-risk@champion`, so **promoting a new version swaps the served model without a code change or redeploy**. (Aliases replaced stages in MLflow 2.9+.)

### The four kinds of monitoring — and why they're different

This is the distinction most people blur:

| | what changed | how it's measured here |
|---|---|---|
| **Data drift** | `P(X)` — the inputs look different | PSI per feature, KS test |
| **Concept drift** | `P(y\|X)` — the *relationship* changed | ROC-AUC / PR-AUC decay on labeled recent data |
| **Feature skew** | serving data violates the training contract | schema, null rates, out-of-range values, unseen categories |
| **Latency** | the model got slower | p50 / p95 / p99 single-row inference |

**Inputs can look identical and the model can still be broken** — that's concept drift, and it's the one you can't catch without labels.

The synthetic data is built to demonstrate this: the "current" vintage is a stressed market (rates up, occupancy down) *and* the underlying relationship shifts — office risk re-prices and DSCR starts mattering more. A model trained on the calm vintage misses it.

Latest monitoring run:

```
interest_rate    PSI 4.44  ALERT      concept drift: ROC-AUC 0.841 -> 0.787
market_vacancy   PSI 1.80  ALERT                     = 6.5% decay (under the 10% bar)
occupancy_rate   PSI 0.70  ALERT      latency: p50 4.4ms / p95 5.4ms
dscr             PSI 0.56  ALERT
ltv              PSI 0.25  ALERT      RETRAIN REQUIRED: true  (data drift)
```

The retrain fires on data drift, not concept drift: the champion's ROC-AUC fell 6.5%, below the 10% threshold. That is the trigger design working as intended. Six features moved far enough that you retrain before the labels confirm the damage, rather than waiting for them.

**One subtlety worth knowing:** PR-AUC actually *rose* on the stressed vintage (0.49 → 0.73) even as the model got worse. PR-AUC is sensitive to class balance, and the default rate rose from 11% to 41%. ROC-AUC is the honest signal here. Monitoring the wrong metric would have told you the model improved.

### Retraining triggers
Explicit rules, not vibes:
- any feature with **PSI ≥ 0.25**, or
- **ROC-AUC relative decay ≥ 10%**, or
- any ALERT-severity feature-skew violation (missing column, unseen category)

### Serving
`POST /predict` returns a probability, a **risk band** (low / moderate / elevated / high) that a credit committee can act on, and the model version that produced it. Pydantic enforces the input contract at the edge — out-of-range LTV or an unseen property type is rejected with a 422 rather than silently scored.

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "loan_amount": 6000000, "ltv": 0.82, "dscr": 1.05, "occupancy_rate": 0.74,
  "interest_rate": 7.9, "year_built": 1985, "loan_term_months": 120,
  "cap_rate": 0.071, "market_vacancy": 0.16, "sponsor_experience_years": 3.0,
  "noi": 350000, "property_type": "office", "region": "west"}'
# -> {"default_probability": 0.885, "risk_band": "high", "model_version": "1", ...}
```

### CI/CD
GitHub Actions on every push: install → generate data → train → test → monitor → **quality gate** (build fails if the champion doesn't beat the explainable baseline) → build and smoke-test the Docker image.

---

## Notes / honest limitations

- The data is **synthetic**, generated from a known non-linear process. It demonstrates the machinery, not a real credit model.
- AUC ~0.72 is modest but realistic for rare-event credit default with noisy features; the point is the *system*, not the score.
- Single-node MLflow with a SQLite backend. A production deployment would use a managed tracking server, object storage for artifacts, and a real feature store.
- No authentication on the API. Production would add authn/authz, TLS, and PII controls.
