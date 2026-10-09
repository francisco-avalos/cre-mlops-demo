# CRE Default Risk — End-to-End MLOps Mini-Stack

A small, complete machine-learning system for **commercial real-estate loan default risk**: train → track → register → serve → monitor → retrain, with Docker and CI.

Built to close a specific gap: I have owned the *substance* of the model lifecycle for years (deployment, monitoring, drift, retraining), but had done it with my own tooling rather than with MLflow and Docker. This project does it with the standard stack.

---

## Why these choices

**The domain is deliberate.** Structured/tabular financial data — LTV, DSCR, occupancy, cap rate, sponsor experience — is where gradient-boosted trees earn their keep, and it mirrors how a real-estate lender actually underwrites.

**The modeling discipline is deliberate too.** An explainable logistic-regression baseline is fit *first*. The gradient-boosted model only ships if it beats that baseline, and CI fails the build if it doesn't. Complexity has to be earned.

Current result on the held-out set:

| model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| `logreg_baseline` (explainable) | 0.718 | 0.267 | 0.201 |
| **`xgb_shallow`** (depth 3) — *champion* | **0.724** | **0.276** | 0.167 |
| `xgb_tuned` (depth 5) | 0.695 | 0.258 | 0.119 |

The shallow boosted model beats the baseline by **+3.5% PR-AUC**. The *deeper* model is worse — it overfits a rare-event target (~9% default rate). That's the whole argument for starting simple.

---

## Quick start

```bash
pip install -r requirements-dev.txt

python data/generate_data.py   # synthetic CRE portfolio (3 vintages)
python src/train.py            # train, track, register champion
python src/monitor.py          # drift / skew / concept drift / latency
pytest tests/ -v               # 15 tests
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
src/train.py            baseline vs GBT, MLflow tracking + artifacts + registry
src/monitor.py          PSI, KS, feature skew, concept drift, latency, retrain trigger
src/api.py              FastAPI serving the registered model by alias
tests/test_api.py       15 tests: drift math + serving contract
Dockerfile              multi-stage, non-root, HEALTHCHECK
docker-compose.yml      MLflow tracking server + API
.github/workflows/ci.yml  test → train → monitor → quality gate → build image
```

### MLflow — the three capabilities, concretely
- **Experiment tracking** — every candidate logs params, metrics (ROC-AUC, PR-AUC, Brier), and tags.
- **Artifact storage** — ROC curves, feature-importance plots and CSVs stored per run.
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
interest_rate    PSI 4.44  ALERT      concept drift: ROC-AUC 0.838 -> 0.724
market_vacancy   PSI 1.80  ALERT                     = 13.6% relative decay
occupancy_rate   PSI 0.70  ALERT      latency: p50 4.9ms / p95 6.9ms
dscr             PSI 0.56  ALERT
ltv              PSI 0.25  ALERT      RETRAIN REQUIRED: true
```

**One subtlety worth knowing:** PR-AUC actually *rose* on the stressed vintage (0.43 → 0.56) even as the model got worse. PR-AUC is sensitive to class balance, and the default rate tripled (9% → 31%). ROC-AUC is the honest signal here. Monitoring the wrong metric would have told you the model improved.

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
