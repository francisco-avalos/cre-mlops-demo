# Instructions
### CRE Default Risk — MLOps Demo: how to run it, and what every file does

Two parts:
- **Part 1** — step-by-step execution, start to finish
- **Part 2** — every file explained (who / what / when / where / why / how)

---
---

# PART 1 — Running the system end to end

## What you need first

| Requirement | Check with | Notes |
|---|---|---|
| Python 3.9–3.12 | `python --version` | 3.11 is safest. Python 3.13 works but some libraries lag. |
| pip | `pip --version` | |
| Docker Desktop | `docker --version` | Only needed for Steps 7–9. Must be **running**, not just installed. |
| ~2 GB free disk | | Docker images are the bulk of it. |

---

## Step 0 — Unpack and enter the project

```bash
unzip cim-mlops-demo.zip
cd cim-mlops-demo
```

You should see `src/`, `data/`, `tests/`, `Dockerfile`, `docker-compose.yml`, `Makefile`.

---

## Step 1 — Create an isolated environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell
```

Your prompt should now show `(.venv)`.

**Using conda instead?** That's fine — `conda create -n mlops_cim_env python=3.11 && conda activate mlops_cim_env` works just as well. On macOS, conda has one advantage: `conda install -c conda-forge xgboost` pulls in the OpenMP runtime automatically (see troubleshooting).

**Why bother:** installing MLflow and XGBoost into your system Python risks breaking other projects. A virtual environment is its own sandbox — and "did you use a venv?" is a real signal of engineering hygiene.

---

## Step 2 — Install dependencies

```bash
pip install -r requirements-dev.txt
```

Takes 1–3 minutes. `requirements-dev.txt` pulls in `requirements.txt` plus the test tools.

Verify:
```bash
python -c "import mlflow, xgboost, sklearn, fastapi; print('ok', mlflow.__version__)"
```

---

## Step 3 — Generate the data

```bash
python data/generate_data.py
```

Expected output:
```
reference.csv       (20000, 14)  default rate 0.091
current.csv         (6000, 13)  (unlabeled, drifted)
current_labeled.csv (6000, 14)  default rate 0.306
```

Three CSVs now exist in `data/`. **The default rate jumping 9% → 31% is the drift you'll detect in Step 5.**

---

## Step 4 — Train, track, and register

```bash
python src/train.py
```

Expected output:
```
logreg_baseline    AUC=0.7184  PR-AUC=0.2670  Brier=0.2014
xgb_shallow        AUC=0.7241  PR-AUC=0.2764  Brier=0.1670
xgb_tuned          AUC=0.6949  PR-AUC=0.2577  Brier=0.1189

Winner: xgb_shallow  (PR-AUC 0.2764, +3.5% vs explainable baseline)
Registered cre-default-risk version 1
  alias @champion -> v1
  alias @serving -> v1
```

What just happened: three models trained, each logged as an MLflow run with params/metrics/plots; the winner registered as `cre-default-risk` v1 and given the `@champion` alias.

A new `mlruns/` folder appeared — that's MLflow's local store.

> **Note the depth-5 model losing to depth-3.** That's the "don't add complexity you haven't earned" point, and it's a real result, not a talking point you invented.

---

## Step 5 — Look at it in the MLflow UI ⭐

```bash
mlflow ui
```

Open **http://localhost:5000**. **Do not skip this step** — this is where MLflow stops being abstract.

Walk through:
1. **Experiments → `cre-default-risk`** — your three runs side by side.
2. Click a run → **Parameters**, **Metrics**, **Artifacts** (ROC curve, feature-importance plot and CSV).
3. Select all three → **Compare** — metrics charted against each other.
4. **Models** tab (top nav) → `cre-default-risk` → version 1, with `@champion` and `@serving` aliases and your tags.

Spend ten minutes clicking around. Stop with `Ctrl+C`.

---

## Step 6 — Run the monitoring

Open a **second terminal** (leave the UI running), activate the venv, then:

```bash
python src/monitor.py
```

Expected (abridged):
```
=== DATA DRIFT (PSI) ===
interest_rate    PSI 4.44  ALERT
market_vacancy   PSI 1.80  ALERT
occupancy_rate   PSI 0.70  ALERT
...
loan_amount      PSI 0.00  OK

=== CONCEPT DRIFT ===
reference_roc_auc 0.8383 -> current_roc_auc 0.7245  (13.6% decay)  ALERT

=== RETRAINING DECISION ===
retrain_required: true
```

Three things to notice:
- Six features alert; five stay OK. **A monitor that flags everything is useless.**
- ROC-AUC decayed 13.6% → concept drift.
- PR-AUC went *up* (0.43 → 0.56) while the model got worse, because the base rate tripled. **Watching the wrong metric would have told you it improved.**

Writes `drift_report.csv`, `monitoring_report.json`, and a run in the `cre-default-risk-monitoring` experiment.

---

## Step 7 — Run the tests

```bash
pytest tests/ -v
```

Expect **15 passed** (14 passed + 1 skipped if you haven't trained yet). These cover the drift math and the serving contract.

---

## Step 8 — Serve the model locally

```bash
uvicorn src.api:app --reload --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger docs. Or from a third terminal:

```bash
curl http://localhost:8000/health

# A distressed office loan
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "loan_amount": 6000000, "ltv": 0.82, "dscr": 1.05, "occupancy_rate": 0.74,
  "interest_rate": 7.9, "year_built": 1985, "loan_term_months": 120,
  "cap_rate": 0.071, "market_vacancy": 0.16, "sponsor_experience_years": 3.0,
  "noi": 350000, "property_type": "office", "region": "west"}'
# -> {"default_probability": 0.885, "risk_band": "high", "model_version": "1", ...}
```

Try a **bad** request to see the contract enforced:
```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"loan_amount": 6000000, "ltv": 9.9, "property_type": "datacenter"}'
# -> 422 Unprocessable Entity
```

That 422 is feature skew being stopped at the edge instead of silently scored. Stop with `Ctrl+C`.

---

## Step 9 — Run it in Docker

Make sure Docker Desktop is **running**, then:

```bash
docker compose up --build
```

First build takes 3–6 minutes. You get two containers:
- **MLflow tracking server** → http://localhost:5001
- **Model API** → http://localhost:8000/docs

```bash
docker ps                    # both containers, health status
docker compose logs api      # API logs
docker compose down          # stop and remove
```

> **Expected on a fresh machine:** the API container starts but reports `model_not_loaded`, because the containerized MLflow server has an empty registry — your model lives in the local `mlruns/` from Step 4. **This is not a bug.** Containers are isolated; that's the point. To populate it, point training at the server first:
> ```bash
> export MLFLOW_TRACKING_URI=http://localhost:5001
> python src/train.py
> docker compose restart api
> ```

---

## Step 10 — Push to GitHub and let CI run

```bash
git init
git add .
git commit -m "CRE default risk MLOps demo"
git branch -M main
git remote add origin https://github.com/<you>/cre-mlops-demo.git
git push -u origin main
```

Open the **Actions** tab. The pipeline runs: install → data → train → test → monitor → quality gate → build and smoke-test the Docker image.

**Get the green check before your interview.** "Here's the repo" lands very differently with a passing build.

---

## The whole thing, condensed

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python data/generate_data.py
python src/train.py
python src/monitor.py
pytest tests/ -v
mlflow ui                      # http://localhost:5000
uvicorn src.api:app --port 8000  # http://localhost:8000/docs
docker compose up --build      # http://localhost:5001 + :8000
```

Or with the Makefile: `make setup && make data && make train && make monitor && make test`

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | venv not active | `source .venv/bin/activate` |
| `No such file: data/reference.csv` | skipped Step 3 | `python data/generate_data.py` |
| `RESOURCE_DOES_NOT_EXIST: cre-default-risk` | no registered model | run `python src/train.py` |
| MLflow UI empty | wrong directory | run `mlflow ui` from the project root (where `mlruns/` is) |
| `Port 5000 already in use` | macOS AirPlay Receiver | `mlflow ui --port 5050`, or disable AirPlay Receiver |
| `Cannot connect to Docker daemon` | Docker Desktop not running | start Docker Desktop, wait for the whale icon |
| `untrusted types ... xgboost` | MLflow 3 skops serialization | already handled — `train.py` uses `serialization_format="cloudpickle"` |
| `XGBoostError: libxgboost.dylib could not be loaded` / `Library not loaded: @rpath/libomp.dylib` | **macOS only.** XGBoost needs the OpenMP runtime, which macOS doesn't ship. | `brew install libomp` — or, in a conda env: `pip uninstall xgboost -y && conda install -c conda-forge xgboost` (conda-forge bundles OpenMP as a dependency). Third option: `conda install -c conda-forge llvm-openmp`. |
| API returns `model_not_loaded` in Docker | container registry is empty | see the note in Step 9 |

**Start over clean:** `make clean` (or `rm -rf mlruns mlartifacts mlflow-store *.json drift_report.csv data/*.csv`)

---
---

# PART 2 — What every file does

Thirteen files. Each entry answers who, what, when, where, why, and how.

---

## `data/generate_data.py` — the data factory

| | |
|---|---|
| **What** | Creates three synthetic CRE loan datasets: a calm training vintage, a stressed serving vintage, and the stressed vintage with realized outcomes. |
| **Who** | You, once at setup. In a real system this would be replaced by a pipeline reading a loan warehouse. |
| **When** | First. Nothing else runs without it. |
| **Where** | Writes `data/reference.csv`, `data/current.csv`, `data/current_labeled.csv`. |
| **Why** | Self-contained (no downloads, no credentials) and — critically — **lets us control the drift**. With real data you can't guarantee a demo contains concept drift. Here we inject it deliberately. |
| **How** | NumPy draws each feature from a plausible distribution, then a hand-written non-linear logit converts features into default probability and a binomial draw produces the 0/1 outcome. |

**Key detail — the two regimes.** `_simulate(stressed=False)` is the calm market. `stressed=True` changes two different things:
1. **Feature distributions shift** — rates 5.4% → 7.6%, occupancy 0.92 → 0.84, vacancy up. That's **data drift**.
2. **The logit formula itself changes** — office gets an extra penalty and DSCR gets a bigger coefficient. That's **concept drift**, and it's why a model trained on the calm vintage systematically misprices office loans even when their inputs look normal.

**Key detail — deliberate non-linearity.** Threshold effects (LTV above 0.75, DSCR below 1.20), an office × vacancy interaction, log-saturating sponsor experience, and non-monotonic building age. **This is what lets a tree model legitimately beat a linear one.** Without it, logistic regression would be the true model and XGBoost could only lose.

---

## `src/train.py` — training and the model registry

| | |
|---|---|
| **What** | Trains three candidates, logs everything to MLflow, picks a winner, registers it, and aliases it for serving. |
| **Who** | You, or CI on every push. |
| **When** | After data generation; re-run whenever a retraining trigger fires. |
| **Where** | Reads `data/reference.csv`; writes to `mlruns/` and `training_summary.json`. |
| **Why** | This is the heart of the MLflow demonstration — tracking, artifacts, and registry in one script. |
| **How** | scikit-learn `Pipeline` (preprocessing + classifier) per candidate, wrapped in `mlflow.start_run()`. |

**The three candidates and why:**
- `logreg_baseline` — the explainable control. **Everything else must beat this.**
- `xgb_shallow` — depth 3, regularized. Usually the right answer for tabular data.
- `xgb_tuned` — depth 5, more trees. Included to *show* that more complexity can hurt — and it does.

**What gets logged per run:** hyperparameters, metrics (ROC-AUC, PR-AUC, Brier), tags, a ROC-curve PNG, a feature-importance PNG + CSV, and the serialized model with an inferred signature.

**Why three metrics:** ROC-AUC for ranking, **PR-AUC because the target is rare (~9%)**, and Brier for calibration — a credit model needs probabilities that *mean* something, not just correct ordering.

**Why `scale_pos_weight`:** with 9% positives, an unweighted model can score well by predicting "no default" constantly. This reweights so the rare class carries its weight.

**Selection and registry:** winner chosen by **PR-AUC** (not accuracy — accuracy is meaningless at a 9% base rate), registered as `cre-default-risk`, then given `@champion` and `@serving` aliases. **Aliases, not stages** — stages were deprecated in MLflow 2.9+.

**Why aliases matter:** the API asks for `models:/cre-default-risk@champion`. Promoting a new version repoints the alias and the API serves the new model **with no code change and no redeploy**. That's the single most useful thing the registry buys you.

**One gotcha already handled:** MLflow 3 defaults to `skops` serialization, which refuses XGBoost objects. The script uses `serialization_format="cloudpickle"`, which works on both MLflow 2.x and 3.x.

---

## `src/monitor.py` — governance and drift

| | |
|---|---|
| **What** | Detects all four production failure modes and decides whether to retrain. |
| **Who** | A scheduler in production (nightly/weekly). You, manually, here. |
| **When** | After a model is registered, against each new batch of serving data. |
| **Where** | Reads all three CSVs + the `@champion` model; writes `drift_report.csv`, `monitoring_report.json`, and an MLflow monitoring run. |
| **Why** | Models fail silently. Accuracy decays with no error, no alert, no crash — just quietly worse decisions. |
| **How** | Statistical comparison of the current batch against the reference, plus performance evaluation where labels exist. |

### The four checks

**1. Data drift — `check_data_drift()`** · *Did P(X) change?*
Computes **PSI** (Population Stability Index) per feature using quantile bins from the reference distribution, plus a **KS test**. PSI is the credit-industry standard; conventional thresholds are 0.10 (warn) and 0.25 (alert).

**2. Feature skew — `check_feature_skew()`** · *Does serving data honor the training contract?*
Missing columns, unexpected columns, null rates, values outside the training range, and **unseen categories**. Different from drift: drift is "the distribution moved," skew is "this input is invalid." A new `property_type` of `datacenter` isn't drift — the model has simply never seen it.

**3. Concept drift — `check_concept_drift()`** · *Did P(y|X) change?*
Scores the reference and the labeled current vintage, compares ROC-AUC and PR-AUC, and alerts on ≥10% relative decay. **Requires labels — this is the one you cannot catch from inputs alone.**

**4. Latency — `check_latency()`** · *Did it get slower?*
p50/p95/p99 on single-row inference, the actual serving path. Batch timing hides tail latency, which is what users feel.

### The retraining trigger — `decide_retraining()`
Fires if **any** of: a feature with PSI ≥ 0.25, ROC-AUC decay ≥ 10%, or an ALERT-severity skew violation. Explicit rules, written down, with the reasons recorded. **A trigger nobody can state precisely is not a trigger.**

### The subtlety worth knowing
On the stressed vintage PR-AUC *rises* (0.43 → 0.56) while ROC-AUC falls 13.6%. PR-AUC depends on class balance, and defaults tripled. **Monitoring PR-AUC alone would have reported an improvement on a model that got materially worse.**

---

## `src/api.py` — the serving layer

| | |
|---|---|
| **What** | FastAPI service exposing the registered model over HTTP. |
| **Who** | Any downstream consumer — an underwriting tool, a dashboard, another service. |
| **When** | Runs continuously once a model is registered. |
| **Where** | Loads from the MLflow registry; listens on port 8000. |
| **Why** | This is the "notebook prototype → production" transition. A model nobody can call isn't deployed. |
| **How** | Loads the model once at startup via a lifespan handler (not per request), then scores incoming JSON. |

**Endpoints:** `GET /health` (liveness + loaded version), `POST /predict` (one loan), `POST /batch` (many), `GET /metrics` (latency percentiles).

**The `Loan` Pydantic model is the serving contract.** Types, numeric bounds, and allowed categories are enforced at the edge — invalid input returns **422** instead of being silently scored. *This is feature-skew prevention, not just input validation.*

**Risk bands** (`<5%` low, `<15%` moderate, `<30%` elevated, else high) translate a probability into something a credit committee can act on. **Every response includes `model_version`** — when someone asks months later why a loan was scored a certain way, you can answer.

**Configurable via environment:** `MODEL_NAME`, `MODEL_ALIAS`, `MLFLOW_TRACKING_URI` — so the same image runs against dev or prod without a rebuild.

---

## `tests/test_api.py` — automated tests

| | |
|---|---|
| **What** | 15 pytest tests covering the drift math and the serving contract. |
| **Who** | You locally; GitHub Actions on every push. |
| **When** | Before every commit; automatically in CI. |
| **Where** | `pytest tests/ -v`. |
| **Why** | Data science repos are notoriously untested. Having tests is a differentiator. |
| **How** | Pure-function tests for the math; FastAPI `TestClient` for the API. |

**Drift math:** PSI is exactly 0 for identical input, stays under the WARN threshold for independent samples of the same population, is never negative (this is the test that catches a flipped log ratio), rises monotonically with shift size, exceeds 0.25 on real drift, and is demonstrably sample-size sensitive. Categorical PSI catches a mix shift; skew detection catches unseen categories and missing columns.
**Contract:** risk bands are monotonic, `/health` responds, and out-of-range LTV / unknown category / missing field each return 422.
**Model-dependent test** is skipped when no `mlruns/` exists, so the suite runs on a clean checkout.

> **Note:** the tests verify *behavior and contracts*, not model accuracy. You can't unit-test "is this model good" — that's what the CI quality gate is for.

---

## `Dockerfile` — the container image

| | |
|---|---|
| **What** | Builds a portable image running the API. |
| **Who** | Docker, locally and in CI. |
| **When** | On `docker compose up --build` or in the CI docker job. |
| **Where** | Produces an image exposing port 8000. |
| **Why** | "Works on my machine" is the oldest failure in deployment. The image pins the OS, Python, and every dependency. |
| **How** | Multi-stage build. |

**Four deliberate choices:**
- **Multi-stage** — dependencies compiled in a builder stage, only the installed packages copied forward. No build toolchain in the final image: smaller and less attack surface.
- **`python:3.11-slim`** — minimal base.
- **Non-root user** — runs as `appuser`. Basic container hygiene, and exactly the kind of thing a firm that cares about data protection asks about.
- **`HEALTHCHECK`** — the orchestrator can tell *running* from *ready*, so it doesn't route traffic to a container still loading its model.

---

## `docker-compose.yml` — multi-service orchestration

| | |
|---|---|
| **What** | Defines and wires two services: an MLflow tracking server and the API. |
| **Who** | You, via `docker compose up`. |
| **When** | When you want the full system rather than a bare API. |
| **Where** | MLflow on **5001**, API on **8000**. |
| **Why** | Shows the model lifecycle as a *system* — a registry service plus a consumer, not one script. |
| **How** | Compose builds the API from the Dockerfile, pulls the official MLflow image, and connects both to a shared network. |

**Details that matter:** MLflow uses a SQLite backend + filesystem artifact root persisted to `./mlflow-store`, so state survives restarts. The API has `depends_on: condition: service_healthy` — it won't start until MLflow passes its health check, avoiding a race on startup. The API reaches MLflow at `http://mlflow:5000` — **container-to-container DNS by service name**, not localhost. The port mapping `5001:5000` avoids clashing with a local `mlflow ui` on 5000.

---

## `.github/workflows/ci.yml` — continuous integration

| | |
|---|---|
| **What** | Runs the whole pipeline automatically on every push and pull request. |
| **Who** | GitHub Actions. |
| **When** | On push/PR to `main`. |
| **Where** | A clean Ubuntu runner — which is the point: it proves the repo works somewhere other than your laptop. |
| **Why** | Catches breakage immediately, and the quality gate stops a worse model from shipping. |
| **How** | Two jobs: `test`, then `docker` (which only runs if `test` passes). |

**The `test` job** runs the full chain — install → generate data → train → pytest → monitor — then the **quality gate**: an inline Python step that reads `training_summary.json` and **fails the build if the champion's PR-AUC doesn't beat the baseline's**. Reports upload as artifacts even on failure.

**The `docker` job** builds the image and smoke-tests it: run the container, curl `/health`, dump logs and fail if it doesn't answer.

> **This is the part most data-science repos lack**, and it's the cheapest way to show you think about reliability, not just modeling. The quality gate is the piece worth pointing at: **an automated rule that a model must justify its own complexity before it ships.**

---

## Supporting files

**`requirements.txt`** — runtime dependencies only (MLflow, scikit-learn, XGBoost, pandas, NumPy, SciPy, matplotlib, FastAPI, uvicorn, Pydantic). Version *ranges*, not exact pins: tight enough to be reproducible, loose enough not to rot. This is what the Docker image installs — the container doesn't need pytest.

**`requirements-dev.txt`** — `-r requirements.txt` plus pytest and httpx. The split keeps test tooling out of production images.

**`Makefile`** — short aliases for long commands (`make train`, `make monitor`, `make test`, `make up`, `make clean`). Self-documenting entry point: a new person can read it and know what the project *does*. (On Windows, run the underlying commands directly.)

**`.gitignore`** — excludes `mlruns/`, `mlflow-store/`, generated CSVs, JSON reports, `__pycache__/`, `.venv/`. **Generated artifacts never belong in git** — they're outputs, reproducible from code. Committing `mlruns/` is a common and telling mistake.

**`README.md`** — the front door: what it is, why the choices, quick start, results table, and honest limitations.

**`data_dictionary.md`** — every field defined with CRE context (LTV, DSCR, cap rate, NOI), how they relate through accounting identities, observed distributions, and the drift made concrete.

---

## How it all fits together

```
generate_data.py
      │  writes reference.csv / current.csv / current_labeled.csv
      ▼
   train.py ──────► MLflow: runs, metrics, artifacts
      │             └─► Registry: cre-default-risk  @champion
      │                        │
      ├────────────────────────┤
      ▼                        ▼
  monitor.py                api.py
  reads @champion           serves @champion
  PSI / KS / concept        POST /predict → probability + risk band
  drift / skew / latency    Pydantic rejects bad input at the edge
      │
      ▼
  retrain required? ──yes──► re-run train.py, register v2,
                             move @champion → v2
                             (API picks it up; no code change)

  tests/ ── run locally and in CI
  ci.yml ── install → data → train → test → monitor → gate → image
  Docker ── packages api.py; compose adds the MLflow server
```

**The loop is the point.** Train → register → serve → monitor → trigger → retrain → re-register, and the alias indirection means serving never has to be touched. That is the lifecycle, and it's what "owning the full model lifecycle" actually looks like in practice.
