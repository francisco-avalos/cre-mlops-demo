"""
Model governance & monitoring.

CIM's posting asks for "automated monitoring for production models to detect
data drift, concept drift, feature skew, and latency degradation" plus
"retraining triggers". This script implements all four, and logs the results
back to MLflow so drift is tracked over time like any other metric.

The four things people conflate -- and how we measure each here:

  DATA DRIFT      P(X) changed. Inputs look different.        -> PSI, KS test
  CONCEPT DRIFT   P(y|X) changed. The RELATIONSHIP changed.   -> AUC/PR-AUC decay
                  Inputs can look identical and the model still goes stale.
  FEATURE SKEW    Serving data doesn't match training contract -> schema/null/range checks
                  (missing columns, new categories, impossible values)
  LATENCY         The model got slower.                       -> p50/p95 inference time
"""
from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

NUMERIC = [
    "loan_amount", "ltv", "dscr", "occupancy_rate", "interest_rate",
    "year_built", "loan_term_months", "cap_rate", "market_vacancy",
    "sponsor_experience_years", "noi",
]
CATEGORICAL = ["property_type", "region"]
TARGET = "default_12m"
MODEL_NAME = "cre-default-risk"

# Industry-conventional PSI thresholds.
PSI_WARN, PSI_ALERT = 0.10, 0.25
# Relative performance decay we're willing to tolerate before retraining.
PERF_DECAY_ALERT = 0.10


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. The standard drift measure in credit risk."""
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if ref.nunique() <= 1:
        return 0.0
    # Quantile bins from the REFERENCE distribution (the world we trained on).
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def categorical_psi(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()))
    r = np.array([(ref == c).mean() for c in cats])
    k = np.array([(cur == c).mean() for c in cats])
    eps = 1e-6
    r, k = np.clip(r, eps, None), np.clip(k, eps, None)
    return float(np.sum((k - r) * np.log(k / r)))


def check_data_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC:
        p = psi(ref[col].values, cur[col].values)
        ks = ks_2samp(ref[col].dropna(), cur[col].dropna())
        rows.append({
            "feature": col, "type": "numeric", "psi": round(p, 4),
            "ks_stat": round(float(ks.statistic), 4),
            "ks_pvalue": round(float(ks.pvalue), 6),
            "status": "ALERT" if p >= PSI_ALERT else ("WARN" if p >= PSI_WARN else "OK"),
        })
    for col in CATEGORICAL:
        p = categorical_psi(ref[col], cur[col])
        rows.append({
            "feature": col, "type": "categorical", "psi": round(p, 4),
            "ks_stat": None, "ks_pvalue": None,
            "status": "ALERT" if p >= PSI_ALERT else ("WARN" if p >= PSI_WARN else "OK"),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def check_feature_skew(ref: pd.DataFrame, cur: pd.DataFrame) -> list[dict]:
    """Serving data vs the training contract: schema, nulls, ranges, new categories."""
    issues = []
    expected = set(NUMERIC + CATEGORICAL)
    missing = expected - set(cur.columns)
    extra = set(cur.columns) - expected - {TARGET}
    if missing:
        issues.append({"check": "missing_columns", "detail": sorted(missing), "severity": "ALERT"})
    if extra:
        issues.append({"check": "unexpected_columns", "detail": sorted(extra), "severity": "WARN"})

    for col in NUMERIC:
        if col not in cur.columns:
            continue
        null_rate = float(cur[col].isna().mean())
        if null_rate > 0.01:
            issues.append({"check": "null_rate", "detail": f"{col}={null_rate:.3f}", "severity": "WARN"})
        lo, hi = ref[col].min(), ref[col].max()
        oob = float(((cur[col] < lo) | (cur[col] > hi)).mean())
        if oob > 0.01:
            issues.append({"check": "out_of_range", "detail": f"{col}={oob:.3f} outside training range",
                           "severity": "WARN"})

    for col in CATEGORICAL:
        if col not in cur.columns:
            continue
        new_cats = set(cur[col].dropna().unique()) - set(ref[col].dropna().unique())
        if new_cats:
            issues.append({"check": "unseen_category", "detail": f"{col}: {sorted(new_cats)}",
                           "severity": "ALERT"})
    return issues


def check_concept_drift(model, ref: pd.DataFrame, cur_labeled: pd.DataFrame) -> dict:
    """Did P(y|X) change? Needs realized labels on the current vintage."""
    feats = NUMERIC + CATEGORICAL
    ref_prob = model.predict_proba(ref[feats])[:, 1]
    cur_prob = model.predict_proba(cur_labeled[feats])[:, 1]

    ref_auc = roc_auc_score(ref[TARGET], ref_prob)
    cur_auc = roc_auc_score(cur_labeled[TARGET], cur_prob)
    ref_pr = average_precision_score(ref[TARGET], ref_prob)
    cur_pr = average_precision_score(cur_labeled[TARGET], cur_prob)

    auc_decay = (ref_auc - cur_auc) / ref_auc
    pr_decay = (ref_pr - cur_pr) / ref_pr
    return {
        "reference_roc_auc": round(float(ref_auc), 4),
        "current_roc_auc": round(float(cur_auc), 4),
        "roc_auc_relative_decay": round(float(auc_decay), 4),
        "reference_pr_auc": round(float(ref_pr), 4),
        "current_pr_auc": round(float(cur_pr), 4),
        "pr_auc_relative_decay": round(float(pr_decay), 4),
        "reference_default_rate": round(float(ref[TARGET].mean()), 4),
        "current_default_rate": round(float(cur_labeled[TARGET].mean()), 4),
        "status": "ALERT" if auc_decay >= PERF_DECAY_ALERT else "OK",
    }


def check_latency(model, sample: pd.DataFrame, n: int = 200) -> dict:
    """Latency degradation: single-row inference, the serving-path case."""
    feats = NUMERIC + CATEGORICAL
    rows = sample[feats].head(n)
    times = []
    for i in range(len(rows)):
        one = rows.iloc[[i]]
        t0 = time.perf_counter()
        model.predict_proba(one)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "n_samples": len(arr),
    }


def decide_retraining(drift_df: pd.DataFrame, concept: dict, skew: list) -> dict:
    """The retraining trigger. Explicit rules beat vibes."""
    reasons = []
    alert_feats = drift_df.loc[drift_df["status"] == "ALERT", "feature"].tolist()
    if alert_feats:
        reasons.append(f"data drift PSI>={PSI_ALERT} on: {', '.join(alert_feats)}")
    if concept["status"] == "ALERT":
        reasons.append(
            f"concept drift: ROC-AUC decayed "
            f"{concept['roc_auc_relative_decay']*100:.1f}% (threshold {PERF_DECAY_ALERT*100:.0f}%)"
        )
    if any(i["severity"] == "ALERT" for i in skew):
        reasons.append("feature skew: schema/category contract violated")
    return {"retrain_required": bool(reasons), "reasons": reasons}


def main() -> dict:
    ref = pd.read_csv("data/reference.csv")
    cur = pd.read_csv("data/current.csv")
    cur_labeled = pd.read_csv("data/current_labeled.csv")

    # Load whatever is currently serving.
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    drift_df = check_data_drift(ref, cur)
    skew = check_feature_skew(ref, cur)
    concept = check_concept_drift(model, ref, cur_labeled)
    latency = check_latency(model, cur)
    decision = decide_retraining(drift_df, concept, skew)

    print("\n=== DATA DRIFT (PSI) ===")
    print(drift_df.to_string(index=False))
    print("\n=== FEATURE SKEW ===")
    print(json.dumps(skew, indent=2) if skew else "  no schema/contract violations")
    print("\n=== CONCEPT DRIFT ===")
    print(json.dumps(concept, indent=2))
    print("\n=== LATENCY ===")
    print(json.dumps(latency, indent=2))
    print("\n=== RETRAINING DECISION ===")
    print(json.dumps(decision, indent=2))

    # Log the monitoring run to MLflow so drift is tracked like any metric.
    mlflow.set_experiment("cre-default-risk-monitoring")
    with mlflow.start_run(run_name="drift_check"):
        for _, r in drift_df.iterrows():
            mlflow.log_metric(f"psi_{r['feature']}", r["psi"])
        mlflow.log_metrics({k: v for k, v in concept.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({f"latency_{k}": v for k, v in latency.items()
                            if isinstance(v, (int, float))})
        mlflow.log_metric("retrain_required", int(decision["retrain_required"]))
        mlflow.set_tag("decision", "; ".join(decision["reasons"]) or "no action")
        drift_df.to_csv("drift_report.csv", index=False)
        mlflow.log_artifact("drift_report.csv", artifact_path="monitoring")

    report = {"data_drift": drift_df.to_dict("records"), "feature_skew": skew,
              "concept_drift": concept, "latency": latency, "decision": decision}
    with open("monitoring_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return report


if __name__ == "__main__":
    main()
