"""
Train CRE loan-default models and manage them with MLflow.

Demonstrates the three MLflow capabilities CIM's posting names explicitly:
  1. Experiment tracking  -- params, metrics, tags for every run
  2. Artifact storage     -- plots, feature importances, the model itself
  3. Model registry       -- register the winner, alias it for serving

Philosophy note (matches the JD): we fit an explainable logistic-regression
baseline FIRST, then only "earn" the gradient-boosted model by beating it.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC = [
    "loan_amount", "ltv", "dscr", "occupancy_rate", "interest_rate",
    "year_built", "loan_term_months", "cap_rate", "market_vacancy",
    "sponsor_experience_years", "noi",
]
CATEGORICAL = ["property_type", "region"]
TARGET = "default_12m"
MODEL_NAME = "cre-default-risk"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def evaluate(y_true, y_prob) -> dict:
    """Metrics that actually matter for a rare-event credit model."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def log_roc_plot(y_true, y_prob, label: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC - {label}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "roc_curve.png")
        fig.savefig(path, dpi=120)
        mlflow.log_artifact(path, artifact_path="plots")
    plt.close(fig)


def log_feature_importance(model: Pipeline, label: str) -> None:
    """Explainability: which features drive the score."""
    try:
        pre = model.named_steps["pre"]
        names = list(NUMERIC) + list(
            pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
        )
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
        else:
            vals = np.abs(clf.coef_[0])
        imp = pd.DataFrame({"feature": names, "importance": vals})
        imp = imp.sort_values("importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(imp["feature"][::-1], imp["importance"][::-1])
        ax.set_title(f"Top features - {label}")
        fig.tight_layout()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "feature_importance.png")
            fig.savefig(p, dpi=120)
            mlflow.log_artifact(p, artifact_path="plots")
            c = os.path.join(d, "feature_importance.csv")
            imp.to_csv(c, index=False)
            mlflow.log_artifact(c, artifact_path="explainability")
        plt.close(fig)
    except Exception as exc:  # explainability is nice-to-have, never fatal
        print(f"[warn] feature importance skipped: {exc}")


def run_experiment(data_path: str = "data/reference.csv") -> dict:
    df = pd.read_csv(data_path)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Handle the class imbalance explicitly (~4% default rate).
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    candidates = {
        # The explainable baseline. If this wins, we ship this.
        "logreg_baseline": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0
        ),
        "xgb_shallow": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            tree_method="hist", random_state=42,
        ),
        "xgb_tuned": XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.8, reg_lambda=3.0,
            min_child_weight=3, gamma=0.1,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            tree_method="hist", random_state=42,
        ),
    }

    mlflow.set_experiment("cre-default-risk")
    results = {}

    for name, clf in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
            pipe.fit(X_tr, y_tr)

            prob = pipe.predict_proba(X_te)[:, 1]
            metrics = evaluate(y_te, prob)

            mlflow.log_param("model_family", name)
            mlflow.log_params(
                {f"hp_{k}": v for k, v in clf.get_params().items()
                 if isinstance(v, (int, float, str, bool)) and v is not None}
            )
            mlflow.log_param("n_train", len(X_tr))
            mlflow.log_param("pos_weight", round(pos_weight, 3))
            mlflow.log_metrics(metrics)
            mlflow.set_tag("vintage", "reference")
            mlflow.set_tag("use_case", "cre_loan_default_12m")

            log_roc_plot(y_te, prob, name)
            log_feature_importance(pipe, name)

            signature = infer_signature(X_te, prob)
            # cloudpickle keeps this portable across MLflow 2.x and 3.x.
            # (MLflow 3 defaults to skops, which rejects XGBoost types.)
            try:
                mlflow.sklearn.log_model(
                    pipe, name="model", signature=signature,
                    input_example=X_te.head(3),
                    serialization_format="cloudpickle",
                )
            except TypeError:  # older MLflow uses artifact_path=
                mlflow.sklearn.log_model(
                    pipe, artifact_path="model", signature=signature,
                    input_example=X_te.head(3),
                    serialization_format="cloudpickle",
                )

            results[name] = {"run_id": run.info.run_id, **metrics}
            print(f"{name:18s} AUC={metrics['roc_auc']:.4f}  "
                  f"PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier']:.4f}")

    # ---- Pick the winner and register it -----------------------------------
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best = results[best_name]
    baseline_pr = results["logreg_baseline"]["pr_auc"]
    lift = (best["pr_auc"] - baseline_pr) / baseline_pr * 100

    print(f"\nWinner: {best_name}  (PR-AUC {best['pr_auc']:.4f}, "
          f"{lift:+.1f}% vs explainable baseline)")

    client = MlflowClient()
    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {mv.version}")

    # Aliases are the modern replacement for stages in MLflow 2.9+/3.x.
    for alias in ("champion", "serving"):
        try:
            client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
            print(f"  alias @{alias} -> v{mv.version}")
        except Exception as exc:
            print(f"  [warn] alias {alias} not set: {exc}")

    client.set_model_version_tag(MODEL_NAME, mv.version, "baseline_pr_auc",
                                 f"{baseline_pr:.4f}")
    client.set_model_version_tag(MODEL_NAME, mv.version, "winner", best_name)

    summary = {
        "best_model": best_name,
        "registered_version": mv.version,
        "metrics": {k: v for k, v in best.items() if k != "run_id"},
        "baseline_pr_auc": baseline_pr,
        "lift_over_baseline_pct": round(lift, 2),
    }
    with open("training_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/reference.csv")
    args = ap.parse_args()
    run_experiment(args.data)
