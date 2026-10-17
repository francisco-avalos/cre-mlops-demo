"""
Train CRE loan-default models and manage them with MLflow.

Demonstrates the three MLflow capabilities CIM's posting names explicitly:
  1. Experiment tracking  -- params, metrics, tags for every run
  2. Artifact storage     -- plots, feature importances, SHAP explanations, the model itself
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
from sklearn.ensemble import RandomForestClassifier
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


def feature_names(pre: ColumnTransformer) -> list[str]:
    """Column names after preprocessing: numerics, then one-hot dummies."""
    return list(NUMERIC) + list(
        pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
    )


def parent_feature(name: str) -> str:
    """Map a one-hot dummy (e.g. 'property_type_office') back to its variable."""
    for cat in CATEGORICAL:
        if name.startswith(cat + "_"):
            return cat
    return name


def importance_method(clf) -> str:
    """Name the built-in importance each model type reports."""
    if isinstance(clf, XGBClassifier):
        return "XGBoost gain (avg. loss reduction from splits on the feature)"
    if isinstance(clf, RandomForestClassifier):
        return "mean decrease in impurity (Gini importance)"
    return "|coefficient| on standardized features"


def log_feature_importance(model: Pipeline, label: str) -> None:
    """Built-in importance: a quick global ranking of features.

    Limits, and why log_shap() exists alongside it:
      - no direction: gain and impurity are always positive, and the logistic
        branch uses |coef|, so you can't tell whether a feature raises or lowers risk;
      - global only: nothing about why one particular loan scored high;
      - different units per model type, so plots aren't comparable across candidates;
      - impurity importance is computed on training data and favors continuous features.
    """
    try:
        pre = model.named_steps["pre"]
        clf = model.named_steps["clf"]
        names = feature_names(pre)
        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
        else:
            vals = np.abs(clf.coef_[0])
        method = importance_method(clf)
        imp = pd.DataFrame({"feature": names, "importance": vals, "method": method})
        imp = imp.sort_values("importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(imp["feature"][::-1], imp["importance"][::-1])
        ax.set_title(f"Top features - {label}\n({method})", fontsize=10)
        fig.tight_layout()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "feature_importance.png")
            fig.savefig(p, dpi=120)
            mlflow.log_artifact(p, artifact_path="plots")
            c = os.path.join(d, "feature_importance.csv")
            imp.to_csv(c, index=False)
            mlflow.log_artifact(c, artifact_path="explainability")
        plt.close(fig)
        mlflow.set_tag("builtin_importance_method", method)
    except Exception as exc:  # explainability is nice-to-have, never fatal
        print(f"[warn] feature importance skipped: {exc}")


def log_shap(model: Pipeline, X_eval: pd.DataFrame, label: str,
             max_rows: int = 1000) -> None:
    """SHAP explanations on held-out loans.

    SHAP splits every prediction into additive feature contributions
    (base value + contributions = model output), so it shows which way each
    feature pushes default risk, by how much, and for each individual loan.
    Built-in importances can't: an inverted DSCR effect in an earlier version
    of the data generator was invisible to them and obvious in SHAP.

    Units: log-odds for XGBoost and logistic regression; probability for the
    random forest (TreeExplainer explains each model's raw output).
    Logs: a beeswarm summary, dependence plots for the top numeric features,
    and a CSV of mean |SHAP| with the direction of each feature's effect.
    """
    try:
        import shap
    except ImportError:
        print("[warn] shap not installed; skipping SHAP (pip install shap)")
        return
    try:
        pre, clf = model.named_steps["pre"], model.named_steps["clf"]
        names = feature_names(pre)
        is_forest = isinstance(clf, RandomForestClassifier)
        n = min(300 if is_forest else max_rows, len(X_eval))  # deep forests are slow to explain
        sample = X_eval.sample(n, random_state=42)
        Xt = pre.transform(sample)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

        if isinstance(clf, (XGBClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.LinearExplainer(clf, Xt)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):          # older shap: one array per class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # newer shap: (rows, features, classes)
            sv = sv[:, :, 1]
        units = "probability" if is_forest else "log-odds"

        S = pd.DataFrame(sv, columns=names)
        raw = sample.reset_index(drop=True)

        # Global view: mean |SHAP|, with one-hot dummies summed back to their variable.
        glob = (S.abs().T.groupby(parent_feature).sum().T.mean()
                .sort_values(ascending=False))
        # Direction: does a higher value push risk up or down? Which category is riskiest?
        direction = {}
        for f in glob.index:
            if f in NUMERIC:
                r = np.corrcoef(raw[f], S[f])[0, 1]
                direction[f] = "higher -> more risk" if r > 0 else "higher -> less risk"
            else:
                lvl = pd.Series({
                    c[len(f) + 1:]: S.loc[Xt[:, names.index(c)] == 1, c].mean()
                    for c in names if parent_feature(c) == f
                }).dropna()
                direction[f] = f"riskiest: {lvl.idxmax()}; safest: {lvl.idxmin()}"
        summary = pd.DataFrame({
            "feature": glob.index,
            f"mean_abs_shap_{units}": glob.values.round(4),
            "direction": [direction[f] for f in glob.index],
        })

        with tempfile.TemporaryDirectory() as d:
            shap.summary_plot(sv, pd.DataFrame(Xt, columns=names),
                              show=False, max_display=12)
            plt.title(f"SHAP summary - {label} ({units})", fontsize=10)
            plt.tight_layout()
            p = os.path.join(d, "shap_beeswarm.png")
            plt.savefig(p, dpi=120)
            plt.close("all")
            mlflow.log_artifact(p, artifact_path="plots")

            top_num = [f for f in glob.index if f in NUMERIC][:3]
            fig, axes = plt.subplots(1, len(top_num), figsize=(4.5 * len(top_num), 3.8))
            for ax, f in zip(np.atleast_1d(axes), top_num):
                ax.scatter(raw[f], S[f], s=5, alpha=0.4)
                ax.axhline(0, color="gray", lw=0.8)
                ax.set_xlabel(f)
                ax.set_ylabel(f"SHAP ({units})")
            fig.suptitle(f"SHAP dependence - {label}")
            fig.tight_layout()
            p = os.path.join(d, "shap_dependence.png")
            fig.savefig(p, dpi=120)
            plt.close(fig)
            mlflow.log_artifact(p, artifact_path="plots")

            c = os.path.join(d, "shap_summary.csv")
            summary.to_csv(c, index=False)
            mlflow.log_artifact(c, artifact_path="explainability")

        for f, v in glob.head(5).items():
            mlflow.log_metric(f"shap_mean_abs_{f}", float(v))
        top3 = "; ".join(f"{f} ({direction[f]})" for f in glob.index[:3])
        print(f"   SHAP top drivers: {top3}")
    except Exception as exc:  # explainability is nice-to-have, never fatal
        print(f"[warn] SHAP skipped: {exc}")


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
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample", random_state=42,
            n_estimators=100, max_depth=10
        ),
        "random_forest_tuned": RandomForestClassifier(
            class_weight="balanced_subsample", random_state=42,
            n_estimators=400, max_depth=25
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
            log_shap(pipe, X_te, name)

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
