"""
Explainability and data-generator regression tests.

The DSCR tests exist because SHAP caught an inverted DSCR effect in an earlier
version of data/generate_data.py: a sign error made low debt-service coverage
*reduce* default risk. These tests pin the intended direction of the key credit
drivers, so that kind of bug fails CI instead of shipping.
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_generator():
    path = os.path.join(ROOT, "data", "generate_data.py")
    spec = importlib.util.spec_from_file_location("generate_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def portfolio():
    """A large calm-market vintage, so default-rate comparisons aren't noise."""
    return _load_generator()._simulate(60_000, np.random.default_rng(123), stressed=False)


def _default_rate(df, mask):
    return df.loc[mask, "default_12m"].mean()


# ---------- the credit logic the synthetic data must respect -----------------
def test_risk_keeps_rising_as_dscr_falls_below_the_covenant(portfolio):
    """The regression test for the sign bug: below DSCR 1.20, lower coverage must
    mean MORE risk. The buggy generator made DSCR < 1.0 safer than 1.10-1.20."""
    d = portfolio["dscr"]
    assert _default_rate(portfolio, d < 1.00) > _default_rate(portfolio, (d >= 1.10) & (d < 1.20))


def test_low_dscr_defaults_more_than_healthy_dscr(portfolio):
    d = portfolio["dscr"]
    assert _default_rate(portfolio, d < 1.10) > _default_rate(portfolio, (d >= 1.35) & (d < 1.60))


def test_high_ltv_defaults_more(portfolio):
    ltv = portfolio["ltv"]
    assert _default_rate(portfolio, ltv > 0.80) > _default_rate(portfolio, ltv < 0.65)


def test_more_sponsor_experience_lowers_risk(portfolio):
    s = portfolio["sponsor_experience_years"]
    assert _default_rate(portfolio, s < 3) > _default_rate(portfolio, s > 15)


# ---------- SHAP itself -------------------------------------------------------
def test_shap_contributions_add_up_to_model_output():
    """SHAP's defining property: base value + contributions = the model's output
    (log-odds for XGBoost). If this fails, the explanations can't be trusted."""
    shap = pytest.importorskip("shap")
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = (X[:, 0] - X[:, 1] + rng.normal(size=500) > 0).astype(int)
    model = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss").fit(X, y)

    explainer = shap.TreeExplainer(model)
    contributions = explainer.shap_values(X)
    base = float(np.atleast_1d(explainer.expected_value)[0])
    margin = model.predict(X, output_margin=True)
    assert np.allclose(base + contributions.sum(axis=1), margin, atol=1e-4)
