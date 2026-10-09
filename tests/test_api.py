"""
Automated tests for the serving path and the drift logic.

CIM lists "Familiarity with Git-based CI/CD workflows and automated testing for
data science repositories" under preferred skills. These run in GitHub Actions
on every push (see .github/workflows/ci.yml).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api import app, risk_band  # noqa: E402
from monitor import categorical_psi, check_feature_skew, psi  # noqa: E402

client = TestClient(app)

VALID_LOAN = {
    "loan_amount": 6_000_000, "ltv": 0.68, "dscr": 1.35,
    "occupancy_rate": 0.91, "interest_rate": 5.8, "year_built": 1998,
    "loan_term_months": 120, "cap_rate": 0.058, "market_vacancy": 0.07,
    "sponsor_experience_years": 12.0, "noi": 350_000,
    "property_type": "office", "region": "west",
}


# ---------- drift math (no model needed) ----------------------------------
def test_psi_is_zero_for_identical_distributions():
    """Degenerate sanity check. Identical input gives identical histograms,
    so this is exactly 0.0 -- not approximately. Weak on its own (a sign-flipped
    implementation would also pass), which is why the tests below exist."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) == 0.0


def test_psi_is_small_for_independent_samples_of_the_same_population():
    """The REAL 'no drift' case: different draws from the same distribution.
    Sampling noise is unavoidable, but it must stay well below the 0.10 WARN
    threshold or the monitor cries wolf on every batch."""
    rng = np.random.default_rng(1)
    worst = max(psi(rng.normal(size=5000), rng.normal(size=5000))
                for _ in range(50))
    assert worst < 0.10


def test_psi_is_never_negative():
    """PSI is a divergence measure -- it cannot be negative. This is the test
    that actually catches a flipped log ratio, which the identity test misses."""
    rng = np.random.default_rng(2)
    ref = rng.normal(size=3000)
    for shift in (0.0, 0.3, 1.0, 2.5):
        assert psi(ref, rng.normal(shift, 1, 3000)) >= 0.0


def test_psi_increases_monotonically_with_shift_size():
    """A bigger distribution shift must produce a bigger PSI. Catches scaling
    and binning bugs that leave the sign and the zero point intact."""
    rng = np.random.default_rng(3)
    ref = rng.normal(size=5000)
    vals = [psi(ref, rng.normal(d, 1, 5000)) for d in (0.0, 0.5, 1.0, 2.0)]
    assert vals == sorted(vals), f"not monotonic: {vals}"


def test_psi_flags_a_shifted_distribution():
    """A large, obvious shift must exceed the ALERT threshold."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(2.5, 1, 5000)
    assert psi(ref, cur) > 0.25


def test_psi_thresholds_are_sample_size_sensitive():
    """Documents a real limitation: at small n, pure sampling noise can push
    PSI past the 0.25 ALERT threshold even with no actual drift. Production use
    needs a minimum batch size (or bootstrapped thresholds), not just the
    conventional 0.10/0.25 cutoffs."""
    rng = np.random.default_rng(4)
    small_n_worst = max(psi(rng.normal(size=200), rng.normal(size=200))
                        for _ in range(200))
    large_n_worst = max(psi(rng.normal(size=5000), rng.normal(size=5000))
                        for _ in range(200))
    assert large_n_worst < small_n_worst


def test_categorical_psi_detects_mix_shift():
    ref = pd.Series(["a"] * 800 + ["b"] * 200)
    cur = pd.Series(["a"] * 200 + ["b"] * 800)
    assert categorical_psi(ref, cur) > 0.25


def test_feature_skew_catches_unseen_category():
    ref = pd.DataFrame({"property_type": ["office"] * 10, "region": ["west"] * 10})
    cur = pd.DataFrame({"property_type": ["datacenter"] * 5, "region": ["west"] * 5})
    issues = check_feature_skew(ref, cur)
    assert any(i["check"] == "unseen_category" for i in issues)


def test_feature_skew_catches_missing_columns():
    ref = pd.DataFrame({"ltv": [0.5], "property_type": ["office"], "region": ["west"]})
    cur = pd.DataFrame({"property_type": ["office"], "region": ["west"]})
    issues = check_feature_skew(ref, cur)
    assert any(i["check"] == "missing_columns" for i in issues)


# ---------- serving contract ----------------------------------------------
def test_risk_bands_are_monotonic():
    assert risk_band(0.01) == "low"
    assert risk_band(0.10) == "moderate"
    assert risk_band(0.20) == "elevated"
    assert risk_band(0.60) == "high"


def test_health_endpoint_responds():
    r = client.get("/health")
    assert r.status_code == 200
    assert "model_name" in r.json()


def test_predict_rejects_out_of_contract_input():
    """LTV above the allowed ceiling must be rejected at the edge, not scored."""
    bad = {**VALID_LOAN, "ltv": 9.9}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_unknown_category():
    bad = {**VALID_LOAN, "property_type": "datacenter"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_missing_field():
    bad = {k: v for k, v in VALID_LOAN.items() if k != "dscr"}
    assert client.post("/predict", json=bad).status_code == 422


@pytest.mark.skipif(
    not os.path.exists("mlruns"), reason="no local MLflow registry; run train.py first"
)
def test_predict_returns_a_valid_probability():
    r = client.post("/predict", json=VALID_LOAN)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "elevated", "high"}
