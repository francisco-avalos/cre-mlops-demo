"""
Generate a synthetic commercial real-estate (CRE) loan portfolio.

Why synthetic: it is self-contained (no downloads), and the schema mirrors the
kind of structured/tabular financial data a real-estate investment manager
actually models -- LTV, DSCR, occupancy, cap rate, sponsor experience.

Produces three files:
  reference.csv  -- the "training" vintage (the world the model learned)
  current.csv    -- a later vintage under market stress (drifted features)
  current_labeled.csv -- same as current, but with realized outcomes, so we can
                         measure CONCEPT drift (does the old model still work?)
"""
import numpy as np
import pandas as pd

PROPERTY_TYPES = ["office", "retail", "industrial", "multifamily"]
REGIONS = ["west", "southwest", "midwest", "northeast", "southeast"]


def _simulate(n: int, rng: np.random.Generator, stressed: bool = False) -> pd.DataFrame:
    """Simulate one vintage of loans. `stressed` shifts the market regime."""
    # Market regime knobs. Under stress: rates up, occupancy down, offices hurt.
    rate_center = 7.6 if stressed else 5.4
    occ_center = 0.84 if stressed else 0.92
    vac_center = 0.12 if stressed else 0.07

    property_type = rng.choice(PROPERTY_TYPES, size=n, p=[0.28, 0.22, 0.25, 0.25])
    region = rng.choice(REGIONS, size=n)

    loan_amount = np.exp(rng.normal(15.6, 0.7, n))                 # ~$6M median
    ltv = np.clip(rng.normal(0.66 if not stressed else 0.71, 0.10, n), 0.25, 0.95)
    dscr = np.clip(rng.normal(1.45 if not stressed else 1.22, 0.30, n), 0.55, 3.0)
    occupancy_rate = np.clip(rng.normal(occ_center, 0.09, n), 0.30, 1.0)
    interest_rate = np.clip(rng.normal(rate_center, 0.9, n), 2.5, 12.0)
    year_built = rng.integers(1955, 2023, n)
    loan_term_months = rng.choice([60, 84, 120, 180, 240], size=n)
    cap_rate = np.clip(rng.normal(0.058 if not stressed else 0.068, 0.012, n), 0.03, 0.12)
    market_vacancy = np.clip(rng.normal(vac_center, 0.035, n), 0.01, 0.35)
    sponsor_experience_years = np.clip(rng.gamma(3.0, 3.5, n), 0, 40)
    noi = loan_amount * cap_rate * rng.normal(1.0, 0.12, n)

    df = pd.DataFrame(
        {
            "loan_amount": loan_amount.round(0),
            "ltv": ltv.round(4),
            "dscr": dscr.round(3),
            "occupancy_rate": occupancy_rate.round(4),
            "interest_rate": interest_rate.round(3),
            "year_built": year_built,
            "loan_term_months": loan_term_months,
            "cap_rate": cap_rate.round(4),
            "market_vacancy": market_vacancy.round(4),
            "sponsor_experience_years": sponsor_experience_years.round(1),
            "noi": noi.round(0),
            "property_type": property_type,
            "region": region,
        }
    )

    # --- Outcome model -------------------------------------------------------
    # Risk rises with leverage, rates and vacancy; falls with coverage,
    # occupancy and sponsor experience. Deliberately NON-linear, because real
    # credit risk is: covenant-style thresholds, interactions, and saturation.
    # This is what lets a gradient-boosted tree genuinely beat a linear model.
    is_office = (df["property_type"] == "office").astype(float)
    is_industrial = (df["property_type"] == "industrial").astype(float)
    age = 2024 - df["year_built"]

    logit = (
        -0.85
        # Threshold effect: breaching 75% LTV is a cliff, not a slope.
        + 2.10 * (df["ltv"] - 0.65)
        + 1.55 * np.maximum(df["ltv"] - 0.75, 0) * 10
        # DSCR below ~1.2 is where lenders actually start losing money.
        - 1.20 * (df["dscr"] - 1.35)
        - 2.40 * np.maximum(1.20 - df["dscr"], 0)
        - 2.20 * (df["occupancy_rate"] - 0.90)
        + 0.22 * (df["interest_rate"] - 5.5)
        + 2.60 * (df["market_vacancy"] - 0.08)
        # Interaction: vacancy hurts office far more than other types.
        + 9.00 * is_office * np.maximum(df["market_vacancy"] - 0.08, 0)
        # Saturating benefit of sponsor experience (first years matter most).
        - 0.85 * np.log1p(df["sponsor_experience_years"])
        + 0.45 * is_office
        - 0.30 * is_industrial
        # Non-monotonic age: brand-new and very old both carry risk.
        + 0.010 * np.abs(age - 25)
    )

    if stressed:
        # CONCEPT DRIFT: the relationship itself changes, not just the inputs.
        # In the stressed regime, office risk re-prices hard and DSCR matters
        # more than it used to. A model trained on the old vintage will miss this.
        logit = (
            logit
            + 0.95 * (df["property_type"] == "office").astype(float)
            - 0.90 * (df["dscr"] - 1.35)
        )

    p = 1.0 / (1.0 + np.exp(-logit))
    df["default_12m"] = rng.binomial(1, p)
    return df


def main(seed: int = 42, n_reference: int = 20000, n_current: int = 6000) -> None:
    rng = np.random.default_rng(seed)

    reference = _simulate(n_reference, rng, stressed=False)
    current_labeled = _simulate(n_current, rng, stressed=True)
    current = current_labeled.drop(columns=["default_12m"])

    reference.to_csv("data/reference.csv", index=False)
    current.to_csv("data/current.csv", index=False)
    current_labeled.to_csv("data/current_labeled.csv", index=False)

    print(f"reference.csv       {reference.shape}  default rate "
          f"{reference['default_12m'].mean():.3f}")
    print(f"current.csv         {current.shape}  (unlabeled, drifted)")
    print(f"current_labeled.csv {current_labeled.shape}  default rate "
          f"{current_labeled['default_12m'].mean():.3f}")


if __name__ == "__main__":
    main()
