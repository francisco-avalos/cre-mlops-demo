# data_dictionary
### CRE Default Risk — field definitions for the MLOps demo dataset

The dataset simulates a portfolio of **commercial real-estate (CRE) loans**. Each row is one loan. The modeling question: *will this loan default in the next 12 months?*

---

## 1. The three files

| File | Rows | Labeled? | What it represents |
|---|---|---|---|
| `reference.csv` | 20,000 | Yes | The **training vintage** — a calm market. This is the world the model learned. |
| `current.csv` | 6,000 | No | A **later vintage under market stress**, as it arrives at serving time (no outcomes yet). Used for data-drift and feature-skew checks. |
| `current_labeled.csv` | 6,000 | Yes | The same stressed vintage *after* outcomes are realized. Needed for **concept drift** — you cannot measure it without labels. |

All three share the same 13 input fields. Only the labeled files carry `default_12m`.

---

## 2. Field definitions

### Loan terms

| Field | Type | Definition |
|---|---|---|
| **`loan_amount`** | float (USD) | Original principal balance of the loan. How much was borrowed. |
| **`interest_rate`** | float (annual %) | Contract interest rate on the loan, e.g. `5.8` = 5.8%. Higher rates mean higher debt service, which squeezes the borrower's ability to pay. |
| **`loan_term_months`** | int | Length of the loan in months. Values: 60, 84, 120, 180, 240 (5 to 20 years). |

### Credit metrics — the two that underwriters actually argue about

| Field | Type | Definition |
|---|---|---|
| **`ltv`** | float (ratio 0–1) | **Loan-to-Value.** Loan amount ÷ property value. `0.68` means the loan is 68% of the property's worth, so the borrower has 32% equity. **This is the lender's cushion.** If the property sells for less than the loan balance, the lender eats the difference. Above ~0.75 is where risk accelerates sharply. |
| **`dscr`** | float (ratio) | **Debt Service Coverage Ratio.** Net operating income ÷ annual debt payments. `1.35` means the property generates 1.35× the cash needed to pay the mortgage. **Below 1.0 the property doesn't cover its own debt** — the borrower is funding payments from somewhere else. Lenders typically want 1.20+ as a covenant floor. |

**LTV asks "if this goes bad, am I covered?" DSCR asks "can they pay me this month?"** They fail in different ways and both matter.

### Property performance

| Field | Type | Definition |
|---|---|---|
| **`noi`** | float (USD/yr) | **Net Operating Income.** Annual rental revenue minus operating expenses, before debt service and taxes. The property's actual earning power — the numerator of DSCR and the basis of its value. |
| **`occupancy_rate`** | float (0–1) | Share of the building's leasable space currently leased. `0.91` = 91% occupied, 9% empty. Directly drives NOI — empty space earns nothing. |
| **`cap_rate`** | float (0–1) | **Capitalization Rate.** NOI ÷ property value. The unlevered yield on the asset, e.g. `0.058` = 5.8%. Inverts to a valuation multiple: **value = NOI ÷ cap rate**. Rising cap rates mean falling property values at the same income — which is precisely how a rate shock destroys equity. |
| **`year_built`** | int (year) | Year the property was constructed. A proxy for capital-expenditure burden and competitiveness. |

### Market and sponsor context

| Field | Type | Definition |
|---|---|---|
| **`market_vacancy`** | float (0–1) | Vacancy rate across the property's *submarket*, not the building. `0.07` = 7% of comparable space sits empty. A weak market makes re-leasing hard and pressures rents. |
| **`sponsor_experience_years`** | float (years) | Years of experience of the borrower/operator ("sponsor"). Experienced sponsors manage through downturns and are likelier to inject capital rather than hand back the keys. |
| **`property_type`** | category | Asset class: `office`, `retail`, `industrial`, `multifamily`. Different demand drivers and risk profiles. |
| **`region`** | category | U.S. region: `west`, `southwest`, `midwest`, `northeast`, `southeast`. |

### Target

| Field | Type | Definition |
|---|---|---|
| **`default_12m`** | int (0/1) | **1** if the loan defaulted within 12 months, **0** otherwise. The binary outcome being predicted. Present only in `reference.csv` and `current_labeled.csv`. |

---

## 3. How the fields relate

These aren't independent columns — they're bound by accounting identities:

```
property value  =  NOI ÷ cap_rate
LTV             =  loan_amount ÷ property value
DSCR            =  NOI ÷ annual debt service      (debt service rises with interest_rate)
NOI             ≈  f(occupancy_rate, rents, operating expenses)
```

So a single shock propagates: **occupancy falls → NOI falls → DSCR falls** *and* **property value falls → LTV rises**. One cause, two credit metrics deteriorating at once. That correlation is why these features drift together in the monitoring output rather than independently.

> **Honest limitation:** in the generator, `noi` is derived as `loan_amount × cap_rate × noise`, which implicitly assumes the loan is roughly the property value. In real data NOI is a property-level operating figure measured independently of the loan. It's a simplification — worth saying out loud if anyone asks.

---

## 4. Observed distributions

### Reference vintage (calm market, n = 20,000)

| Field | Min | 25% | Median | 75% | Max |
|---|---|---|---|---|---|
| `loan_amount` | $302K | $3.69M | $5.88M | $9.45M | $198M |
| `ltv` | 0.269 | 0.593 | 0.659 | 0.727 | 0.950 |
| `dscr` | 0.550 | 1.245 | 1.445 | 1.651 | 2.659 |
| `occupancy_rate` | 0.576 | 0.858 | 0.918 | 0.980 | 1.000 |
| `interest_rate` | 2.50% | 4.79% | 5.40% | 6.01% | 8.82% |
| `year_built` | 1955 | 1971 | 1988 | 2005 | 2022 |
| `loan_term_months` | 60 | 84 | 120 | 180 | 240 |
| `cap_rate` | 0.030 | 0.050 | 0.058 | 0.066 | 0.114 |
| `market_vacancy` | 0.010 | 0.047 | 0.070 | 0.094 | 0.245 |
| `sponsor_experience_years` | 0.1 | 6.0 | 9.4 | 13.6 | 40.0 |
| `noi` | $14.3K | $203K | $333K | $551K | $11.8M |

**Categorical mix:** `property_type` — office 28.2%, industrial 25.5%, multifamily 24.8%, retail 21.5%. `region` — roughly 20% each.

**Base default rate: 9.15%** (a rare-event problem, which is why PR-AUC and `scale_pos_weight` appear in the training code).

### What shifts in the stressed vintage

| Field | Reference median | Stressed median | Direction |
|---|---|---|---|
| `interest_rate` | 5.40% | **7.61%** | Rate shock |
| `market_vacancy` | 0.070 | **0.120** | Market weakens |
| `occupancy_rate` | 0.918 | **0.839** | Buildings empty out |
| `dscr` | 1.445 | **1.214** | Coverage erodes toward the 1.20 floor |
| `cap_rate` | 0.058 | **0.068** | Values compress |
| `ltv` | 0.659 | **0.710** | Leverage rises as values fall |
| `loan_amount`, `year_built`, `loan_term_months`, `sponsor_experience_years`, `property_type`, `region` | — | unchanged | Stable by design |

**Default rate rises 9.15% → 30.62%.**

The stable fields are deliberate: a monitoring system that flags *everything* is useless. Drift detection has to distinguish what actually moved from what didn't.

---

## 5. How the outcome is generated

Risk is modeled as a non-linear function of the inputs — this is what makes a tree model worth using:

- **Threshold effect on LTV** — risk rises gradually, then accelerates above 0.75. A cliff, not a slope.
- **Threshold effect on DSCR** — an extra penalty once coverage drops below 1.20, mirroring a covenant breach.
- **Interaction: office × market_vacancy** — vacancy hurts office far more than other asset classes.
- **Saturating sponsor experience** — the first few years matter most; year 30 adds little over year 25 (modeled as a log).
- **Non-monotonic age** — both brand-new and very old properties carry elevated risk; risk is lowest around 25 years old.

A linear model can't represent thresholds or interactions. That's exactly why the depth-3 XGBoost beats the logistic baseline here — and why the win is real rather than an artifact.

### The concept drift, made concrete

In the stressed vintage the *relationship itself* changes, not just the inputs: office risk re-prices sharply upward and DSCR carries more weight than before.

**Default rate by property type:**

| Property type | Reference | Stressed | Change |
|---|---|---|---|
| **office** | 12.9% | **55.9%** | **4.3×** |
| retail | 8.1% | 22.4% | 2.8× |
| multifamily | 8.6% | 22.0% | 2.6× |
| industrial | 6.3% | 18.9% | 3.0× |

Office goes from "somewhat riskier" to "more than half the book defaults." A model trained on the calm vintage encoded the old office penalty and systematically under-predicts the new one — **even for an office loan whose input features look perfectly ordinary.** That's the definition of concept drift, and it's why monitoring inputs alone isn't enough.

---

## 6. Serving contract

The API (`src/api.py`) enforces these at the edge via Pydantic. Anything violating the contract is rejected with HTTP 422 rather than silently scored:

| Field | Constraint |
|---|---|
| `loan_amount` | > 0 |
| `ltv` | 0 to 1.5 |
| `dscr` | > 0 |
| `occupancy_rate` | 0 to 1 |
| `interest_rate` | > 0 |
| `year_built` | 1800 to 2100 |
| `loan_term_months` | > 0 |
| `cap_rate` | 0 to 1 |
| `market_vacancy` | 0 to 1 |
| `sponsor_experience_years` | ≥ 0 |
| `noi` | any float |
| `property_type` | one of: office, retail, industrial, multifamily |
| `region` | one of: west, southwest, midwest, northeast, southeast |

---

## 7. Quick glossary

| Term | Plain meaning |
|---|---|
| **CRE** | Commercial real estate — income-producing property, not single-family homes |
| **LTV** | How much of the property's value is borrowed. The lender's cushion. |
| **DSCR** | How many times over the property's income covers its debt payments |
| **NOI** | Annual income after operating costs, before debt and taxes |
| **Cap rate** | Unlevered yield (NOI ÷ value). Inverts to a valuation multiple. |
| **Sponsor** | The borrower/operator behind the deal |
| **Debt service** | The required loan payments |
| **Submarket** | The local competitive set the property sits in |
| **Vintage** | A cohort of loans originated in the same period |

> All data here is **synthetic**, generated from a known process in `data/generate_data.py`. It demonstrates the MLOps machinery and the drift behavior — it is not a real credit model and the field distributions are illustrative, not market-calibrated.
