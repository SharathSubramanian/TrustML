"""Loan-default dataset used as the first BLIND test (planted: leak, 99 sentinel, duplicates, repeated
customers, informative missingness, imbalance, base-rate drift, subgroup interaction, noise features)."""
import json
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n_cust = 3200
cust = pd.DataFrame({
    "customer_id": [f"CU{70000 + i}" for i in range(n_cust)],
    "annual_income": np.round(rng.lognormal(10.9, 0.5, n_cust), -2),
    "employment_years": rng.integers(0, 30, n_cust),
    "home_ownership": rng.choice(["rent", "mortgage", "own"], n_cust, p=[.45, .4, .15]),
    "state": rng.choice(["FL", "GA", "TX", "CA", "NY", "NC"], n_cust),
})
# ~1.3 applications per customer -> repeated IDs (group split needed)
rows = cust.sample(n=4200, replace=True, random_state=1).reset_index(drop=True)
n = len(rows)
app_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
loan_amount = np.round(rng.uniform(2000, 40000, n), -2)
term = rng.choice([36, 60], n, p=[.65, .35])
purpose = rng.choice(["debt_consolidation", "car", "home_improvement", "small_business", "medical"], n,
                     p=[.45, .2, .15, .1, .1])
credit_score = np.clip(np.round(rng.normal(690, 55, n)), 480, 850)
dti = np.round(np.clip(rng.normal(18, 8, n), 0, 60), 1)
# later applications are riskier (drift): policy loosened in 2024
late = (app_date >= pd.Timestamp("2024-01-01")).astype(int)
logit = (-3.2 + 0.035 * (dti - 18) - 0.018 * (credit_score - 690) + 0.5 * (term == 60)
         + 1.1 * (purpose == "small_business") + 0.6 * late
         + 0.000025 * (loan_amount - 15000) - 0.000008 * (rows["annual_income"].to_numpy() - 55000))
# subgroup failure: in small_business, risk is driven by employment_years (interaction)
logit += np.where(purpose == "small_business", -0.15 * (rows["employment_years"].to_numpy() - 8), 0)
default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame({
    "application_id": [f"A{900000 + i}" for i in range(n)],
    "customer_id": rows["customer_id"], "application_date": app_date.strftime("%Y-%m-%d"),
    "loan_amount": loan_amount, "term_months": term, "purpose": purpose, "credit_score": credit_score,
    "debt_to_income": dti, "annual_income": rows["annual_income"], "employment_years": rows["employment_years"],
    "home_ownership": rows["home_ownership"], "state": rows["state"],
    "num_open_accounts": rng.poisson(8, n), "marketing_channel": rng.choice(["web", "branch", "partner"], n),
})
# LEAK: collections status is only known after default
noise = rng.random(n) < 0.03
df["collections_flag"] = np.where(noise, 1 - default, default)
# sentinel: employment_years 99 = unknown
df.loc[rng.choice(n, 60, replace=False), "employment_years"] = 99
# missing credit_score more often for defaulters (informative missingness)
miss = rng.random(n) < np.where(default == 1, 0.25, 0.06)
df.loc[miss, "credit_score"] = np.nan
df["defaulted"] = np.where(default == 1, "Yes", "No")
# duplicates
df = pd.concat([df, df.sample(25, random_state=3)], ignore_index=True).sample(frac=1, random_state=4).reset_index(drop=True)
out = sys.argv[1] if len(sys.argv) > 1 else "data/benchmarks"
df.to_csv(f"{out}/loan_applications.csv", index=False)
truth = {
  "target": "defaulted", "positive": "Yes", "default_rate_pct": round(100 * (df.defaulted == "Yes").mean(), 2),
  "planted": [
    "LEAK: collections_flag (post-outcome, equals target with 3% noise)",
    "SENTINEL: employment_years = 99 (60 rows) means unknown",
    "DUPLICATES: 25 exact duplicate rows",
    "REPEATED IDS: customer_id appears in several applications -> group split by customer_id recommended",
    "INFORMATIVE MISSINGNESS: credit_score missing 25% for defaulters vs 6%",
    "CLASS IMBALANCE: minority 'Yes' well under 20%",
    "DRIFT: applications from 2024 are riskier -> forward-in-time check should show a drop",
    "SUBGROUP: purpose=small_business behaves differently (employment interaction)",
    "IDENTIFIER: application_id",
    "NOISE: state, marketing_channel, num_open_accounts carry no signal",
    "TRUE DRIVERS: debt_to_income, credit_score, term_months, purpose, loan_amount, annual_income, application date"],
}
json.dump(truth, open(f"{out}/loan_applications.truth.json", "w"), indent=2)
print(len(df), truth["default_rate_pct"])
