"""Synthetic churn dataset with KNOWN injected issues + a ground-truth file.

The ground truth is what makes the TrustML vs Muse comparison objective:
we know exactly which issues exist, so detection can be scored by precision/recall.

Usage:  python -m benchmarks.make_churn --out data/benchmarks --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_churn(n: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    monthly = np.round(rng.normal(70, 25, n).clip(18, 140), 2)
    support_calls = rng.poisson(1.5, n)
    payment = rng.choice(["card", "bank_transfer", "electronic_check", "mailed_check"], n)
    region = rng.choice(["north", "south", "east", "west"], n)
    age = rng.integers(18, 85, n).astype(float)
    income = np.round(rng.lognormal(10.8, 0.45, n), 0)

    # True churn mechanism (known): short tenure, month-to-month, high charges, many support calls
    logit = (-2.6 - 0.045 * tenure + 1.4 * (contract == "month-to-month") - 0.8 * (contract == "two-year")
             + 0.018 * (monthly - 70) + 0.45 * support_calls + 0.4 * (payment == "electronic_check"))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "signup_date": (pd.Timestamp("2024-12-31") - pd.to_timedelta(tenure * 30, unit="D")).strftime("%Y-%m-%d"),
        "tenure_months": tenure,
        "contract_type": contract,
        "monthly_charges": monthly,
        "total_charges": np.round(monthly * tenure, 2).astype(str),
        "support_calls_90d": support_calls,
        "payment_method": payment,
        "region": region,
        "age": age,
        "annual_income": income,
        "country": "US",
        "churned": churn,
    })

    # ---- Injected issues (ground truth) ----
    # 1. LEAKAGE: post-outcome feature, nearly a copy of the target (2% noise)
    noise = rng.random(n) < 0.02
    df.insert(11, "cancellation_fee_charged", np.where(noise, 1 - churn, churn))

    # 2. Missing values (~12%) in income, MAR-ish: more missing for short tenure
    miss_p = np.where(tenure < 12, 0.22, 0.08)
    df.loc[rng.random(n) < miss_p, "annual_income"] = np.nan

    # 3. Sentinel value -999 in age (40 rows)
    sent_idx = rng.choice(n, 40, replace=False)
    df.loc[sent_idx, "age"] = -999

    # 4. Numbers-stored-as-text with blank tokens in total_charges (new customers)
    blank_idx = rng.choice(np.where(tenure <= 2)[0], 25, replace=False)
    df.loc[blank_idx, "total_charges"] = " "

    # 5. Exact duplicate rows (30)
    dup_src = rng.choice(n, 30, replace=False)
    df = pd.concat([df, df.iloc[dup_src]], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    truth = {
        "dataset": "churn_synthetic_v1",
        "seed": seed,
        "target": "churned",
        "task_type": "binary_classification",
        "n_rows": len(df),
        "churn_rate_pct": round(100 * df["churned"].mean(), 2),
        "true_drivers": ["tenure_months", "contract_type", "monthly_charges", "support_calls_90d", "payment_method"],
        "noise_features": ["region", "age", "annual_income"],
        "injected_issues": [
            {"check": "suspicious_predictor", "columns": ["cancellation_fee_charged"],
             "note": "Post-outcome feature: equals churned with 2% label noise. TRUE LEAKAGE."},
            {"check": "missing_values", "columns": ["annual_income"], "note": "~12% missing, more for short tenure."},
            {"check": "sentinel_value", "columns": ["age"], "note": "40 rows of -999."},
            {"check": "mixed_type_column", "columns": ["total_charges"], "note": "Numeric stored as text; 25 blank ' ' tokens."},
            {"check": "duplicate_rows", "columns": [], "note": "30 exact duplicate rows."},
            {"check": "id_like_column", "columns": ["customer_id"], "note": "Unique identifier."},
            {"check": "constant_column", "columns": ["country"], "note": "Always 'US'."},
            {"check": "class_imbalance", "columns": ["churned"], "note": "Minority class well under 20%."},
            {"check": "datetime_present", "columns": ["signup_date"], "note": "Date column; temporal-split caveat."},
        ],
        "must_not_flag_as_leakage": ["tenure_months", "contract_type", "monthly_charges", "support_calls_90d"],
    }
    return df, truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmarks")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, truth = make_churn(seed=args.seed)
    df.to_csv(out / "churn_synthetic_v1.csv", index=False)
    (out / "churn_synthetic_v1.truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {len(df)} rows; churn rate {truth['churn_rate_pct']}% -> {out}")


if __name__ == "__main__":
    main()
