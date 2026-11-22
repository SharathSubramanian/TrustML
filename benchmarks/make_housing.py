"""Synthetic house-price REGRESSION dataset with known planted issues + ground truth.

Planted:
- post-outcome leak: assessed_value_after_sale ~= sale_price
- engineered-feature opportunity: price depends on lot AREA, but only width and depth are given
- subgroup failure: in the 'historic' neighbourhood older houses are worth MORE (opposite sign)
- informative missingness: garage_area missing far more often for cheap houses
- identifier, datetime column, a few duplicates

Usage: python -m benchmarks.make_housing --out data/benchmarks --seed 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_housing(n: int = 3000, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    hood = rng.choice(["downtown", "suburb", "riverside", "historic", "industrial"], n, p=[.2, .3, .2, .15, .15])
    lot_w = np.round(rng.uniform(10, 40, n), 1)
    lot_d = np.round(rng.uniform(15, 60, n), 1)
    sqft = np.round(rng.normal(1800, 450, n).clip(500, 4500))
    bedrooms = np.clip(np.round(sqft / 600 + rng.normal(0, 0.7, n)), 1, 7).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.6 + rng.normal(0, 0.5, n)), 1, 5).astype(int)
    age = rng.integers(0, 101, n)
    condition = rng.choice(["poor", "fair", "good", "excellent"], n, p=[.1, .3, .45, .15])
    hood_eff = pd.Series(hood).map({"downtown": 60000, "suburb": 20000, "riverside": 45000,
                                    "historic": 10000, "industrial": -30000}).to_numpy()
    cond_eff = pd.Series(condition).map({"poor": -25000, "fair": -8000, "good": 0, "excellent": 30000}).to_numpy()
    age_eff = np.where(hood == "historic", 1500 * age, -800 * age)   # planted subgroup failure
    price = (50000 + 110 * sqft + 95 * lot_w * lot_d + hood_eff + cond_eff + age_eff
             + 4000 * bathrooms + rng.normal(0, 20000, n))
    price = np.round(price.clip(40000), -2)

    garage = np.round(rng.normal(30, 8, n).clip(0), 1)
    cheap = price < np.quantile(price, 0.3)
    garage = np.where(rng.random(n) < np.where(cheap, 0.45, 0.05), np.nan, garage)  # informative missingness

    df = pd.DataFrame({
        "house_id": [f"H{50000 + i}" for i in range(n)],
        "listing_date": (pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")).strftime("%Y-%m-%d"),
        "neighborhood": hood, "lot_width_m": lot_w, "lot_depth_m": lot_d, "living_sqft": sqft,
        "bedrooms": bedrooms, "bathrooms": bathrooms, "age_years": age, "condition": condition,
        "garage_area_m2": garage,
        "assessed_value_after_sale": np.round(price * rng.uniform(0.98, 1.02, n), -2),  # planted leak
        "sale_price": price,
    })
    dup = rng.choice(n, 15, replace=False)
    df = pd.concat([df, df.iloc[dup]], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    truth = {
        "dataset": "housing_synthetic_v1", "seed": seed, "target": "sale_price", "task_type": "regression",
        "n_rows": len(df),
        "injected_issues": [
            {"check": "suspicious_predictor", "columns": ["assessed_value_after_sale"], "note": "post-sale value; TRUE LEAKAGE"},
            {"check": "informative_missingness", "columns": ["garage_area_m2"], "note": "missing 45% for cheapest 30% vs 5%"},
            {"check": "missing_values", "columns": ["garage_area_m2"], "note": ""},
            {"check": "id_like_column", "columns": ["house_id"], "note": ""},
            {"check": "datetime_present", "columns": ["listing_date"], "note": ""},
            {"check": "duplicate_rows", "columns": [], "note": "15 exact duplicates"},
        ],
        "engineered_opportunity": {"sources": ["lot_width_m", "lot_depth_m"], "op": "product",
                                   "note": "price depends on lot area = width x depth"},
        "subgroup_failure": {"column": "neighborhood", "group": "historic",
                             "note": "age effect has opposite sign; linear model errs most here"},
        "must_not_flag_as_leakage": ["living_sqft", "neighborhood", "lot_width_m", "lot_depth_m"],
    }
    return df, truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmarks")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, truth = make_housing(seed=a.seed)
    df.to_csv(out / "housing_synthetic_v1.csv", index=False)
    (out / "housing_synthetic_v1.truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
