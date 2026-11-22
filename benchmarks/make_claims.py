"""Blind test #3: medical insurance claim cost (REGRESSION). Written before any TrustML run; not tuned afterwards."""
import json
import numpy as np, pandas as pd
rng = np.random.default_rng(31337)
n_mem = 3600
mem = pd.DataFrame({"member_id": [f"M{700000+i}" for i in range(n_mem)],
                    "age": rng.integers(18, 80, n_mem),
                    "sex": rng.choice(["female", "male"], n_mem),
                    "smoker": rng.choice(["no", "yes"], n_mem, p=[.78, .22]),
                    "height_cm": rng.normal(170, 9, n_mem).round(0),
                    "region": rng.choice(["northeast", "southeast", "midwest", "west"], n_mem),
                    "plan_type": rng.choice(["bronze", "silver", "gold"], n_mem, p=[.4, .4, .2]),
                    "signup_channel": rng.choice(["web", "phone", "agent", "employer"], n_mem),
                    "annual_income": rng.lognormal(10.9, 0.45, n_mem).round(-2)})
mem["weight_kg"] = (rng.normal(27, 4.5, n_mem) * (mem.height_cm / 100) ** 2).round(1)
k = rng.choice([1, 1, 1, 2, 2, 3], n_mem)
df = mem.loc[mem.index.repeat(k)].reset_index(drop=True)
n = len(df)
df["claim_date"] = (pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")).strftime("%Y-%m-%d")
df["chronic_conditions"] = np.clip(rng.poisson(0.6 + df.age / 40), 0, 8)
df["policy_suffix"] = rng.integers(1000, 9999, n)                                   # noise
bmi = df.weight_kg / (df.height_cm / 100) ** 2
smk = (df.smoker == "yes").to_numpy()
logc = (7.2 + np.where(smk, 0.045, 0.010) * (df.age - 45)                         # INTERACTION age x smoker
        + 0.55 * smk + 0.06 * (bmi - 27) + 0.22 * df.chronic_conditions
        + np.select([df.plan_type == "gold", df.plan_type == "silver"], [0.25, 0.1], 0)
        + rng.normal(0, 0.45, n))                                                    # no sex effect, no time trend
df["claim_cost"] = np.exp(logc).round(2)                                             # heavy-tailed target
df["amount_approved"] = (df.claim_cost * rng.uniform(0.85, 0.97, n)).round(2)       # LEAK (decided after the claim)
pc = rng.poisson(1.0 + 0.6 * df.chronic_conditions).astype(float)
q = df.claim_cost.rank(pct=True).to_numpy()
pc[rng.random(n) < np.where(q > 0.75, 0.35, 0.07)] = np.nan                         # informative missingness
df["prior_claims_count"] = pc
df.loc[rng.random(n) < 0.05, "annual_income"] = -1                                  # SENTINEL
dups = df.sample(25, random_state=9)
df = pd.concat([df, dups]).sample(frac=1, random_state=10).reset_index(drop=True)
cols = ["member_id", "claim_date", "age", "sex", "smoker", "height_cm", "weight_kg", "region", "plan_type",
        "signup_channel", "annual_income", "chronic_conditions", "prior_claims_count", "policy_suffix",
        "amount_approved", "claim_cost"]
df[cols].to_csv("data/benchmarks/insurance_claims.csv", index=False)
truth = {"task": "regression", "target": "claim_cost", "n_rows": len(df), "facts": {
  "F1_leak": "amount_approved is set after the claim (≈0.85-0.97 × claim_cost) → must be flagged and excluded",
  "F2_duplicates": "25 exact duplicate rows",
  "F3_group_split": "member_id repeats (members file several claims) → group split",
  "F4_sentinel": "annual_income = -1 placeholder (~5%)",
  "F5_log_target": "claim_cost is heavy-tailed (log-normal) → log target should help",
  "F6_interaction": "age matters strongly only for smoker=yes (slope 0.045 vs 0.010 on log scale)",
  "F7_engineered": "BMI = weight/height² drives cost; a weight/height ratio should help vs raw columns",
  "F8_noise": "policy_suffix, signup_channel, region and annual_income carry no signal",
  "F9_informative_missing": "prior_claims_count is missing more often for high-cost claims",
  "N1_fairness_negative": "sex has NO effect → fairness on sex should be 'not material' (false-alarm check)",
  "N2_time_negative": "no trend over claim_date → time check should say stable (false-alarm check)"}}
json.dump(truth, open("data/benchmarks/insurance_claims.truth.json", "w"), indent=2)
print(n + 25, round(df.claim_cost.median()), round(df.claim_cost.max()), round(df.member_id.duplicated().mean(), 3))
