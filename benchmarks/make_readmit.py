"""Blind test #2: hospital readmission (binary). Written BEFORE running TrustML on it; not tuned afterwards."""
import json
import numpy as np, pandas as pd
rng = np.random.default_rng(2026)
n_pat = 3300
pat_ids = [f"P{100000+i}" for i in range(n_pat)]
visits = rng.choice([1, 1, 1, 2, 2, 3], n_pat)
rows = []
for pid, k in zip(pat_ids, visits):
    sex = rng.choice(["F", "M"]); ageg = rng.choice(["18-44", "45-64", "65-79", "80+"], p=[.22, .33, .3, .15])
    chronic = rng.poisson(1.5 + {"18-44": 0, "45-64": .6, "65-79": 1.2, "80+": 1.6}[ageg])
    for _ in range(k):
        rows.append({"patient_id": pid, "sex": sex, "age_group": ageg, "chronic_conditions": chronic})
df = pd.DataFrame(rows)
n = len(df)
df["admission_date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
df["admission_type"] = rng.choice(["emergency", "elective", "urgent"], n, p=[.45, .35, .2])
df["length_of_stay"] = np.clip(rng.gamma(2.2, 2.3, n).round(0) + 1, 1, 40)
df["num_medications"] = np.clip(rng.poisson(9, n) + df.chronic_conditions * 2, 0, 60)
df["prior_admissions_12m"] = rng.poisson(0.8, n)
df["bmi"] = rng.normal(28, 5.5, n).round(1)
df["hba1c"] = rng.normal(6.4, 1.1, n).round(1)
df["discharge_to"] = rng.choice(["home", "home_health", "skilled_nursing"], n, p=[.62, .23, .15])
df["room_number"] = rng.integers(100, 999, n)                      # noise
df["admission_weekday"] = df.admission_date.dt.day_name()            # noise
emerg = (df.admission_type == "emergency").to_numpy()
t_years = ((df.admission_date - pd.Timestamp("2023-01-01")).dt.days / 365).to_numpy()
logit = (-3.0 + 0.35 * df.chronic_conditions + 0.45 * df.prior_admissions_12m
         + np.where(emerg, 0.16 * (df.length_of_stay - 5), -0.01 * (df.length_of_stay - 5))   # INTERACTION
         + 0.6 * (df.discharge_to == "skilled_nursing") + 0.03 * (df.num_medications - 12)
         + 0.08 * (df.hba1c - 6.4) + 0.55 * t_years).to_numpy()                                # DRIFT
old = (df.age_group == "80+").to_numpy()
logit = np.where(old, -2.0 + 0.15 * (logit + 2.0) + rng.normal(0, 1.2, n), logit)          # 80+: hard to predict
p = 1 / (1 + np.exp(-logit))
df["readmitted_30d"] = (rng.random(n) < p).astype(int)
y = df.readmitted_30d.to_numpy()
# LEAK: recorded at the 30-day follow-up, i.e. after the outcome
df["followup_visit_type"] = np.where(y == 1, rng.choice(["inpatient", "clinic"], n, p=[.93, .07]),
                                     rng.choice(["inpatient", "clinic", "none"], n, p=[.03, .55, .42]))
# informative missingness: hba1c missing more often for readmitted
miss = rng.random(n) < np.where(y == 1, 0.30, 0.08)
df.loc[miss, "hba1c"] = np.nan
# sentinel: bmi = 999 for 4%
df.loc[rng.random(n) < 0.04, "bmi"] = 999
df["admission_date"] = df.admission_date.dt.strftime("%Y-%m-%d")
dups = df.sample(30, random_state=3)
df = pd.concat([df, dups]).sample(frac=1, random_state=4).reset_index(drop=True)
df.to_csv("data/benchmarks/hospital_readmissions.csv", index=False)
truth = {
  "n_rows": len(df), "positive_rate": round(float(df.readmitted_30d.mean()), 4),
  "facts": {
    "leak": "followup_visit_type (recorded at the 30-day follow-up, after the outcome)",
    "duplicates": "30 exact duplicate rows",
    "group_split": "patient_id repeats (multiple admissions per patient)",
    "sentinel": "bmi = 999 placeholder (~4%)",
    "imbalance": "minority class handled so recall reaches the target",
    "interaction": "length_of_stay matters only for admission_type=emergency",
    "fairness": "age_group: recall clearly lowest for 80+ (outcome barely predictable there)",
    "noise": "room_number and admission_weekday carry no signal",
    "drift": "readmission rate rises over admission_date (2023 -> 2024)",
    "informative_missing": "hba1c is missing more often for readmitted patients",
  }}
json.dump(truth, open("data/benchmarks/hospital_readmissions.truth.json", "w"), indent=2)
print(len(df), df.readmitted_30d.mean(), df.patient_id.duplicated().mean())
