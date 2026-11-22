"""Run M1 tools on a dataset and print / save structured results.

Usage:
  python -m scripts.run_audit data/benchmarks/churn_synthetic_v1.csv --target churned
  python -m scripts.run_audit my.xlsx --target price --out runs/audit.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split

from trustml.io import load_dataset
from trustml.schemas import TaskType
from trustml.tools.leakage import screen_leakage
from trustml.tools.profiler import profile_dataset
from trustml.tools.quality import audit_quality


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", choices=[t.value for t in TaskType], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t0 = time.time()
    df, handle = load_dataset(a.path)
    prof = profile_dataset(df, handle, target=a.target, task_type=TaskType(a.task) if a.task else None)
    quality = audit_quality(df, prof)

    # Leakage screen runs on the TRAINING split only; the 20% test split is never looked at.
    labelled = df[df[a.target].notna()]
    strat = labelled[a.target] if prof.target.task_type != TaskType.REGRESSION else None
    train, _test = train_test_split(labelled, test_size=0.2, random_state=a.seed, stratify=strat)
    leak = screen_leakage(train, prof)

    print(f"\nDataset: {handle.path}  rows={handle.n_rows} cols={handle.n_cols}  sha256={handle.sha256[:12]}…")
    t = prof.target
    print(f"Target: {t.name} ({t.task_type.value})", f"classes={t.class_counts}" if t.class_counts else "")
    print("\n== Quality findings ==")
    for f in quality.findings:
        print(f"[{f.severity.value.upper():6}] {f.check:20} {f.message}")
    print("\n== Leakage screen (train split only) ==")
    for f in leak.findings:
        if f.check == "single_feature_ranking":
            print("Ranking:", json.dumps(f.evidence["ranking"]))
        else:
            print(f"[{f.severity.value.upper():6}] {f.check:20} {f.message}")
    print(f"\nElapsed: {time.time() - t0:.1f}s")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "profile": prof.model_dump(mode="json"),
            "quality": quality.model_dump(mode="json"),
            "leakage": leak.model_dump(mode="json"),
        }, indent=2))
        print(f"Saved -> {a.out}")


if __name__ == "__main__":
    main()
