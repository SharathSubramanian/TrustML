"""Purposeful charts: each chart is produced only when the run gives it a job.

Static PNGs (embedded in the HTML report and shown in the dashboard).
Palette: validated reference categorical slots - blue = base/before,
orange = new/after/flagged, gray = context. Text uses ink colors, not series colors.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trustml.io import load_predictions  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2a9"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
    "font.size": 10,
})


class Chart(dict):
    """{'id','title','purpose','path','interpretation'}"""


def _save(fig, out: Path, cid: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{cid}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)


def _hbar(ax, labels, values, colors, fmt="{:.3f}"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.6)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    span = max(abs(v) for v in values) or 1
    for i, v in enumerate(values):
        ax.text(v + span * 0.01, i, fmt.format(v), va="center", fontsize=8, color=INK2)


def build_charts(run, explanations: dict | None = None) -> list[Chart]:
    out = run.run_dir / "charts"
    ctx, charts = run.ctx, []
    task = ctx.task.value
    E = ctx.experiments

    # 1. target distribution - always: sets the context for every metric
    fig, ax = plt.subplots(figsize=(6, 3))
    y = ctx.data[ctx.target]
    if task == "regression":
        ax.hist(y, bins=40, color=BLUE)
        if y.abs().max() >= 10000:
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
        ax.set_xlabel(ctx.target); ax.set_ylabel("rows")
        interp = f"Target mean {y.mean():,.0f}, median {y.median():,.0f}."
    else:
        vc = y.astype(str).value_counts()
        cols = [ORANGE if k == ctx.positive_class else BLUE for k in vc.index]
        _hbar(ax, [f"{k}" for k in vc.index], (100 * vc / vc.sum()).tolist(), cols, "{:.1f}%")
        ax.set_xlabel("% of labelled rows")
        interp = f"Minority class '{ctx.positive_class}' is {100 * (y.astype(str) == ctx.positive_class).mean():.1f}% of rows; accuracy alone would be misleading."
    ax.set_title(f"Target distribution: {ctx.target}")
    charts.append(Chart(id="target", title="Target distribution", purpose="Context for metric choice",
                        path=_save(fig, out, "target"), interpretation=interp))

    # 2. missingness - only if something is missing
    miss = [c for c in run.profile.columns if c.n_missing and c.name != ctx.target]
    if miss:
        miss = sorted(miss, key=lambda c: c.pct_missing, reverse=True)[:12]
        flagged = {c for f in run.missingness.findings for c in f.columns}
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(miss) + 1.2))
        _hbar(ax, [c.name for c in miss], [c.pct_missing for c in miss],
              [ORANGE if c.name in flagged else BLUE for c in miss], "{:.1f}%")
        ax.set_xlabel("% missing"); ax.set_title("Missing values by column")
        txt = "Orange = missingness associated with the target (training split)." if flagged else \
            "No column's missingness is associated with the target."
        charts.append(Chart(id="missing", title="Missing values", purpose="Data-quality finding",
                            path=_save(fig, out, "missing"), interpretation=txt))

    # 3. leakage screen - only if a suspicion was raised
    if run.facts["leak_candidates"]:
        items = list(run.leak_scores.items())[:10]
        metric = "R²" if task == "regression" else "ROC-AUC"
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(items) + 1.2))
        _hbar(ax, [k for k, _ in items], [v for _, v in items],
              [ORANGE if k in run.facts["leak_candidates"] else BLUE for k, _ in items])
        ax.set_xlabel(f"single-feature CV {metric} (training split)")
        ax.set_title("Leakage screen: how well each feature predicts the target alone")
        charts.append(Chart(id="leak_screen", title="Leakage screen", purpose="Why a feature was suspected",
                            path=_save(fig, out, "leak_screen"),
                            interpretation=f"{run.facts['leak_candidates']} stand far above every other feature. "
                                           "High predictive power alone is not proof of leakage."))

    # 4. paired fold comparisons for every experiment-based verdict
    for v in [v for v in run.facts["verdicts"] if "base" in v and "new" in v]:
        m = v["metric"]
        if m not in E[v["base"]].cv_metrics:
            continue
        a, b = E[v["base"]].cv_metrics[m].folds, E[v["new"]].cv_metrics[m].folds
        k = np.arange(1, len(a) + 1)
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(k, a, "-o", color=BLUE, lw=2, ms=7, label=f"before ({v['base']})")
        ax.plot(k, b, "-o", color=ORANGE, lw=2, ms=7, label=f"after ({v['new']})")
        ax.set_xticks(k, [f"fold {i}" for i in k]); ax.set_ylabel(f"CV {m}")
        label = v.get("feature") or v.get("intervention") or v.get("model") or ", ".join(v.get("features", []))
        ax.set_title(f"{v['topic'].replace('_', ' ').capitalize()}: {label}")
        ax.legend(loc="best")
        cid = f"cmp_{v['new']}"
        charts.append(Chart(id=cid, title=f"{v['topic']}: {label}", purpose="Paired evidence on identical folds",
                            path=_save(fig, out, cid),
                            interpretation=f"Verdict: {v['verdict']}. Mean {m} {np.mean(a):.3f} → {np.mean(b):.3f}.",
                            data={"kind": "lines", "x": [f"fold {i}" for i in k], "ylabel": f"CV {m}",
                                  "series": [{"name": f"before ({v['base']})", "values": list(map(float, a)), "color": BLUE},
                                             {"name": f"after ({v['new']})", "values": list(map(float, b)), "color": ORANGE}]}))

    # 5. permutation importance - only if it was computed
    imp = run.analyses.get("importance")
    if imp:
        rows = imp["ranking"][:12]
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(rows) + 1.2))
        yy = np.arange(len(rows))
        ax.barh(yy, [r["mean"] for r in rows], xerr=[r["std"] for r in rows], height=0.6, ecolor=INK2,
                color=[ORANGE if r["feature"] in imp["noise_candidates"] else BLUE for r in rows])
        ax.set_yticks(yy, [r["feature"] for r in rows]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f"drop in CV {imp['scoring']} when shuffled (mean ± sd over folds)")
        ax.set_title(f"Permutation importance ({imp['exp_id']})")
        charts.append(Chart(id="importance", title="Permutation importance", purpose="Which features the model uses",
                            path=_save(fig, out, "importance"),
                            interpretation=f"Orange = noise candidates {imp['noise_candidates'] or 'none'}; "
                                           "they were tested by ablation, not removed on this evidence alone.",
                            data={"kind": "hbar", "labels": [r["feature"] for r in rows],
                                  "values": [float(r["mean"]) for r in rows], "errors": [float(r["std"]) for r in rows],
                                  "colors": [ORANGE if r["feature"] in imp["noise_candidates"] else BLUE for r in rows],
                                  "xlabel": f"drop in CV {imp['scoring']} when shuffled"}))

    # 6. subgroup errors - only if groups were flagged
    hist = run.analyses.get("subgroups_history", [])
    flagged_any = [h for h in hist if h.get("flagged")]
    if flagged_any:
        first = flagged_any[0]
        key = first["metric"]
        groups = [f"{g['column']}={g['group']}" for g in first["flagged"][:5]]
        before = [g[key] for g in first["flagged"][:5]]
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(groups) + 2.6))
        yy = np.arange(len(groups))
        later = [h for h in hist if h["exp_id"] != first["exp_id"]]
        if later:
            idx = {f"{g['column']}={g['group']}": g for g in later[-1]["all_groups"]}
            after = [idx.get(g, {}).get(key, np.nan) for g in groups]
            ax.barh(yy - 0.18, before, height=0.34, color=BLUE, label=f"{first['exp_id']}")
            ax.barh(yy + 0.18, after, height=0.34, color=ORANGE, label=f"{later[-1]['exp_id']}")
        else:
            ax.barh(yy, before, height=0.5, color=ORANGE, label=first["exp_id"])
        ax.axvline(first["overall"], color=INK2, ls="--", lw=1, label=f"overall ({first['exp_id']})")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.3), ncol=1, fontsize=8)
        ax.set_yticks(yy, groups); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"out-of-fold {key}"); ax.set_title("Where the model fails: flagged subgroups")
        charts.append(Chart(id="subgroups", title="Subgroup errors", purpose="Error analysis",
                            path=_save(fig, out, "subgroups"),
                            interpretation="Groups clearly worse than overall (|z| ≥ 3); orange bars show the "
                                           "same groups for the later model, if it changed."))

    # 6b. forward-in-time check - only if it was run
    ts = run.analyses.get("time_split")
    if ts and ts.get("folds"):
        mname = ts["metric"]
        labels = [f"≤{f['train_until']}\n→ {f['test_to']}" for f in ts["folds"]]
        vals = [f[mname] for f in ts["folds"]]
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        x = np.arange(len(vals))
        ax.bar(x, vals, color=BLUE, width=0.55, label="forward-in-time fold")
        ax.axhline(ts["random_cv_mean"], color=ORANGE, lw=2, ls="--", label=f"shuffled CV mean {ts['random_cv_mean']:.3f}")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, labels, fontsize=7)
        lo = min(vals + [ts["random_cv_mean"]])
        ax.set_ylim(max(0, lo - 0.15), min(1.02, max(vals + [ts["random_cv_mean"]]) + 0.06) if mname != "r2" else None)
        ax.set_ylabel(mname); ax.set_title(f"Forward-in-time validation by {ts['date_col']}")
        ax.legend(loc="lower left", fontsize=8)
        charts.append(Chart(id="time_split", title="Forward-in-time validation", purpose="Validation over time",
                            path=_save(fig, out, "time_split"),
                            interpretation=f"Train on earlier rows, test on later ones: mean {mname} "
                                           f"{ts['time_split_mean']:.3f} vs shuffled CV {ts['random_cv_mean']:.3f}.",
                            data={"kind": "bars", "x": [l.replace(chr(10), " ") for l in labels], "values": list(map(float, vals)),
                                  "hline": float(ts["random_cv_mean"]), "ylabel": mname}))

    # 6c. interaction screen - only if it found a candidate
    it = run.analyses.get("interactions")
    if it and it.get("top10"):
        rows = it["top10"][:8]
        labels = [f"{r['feature']} | {r['group_col']}={r['level']}" for r in rows]
        vals = [abs(r["z"]) for r in rows]
        cand = {(r["feature"], r["group_col"], r["level"]) for r in it["candidates"]}
        cols = [ORANGE if (r["feature"], r["group_col"], r["level"]) in cand else BLUE for r in rows]
        fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.4))
        _hbar(ax, labels, vals, cols, "{:.1f}")
        ax.axvline(it["z_threshold"], color=INK2, ls="--", lw=1)
        ax.set_xlabel("|z|: how differently the feature relates to residuals inside vs outside the group")
        ax.set_title("Interaction screen (out-of-fold residuals)")
        charts.append(Chart(id="interactions", title="Interaction screen", purpose="Group-specific effects",
                            path=_save(fig, out, "interactions"),
                            interpretation=f"Orange = candidates (|z| ≥ {it['z_threshold']:g} and a residual-correlation "
                                           f"difference ≥ {it.get('min_corr_diff', 0):g}); they are hypotheses until "
                                           "test_interaction confirms them on identical folds.",
                            data={"kind": "hbar", "labels": labels, "values": vals, "colors": cols,
                                  "xlabel": "|z| (inside vs outside group)", "vline": it["z_threshold"]}))

    # 6d. fairness - one chart per checked column
    for col, fr in (run.analyses.get("fairness") or {}).items():
        key = "recall" if fr.get("metric") == "recall" else "mae"
        rows = [g for g in fr["groups"] if g.get(key) is not None]
        if not rows:
            continue
        labels = [f"{col}={g['group']} (n={g['n']})" for g in rows]
        vals = [g[key] for g in rows]
        worst = fr.get("worst_group")
        cols = [ORANGE if g["group"] == worst else BLUE for g in rows]
        fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.4))
        _hbar(ax, labels, vals, cols)
        ax.set_xlabel(f"out-of-fold {key}"); ax.set_title(f"Fairness check: {key} by {col}")
        cid = f"fairness_{col}"
        gap = f"gap {fr['gap']:.3f}" + (f", 95% CI {fr['gap_ci95']}" if fr.get("gap_ci95") else "") if "gap" in fr else ""
        charts.append(Chart(id=cid, title=f"Fairness: {col}", purpose="Error rates across groups",
                            path=_save(fig, out, cid),
                            interpretation=f"Orange = worst group. {gap}. "
                                           f"{'Material gap: review before use.' if fr.get('material') else 'No material gap.'}",
                            data={"kind": "hbar", "labels": labels, "values": vals, "colors": cols,
                                  "xlabel": f"out-of-fold {key}"}))

    # 6e. calibration - reliability diagram
    cal = run.analyses.get("calibration")
    if cal and cal.get("table"):
        t = cal["table"]
        xs, ys, ns = [r["mean_predicted"] for r in t], [r["observed_rate"] for r in t], [r["n"] for r in t]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1, label="perfect calibration")
        ax.plot(xs, ys, "-o", color=ORANGE if not cal["well_calibrated"] else BLUE, lw=2,
                label=f"{cal['exp_id']} (ECE {cal['ece']:.3f})")
        for x_, y_, n_ in zip(xs, ys, ns):
            ax.annotate(f"n={n_}", (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=7, color=INK2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed rate")
        ax.set_title("Calibration (out-of-fold, reliability)"); ax.legend(loc="upper left", fontsize=8)
        charts.append(Chart(id="calibration", title="Calibration", purpose="Can the probabilities be trusted?",
                            path=_save(fig, out, "calibration"),
                            interpretation=(f"Brier {cal['brier']:.4f}, ECE {cal['ece']:.3f}. Points below the diagonal = "
                                            "the model over-states risk." + (" Class weighting inflates probabilities."
                                                                              if cal["class_weighted"] else "")),
                            data={"kind": "scatterline", "x": xs, "y": ys, "n": ns, "xlabel": "mean predicted probability",
                                  "ylabel": "observed rate", "diagonal": True}))

    # 7. final model on the untouched test set
    w = E.get(run.facts["working_id"]) if run.facts["working_id"] else None
    if w is not None:
        p = load_predictions(run.run_dir / w.predictions_path)
        if task == "binary_classification":
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
            yt, yp = p.y_true.to_numpy(), p.y_pred.to_numpy()
            cm = np.array([[np.sum((yt == 0) & (yp == 0)), np.sum((yt == 0) & (yp == 1))],
                           [np.sum((yt == 1) & (yp == 0)), np.sum((yt == 1) & (yp == 1))]])
            ax = axes[0]
            ax.imshow(cm, cmap="Blues"); ax.grid(False)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=11)
            ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
            ax.set_title(f"Confusion matrix ({w.exp_id})")
            ax = axes[1]
            b = E[run.facts["baseline_id"]]
            for exp, col in [(b, GRAY), (w, ORANGE)]:
                q = load_predictions(run.run_dir / exp.predictions_path)
                order = np.argsort(-q.y_score.to_numpy())
                tp = np.cumsum(q.y_true.to_numpy()[order]); prec = tp / np.arange(1, len(q) + 1); rec = tp / tp[-1]
                ax.plot(rec, prec, color=col, lw=2, label=f"{exp.exp_id} (PR-AUC {exp.test_metrics['pr_auc']:.2f})")
            ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Precision-recall, test set")
            ax.legend(loc="upper right", fontsize=7)
            interp = (f"Test recall {w.test_metrics['recall']:.3f}, precision {w.test_metrics['precision']:.3f} at "
                      f"threshold {w.threshold:.2f}. Gray = original baseline (includes any leaking feature).")
        elif task == "multiclass_classification":
            classes = getattr(ctx, "classes_", None) or sorted(set(p.y_true) | set(p.y_pred))
            K = len(classes)
            yt, yp = p.y_true.to_numpy().astype(int), p.y_pred.to_numpy().astype(int)
            cm = np.zeros((K, K), dtype=int)
            for a_, b_ in zip(yt, yp):
                cm[a_, b_] += 1
            fig, ax = plt.subplots(figsize=(0.9 * K + 3, 0.8 * K + 2))
            ax.imshow(cm, cmap="Blues"); ax.grid(False)
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=10)
            ax.set_xticks(range(K), [f"pred {c}" for c in classes], rotation=30, ha="right")
            ax.set_yticks(range(K), [f"true {c}" for c in classes])
            ax.set_title(f"Confusion matrix ({w.exp_id})")
            interp = (f"Test balanced accuracy {w.test_metrics['balanced_accuracy']:.3f}, "
                      f"macro F1 {w.test_metrics['f1_macro']:.3f}.")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
            ax = axes[0]
            ax.scatter(p.y_true, p.y_pred, s=8, color=BLUE, alpha=0.5)
            lo, hi = p.y_true.min(), p.y_true.max()
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
            ax.set_xlabel("actual"); ax.set_ylabel("predicted"); ax.set_title(f"Predicted vs actual ({w.exp_id})")
            ax = axes[1]
            ax.scatter(p.y_pred, p.y_true - p.y_pred, s=8, color=BLUE, alpha=0.5)
            ax.axhline(0, color=INK2, lw=1); ax.set_xlabel("predicted"); ax.set_ylabel("residual")
            ax.set_title("Residuals")
            from matplotlib.ticker import FuncFormatter, MaxNLocator
            big = float(p.y_true.abs().max()) >= 10000
            for a_ in axes:
                a_.xaxis.set_major_locator(MaxNLocator(5))
                a_.yaxis.set_major_locator(MaxNLocator(6))
                if big:
                    a_.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
                    a_.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
            axes[0].set_title(f"Predicted vs actual ({w.exp_id})", fontsize=10)
            interp = f"Test R²{w.test_metrics['r2']:.3f}, RMSE {w.test_metrics['rmse']:,.0f}."
        charts.append(Chart(id="final", title="Final model on the test set", purpose="Final evaluation",
                            path=_save(fig, out, "final"), interpretation=interp))

    # 8. SHAP global importance (final model, and the original baseline if it differs)
    for key, ex in (explanations or {}).items():
        if not ex.get("supported"):
            continue
        rows = ex["global"][:10]
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(rows) + 1.2))
        _hbar(ax, [r["feature"] for r in rows], [r["mean_abs_shap"] for r in rows],
              [ORANGE if r["feature"] in run.facts["excluded_as_leak"] else BLUE for r in rows],
              "{:,.3g}")
        ax.set_xlabel(f"mean |SHAP| ({ex['output_space']})")
        ax.set_title(f"SHAP global importance: {ex['exp_id']}")
        cid = f"shap_{key}"
        charts.append(Chart(id=cid, title=f"SHAP ({key})", purpose="Explainability",
                            path=_save(fig, out, cid),
                            interpretation="How strongly each feature moves this model's output on test rows. "
                                           "Describes the model, not causes. Orange = feature later excluded as leakage."))
    return charts
