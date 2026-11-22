"""Assemble the HTML report from stored, verified run data.

finalize_run() = explanations + charts + narratives + report.html, all written to the run folder.
The report contains only numbers read from experiment records or tool outputs.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Optional

from jinja2 import Environment

from trustml.charts import build_charts
from trustml.explain import explain_model
from trustml.narrative import write_narratives
from trustml.schemas import Severity


def _img(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _recommendations(run) -> list[str]:
    recs, f = [], run.facts
    if f["excluded_as_leak"]:
        recs.append(f"Confirm with the data owner when {f['excluded_as_leak']} is recorded. If it is only known after "
                    "the outcome, remove it from the data pipeline permanently.")
    checks = {x.check for x in run.quality.findings + run.split_findings}
    if run.facts["done"].get("dedupe"):
        recs.append("Duplicate rows were removed before splitting in this analysis; fix the duplication at the source.")
    elif "duplicate_rows" in checks or "train_test_overlap" in checks:
        recs.append("Deduplicate before training. Exact duplicates fall on both sides of the split and inflate test scores.")
    if run.ctx.group_col:
        recs.append(f"Keep using a split grouped by '{run.ctx.group_col}' (as this analysis did) whenever the model is "
                    "re-validated, so the same entity never appears in training and evaluation.")
    elif "repeated_ids" in checks:
        ids = [x.columns[0] for x in run.split_findings if x.check == "repeated_ids"]
        recs.append(f"If {ids} identify real entities, rerun with a group split (--group-col {ids[0]}).")
    if "sentinel_value" in checks:
        recs.append("Replace placeholder values (e.g. -999) with missing values at the source.")
    ts = run.analyses.get("time_split")
    if ts and ts.get("time_split_mean") is not None:
        if any(v["topic"] == "time_split" and v["verdict"] in ("degrades_forward_in_time", "target_drift")
               for v in run.facts["verdicts"]):
            recs.append(f"Performance drops on newer rows ({ts['metric']} {ts['time_split_mean']:.3f} forward-in-time vs "
                        f"{ts['random_cv_mean']:.3f} shuffled). Quote the forward-in-time estimate and retrain regularly.")
    elif "datetime_present" in checks:
        recs.append("If rows are time-ordered, validate on a later time period before deployment.")
    for h in run.analyses.get("subgroups_history", [])[-1:]:
        if h.get("flagged"):
            g = h["flagged"][0]
            recs.append(f"The final model is weakest for {g['column']}={g['group']}. Collect more examples or "
                        "group-specific information for it, and monitor it separately.")
    for v in f["verdicts"]:
        if v["topic"] == "interaction" and v["verdict"] == "interaction_confirmed":
            recs.append(f"Keep the interaction term '{v['feature']}': the effect differs inside {v['group']}. Ask the "
                        "domain owner whether this group needs its own policy or model.")
        if v["topic"] == "fairness" and v["material"]:
            recs.append(f"Review the error-rate gap across '{v['column']}' ({v['which']}) with domain and legal owners "
                        "before deployment; consider group-aware thresholds only if policy allows.")
        if v["topic"] == "calibration" and not v["well_calibrated"]:
            recs.append("Do not quote the raw predicted probabilities as risk percentages; recalibrate (e.g. Platt or "
                        "isotonic on held-out training folds) if probabilities are needed.")
        if v["topic"] == "threshold" and v["verdict"] in ("recall_improved", "cost_reduced"):
            recs.append(f"Use the tuned decision threshold from {v['new']} rather than 0.5.")
    return recs


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>TrustML report – {{ run_id }}</title>
<style>
:root{--blue:#1f5fbf;--blue-bg:#eaf2fc;--ink:#0b0b0b;--ink2:#52514e;--line:#e3e2dd;--ok:#1b7f3b;--okbg:#e7f5ec;--warn:#8a5a00;--warnbg:#fdf3dc;--bad:#a12a2a;--badbg:#fbe9e9}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);max-width:1000px;margin:0 auto;padding:28px 28px 80px;line-height:1.55;background:#fff}
h1{font-size:26px;font-weight:600;margin:0 0 4px;color:var(--blue)}h2{font-size:19px;font-weight:600;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--blue-bg)}
h3{font-size:15px;font-weight:600;margin:18px 0 6px}.muted{color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:#f7f7f5;font-weight:600}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.ok{background:var(--okbg);color:var(--ok)}.warn{background:var(--warnbg);color:var(--warn)}.bad{background:var(--badbg);color:var(--bad)}.info{background:var(--blue-bg);color:var(--blue)}
.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}.card .k{color:var(--ink2);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.summary{background:var(--blue-bg);border-radius:10px;padding:14px 16px}figure{margin:14px 0}figure img{max-width:100%;border:1px solid var(--line);border-radius:8px}figcaption{font-size:13px;color:var(--ink2)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.tile{background:#f7f7f5;border-radius:8px;padding:10px}.tile b{display:block;font-size:20px}
code{background:#f2f2ef;padding:1px 4px;border-radius:4px;font-size:12px}
.printbtn{position:fixed;top:14px;right:14px;background:var(--blue);color:#fff;border:0;border-radius:8px;padding:7px 14px;font-weight:600;cursor:pointer}
@media print{.printbtn{display:none}body{padding:0;max-width:none}.card,figure,table{break-inside:avoid}h2{break-after:avoid}}
</style></head><body>
<button class="printbtn" onclick="window.print()">Save as PDF</button>
<h1>TrustML analysis report</h1>
<div class="muted">Run {{ run_id }} · dataset <code>{{ data_path }}</code> (sha256 {{ sha[:12] }}…) · planner {{ planner }} · {{ steps|length }} steps · {{ elapsed }} s</div>
<p><b>Objective:</b> {{ objective }}</p>
<p>{% if verified %}<span class="badge ok">All {{ n_claims }} reported numbers independently verified</span>{% else %}<span class="badge bad">Verification failures – see appendix</span>{% endif %}
 <span class="badge info">Stop reason: {{ stop_reason }}</span></p>

<h2>1. Executive summary</h2>
<div class="summary">{{ narr.summary }}</div>
<p class="muted">Source: {{ narr.summary_meta.source }}{% if narr.summary_meta.numbers_checked %} – every number checked against verified results{% endif %}{% if narr.summary_meta.rejected_llm_text %} – LLM draft rejected (unsupported numbers {{ narr.summary_meta.unsupported_numbers }}){% endif %}.</p>
{% if out_of_scope %}<div class="card"><span class="badge warn">Not supported</span>{% for i in out_of_scope %}<div>“{{ i.matched }}” – {{ i.reason }}</div>{% endfor %}<div class="muted">The rest of the request was carried out.</div></div>{% endif %}
<h3>Key conclusions</h3><ul>{% for c in conclusions %}<li>{{ c }}</li>{% endfor %}</ul>

<h2>2. Dataset overview</h2>
<div class="grid"><div class="tile">rows<b>{{ "{:,}".format(n_rows) }}</b></div><div class="tile">columns<b>{{ n_cols }}</b></div>
<div class="tile">target<b>{{ target }}</b></div><div class="tile">task<b style="font-size:15px">{{ task }}</b></div></div>
{{ chart('target') }}
<table><tr><th>column</th><th>kind</th><th>missing %</th><th>unique</th><th>used as feature</th></tr>
{% for c in columns %}<tr><td>{{ c.name }}</td><td>{{ c.inferred_kind }}</td><td>{{ "%.1f"|format(c.pct_missing) }}</td><td>{{ c.n_unique }}</td><td>{{ used.get(c.name, '') }}</td></tr>{% endfor %}</table>

<h2>3. Data quality and leakage findings</h2>
<table><tr><th>severity</th><th>check</th><th>columns</th><th>finding</th><th>recommendation</th></tr>
{% for f in findings %}<tr><td><span class="badge {{ sevcls[f.severity] }}">{{ f.severity }}</span></td><td>{{ f.check }}</td><td>{{ f.columns|join(', ') }}</td><td>{{ f.message }}</td><td class="muted">{{ f.recommendation }}</td></tr>{% endfor %}</table>
{{ chart('missing') }}{{ chart('leak_screen') }}
<p class="muted">Leakage findings are suspicions from the training split only; the ablation in section 5 tests them. Correlation is not proof of leakage.</p>

<h2>4. Preprocessing and validation</h2>
<ul><li>Split: {{ split.strategy }} {{ split.n_train }} train / {{ split.n_test }} test (seed {{ split.seed }}, test-index hash {{ split.test_index_sha[:12] }}…). The test set is fixed for the whole run.</li>
<li>{{ split.cv_folds }}-fold cross-validation{% if cv_repeats > 1 %}, repeated {{ cv_repeats }}× with different shuffles{% endif %} on the training split. All agent decisions use CV results only; the planner never sees test metrics.</li>
<li>Preprocessing inside a scikit-learn pipeline fitted on training folds only: median imputation + missing indicators (+ scaling for linear models); most-frequent imputation + one-hot encoding (rare levels grouped) for categoricals.</li>
<li>Original file never modified (hash recorded). Transformations applied to a derived copy:</li></ul>
<table><tr><th>operation</th><th>details</th></tr>{% for t in transforms %}<tr><td>{{ t.op }}</td><td class="muted">{{ t|tojson }}</td></tr>{% endfor %}</table>

<h2>5. Agent investigation</h2>
<p class="muted">Each step: what the agent observed → what it decided and why → what the tool returned → independent verification.</p>
{% for s in steps %}<div class="card"><div><b>Step {{ s.step }} · {{ s.action.action }}</b> <span class="muted">planner: {{ s.planner }}</span>
{% if s.verification %}{% if s.verification.passed %}<span class="badge ok">verified</span>{% else %}<span class="badge bad">verification failed</span>{% endif %}{% endif %}</div>
<div class="k">Observed</div><div>{{ s.observation }}</div>
<div class="k">Decided</div><div>{{ s.action.rationale or s.action.reason }} <code>{{ args(s.action) }}</code></div>
{% if s.planner_note %}<div class="muted">Note: {{ s.planner_note }}</div>{% endif %}
{% if s.result.interpretation %}<div class="k">Result</div><div>{{ s.result.interpretation }}</div>{% endif %}
{% if s.verification %}<div class="muted">{{ nchecks(s.verification) }} independent checks{% if s.verification.note %} – {{ s.verification.note }}{% endif %}</div>{% endif %}
</div>{% for cid in step_charts.get(s.step, []) %}{{ chart(cid) }}{% endfor %}{% endfor %}

<h2>6. Experiments</h2>
<table><tr><th>id</th><th>model</th><th>changes vs base</th><th>CV {{ primary }} (mean ± sd)</th>{% if binary %}<th>CV recall</th>{% endif %}<th>test {{ key_test }} [95% CI]</th></tr>
{% for e in experiments %}<tr{% if e.exp_id == final_id %} style="background:var(--blue-bg)"{% endif %}><td>{{ e.exp_id }}{% if e.exp_id == final_id %} <span class="badge info">final</span>{% endif %}</td><td>{{ e.spec.model }}</td><td class="muted">{{ changes(e) }}</td>
<td>{{ "%.4f"|format(e.cv_metrics[primary].mean) }} ± {{ "%.4f"|format(e.cv_metrics[primary].std) }}</td>{% if binary %}<td>{{ "%.3f"|format(e.cv_metrics['recall'].mean) }}</td>{% endif %}
<td>{{ fmt(e.test_metrics[key_test]) }} {% if e.test_ci95.get(key_test) %}<span class="muted">[{{ fmt(e.test_ci95[key_test][0]) }}, {{ fmt(e.test_ci95[key_test][1]) }}]</span>{% endif %}</td></tr>{% endfor %}</table>
<p class="muted">Test scores are shown for transparency; they were not used for any decision.</p>
{{ chart('final') }}

<h2>7. Feature recommendations</h2>
<h3>Engineered features (derived from existing columns – tested)</h3>
{% if eng_tested %}<table><tr><th>feature</th><th>recipe</th><th>verdict</th><th>prediction-time availability</th></tr>{% for v in eng_tested %}<tr><td>{{ v.feature }}</td><td>{{ v.recipe }}</td><td>{{ v.verdict }}</td><td class="muted">computed row-wise from its source columns; available whenever they are</td></tr>{% endfor %}</table>{% else %}<p class="muted">No engineered feature was tested in this run.</p>{% endif %}
<p class="muted">Screened but not tested in the full model: {% for s in eng_untested %}{{ s.name }} (screen gain {{ "%+.4f"|format(s.screen_gain) }}){% if not loop.last %}, {% endif %}{% else %}none{% endfor %}.</p>
<h3>New variables worth collecting (suggestions – NOT tested)</h3>
{% if narr.collection %}<table><tr><th>variable</th><th>why</th><th>how to collect</th><th>known at prediction time?</th><th>feasibility</th><th>privacy / fairness</th></tr>
{% for s in narr.collection %}<tr><td>{{ s.name }}</td><td>{{ s.rationale }}</td><td class="muted">{{ s.how_to_collect }}</td><td>{{ s.available_at_prediction_time }} – <span class="muted">{{ s.timing_note }}</span></td><td>{{ s.feasibility }}</td><td class="muted">{{ s.privacy_fairness }}</td></tr>{% endfor %}</table>
<p class="muted">Generated by {{ narr.collection_meta.source }} from column names only (no data values sent). These are hypotheses; their value is unknown until the data is collected and tested.</p>
{% else %}<p class="muted">No suggestions ({{ narr.collection_meta.error or 'LLM not configured' }}).</p>{% endif %}

<h2>8. Error analysis</h2>
{% if subgroup_runs %}{% for h in subgroup_runs %}<h3>{{ h.exp_id }} – out-of-fold {{ h.metric }} overall {{ fmt(h.overall) }}</h3>
{% if h.flagged %}<table><tr><th>group</th><th>{{ h.metric }}</th><th>n</th><th>z</th></tr>{% for g in h.flagged %}<tr><td>{{ g.column }}={{ g.group }}</td><td>{{ fmt(g.get('recall', g.get('mae'))) }}</td><td>{{ g.get('n_positives', g.get('n_rows')) }}</td><td>{{ g.z }}</td></tr>{% endfor %}</table>{% else %}<p>No group clearly worse than overall.</p>{% endif %}{% endfor %}{{ chart('subgroups') }}
{% else %}<p class="muted">Subgroup analysis was not run.</p>{% endif %}
{% if time_split and time_split.folds %}<h3>Forward-in-time validation ({{ time_split.date_col }})</h3>
<table><tr><th>fold</th><th>trained until</th><th>tested on</th><th>n test</th><th>{{ time_split.metric }}</th></tr>
{% for f in time_split.folds %}<tr><td>{{ f.fold }}</td><td>{{ f.train_until }}</td><td>{{ f.test_from }} – {{ f.test_to }}</td><td>{{ f.n_test }}</td><td>{{ "%.3f"|format(f[time_split.metric]) }}</td></tr>{% endfor %}</table>
<p>Mean {{ "%.3f"|format(time_split.time_split_mean) }} vs shuffled CV {{ "%.3f"|format(time_split.random_cv_mean) }}. Forward-in-time folds use training rows only; the test set is untouched.</p>{{ chart('time_split') }}{% endif %}

{% if interactions %}<h3>Group-specific effects (interaction screen)</h3>
<p class="muted">{{ interactions.method }}; {{ interactions.n_tested }} feature × group pairs tested on out-of-fold training residuals; threshold |z| ≥ {{ interactions.z_threshold }}.</p>
{% if interactions.candidates %}<table><tr><th>feature</th><th>group</th><th>n in group</th><th>residual corr. inside</th><th>outside</th><th>z</th></tr>{% for c in interactions.candidates %}<tr><td>{{ c.feature }}</td><td>{{ c.group_col }}={{ c.level }}</td><td>{{ c.n_in }}</td><td>{{ "%+.3f"|format(c.r_in) }}</td><td>{{ "%+.3f"|format(c.r_out) }}</td><td>{{ "%+.1f"|format(c.z) }}</td></tr>{% endfor %}</table>{% else %}<p>No group-specific effect detected.</p>{% endif %}
{% for v in interaction_verdicts %}<p><b>Tested:</b> {{ v.feature }} – <span class="badge {{ 'ok' if v.verdict == 'interaction_confirmed' else 'warn' }}">{{ v.verdict }}</span> ({{ v.group }} {{ v.group_metric }} {{ "%.3f"|format(v.group_before) }} → {{ "%.3f"|format(v.group_after) }}, out-of-fold).</p>{% endfor %}
{{ chart('interactions') }}{% endif %}

<h2>9. Trust checks: fairness and calibration</h2>
{% if fairness %}{% for col, fr in fairness.items() %}<h3>Fairness by {{ col }} ({{ fr.exp_id }}, out-of-fold)</h3>
<table><tr><th>group</th><th>n</th>{% if fr.metric == 'recall' %}<th>positives</th><th>actual rate</th><th>flagged rate</th><th>recall</th><th>false-alarm rate</th>{% else %}<th>MAE</th><th>mean actual</th><th>relative MAE</th><th>mean error</th>{% endif %}</tr>
{% for g in fr.groups %}<tr><td>{{ g.group }}</td><td>{{ g.n }}</td>{% if fr.metric == 'recall' %}<td>{{ g.n_pos }}</td><td>{{ fmt(g.actual_rate) }}</td><td>{{ fmt(g.flagged_rate) }}</td><td>{{ fmt(g.recall) }}</td><td>{{ fmt(g.false_alarm_rate) }}</td>{% else %}<td>{{ fmt(g.mae) }}</td><td>{{ fmt(g.mean_actual) }}</td><td>{{ fmt(g.relative_mae) }}</td><td>{{ fmt(g.mean_error) }}</td>{% endif %}</tr>{% endfor %}</table>
{% if fr.gap is defined %}<p>{{ fr.metric }} gap {{ "%.3f"|format(fr.gap) }}{% if fr.gap_ci95 %} (bootstrap 95% CI {{ fr.gap_ci95 }}){% endif %}{% if fr.false_alarm_gap is not none and fr.false_alarm_gap is defined %}; false-alarm gap {{ "%.3f"|format(fr.false_alarm_gap) }} (95% CI {{ fr.false_alarm_gap_ci95 }}){% endif %} – {% if fr.material %}<span class="badge bad">material</span>{% else %}<span class="badge ok">not material</span>{% endif %}</p>{% endif %}
{{ chart('fairness_' ~ col) }}{% endfor %}
<p class="muted">A gap in error rates is a signal for review, not a legal finding. Removing a sensitive column does not remove proxies for it.</p>
{% else %}<p class="muted">No fairness column was chosen or detected (columns such as gender, sex, race, age or marital status are checked automatically; others can be chosen in the dashboard or with --fairness-col).</p>{% endif %}
{% if calibration %}<h3>Calibration ({{ calibration.exp_id }}, out-of-fold)</h3>
<p>Brier score {{ "%.4f"|format(calibration.brier) }}, expected calibration error {{ "%.3f"|format(calibration.ece) }}; mean predicted probability {{ "%.3f"|format(calibration.mean_predicted) }} vs actual rate {{ "%.3f"|format(calibration.actual_rate) }} –
{% if calibration.well_calibrated %}<span class="badge ok">well calibrated</span>{% else %}<span class="badge warn">needs recalibration before probabilities are quoted</span>{% endif %}</p>
{{ chart('calibration') }}{% endif %}

<h2>10. Explainability</h2>
<p class="muted">SHAP values describe how the model uses its inputs on test rows. They are not causal effects.</p>
{% for key, ex in explanations.items() %}{% if ex.supported %}{{ chart('shap_' ~ key) }}
<p class="muted">Additivity check (base value + SHAP = model output): max error {{ "%.2e"|format(ex.additivity_max_error) }} {% if ex.additivity_ok %}<span class="badge ok">passed</span>{% else %}<span class="badge bad">failed</span>{% endif %}</p>
{% if key == 'final' %}{% for l in ex.local %}<div class="card"><b>{{ l.case }}</b> <span class="muted">(row {{ l.row_index }}; true {{ fmt(l.y_true) }}, predicted {{ fmt(l.y_pred) }}; base {{ fmt(l.base_value) }} → output {{ fmt(l.model_output) }} in {{ l.output_space }})</span>
<table><tr><th>feature</th><th>value</th><th>SHAP contribution</th></tr>{% for c in l.contributions %}<tr><td>{{ c.feature }}</td><td>{{ c.value }}</td><td>{{ "%+.4g"|format(c.shap) }}</td></tr>{% endfor %}</table></div>{% endfor %}{% endif %}{% endif %}{% endfor %}

<h2>11. Limitations and caveats</h2><ul>{% for c in caveats %}<li>{{ c }}</li>{% endfor %}
<li>Models are limited to dummy, logistic/linear regression, random forest and gradient boosting, with fixed settings (no hyperparameter search).</li>
<li>Leakage and noise judgements are evidence-based heuristics, not proofs; domain review is required.</li></ul>

<h2>12. Recommendations</h2><ol>{% for r in recs %}<li>{{ r }}</li>{% endfor %}</ol>

<h2>Appendix – verification</h2>
<table><tr><th>claim</th><th>result</th><th>detail</th></tr>{% for c in claim_checks %}<tr><td>{{ c.name }}</td><td>{% if c.passed %}<span class="badge ok">pass</span>{% else %}<span class="badge bad">fail</span>{% endif %}</td><td class="muted">{{ c.detail }}</td></tr>{% endfor %}</table>
<p class="muted">Every experiment was also checked by trustml/verify.py: fixed test set, labels match the original file, target not a feature, exclusions respected, threshold applied, and every test metric recomputed with independent code ({{ total_checks }} checks in total).</p>
</body></html>"""


def finalize_run(run, llm_client=None, llm_model: Optional[str] = None) -> dict:
    f = run.facts
    explanations = {}
    if f["working_id"]:
        explanations["final"] = explain_model(run.ctx, f["working_id"])
        if f["baseline_id"] and f["baseline_id"] != f["working_id"]:
            explanations["baseline"] = explain_model(run.ctx, f["baseline_id"])
    (run.run_dir / "explanations.json").write_text(json.dumps(explanations, indent=2, default=str))
    charts = build_charts(run, explanations)
    (run.run_dir / "charts.json").write_text(json.dumps(charts, indent=2))
    narr = write_narratives(run, llm_client, llm_model)
    (run.run_dir / "narratives.json").write_text(json.dumps(narr, indent=2))

    by_id = {c["id"]: c for c in charts}
    step_charts: dict[int, list[str]] = {}
    for s in run.steps:
        for e in s.result.get("new_experiments", []):
            if f"cmp_{e}" in by_id:
                step_charts.setdefault(s.step, []).append(f"cmp_{e}")
        if s.action["action"] == "rank_feature_importance":
            step_charts.setdefault(s.step, []).append("importance")

    def chart(cid):
        c = by_id.get(cid)
        if not c:
            return ""
        return (f'<figure><img alt="{html.escape(c["title"])}" src="{_img(c["path"])}"><figcaption><b>{html.escape(c["title"])}</b> – '
                f'{html.escape(c["interpretation"])}</figcaption></figure>')

    E = run.ctx.experiments
    binary = run.ctx.task.value == "binary_classification"
    key_test = "recall" if binary else ("r2" if run.ctx.task.value == "regression" else "balanced_accuracy")
    used = {c: "yes" for c in run.ctx.candidate_features}
    for t in run.ctx.transform_log:
        if t["op"] == "exclude_feature":
            used[t["column"]] = f"no ({t['reason']})"
    for c in f["excluded_as_leak"]:
        used[c] = "no (suspected leakage)"
    sev_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}
    findings = sorted([x for x in run.all_findings() if x.check != "single_feature_ranking"],
                      key=lambda x: sev_order[x.severity])
    tested_names = {v["feature"] for v in f["verdicts"] if v["topic"] == "feature_addition"}
    eng_tested = [{"feature": v["feature"], "verdict": v["verdict"],
                   "recipe": run.ctx.engineered.get(v["feature"], {})} for v in f["verdicts"] if v["topic"] == "feature_addition"]

    def changes(e):
        s, parts = e.spec, []
        if s.parent_id:
            parent = E[s.parent_id].spec
            new_ex = [x for x in s.exclude_features if x not in parent.exclude_features]
            new_add = [x for x in s.extra_features if x not in parent.extra_features]
            if new_ex: parts.append(f"− {', '.join(new_ex)}")
            if new_add: parts.append(f"+ {', '.join(new_add)}")
            if s.class_weight != parent.class_weight: parts.append("class weights balanced")
            if s.threshold_strategy != parent.threshold_strategy: parts.append(f"threshold {s.threshold_strategy} = {e.threshold:.2f}")
            if s.model != parent.model: parts.append(f"model → {s.model}")
            if s.sentinel_to_missing != parent.sentinel_to_missing: parts.append(f"placeholders → missing {s.sentinel_to_missing}")
            if s.target_transform != parent.target_transform: parts.append(f"target {s.target_transform}")
            return f"from {s.parent_id}: " + "; ".join(parts)
        return "reference" if s.model == "dummy" else "all candidate features"

    def fmt(v):
        if v is None: return "–"
        v = float(v)
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"

    env = Environment(autoescape=True)
    env.globals.update(chart=lambda cid: __import__("markupsafe").Markup(chart(cid)), fmt=fmt, changes=changes,
                       args=lambda a: json.dumps({k: v for k, v in a.items() if k not in ("action", "rationale", "reason")}),
                       nchecks=lambda v: sum(x["n_checks"] for x in v["experiments"].values()) + len(v.get("analysis_checks", [])))
    s = run.summary
    html_out = env.from_string(TEMPLATE).render(
        run_id=run.run_id, data_path=str(run.handle.path), sha=run.handle.sha256, planner=getattr(run.planner, "name", "?"),
        steps=[st.model_dump() for st in run.steps], elapsed=s["elapsed_s"], objective=run.objective,
        verified=s["all_claims_verified"] and all(st.verification.get("passed", True) for st in run.steps),
        n_claims=len(s["claim_checks"]), stop_reason=s["stop_reason"], narr=narr, out_of_scope=s["out_of_scope"],
        conclusions=s["conclusions"], n_rows=run.handle.n_rows, n_cols=run.handle.n_cols, target=run.ctx.target,
        task=run.ctx.task.value.replace("_", " "), columns=run.profile.columns, used=used,
        findings=[x.model_dump(mode="json") for x in findings],
        sevcls={"high": "bad", "medium": "warn", "low": "info", "info": "info"},
        split=run.ctx.split.model_dump(), transforms=run.ctx.transform_log, step_charts=step_charts,
        experiments=list(E.values()), final_id=f["working_id"], primary=run.primary(), binary=binary, key_test=key_test,
        eng_tested=eng_tested,
        eng_untested=[x for x in f["engineered_suggestions"] if x["name"] not in tested_names],
        subgroup_runs=run.analyses.get("subgroups_history", []), explanations=explanations,
        time_split=run.analyses.get("time_split"),
        interactions=run.analyses.get("interactions"), fairness=run.analyses.get("fairness") or {},
        calibration=run.analyses.get("calibration"), cv_repeats=getattr(run, "cv_repeats", 1),
        interaction_verdicts=[v for v in f["verdicts"] if v["topic"] == "interaction"],
        caveats=s["caveats"], recs=_recommendations(run), claim_checks=s["claim_checks"],
        total_checks=sum(st.verification["experiments"][k]["n_checks"] for st in run.steps if st.verification
                         for k in st.verification["experiments"]),
    )
    path = run.run_dir / "report.html"
    path.write_text(html_out, encoding="utf-8")
    return {"report": str(path), "charts": charts, "explanations": explanations, "narratives": narr}
