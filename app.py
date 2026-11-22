"""TrustML dashboard.   Run:  python -m streamlit run app.py

Everything shown is read from a run folder (runs/<run_id>/), so a saved run can be reopened and
presented exactly as it was produced.
"""
from __future__ import annotations

import base64
import html
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
BENCH = ROOT / "data" / "benchmarks"
LOGO_SVG = ROOT / "assets" / "trustml_logo.svg"
LOGO_PNG = ROOT / "assets" / "trustml_logo.png"


def logo_img(px: int) -> str:
    if LOGO_SVG.exists():
        b64 = base64.b64encode(LOGO_SVG.read_bytes()).decode()
        return f'<img src="data:image/svg+xml;base64,{b64}" width="{px}" height="{px}" alt="TrustML logo">'
    return f'<div class="tm-logo" style="width:{px}px;height:{px}px">T</div>'


DEMOS = {
    "Churn – leakage, imbalance and cost": {
        "file": "churn_synthetic_v1.csv", "target": "churned", "use_cost": True, "cost_fn": 5.0, "cost_fp": 1.0,
        "objective": "Predict which customers will churn. Check the data for problems and leakage, explain why the "
                     "model misses churners, and recommend a decision threshold if missing a churner costs 5 times "
                     "more than a false alarm."},
    "House prices – features, models and weak spots": {
        "file": "housing_synthetic_v1.csv", "target": "sale_price", "use_cost": False,
        "objective": "Predict sale price. Audit the data for quality problems and leakage, test whether any engineered "
                     "feature helps, and tell me where the model is least accurate."},
}

ACTION_STYLE = {  # category -> (label, colour)
    "deduplicate_rows": ("Data fix", "#0f766e"), "treat_sentinels_as_missing": ("Data fix", "#0f766e"),
    "log_transform_target": ("Data fix", "#0f766e"), "run_baseline": ("Baseline", "#475569"),
    "ablate_feature": ("Leakage", "#b42318"), "apply_class_weighting": ("Imbalance", "#b54708"),
    "tune_threshold": ("Threshold", "#b54708"), "rank_feature_importance": ("Features", "#6941c6"),
    "ablate_feature_group": ("Features", "#6941c6"), "add_engineered_feature": ("Features", "#6941c6"),
    "try_model": ("Model", "#1f5fbf"), "analyze_subgroups": ("Error analysis", "#475569"),
    "check_time_split": ("Validation", "#475569"), "finish": ("Stop", "#344054"),
    "use_group_split": ("Data fix", "#0f766e"), "screen_interactions": ("Interactions", "#7a2e98"),
    "test_interaction": ("Interactions", "#7a2e98"), "check_fairness": ("Trust check", "#067647"),
    "check_calibration": ("Trust check", "#067647"),
}

st.set_page_config(page_title="TrustML", page_icon=str(LOGO_PNG) if LOGO_PNG.exists() else "◆", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family: 'Inter', -apple-system, Segoe UI, sans-serif; }
footer, [data-testid="stDecoration"] {visibility: hidden;}
header[data-testid="stHeader"] { background: transparent; }
/* keep the "reopen sidebar" control visible and obvious in every Streamlit version */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; z-index: 1000; }
[data-testid="stExpandSidebarButton"] button, [data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button, button[data-testid="stExpandSidebarButton"] {
  background: #1f5fbf !important; color: #fff !important; border-radius: 8px !important; }
.stApp { background: #f6f8fb; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e4e7ec; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
h1, h2, h3 { color: #101828; letter-spacing: -0.01em; }
.tm-brand { display:flex; align-items:center; gap:14px; margin: 2px 0 4px; }
.tm-brand img { border-radius:14px; box-shadow: 0 4px 14px rgba(31,95,191,.25); }
.tm-logo { border-radius:12px; background:#1f5fbf; color:#fff; display:flex; align-items:center;
           justify-content:center; font-weight:700; font-size:22px; }
.tm-name { font-weight:800; font-size:31px; color:#101828; letter-spacing:-0.02em; line-height:1; }
.tm-name span { color:#1f5fbf; }
.tm-tag { color:#667085; font-size:12.5px; margin: 6px 0 12px; }
.tm-plan { background:#f9fafb; border:1px solid #e4e7ec; border-radius:10px; padding:10px 12px; font-size:12.5px; color:#475467; margin:4px 0 6px; }
.tm-plan b { color:#101828; }
.tm-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.tm-heroband { display:flex; align-items:center; gap:22px; }
.tm-heroband .tm-name { font-size:44px; }
.tm-trace { font-size:12.5px; color:#475467; border-left:3px solid #c7d7fe; padding:2px 0 2px 10px; margin:4px 0; }
.tm-side-h { font-size:11px; font-weight:600; color:#667085; text-transform:uppercase; letter-spacing:.06em; margin:14px 0 4px; }
.tm-header { background:#fff; border:1px solid #e4e7ec; border-radius:14px; padding:18px 22px; margin-bottom:14px; }
.tm-title { font-size:22px; font-weight:700; color:#101828; margin:0; }
.tm-sub { color:#475467; font-size:13.5px; margin-top:4px; }
.tm-chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:500; margin:6px 6px 0 0;
           background:#f2f4f7; color:#344054; border:1px solid #e4e7ec; }
.tm-ok { background:#ecfdf3; color:#067647; border-color:#abefc6; }
.tm-bad { background:#fef3f2; color:#b42318; border-color:#fecdca; }
.tm-blue { background:#eff4ff; color:#1f5fbf; border-color:#c7d7fe; }
.tm-kpis { display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:12px; margin-bottom:14px; }
.tm-kpi { background:#fff; border:1px solid #e4e7ec; border-radius:12px; padding:14px 16px; }
.tm-kpi .l { color:#667085; font-size:12px; font-weight:500; }
.tm-kpi .v { color:#101828; font-size:24px; font-weight:700; margin-top:2px; }
.tm-kpi .d { font-size:12px; margin-top:2px; color:#667085; }
.tm-kpi .d.pos { color:#067647; }
.tm-kpi .d.neg { color:#b42318; }
.tm-card { background:#fff; border:1px solid #e4e7ec; border-radius:12px; padding:16px 18px; margin-bottom:12px; }
.tm-card h4 { margin:0 0 8px; font-size:15px; color:#101828; }
.tm-summary { background:#eff4ff; border:1px solid #c7d7fe; border-radius:12px; padding:14px 18px; color:#1d2939; line-height:1.6; }
.tm-step { display:flex; gap:14px; margin-bottom:10px; }
.tm-num { flex:0 0 30px; height:30px; border-radius:50%; background:#fff; border:2px solid var(--c); color:var(--c);
          display:flex; align-items:center; justify-content:center; font-weight:700; font-size:13px; margin-top:6px; }
.tm-body { flex:1; background:#fff; border:1px solid #e4e7ec; border-left:4px solid var(--c); border-radius:10px; padding:12px 16px; }
.tm-body .t { font-weight:600; color:#101828; font-size:14.5px; }
.tm-cat { display:inline-block; font-size:11px; font-weight:600; color:var(--c); background:color-mix(in srgb, var(--c) 10%, white);
          border-radius:6px; padding:1px 7px; margin-right:6px; text-transform:uppercase; letter-spacing:.04em; }
.tm-row { display:grid; grid-template-columns: 88px 1fr; gap:8px; margin-top:6px; font-size:13.5px; color:#344054; }
.tm-row .k { color:#667085; font-size:11.5px; font-weight:600; text-transform:uppercase; letter-spacing:.04em; padding-top:2px; }
.tm-note { color:#b54708; font-size:12.5px; margin-top:6px; }
.tm-muted { color:#667085; font-size:12.5px; }
.tm-hero { background:#fff; border:1px solid #e4e7ec; border-radius:16px; padding:34px 36px; }
.tm-hero h1 { font-size:30px; margin:0 0 6px; }
.tm-steps3 { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:14px; margin-top:22px; }
.tm-steps3 div { background:#f9fafb; border:1px solid #eaecf0; border-radius:12px; padding:14px 16px; font-size:13.5px; color:#344054; }
.tm-steps3 b { display:block; color:#101828; margin-bottom:4px; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background:#fff; border:1px solid #e4e7ec; border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:6px 14px; height:auto; }
.stTabs [aria-selected="true"] { background:#eff4ff; color:#1f5fbf; }
div[data-testid="stExpander"] { background:#fff; border-radius:10px; }
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def _full_width(fn, *args, **kwargs):
    """Newer Streamlit: width='stretch'; older: use_container_width / use_column_width."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            return fn(*args, **extra, **kwargs)
        except TypeError:
            continue
    return None


def esc(x) -> str:
    return html.escape(str(x))


def load_run(run_dir: Path) -> dict:
    j = lambda name: json.loads((run_dir / name).read_text()) if (run_dir / name).exists() else {}  # noqa: E731
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()] \
        if (run_dir / "steps.jsonl").exists() else []
    exps = [json.loads(p.read_text()) for p in sorted((run_dir / "experiments").glob("*.json"))]
    return {"dir": run_dir, "run": j("run.json"), "summary": j("summary.json"), "findings": j("findings.json"),
            "steps": steps, "experiments": exps, "charts": j("charts.json") or [],
            "explanations": j("explanations.json"), "narratives": j("narratives.json")}


def saved_runs() -> list[Path]:
    if not RUNS.exists():
        return []
    return sorted([p for p in RUNS.iterdir() if (p / "summary.json").exists()], reverse=True)


def plotly_figure(c: dict):
    """Interactive version of a chart, built from the same stored numbers as the PNG. None if not possible."""
    d = c.get("data")
    if not d:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    fig = go.Figure()
    k = d["kind"]
    if k == "lines":
        for sr in d["series"]:
            fig.add_trace(go.Scatter(x=d["x"], y=sr["values"], mode="lines+markers", name=sr["name"],
                                     line=dict(color=sr["color"], width=2.5), marker=dict(size=8)))
        fig.update_yaxes(title=d.get("ylabel"))
    elif k == "hbar":
        fig.add_trace(go.Bar(x=d["values"], y=d["labels"], orientation="h", marker_color=d.get("colors"),
                             error_x=dict(type="data", array=d["errors"]) if d.get("errors") else None,
                             hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title=d.get("xlabel"))
        if d.get("vline") is not None:
            fig.add_vline(x=d["vline"], line_dash="dash", line_color="#52514e")
    elif k == "bars":
        fig.add_trace(go.Bar(x=d["x"], y=d["values"], marker_color="#2a78d6", name="forward-in-time fold"))
        if d.get("hline") is not None:
            fig.add_hline(y=d["hline"], line_dash="dash", line_color="#eb6834", annotation_text="shuffled CV mean")
        fig.update_yaxes(title=d.get("ylabel"))
    elif k == "scatterline":
        if d.get("diagonal"):
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                                     line=dict(color="#b4b2a9", dash="dash")))
        fig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="lines+markers", name="model", text=[f"n={n}" for n in d["n"]],
                                 line=dict(color="#eb6834", width=2.5), marker=dict(size=9),
                                 hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<br>%{text}<extra></extra>"))
        fig.update_xaxes(title=d.get("xlabel"), range=[0, 1]); fig.update_yaxes(title=d.get("ylabel"), range=[0, 1])
    else:
        return None
    n = len(d.get("labels", [])) if k == "hbar" else 0
    fig.update_layout(title=dict(text=c["title"], font=dict(size=14)), template="plotly_white",
                      height=max(300, 34 * n + 140), margin=dict(l=10, r=10, t=46, b=10),
                      legend=dict(orientation="h", y=-0.2), showlegend=k in ("lines", "scatterline"),
                      font=dict(family="Inter, sans-serif"))
    return fig


_CHART_N = [0]


def chart(R: dict, cid: str, caption: bool = True) -> None:
    c = next((c for c in R["charts"] if c["id"] == cid), None)
    if not c:
        return
    fig = plotly_figure(c) if st.session_state.get("interactive", True) else None
    if fig is not None:
        _CHART_N[0] += 1
        _full_width(st.plotly_chart, fig, key=f"pl::{cid}::{_CHART_N[0]}")
    elif Path(c["path"]).exists():
        _full_width(st.image, c["path"])
    else:
        return
    if caption:
        st.caption(f"**{c['title']}** – {c['interpretation']}")


def evidence_zip(run_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(run_dir.parent))
    return buf.getvalue()


def apply_demo():
    d = DEMOS.get(st.session_state.get("demo"))
    if not d:
        return
    st.session_state["src"] = "Benchmark dataset"
    st.session_state["bench"] = d["file"]
    st.session_state["target"] = d["target"]
    st.session_state["objective"] = d["objective"]
    st.session_state["use_cost"] = d["use_cost"]
    if d["use_cost"]:
        st.session_state["cost_fn"], st.session_state["cost_fp"] = d["cost_fn"], d["cost_fp"]


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown(f'<div class="tm-brand">{logo_img(58)}<div class="tm-name">Trust<span>ML</span></div></div>'
                    '<div class="tm-tag">An agent that investigates tabular data and verifies every number it reports</div>',
                    unsafe_allow_html=True)
mode = st.sidebar.radio("Mode", ["New analysis", "Open saved run", "Compare runs"], key="mode", horizontal=True,
                        label_visibility="collapsed")

if mode == "New analysis":
    st.sidebar.markdown('<div class="tm-side-h">Quick start</div>', unsafe_allow_html=True)
    st.sidebar.selectbox("Demo prompt", ["(none)"] + list(DEMOS), key="demo", on_change=apply_demo,
                         label_visibility="collapsed")
    st.sidebar.markdown('<div class="tm-side-h">1 · Data</div>', unsafe_allow_html=True)
    src = st.sidebar.radio("Source", ["Upload CSV/Excel", "Benchmark dataset"], key="src", horizontal=True,
                           label_visibility="collapsed")
    data_path = None
    if src == "Upload CSV/Excel":
        up = st.sidebar.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        if up is not None:
            dest = RUNS / "uploads" / up.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(up.getvalue())
            data_path = dest
    else:
        files = sorted(BENCH.glob("*.csv")) if BENCH.exists() else []
        if files:
            pick = st.sidebar.selectbox("Benchmark file", [f.name for f in files], key="bench")
            data_path = BENCH / pick
        else:
            st.sidebar.warning("No benchmark files. Run: python -m benchmarks.make_churn")

    if data_path is not None:
        try:
            from trustml.io import load_dataset
            cols = list(load_dataset(data_path)[0].columns)
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Could not read the file: {e}")
            cols = []
        if cols:
            if st.session_state.get("target") not in cols:
                guess = next((c for c in cols if c.lower() in ("churned", "churn", "sale_price", "target", "label", "y")),
                             cols[-1])
                st.session_state["target"] = guess
            target = st.sidebar.selectbox("Target column", cols, key="target")
            st.sidebar.markdown('<div class="tm-side-h">2 · Objective</div>', unsafe_allow_html=True)
            if "objective" not in st.session_state:
                st.session_state["objective"] = "Build a reliable model and investigate its weaknesses."
            objective = st.sidebar.text_area("Objective / prompt", key="objective", height=120,
                                             label_visibility="collapsed")
            from trustml.llm import planner_model, try_client
            has_key = try_client() is not None
            st.sidebar.markdown('<div class="tm-side-h">3 · Planner – who decides the next step</div>',
                                unsafe_allow_html=True)
            planner_kind = st.sidebar.radio(
                "Planner", ["llm (NaviGator)", "rule"] if has_key else ["rule", "llm (NaviGator)"], key="planner",
                horizontal=True, label_visibility="collapsed",
                format_func=lambda x: "🧠 LLM (NaviGator)" if x.startswith("llm") else "⚙️ Rule-based")
            if planner_kind.startswith("llm"):
                llm_model = st.sidebar.text_input("NaviGator model", value=planner_model(), key="llm_model")
                st.sidebar.markdown(
                    '<div class="tm-plan"><span class="tm-dot" style="background:'
                    + ("#12b76a" if has_key else "#f04438") + '"></span><b>'
                    + ("NaviGator key found" if has_key else "No NaviGator key – will fall back to rules")
                    + '</b><br>The LLM reads the audit and CV results (never test scores) and picks the next tool. '
                      'Every choice is schema-validated and checked against the run state; an invalid choice falls back '
                      'to the rule planner for that step, and the fallback is logged.</div>', unsafe_allow_html=True)
            else:
                llm_model = planner_model()
                st.sidebar.markdown('<div class="tm-plan"><span class="tm-dot" style="background:#1f5fbf"></span>'
                                    '<b>Deterministic policy</b><br>Same data → same steps. Fast and reproducible; '
                                    'best for rehearsed demos and as the LLM planner\'s safety net.</div>',
                                    unsafe_allow_html=True)
            llm_text = st.sidebar.checkbox("NaviGator writes the summary wording and data-collection ideas",
                                           value=has_key, disabled=not has_key, key="llm_text",
                                           help="Needs NAVIGATOR_TOOLKIT_API_KEY in .env. Numbers are never written "
                                                "by the LLM; its text is checked against verified results.")
            with st.sidebar.expander("4 · Budget and options"):
                max_steps = st.slider("Max agent steps", 2, 30, 22, key="max_steps")
                max_seconds = st.slider("Time limit (s)", 60, 900, 600, step=30, key="max_seconds")
                from trustml.agent.loop import infer_recall_target
                rt = infer_recall_target(objective or "")
                auto_recall = st.checkbox("Read the recall target from the objective"
                                          + (f" (found {rt[0]:.0%})" if rt else " (none found → 0.6)"),
                                          value=True, key="auto_recall")
                min_recall = st.slider("Target minority recall (binary)", 0.3, 0.95, 0.6, 0.05, key="min_recall",
                                       disabled=auto_recall)
                use_cost = st.checkbox("Misclassification costs (binary)", key="use_cost")
                cost_fn = st.number_input("Cost of a missed positive", 1.0, 100.0, 5.0, key="cost_fn")
                cost_fp = st.number_input("Cost of a false alarm", 0.1, 100.0, 1.0, key="cost_fp")
                group_col = st.selectbox("Group split column", ["(none)"] + cols, key="group_col")
                trusted = st.multiselect("Known at prediction time (never excluded as leakage)",
                                         [c for c in cols if c != target], key="trusted")
                from trustml.tools.diagnostics import detect_sensitive
                auto_fair = detect_sensitive([c for c in cols if c != target])
                fair_cols = st.multiselect("Fairness columns (checked for error-rate gaps)",
                                           [c for c in cols if c != target], key="fair_cols",
                                           help="Detected automatically: " + (", ".join(auto_fair) or "none") +
                                                ". Add others here (e.g. region).")
                cv_repeats = st.select_slider("Cross-validation repeats", [1, 2, 3], value=1, key="cv_repeats",
                                              help="Repeat 5-fold CV with different shuffles for steadier comparisons "
                                                   "(slower). Not used with a group split.")
            if _full_width(st.sidebar.button, "Run investigation", type="primary", key="run_btn"):
                from trustml.agent.loop import Budget, run_agent
                from trustml.agent.planner import LLMPlanner, RulePlanner
                from trustml.report import finalize_run
                planner = RulePlanner()
                if planner_kind.startswith("llm"):
                    try:
                        planner = LLMPlanner(model=llm_model)
                    except Exception as e:  # noqa: BLE001
                        st.warning(f"LLM planner unavailable ({e}); using the rule planner.")
                live = st.status("Agent running…", expanded=True)

                def on_step(s):
                    a = s.action
                    cat = ACTION_STYLE.get(a["action"], ("Step", "#475569"))[0]
                    live.write(f"**{s.step}. {a['action'].replace('_', ' ')}** · {cat} · {s.planner}  \n"
                               f"{(s.result or {}).get('interpretation') or a.get('reason') or ''}")

                try:
                    run = run_agent(str(data_path), target, objective, planner=planner, on_step=on_step,
                                    run_root=str(RUNS), trusted_features=trusted, fairness_cols=fair_cols,
                                    cv_repeats=cv_repeats,
                                    budget=Budget(max_steps=max_steps, max_seconds=max_seconds,
                                                  **({} if auto_recall else {"min_recall": min_recall})),
                                    costs={"fn": cost_fn, "fp": cost_fp} if use_cost else None,
                                    group_col=None if group_col == "(none)" else group_col)
                    live.write("Building explanations, charts and report…")
                    text_client = getattr(planner, "client", None) or (try_client() if llm_text else None)
                    finalize_run(run, llm_client=text_client if (llm_text or planner_kind.startswith("llm")) else None,
                                 llm_model=getattr(planner, "model", None) or llm_model)
                    live.update(label=f"Done – run {run.run_id}", state="complete", expanded=False)
                    st.session_state["run_dir"] = str(run.run_dir)
                except Exception as e:  # noqa: BLE001
                    live.update(label="Run failed", state="error")
                    st.error(f"{type(e).__name__}: {e}")
else:
    runs = saved_runs()
    labels = []
    for p in runs:
        r = json.loads((p / "run.json").read_text())
        labels.append(f"{Path(r['data_path']).name} · {r['target']} · {r.get('planner', '?')} · {p.name}")
    if not runs:
        st.sidebar.info("No saved runs yet.")
    elif mode == "Open saved run":
        i = st.sidebar.selectbox("Saved run", range(len(runs)), format_func=lambda k: labels[k], key="saved")
        st.session_state["run_dir"] = str(runs[i])
    else:
        st.sidebar.markdown('<div class="tm-side-h">Compare two runs</div>', unsafe_allow_html=True)
        ia = st.sidebar.selectbox("Run A", range(len(runs)), format_func=lambda k: labels[k], key="cmp_a")
        ib = st.sidebar.selectbox("Run B", range(len(runs)), format_func=lambda k: labels[k], key="cmp_b",
                                  index=min(1, len(runs) - 1))
        st.session_state["compare"] = (str(runs[ia]), str(runs[ib]))
_ = (st.sidebar.toggle if hasattr(st.sidebar, "toggle") else st.sidebar.checkbox)(
    "Interactive charts", value=True, key="interactive")

# ---------------------------------------------------------------- landing page
# ---------------------------------------------------------------- compare runs
if mode == "Compare runs" and st.session_state.get("compare"):
    from trustml.compare_runs import compare_runs
    a, b = st.session_state["compare"]
    C = compare_runs(Path(a), Path(b))
    st.markdown(f'<div class="tm-header"><div class="tm-heroband">{logo_img(44)}<div><div class="tm-title">'
                f'Compare runs</div><div class="tm-sub">Same dataset: <b>{"yes" if C["same_data"] else "no"}</b> · '
                f'same test split: <b>{"yes" if C["same_split"] else "no"}</b>'
                f'{"" if C["same_split"] else " – test scores are not directly comparable"}</div></div></div></div>',
                unsafe_allow_html=True)
    _full_width(st.dataframe, pd.DataFrame(C["rows"]), hide_index=True)
    c1, c2 = st.columns(2, gap="large")
    for col, side in ((c1, "a"), (c2, "b")):
        with col:
            st.markdown(f"#### Run {side.upper()} · {C[side]['label']}")
            st.markdown("**Agent path:** " + " → ".join(C[side]["path"]))
            for x in C[side]["conclusions"]:
                st.markdown(f"- {x}")
    if C["only_a"] or C["only_b"]:
        st.markdown("#### Actions taken by only one run")
        st.write({"only in A": C["only_a"], "only in B": C["only_b"]})
    st.stop()

if "run_dir" not in st.session_state or not Path(st.session_state["run_dir"], "summary.json").exists():
    st.markdown(f"""<div class="tm-hero"><div class="tm-heroband">{logo_img(92)}<div class="tm-name">Trust<span>ML</span></div></div>
<div class="tm-sub" style="font-size:15px;margin-top:14px">An AI agent that audits a tabular dataset, decides what to investigate,
runs bounded experiments, and verifies every number it reports.</div>
<div class="tm-steps3">
<div><b>1 · Audit</b>Profiles the data and checks quality, duplicates, placeholders, imbalance and possible leakage
– on the training split only.</div>
<div><b>2 · Investigate</b>Plans the next experiment from what it observed: leak ablation, class weighting, thresholds,
feature removal or engineering, interactions, model choice, subgroup, fairness, calibration and time checks.</div>
<div><b>3 · Verify &amp; explain</b>Every metric is recomputed with independent code; SHAP explanations, a report
and a conversational assistant answer only from verified results.</div>
</div><div class="tm-muted" style="margin-top:18px">Start in the sidebar: pick a demo prompt, or upload a CSV/Excel file,
choose the target column, describe your objective and pick a planner.</div></div>""", unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------- run header
R = load_run(Path(st.session_state["run_dir"]))
S, meta, F = R["summary"], R["run"], R["findings"]
prof = F["profile"]
verified = S.get("all_claims_verified") and all(s.get("verification", {}).get("passed", True) for s in R["steps"])
n_checks = sum(sum(x["n_checks"] for x in s["verification"].get("experiments", {}).values())
               + len(s["verification"].get("analysis_checks", [])) for s in R["steps"] if s.get("verification"))
chips = [f'<span class="tm-chip">{esc(Path(meta["data_path"]).name)}</span>',
         f'<span class="tm-chip">target: {esc(meta["target"])}</span>',
         f'<span class="tm-chip">{esc(prof["target"]["task_type"].replace("_", " "))}</span>',
         f'<span class="tm-chip">planner: {esc(meta["planner"])}</span>',
         f'<span class="tm-chip">{S["steps"]} steps · {S["elapsed_s"]} s</span>',
         (f'<span class="tm-chip tm-ok">✓ all {len(S["claim_checks"])} reported numbers verified</span>' if verified
          else '<span class="tm-chip tm-bad">verification failures – see Report</span>')]
st.markdown(f'<div class="tm-header"><div class="tm-heroband">{logo_img(48)}<div style="flex:1">'
            f'<div class="tm-title">Analysis report</div>'
            f'<div class="tm-sub">{esc(meta["objective"])}</div></div></div>{"".join(chips)}'
            f'<div class="tm-muted" style="margin-top:8px">Run {esc(meta["run_id"])}</div></div>',
            unsafe_allow_html=True)

all_find = [f for g in ("quality", "missingness_train_only", "leakage_train_only")
            for f in F.get(g, {}).get("findings", []) if f["check"] != "single_feature_ranking"] + F.get("split_checks", [])
n_hi = sum(f["severity"] == "high" for f in all_find)
n_med = sum(f["severity"] == "medium" for f in all_find)
fm = S.get("final_model") or {}
tm, bm = fm.get("test_metrics", {}), S.get("baseline_test_metrics") or {}
key = "recall" if "recall" in tm else ("r2" if "r2" in tm else "balanced_accuracy")
delta = (tm.get(key, 0) - bm.get(key, 0)) if bm and key in bm else None
leak_txt = "leak removed" if any(v.get("verdict") == "likely_leakage" for v in S.get("verdicts", [])) else ""
if delta is None:
    d_txt, d_cls = "", ""
elif leak_txt and delta < 0:
    d_txt, d_cls = f"{delta:+.3f} vs leaky baseline – an honest estimate", ""
else:
    d_txt, d_cls = f"{delta:+.3f} vs original baseline", ("pos" if delta > 0 else "neg" if delta < 0 else "")
kpis = [("Rows", f"{prof['handle']['n_rows']:,}", f"{prof['handle']['n_cols']} columns", ""),
        ("Data issues", f"{n_hi + n_med}", f"{n_hi} high · {n_med} medium", "neg" if n_hi else ""),
        ("Experiments", f"{len(R['experiments'])}", f"{S['steps']} agent steps", ""),
        (f"Final test {key.replace('_', ' ')}", f"{tm.get(key, float('nan')):.3f}" if tm else "–", d_txt, d_cls),
        ("Independent checks", f"{n_checks}", "all passed" if verified else "failures", "pos" if verified else "neg")]
st.markdown('<div class="tm-kpis">' + "".join(
    f'<div class="tm-kpi"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div>'
    f'<div class="d {cls}">{esc(d)}</div></div>' for l, v, d, cls in kpis) + "</div>",
    unsafe_allow_html=True)

TAB_NAMES = ["Overview", "Data quality", "Agent investigation", "Planner", "Models", "Trust checks", "Explainability",
             "Ask TrustML", "Report"]
tabs = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

# ---------------------------------------------------------------- overview
with tabs["Overview"]:
    narr = R["narratives"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        if narr:
            st.markdown(f'<div class="tm-summary">{esc(narr["summary"])}</div>', unsafe_allow_html=True)
            st.caption(f"Summary source: {narr['summary_meta']['source']}"
                       + (" – every number checked against verified results"
                          if narr["summary_meta"].get("numbers_checked") else ""))
        for i in S.get("out_of_scope", []):
            st.warning(f"Not supported: “{i['matched']}” – {i['reason']}")
        st.markdown("#### Conclusions")
        for x in S["conclusions"]:
            st.markdown(f"- {x}")
    with right:
        st.markdown('<div class="tm-card"><h4>Caveats</h4>' + "".join(
            f'<div style="font-size:13.5px;color:#344054;margin:6px 0">• {esc(c)}</div>' for c in S["caveats"])
            + "</div>", unsafe_allow_html=True)
        chart(R, "target")

# ---------------------------------------------------------------- data quality
with tabs["Data quality"]:
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    all_find.sort(key=lambda f: order[f["severity"]])
    counts = {s: sum(f["severity"] == s for f in all_find) for s in order}
    st.markdown("".join(f'<span class="tm-chip {"tm-bad" if s == "high" else "tm-blue" if s == "info" else ""}">'
                        f'{s}: {n}</span>' for s, n in counts.items()), unsafe_allow_html=True)
    sev = st.multiselect("Show severities", list(order), default=["high", "medium", "low"], key="sev_filter")
    _full_width(st.dataframe, pd.DataFrame([{"severity": f["severity"], "check": f["check"],
                                             "columns": ", ".join(f["columns"]), "finding": f["message"],
                                             "recommendation": f["recommendation"]}
                                            for f in all_find if f["severity"] in sev]), hide_index=True)
    with st.expander("Evidence – raw tool output"):
        st.json([{"check": f["check"], "columns": f["columns"], "evidence": f["evidence"]} for f in all_find])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart(R, "missing")
    with c2:
        chart(R, "leak_screen")
    with st.expander("Column profile"):
        _full_width(st.dataframe, pd.DataFrame([{"column": c["name"], "kind": c["inferred_kind"],
                                                 "missing %": c["pct_missing"], "unique": c["n_unique"],
                                                 "sample": str(c["sample_values"][:3])} for c in prof["columns"]]),
                    hide_index=True)

# ---------------------------------------------------------------- investigation timeline
with tabs["Agent investigation"]:
    st.caption("Each step: what the agent observed → what it decided and why → what the tool returned → "
               "independent verification. Expand a step for its evidence.")
    for s in R["steps"]:
        a = s["action"]
        cat, col = ACTION_STYLE.get(a["action"], ("Step", "#475569"))
        v = s.get("verification") or {}
        n = sum(x["n_checks"] for x in v.get("experiments", {}).values()) + len(v.get("analysis_checks", []))
        badge = ((f'<span class="tm-chip tm-ok">✓ verified · {n} checks</span>' if v.get("passed")
                  else '<span class="tm-chip tm-bad">verification failed</span>') if v else "")
        args = {k: val for k, val in a.items() if k not in ("action", "rationale", "reason")}
        res = (s.get("result") or {}).get("interpretation", "")
        st.markdown(f"""<div class="tm-step" style="--c:{col}"><div class="tm-num">{s['step']}</div>
<div class="tm-body"><div><span class="tm-cat">{esc(cat)}</span><span class="t">{esc(a['action'].replace('_', ' '))}</span>
<span class="tm-chip">{esc(s['planner'])}</span>{badge}</div>
<div class="tm-row"><div class="k">Observed</div><div>{esc(s['observation'])}</div></div>
<div class="tm-row"><div class="k">Decided</div><div>{esc(a.get('rationale') or a.get('reason') or '')}
{f'<code style="font-size:11.5px">{esc(json.dumps(args))}</code>' if args else ''}</div></div>
{f'<div class="tm-row"><div class="k">Result</div><div>{esc(res)}</div></div>' if res else ''}
{f'<div class="tm-note">Planner note: {esc(s["planner_note"])}</div>' if s.get('planner_note') else ''}
{f'<div class="tm-muted" style="margin-top:4px">{esc(v["note"])}</div>' if v.get('note') else ''}
</div></div>""", unsafe_allow_html=True)
        new = (s.get("result") or {}).get("new_experiments", [])
        step_charts = [f"cmp_{e}" for e in new] + {
            "rank_feature_importance": ["importance"], "screen_interactions": ["interactions"],
            "check_calibration": ["calibration"], "check_time_split": ["time_split"],
            "check_fairness": [f"fairness_{a.get('column')}"]}.get(a["action"], [])
        if s.get("result"):
            with st.expander(f"Evidence for step {s['step']}"):
                for cid in step_charts:
                    chart(R, cid)
                checks = (s.get("verification") or {}).get("analysis_checks", [])
                if checks:
                    st.markdown("**Independent checks**")
                    _full_width(st.dataframe, pd.DataFrame([{"check": c["name"], "passed": c["passed"],
                                                             "detail": c.get("detail", "")} for c in checks]),
                                hide_index=True)
                st.json(s["result"], expanded=False)

# ---------------------------------------------------------------- planner
with tabs["Planner"]:
    steps = R["steps"]
    by = {}
    for s_ in steps:
        by[s_["planner"]] = by.get(s_["planner"], 0) + 1
    llm_n = sum(n for k, n in by.items() if k.startswith("llm"))
    fb_n = sum(n for k, n in by.items() if "fallback" in k)
    kind = meta.get("planner", "?")
    st.markdown(
        '<div class="tm-kpis" style="grid-template-columns:repeat(4,minmax(0,1fr))">' + "".join(
            f'<div class="tm-kpi"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div><div class="d">{esc(d)}</div></div>'
            for l, v, d in [("Planner for this run", "LLM" if kind.startswith("llm") else "Rule",
                             kind), ("Decisions", str(len(steps)), "including finish"),
                            ("Chosen by the LLM", str(llm_n), "validated before execution"),
                            ("Rule fallbacks", str(fb_n), "invalid / failed LLM choices")]) + "</div>",
        unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("#### Decision log")
        _full_width(st.dataframe, pd.DataFrame([{
            "step": s_["step"], "decided by": s_["planner"], "action": s_["action"]["action"],
            "arguments": json.dumps({k: v for k, v in s_["action"].items() if k not in ("action", "rationale", "reason")}),
            "why": s_["action"].get("rationale") or s_["action"].get("reason") or "",
            "note": s_.get("planner_note") or ""} for s_ in steps]), hide_index=True)
    with c2:
        st.markdown("""<div class="tm-card"><h4>How planning works</h4>
<div class="tm-row"><div class="k">Sees</div><div>Audit findings, CV-only experiment results, verdicts so far, the
objective and remaining budget. Never test-set metrics.</div></div>
<div class="tm-row"><div class="k">Chooses</div><div>One of 19 typed tools (e.g. ablate_feature, screen_interactions,
check_fairness) with arguments.</div></div>
<div class="tm-row"><div class="k">Checked</div><div>Pydantic schema + state rules (e.g. no baseline before
deduplication, only screened interactions, no repeats). A rejected LLM choice is replaced by the rule policy for that
step and logged here.</div></div>
<div class="tm-row"><div class="k">Bounded</div><div>Step and time budgets; the agent stops when no triggered
investigation remains.</div></div></div>""", unsafe_allow_html=True)
        if by:
            st.markdown("#### Who decided")
            st.bar_chart(pd.Series(by, name="decisions"))

# ---------------------------------------------------------------- models
with tabs["Models"]:
    exps = R["experiments"]
    if exps:
        task = exps[0]["task_type"]
        prim = {"binary_classification": "roc_auc", "regression": "r2"}.get(task, "balanced_accuracy")
        k2 = "recall" if task == "binary_classification" else prim
        final_id = fm.get("exp_id")
        by_id = {e["exp_id"]: e for e in exps}
        rows = []
        for e in exps:
            sp = e["spec"]
            par = by_id.get(sp.get("parent_id") or "", {}).get("spec", {})
            changes = []
            new_ex = [x for x in sp["exclude_features"] if x not in par.get("exclude_features", [])]
            new_add = [x for x in sp.get("extra_features", []) if x not in par.get("extra_features", [])]
            if new_ex:
                changes.append("− " + ", ".join(new_ex))
            if new_add:
                changes.append("+ " + ", ".join(new_add))
            if sp.get("sentinel_to_missing") and sp.get("sentinel_to_missing") != par.get("sentinel_to_missing"):
                changes.append("placeholders → missing")
            if sp.get("target_transform") and sp.get("target_transform") != par.get("target_transform"):
                changes.append("log target")
            if sp.get("class_weight") and sp.get("class_weight") != par.get("class_weight"):
                changes.append("balanced class weights")
            if sp.get("threshold_strategy", "default") != par.get("threshold_strategy", "default") and \
                    sp.get("threshold_strategy", "default") != "default":
                changes.append(f"threshold {sp['threshold_strategy']} = {e['threshold']:.2f}")
            if par and sp["model"] != par.get("model"):
                changes.append(f"model → {sp['model']}")
            if not par:
                changes.append("reference" if sp["model"] == "dummy" else "all candidate features")
            rows.append({"": "★ final" if e["exp_id"] == final_id else "", "experiment": e["exp_id"],
                         "model": sp["model"], "built from": sp.get("parent_id") or "–",
                         "change vs parent": "; ".join(changes),
                         f"CV {prim}": f"{e['cv_metrics'][prim]['mean']:.4f} ± {e['cv_metrics'][prim]['std']:.4f}",
                         **({"CV recall": round(e["cv_metrics"]["recall"]["mean"], 3)} if task == "binary_classification" else {}),
                         f"test {k2}": round(e["test_metrics"][k2], 4),
                         "95% CI": str(e["test_ci95"].get(k2, ""))})
        _full_width(st.dataframe, pd.DataFrame(rows), hide_index=True)
        st.caption("Agent decisions used cross-validation only; test scores are shown for transparency.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart(R, "final")
    with c2:
        chart(R, "subgroups")
        chart(R, "time_split")

# ---------------------------------------------------------------- trust checks
with tabs["Trust checks"]:
    an_dir = R["dir"] / "analyses"
    rd = lambda n: json.loads((an_dir / n).read_text()) if (an_dir / n).exists() else None  # noqa: E731
    it, cal = rd("interactions.json"), rd("calibration.json")
    fair = {p.stem[len("fairness_"):]: json.loads(p.read_text()) for p in sorted(an_dir.glob("fairness_*.json"))} \
        if an_dir.exists() else {}
    vmap = {v["topic"]: v for v in S.get("verdicts", [])}
    st.markdown("#### Group-specific effects (interactions)")
    if it:
        iv = vmap.get("interaction")
        if iv:
            ok = iv["verdict"] == "interaction_confirmed"
            st.markdown(f'<span class="tm-chip {"tm-ok" if ok else ""}">{esc(iv["verdict"])}</span> '
                        f'<b>{esc(iv["feature"])}</b> – {esc(iv["group"])} {esc(iv["group_metric"])} '
                        f'{iv["group_before"]:.3f} → {iv["group_after"]:.3f} (out-of-fold)', unsafe_allow_html=True)
        elif not it.get("candidates"):
            st.caption(f"No group-specific effect: {it['n_tested']} feature × group pairs screened.")
        chart(R, "interactions")
    else:
        st.caption("The interaction screen was not run.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Fairness")
        if not fair:
            st.caption("No fairness column was detected or chosen. Pick one under '4 · Budget and options'.")
        for col, fr in fair.items():
            badge = "tm-bad" if fr.get("material") else "tm-ok"
            st.markdown(f'<span class="tm-chip {badge}">{"material gap" if fr.get("material") else "no material gap"}'
                        f'</span> <b>{esc(col)}</b>', unsafe_allow_html=True)
            _full_width(st.dataframe, pd.DataFrame(fr["groups"]).drop(columns=["interval"], errors="ignore"),
                        hide_index=True)
            chart(R, f"fairness_{col}")
    with c2:
        st.markdown("#### Calibration")
        if cal:
            st.markdown(f'<span class="tm-chip {"tm-ok" if cal["well_calibrated"] else "tm-bad"}">'
                        f'{"well calibrated" if cal["well_calibrated"] else "needs recalibration"}</span> '
                        f'Brier {cal["brier"]:.4f} · ECE {cal["ece"]:.3f}', unsafe_allow_html=True)
            chart(R, "calibration")
        else:
            st.caption("Calibration applies to binary classification and was not run here.")

# ---------------------------------------------------------------- explainability
with tabs["Explainability"]:
    ex = R["explanations"]
    if not ex:
        st.info("No explanations in this run.")
    cols_ = st.columns(2, gap="large")
    for (k, e), col in zip([(k, e) for k, e in ex.items() if e.get("supported")], cols_ * 2):
        with col:
            st.markdown(f"#### {'Final model' if k == 'final' else 'Original baseline (for comparison)'}")
            chart(R, f"shap_{k}")
            st.caption(f"Additivity check: max error {e['additivity_max_error']:.2e} – "
                       f"{'passed' if e['additivity_ok'] else 'FAILED'}. {e['note']}")
    fin = ex.get("final", {})
    for loc in fin.get("local", []) if fin.get("supported") else []:
        with st.expander(f"Why this prediction? {loc['case']} (row {loc['row_index']})"):
            st.write(f"True {loc['y_true']:g}, predicted {loc['y_pred']:g}; base value {loc['base_value']:.4g} → "
                     f"model output {loc['model_output']:.4g} ({loc['output_space']}).")
            _full_width(st.dataframe, pd.DataFrame(loc["contributions"]).astype({"value": str}), hide_index=True)
    narr = R["narratives"]
    st.markdown("#### New data worth collecting")
    st.caption("Suggestions only – not tested. Generated from column names; no data values were sent to the LLM.")
    if narr and narr.get("collection"):
        _full_width(st.dataframe, pd.DataFrame(narr["collection"]), hide_index=True)
    else:
        st.info((narr or {}).get("collection_meta", {}).get("error", "No suggestions for this run."))

# ---------------------------------------------------------------- ask TrustML
with tabs["Ask TrustML"]:
    from trustml.llm import planner_model, try_client
    conv_key = f"conv::{R['dir']}"
    conv = st.session_state.get(conv_key)
    if conv is None:
        from trustml.chat import Conversation
        client = try_client()
        conv = Conversation(R["dir"], llm_client=client, llm_model=planner_model() if client else None)
        st.session_state[conv_key] = conv
    live = conv.client is not None
    st.markdown(
        f'<div class="tm-card"><div class="tm-heroband" style="gap:12px">{logo_img(34)}<div><h4 style="margin:0">'
        f'Chat with TrustML</h4><div class="tm-muted">'
        + (f"Conversational mode via NaviGator <b>{esc(conv.model)}</b>: it remembers the conversation, picks TrustML "
           "tools (stored verified results, or a new verified experiment on this run's exact split) and explains the "
           "results. Every number in a reply is checked against tool output; an unsupported reply is replaced."
           if live else "No NaviGator key found: answers come from TrustML's deterministic router (add "
                        "NAVIGATOR_TOOLKIT_API_KEY to .env for conversational replies).")
        + "</div></div></div></div>", unsafe_allow_html=True)
    feats = (fm.get("features_used") or [])[:2]
    examples = ["Why does the model miss positives?" if "recall" in tm else "Where is the model least accurate?",
                f"What if we remove {feats[0]}?" if feats else "Which features matter most?",
                "Can I trust its probabilities?" if "recall" in tm else "Does the model hold up over time?",
                "Is the model fair across groups?"]
    ex_cols = st.columns(len(examples))
    clicked = None
    for c, q in zip(ex_cols, examples):
        if _full_width(c.button, q, key=f"ex::{q}"):
            clicked = q
    if conv.history and st.button("Clear conversation", key="clear_chat"):
        conv.history.clear()
        st.rerun() if hasattr(st, "rerun") else None
    box = st.container()  # messages above, input box below (like a chat app)
    q = st.chat_input("Message TrustML about this analysis…") or clicked

    def show_turn(t):
        st.markdown(t.content)
        v = t.verification or {}
        bits = []
        if v.get("n_new_experiments"):
            bits.append("✓ new experiments verified" if v.get("experiments_verified") else "✗ verification failed")
        elif t.tools:
            bits.append("from stored, verified results")
        st.caption(" · ".join(bits + [t.source, f"{t.seconds} s"]))
        if t.tools:
            with st.expander(f"Tool trace ({len(t.tools)} tool call{'s' if len(t.tools) > 1 else ''})"):
                for x in t.tools:
                    vv = x.get("verification") or {}
                    st.markdown(f'<div class="tm-trace"><b>{esc(x["tool"])}</b>({esc(x["question"])})<br>'
                                f'{esc(x["text"])}<br><span class="tm-muted">{esc(x["source"])}'
                                + (f' · new experiments {esc(", ".join(x["new_experiments"]))}' if x["new_experiments"] else "")
                                + (f' · {sum(vv["checks"].values())} checks {"passed" if vv.get("passed") else "FAILED"}'
                                   if vv.get("checks") else "") + "</span></div>", unsafe_allow_html=True)

    turns = conv.history
    avatar = str(LOGO_PNG) if LOGO_PNG.exists() else None
    for i in range(0, len(turns) - 1, 2):
        with box.chat_message("user"):
            st.markdown(turns[i].content)
        with box.chat_message("assistant", avatar=avatar):
            show_turn(turns[i + 1])
    if q:
        with box.chat_message("user"):
            st.markdown(q)
        with box.chat_message("assistant", avatar=avatar):
            with st.spinner("Thinking – new experiments take a few seconds…"):
                try:
                    t = conv.send(q)
                    show_turn(t)
                except Exception as e:  # noqa: BLE001
                    st.error(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- report
with tabs["Report"]:
    rp = R["dir"] / "report.html"
    c1, c2, _ = st.columns([1, 1, 3])
    if rp.exists():
        report_html = rp.read_text(encoding="utf-8")
        c1.download_button("Download report (HTML)", report_html, file_name=f"trustml_report_{meta['run_id']}.html",
                           mime="text/html", key="dl")
        c2.download_button("Download evidence pack (.zip)", evidence_zip(R["dir"]),
                           file_name=f"trustml_evidence_{meta['run_id']}.zip", mime="application/zip", key="dlzip",
                           help="Everything behind this run: inputs, findings, every experiment and prediction file, "
                                "agent steps, verification results, charts and the report.")
        if hasattr(st, "iframe"):  # report text is HTML-escaped by the Jinja2 template (autoescape)
            st.iframe(report_html, height=900)
        else:
            components.html(report_html, height=900, scrolling=True)
    else:
        st.info("No report in this run folder.")
