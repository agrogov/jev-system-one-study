from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        if np.isnan(float(x)):
            return "n/a"
    except Exception:
        pass
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _plot_reliability(df: pd.DataFrame, out: Path, primitive: str) -> None:
    d = df[df["type"] == primitive].copy()
    if d.empty or "confidence_top" not in d or "correct" not in d:
        return
    bins = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for i in range(10):
        lo, hi = bins[i], bins[i+1]
        mask = (d["confidence_top"] >= lo) & (d["confidence_top"] <= hi if i == 9 else d["confidence_top"] < hi)
        b = d[mask]
        if len(b):
            xs.append(float(b["confidence_top"].mean()))
            ys.append(float(b["correct"].mean()))
            ns.append(len(b))
    if not xs:
        return
    plt.figure(figsize=(6.4, 5.2))
    plt.plot([0,1], [0,1], linestyle="--", label="perfect calibration")
    plt.plot(xs, ys, marker="o", label=primitive)
    for x,y,n in zip(xs,ys,ns):
        plt.annotate(str(n), (x,y), xytext=(3,3), textcoords="offset points", fontsize=8)
    plt.xlabel("Mean predicted top-class probability")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability diagram: {primitive}")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_exact_probability(df: pd.DataFrame, out: Path) -> None:
    if "target_probability" not in df.columns or "p_true" not in df.columns:
        return
    d = df.dropna(subset=["target_probability", "p_true"]).copy()
    if d.empty:
        return
    # Aggregate identical target probabilities to show systematic bias rather than point overlap.
    g = d.groupby("target_probability", as_index=False).agg(mean_pred=("p_true", "mean"), n=("p_true", "size"))
    plt.figure(figsize=(6.4, 5.2))
    plt.plot([0,1], [0,1], linestyle="--", label="ideal")
    plt.scatter(g["target_probability"], g["mean_pred"], s=np.clip(g["n"]*3, 20, 180), label="Jev")
    plt.xlabel("Analytically correct probability")
    plt.ylabel("Mean Jev probability")
    plt.title("Exact probability recovery")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_event_reliability(df: pd.DataFrame, out: Path) -> None:
    if not {"p_true", "label"}.issubset(df.columns):
        return
    d = df.dropna(subset=["p_true", "label"]).copy()
    d = d[d["type"] == "noul"]
    if d.empty:
        return
    bins = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for i in range(10):
        lo, hi = bins[i], bins[i+1]
        mask = (d["p_true"] >= lo) & (d["p_true"] <= hi if i == 9 else d["p_true"] < hi)
        b = d[mask]
        if len(b):
            xs.append(float(b["p_true"].mean()))
            ys.append(float(b["label"].astype(float).mean()))
            ns.append(len(b))
    if not xs:
        return
    plt.figure(figsize=(6.4, 5.2))
    plt.plot([0,1], [0,1], linestyle="--", label="perfect calibration")
    plt.plot(xs, ys, marker="o", label="empirical outcomes")
    for x,y,n in zip(xs,ys,ns):
        plt.annotate(str(n), (x,y), xytext=(3,3), textcoords="offset points", fontsize=8)
    plt.xlabel("Mean Jev P(true)")
    plt.ylabel("Observed true frequency")
    plt.title("Event-probability calibration")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_probability_error_by_experiment(df: pd.DataFrame, out: Path) -> None:
    if "prob_abs_error" not in df.columns:
        return
    d = df.dropna(subset=["prob_abs_error"]).copy()
    if d.empty:
        return
    g = d.groupby("experiment")["prob_abs_error"].mean().sort_values()
    plt.figure(figsize=(8.0, max(4.2, 0.45*len(g)+1.5)))
    plt.barh(range(len(g)), g.values)
    plt.yticks(range(len(g)), g.index)
    plt.xlabel("Mean absolute probability error")
    plt.title("Probability error by experiment")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_semantic_confidence(df: pd.DataFrame, out: Path) -> None:
    if not {"target_distribution_json", "pred_distribution_json"}.issubset(df.columns):
        return
    d = df[df["experiment"].isin(["semantic_adjudication", "semantic_repeatability"])].dropna(subset=["target_distribution_json", "pred_distribution_json"]).copy()
    if d.empty:
        return
    xs, ys = [], []
    for _, row in d.iterrows():
        target = json.loads(row["target_distribution_json"])
        pred = json.loads(row["pred_distribution_json"])
        xs.append(max(target.values()))
        ys.append(max(pred.values()))
    plt.figure(figsize=(6.4, 5.2))
    plt.plot([0,1], [0,1], linestyle="--", label="matched confidence")
    plt.scatter(xs, ys, s=18, alpha=0.45, label="cases")
    plt.xlabel("Reference modal probability")
    plt.ylabel("Jev top probability")
    plt.title("Semantic ambiguity: reference vs Jev confidence")
    plt.xlim(0,1); plt.ylim(0,1)
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()


def _plot_semantic_distance_by_domain(df: pd.DataFrame, out: Path) -> None:
    if "distribution_tv" not in df.columns or "meta_domain" not in df.columns:
        return
    d = df[df["experiment"] == "semantic_adjudication"].dropna(subset=["distribution_tv"]).copy()
    if d.empty:
        return
    g = d.groupby("meta_domain")["distribution_tv"].mean().sort_values()
    plt.figure(figsize=(8.0, max(4.2, 0.55*len(g)+1.4)))
    plt.barh(range(len(g)), g.values)
    plt.yticks(range(len(g)), g.index)
    plt.xlabel("Mean total-variation distance")
    plt.title("Semantic reference-distribution error by domain")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()


def _plot_risk_coverage(df: pd.DataFrame, out: Path) -> None:
    d = df[df["type"].isin(["noul","choice"])].copy()
    if "correct" not in d.columns:
        return
    d = d.dropna(subset=["confidence_top", "correct"])
    if d.empty:
        return
    d = d.sort_values("confidence_top", ascending=False)
    correct = d["correct"].astype(float).to_numpy()
    coverage = np.arange(1, len(d)+1) / len(d)
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(d)+1)
    plt.figure(figsize=(6.4, 5.2))
    plt.plot(coverage, risk)
    plt.xlabel("Coverage (fraction auto-accepted)")
    plt.ylabel("Risk (error rate among accepted)")
    plt.title("Selective prediction: risk vs coverage")
    plt.xlim(0,1)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_scaling(summary: dict[str, Any], outdir: Path) -> None:
    for exp, xkey, title in [
        ("parallel_question_scaling", "question_count", "Latency vs parallel question count"),
        ("choice_cardinality_scaling", "cardinality", "Latency vs Choice cardinality"),
    ]:
        vals = (summary.get("scaling") or {}).get(exp)
        if not vals:
            continue
        x = [v[xkey] for v in vals]
        y = [v["latency_ms"] for v in vals]
        plt.figure(figsize=(6.4, 5.2))
        plt.plot(x, y, marker="o")
        plt.xlabel(xkey.replace("_", " ").title())
        plt.ylabel("Latency (ms)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outdir / f"{exp}.png", dpi=160)
        plt.close()

    vals = (summary.get("scaling") or {}).get("context_length_position")
    if vals:
        plt.figure(figsize=(6.4, 5.2))
        for pos in ["start", "middle", "end"]:
            subset = sorted([v for v in vals if v["evidence_position"] == pos], key=lambda v: v["context_chars"])
            if subset:
                plt.plot([v["context_chars"] for v in subset], [v["latency_ms"] for v in subset], marker="o", label=pos)
        plt.xlabel("State size (characters)")
        plt.ylabel("Latency (ms)")
        plt.title("Latency vs context length / evidence position")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "context_length_position.png", dpi=160)
        plt.close()


def _risk_at_coverage(df: pd.DataFrame, coverage_target: float) -> dict[str, float] | None:
    d = df[df["type"].isin(["noul","choice"])].dropna(subset=["confidence_top", "correct"]).sort_values("confidence_top", ascending=False)
    if d.empty:
        return None
    n = max(1, int(np.ceil(len(d) * coverage_target)))
    accepted = d.iloc[:n]
    return {
        "coverage": n / len(d),
        "risk": 1.0 - float(accepted["correct"].mean()),
        "threshold": float(accepted["confidence_top"].min()),
        "n": n,
    }


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "normalized.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)

    for primitive in ["noul", "choice"]:
        _plot_reliability(df, figdir / f"reliability_{primitive}.png", primitive)
    _plot_risk_coverage(df, figdir / "risk_coverage.png")
    _plot_scaling(summary, figdir)
    _plot_exact_probability(df, figdir / "exact_probability_recovery.png")
    _plot_event_reliability(df, figdir / "event_probability_reliability.png")
    _plot_probability_error_by_experiment(df, figdir / "probability_error_by_experiment.png")
    _plot_semantic_confidence(df, figdir / "semantic_confidence.png")
    _plot_semantic_distance_by_domain(df, figdir / "semantic_distance_by_domain.png")

    lines = [
        "# Jev Black-Box Benchmark Report",
        "",
        "> Private evaluation report. Verify your contractual permissions before publishing benchmark results.",
        "",
        "## Run health",
        "",
        f"- Requests: **{summary.get('requests_total')}** total, **{summary.get('requests_ok')}** successful, **{summary.get('requests_failed')}** failed.",
        f"- Latency: p50 **{_fmt(summary.get('latency_ms',{}).get('p50'),1)} ms**, p95 **{_fmt(summary.get('latency_ms',{}).get('p95'),1)} ms**, p99 **{_fmt(summary.get('latency_ms',{}).get('p99'),1)} ms**.",
        "",
        "## Calibration and correctness",
        "",
    ]
    for typ, m in (summary.get("calibration") or {}).items():
        parts = [f"n={m.get('n')}"]
        for key in ["accuracy","brier_mean","nll_mean","ece_15","mae_mean"]:
            if key in m:
                parts.append(f"{key}={_fmt(m[key])}")
        lines.append(f"- **{typ}**: " + ", ".join(parts))

    lines += ["", "### By experiment", ""]
    for exp, m in sorted((summary.get("by_experiment") or {}).items()):
        parts = [f"n={m.get('n')}"]
        for key in ["accuracy", "brier_mean", "nll_mean", "mae_mean"]:
            if key in m:
                parts.append(f"{key}={_fmt(m[key])}")
        lines.append(f"- **{exp}**: " + ", ".join(parts))

    lines += ["", "### Selective prediction", ""]
    for cov in [0.50, 0.75, 0.90, 0.95, 1.00]:
        r = _risk_at_coverage(df, cov)
        if r:
            lines.append(f"- At ~{int(cov*100)}% coverage: error/risk **{_fmt(r['risk']*100,2)}%**, threshold **{_fmt(r['threshold'],4)}**, n={r['n']}.")

    prob = summary.get("probabilistic") or {}
    if prob:
        lines += ["", "## Probabilistic calibration", ""]
        b = prob.get("binary_exact_probability")
        if b:
            lines.append(f"- **Binary exact-probability recovery:** n={b.get('n')}, MAE={_fmt(b.get('mae'))}, RMSE={_fmt(b.get('rmse'))}, bias={_fmt(b.get('bias'))}, max error={_fmt(b.get('max_abs_error'))}, Pearson r={_fmt(b.get('pearson_r'))}.")
            if b.get("empirical_outcome_n"):
                lines.append(f"- **Sampled-world empirical calibration:** n={b.get('empirical_outcome_n')}, event ECE(15)={_fmt(b.get('event_ece_15'))}, Brier={_fmt(b.get('outcome_brier'))}, NLL={_fmt(b.get('outcome_nll'))}.")
            lines.append(f"- Returned {b.get('unique_returned_probabilities')} distinct binary probability values in this run.")
            lines += ["", "### Probability bins", "", "| Jev bin | n | mean Jev p | mean exact p | empirical true rate |", "|---|---:|---:|---:|---:|"]
            for row in b.get("bins", []):
                lines.append(f"| {row['lo']:.1f}-{row['hi']:.1f} | {row['n']} | {_fmt(row['mean_pred'])} | {_fmt(row['mean_target'])} | {_fmt(row.get('empirical_rate'))} |")
        m = prob.get("multiclass_exact_distribution")
        if m:
            lines += ["", f"- **Multiclass posterior recovery:** n={m.get('n')}, mean TV={_fmt(m.get('tv_mean'))}, p95 TV={_fmt(m.get('tv_p95'))}, mean JSD={_fmt(m.get('jsd_mean'))}, mean KL(target||Jev)={_fmt(m.get('kl_target_pred_mean'))}, soft Brier={_fmt(m.get('soft_brier_mean'))}, sampled-label accuracy={_fmt(m.get('hard_accuracy'))}."]

    sem = summary.get("semantic") or {}
    if sem:
        lines += ["", "## Semantic adjudication calibration", ""]
        lines.append(f"- Cases: **{sem.get('n')}**; modal accuracy={_fmt(sem.get('modal_accuracy'))}; mean TV={_fmt(sem.get('tv_mean'))}; mean JSD={_fmt(sem.get('jsd_mean'))}; soft Brier={_fmt(sem.get('soft_brier_mean'))}.")
        lines.append(f"- Mean reference top probability={_fmt(sem.get('reference_max_mean'))}; mean Jev top probability={_fmt(sem.get('jev_max_mean'))}; mean overconfidence gap={_fmt(sem.get('overconfidence_gap_mean'))}.")
        lines.append(f"- Mean normalized reference entropy={_fmt(sem.get('reference_entropy_mean'))}; Jev entropy={_fmt(sem.get('jev_entropy_mean'))}; entropy gap={_fmt(sem.get('entropy_gap_mean'))}.")
        if sem.get("by_domain"):
            lines += ["", "### By semantic domain", "", "| Domain | n | modal accuracy | mean TV | mean overconfidence gap | ref entropy | Jev entropy |", "|---|---:|---:|---:|---:|---:|---:|"]
            for dom, v in sorted(sem["by_domain"].items()):
                lines.append(f"| {dom} | {v.get('n')} | {_fmt(v.get('modal_accuracy'))} | {_fmt(v.get('tv_mean'))} | {_fmt(v.get('overconfidence_gap_mean'))} | {_fmt(v.get('reference_entropy_mean'))} | {_fmt(v.get('jev_entropy_mean'))} |")

    meta = summary.get("metamorphic") or {}
    lines += ["", "## Metamorphic / architectural probes", ""]
    coi = meta.get("candidate_order_invariance")
    if coi:
        lines.append(f"- **Candidate-order invariance:** {coi['trials']} trials; mean JSD={_fmt(coi['jsd_mean'],6)}, max JSD={_fmt(coi['jsd_max'],6)}, max absolute probability delta={_fmt(coi['max_abs_probability_delta'],6)}.")
    qpi = meta.get("question_paraphrase_invariance")
    if qpi:
        lines.append(f"- **Question paraphrase invariance:** {qpi['trials']} trials; mean JSD={_fmt(qpi['jsd_mean'],6)}, max JSD={_fmt(qpi['jsd_max'],6)}.")
    qi = meta.get("question_independence")
    if qi:
        worst = max(qi, key=lambda x: x["max_abs_delta"])
        lines.append(f"- **Question independence:** worst max probability delta={_fmt(worst['max_abs_delta'],6)} at bundle size {worst['bundle_size']} (JSD={_fmt(worst['jsd'],6)}).")
    rep = meta.get("repeatability")
    if rep:
        for qid, v in rep.items():
            lines.append(f"- **Repeatability/{qid}:** max probability std={_fmt(v['max_std_across_dimensions'],8)}, max range={_fmt(v['max_range_across_dimensions'],8)} across {v['repetitions']} identical calls.")
    prep = meta.get("prob_repeatability")
    if prep:
        for group, v in prep.items():
            lines.append(f"- **Probabilistic repeatability/{group}:** target={_fmt(v['target_probability'])}, mean={_fmt(v['mean'])}, std={_fmt(v['std'],6)}, range={_fmt(v['range'],6)} across {v['n']} identical calls.")
    pinv = meta.get("prob_representation_invariance")
    if pinv:
        lines.append(f"- **Probability representation invariance:** mean within-group range={_fmt(pinv.get('mean_range'),6)}, max range={_fmt(pinv.get('max_range'),6)} across percent/decimal/frequency renderings.")

    if meta.get("evidence_removal"):
        lines += ["", "### Evidence removal", "", "Expected healthy behavior: decisive-class probability/confidence should generally fall, while `insufficient` mass or entropy rises as decisive evidence is removed.", ""]
        for v in meta["evidence_removal"]:
            probs = v["probabilities"]
            top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:2]
            lines.append(f"- level {v['removal_level']}: top={top}, confidence={_fmt(v.get('reported_confidence'))}, normalized_entropy={_fmt(v.get('entropy_norm'))}")

    lines += ["", "## Figures", ""]
    for fn in ["reliability_noul.png", "reliability_choice.png", "risk_coverage.png", "exact_probability_recovery.png", "event_probability_reliability.png", "probability_error_by_experiment.png", "semantic_confidence.png", "semantic_distance_by_domain.png", "parallel_question_scaling.png", "choice_cardinality_scaling.png", "context_length_position.png"]:
        if (figdir / fn).exists():
            lines.append(f"![{fn}](figures/{fn})")
            lines.append("")

    lines += [
        "## Interpretation guardrails",
        "",
        "- Good top-1 accuracy does **not** establish calibration.",
        "- Good in-distribution calibration does **not** establish epistemic uncertainty under distribution shift.",
        "- Near-zero drift under reordered options/questions supports invariance claims, but does not reveal the hidden architecture uniquely.",
        "- A flat latency curve versus question count supports parallel execution; it does not by itself prove a particular neural architecture.",
        "- Treat this report as a black-box behavioral evaluation, not reverse-engineering of weights or proprietary training data.",
    ]
    report = run_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
