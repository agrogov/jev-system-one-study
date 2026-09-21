#!/usr/bin/env python3
"""Three-way comparison: recorded Jev runs vs the Laya and SemIf local replays.

All quality metrics are computed on the *same* decisions: a (case_id, question_id) pair is used only
if Jev completed it and both Laya and SemIf consumed the complete request. Coverage, compatibility
outcomes and latency are reported separately over all requests, so unsupported cases are never hidden.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_jev_laya as cl  # noqa: E402

SYSTEMS = ("jev", "laya", "semif")
LABEL = {"jev": "Jev", "laya": "Laya", "semif": "SemIf"}
KEYS = ["case_id", "question_id"]
ARCH_RUNS = {"core": cl.SOURCE_RUNS[0], "iso": cl.SOURCE_RUNS[1], "full": cl.SOURCE_RUNS[2]}
SUITES = (
    ("architecture core", cl.SOURCE_RUNS[0]),
    ("isolated core scaling", cl.SOURCE_RUNS[1]),
    ("full boundary scaling", cl.SOURCE_RUNS[2]),
    ("probability calibration", cl.SOURCE_RUNS[3]),
    ("semantic", cl.SOURCE_RUNS[4]),
)
VALUE_COLS = ("correct", "prob_abs_error", "distribution_tv", "brier", "confidence_top", "meta_reference_max_probability",
              "meta_question_count", "meta_cardinality", "meta_context_chars", "meta_domain", "experiment", "type")


def run_dirs(study: Path, source: str, laya_slug: str, semif_slug: str) -> dict[str, Path]:
    return {
        "jev": study / "jev-results" / source,
        "laya": study / "laya-results" / f"{source}__laya_{laya_slug}",
        "semif": study / "semif-results" / f"{source}__semif_{semif_slug}",
    }


def comparable_ids(system: str, run_dir: Path) -> set[str] | None:
    if system == "jev":
        return None
    return {str(r["case_id"]) for r in cl.read_jsonl(run_dir / "raw.jsonl")
            if r.get("case_id") is not None and bool(r.get("comparable", r.get("ok") and r.get("compatibility_status") in {None, "full_input"}))}


def load_side(system: str, run_dir: Path) -> pd.DataFrame:
    df = cl.load_norm(run_dir)
    if df.empty:
        return df
    df = cl.filter_comparable(df, comparable_ids(system, run_dir))
    return df[[c for c in df.columns if c in KEYS or c in VALUE_COLS]].copy()


def aligned(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join the three systems on (case_id, question_id); value columns get a system suffix."""
    out = None
    for sys_name in SYSTEMS:
        f = frames[sys_name]
        if f.empty:
            return pd.DataFrame()
        renamed = f.rename(columns={c: f"{c}__{sys_name}" for c in f.columns if c not in KEYS})
        out = renamed if out is None else out.merge(renamed, on=KEYS, how="inner")
    return out


def joint(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows where ``col`` is defined for all three systems (so every system is scored on identical rows)."""
    cols = [f"{col}__{s}" for s in SYSTEMS]
    if df.empty or not all(c in df.columns for c in cols):
        return pd.DataFrame()
    sub = df.copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub.dropna(subset=cols)


def m(df: pd.DataFrame, col: str, system: str) -> float | None:
    sub = joint(df, col)
    return cl.mean_col(sub, f"{col}__{system}") if len(sub) else None


def n_joint(df: pd.DataFrame, col: str) -> int:
    return len(joint(df, col))


def raw_summary(system: str, run_dir: Path) -> dict[str, Any]:
    rows = cl.read_jsonl(run_dir / "raw.jsonl")
    ok = [r for r in rows if r.get("ok")]
    comparable = [r for r in rows if system == "jev" or bool(r.get("comparable", r.get("ok") and r.get("compatibility_status") in {None, "full_input"}))]
    if system == "jev":
        comparable = ok
    wall = [float(r["latency_ms"]) for r in ok if r.get("latency_ms") is not None]
    statuses: dict[str, int] = {}
    for r in rows:
        s = str(r.get("compatibility_status") or ("full_input" if r.get("ok") else "backend_error"))
        statuses[s] = statuses.get(s, 0) + 1
    return {"total": len(rows), "ok": len(ok), "comparable": len(comparable), "wall_p50_ms": cl.quantile(wall, 0.5), "statuses": statuses}


def server_ms(row: dict[str, Any]) -> float | None:
    h = row.get("response_headers") or {}
    for name in ("x-envoy-upstream-service-time", "x-laya-inference-time-ms", "x-semif-inference-time-ms"):
        if h.get(name) is not None:
            return float(h[name])
    return None


def pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"


def num(x: float | None, digits: int = 4) -> str:
    return cl.fmt(x, digits)


def table(header: list[str], rows: list[list[str]]) -> str:
    aligns = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(aligns) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def scaling_table(study: Path, laya_slug: str, semif_slug: str, experiment: str, key: str, label: str) -> tuple[str, list[dict[str, Any]]]:
    """Per-x server time and success by system, pooled over the three architecture runs."""
    records: list[dict[str, Any]] = []
    for source in ARCH_RUNS.values():
        for system, d in run_dirs(study, source, laya_slug, semif_slug).items():
            for r in cl.read_jsonl(d / "raw.jsonl"):
                if r.get("experiment") != experiment:
                    continue
                x = (r.get("metadata") or {}).get(key)
                if x is None:
                    continue
                cmp_ok = system == "jev" and bool(r.get("ok")) or bool(r.get("comparable", False))
                records.append({"system": system, "x": x, "ms": server_ms(r), "ok": bool(r.get("ok")), "comparable": cmp_ok,
                                "status": r.get("compatibility_status") or ("full_input" if r.get("ok") else "backend_error")})
    df = pd.DataFrame(records)
    if df.empty:
        return "", records
    rows = []
    for x, g in df.groupby("x"):
        cells = [str(int(x)) if float(x).is_integer() else str(x)]
        for system in SYSTEMS:
            s = g[g.system == system]
            if s.empty:
                cells.append("—")
                continue
            ms = s[s.ok]["ms"].dropna()
            outcome = "ok" if s.comparable.all() else ("/".join(sorted(set(s[~s.comparable].status))) if not s.ok.all() or True else "ok")
            cells.append(f"{ms.median():.0f} ms" + ("" if s.comparable.all() else f" ({outcome})") if len(ms) else f"({outcome})")
        rows.append(cells)
    return table([label] + [f"{LABEL[s]} server time" for s in SYSTEMS], rows), records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--laya-slug", default="typed-decisions_mlx")
    p.add_argument("--semif-slug", default="qwen3-5-4b_mlx")
    p.add_argument("--output-dir", type=Path, help="Default: <study-root>/comparisons/three_way")
    args = p.parse_args()
    study = args.study_root.expanduser().resolve()
    out = (args.output_dir or study / "comparisons" / "three_way").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    cov_rows, qual_rows, prob_rows, sem_rows, latency_rows = [], [], [], [], []
    summary: list[dict[str, Any]] = []
    sem_common = pd.DataFrame()
    for suite, source in SUITES:
        dirs = run_dirs(study, source, args.laya_slug, args.semif_slug)
        for s, d in dirs.items():
            if not d.is_dir():
                raise FileNotFoundError(f"Missing {s} run: {d}")
            cl.ensure_normalized(d, study)
        raws = {s: raw_summary(s, d) for s, d in dirs.items()}
        common = aligned({s: load_side(s, d) for s, d in dirs.items()})
        cov_rows.append([suite] + [f"{raws[s]['comparable']}/{raws[s]['total']}" for s in SYSTEMS])
        latency_rows.append([suite] + [num(raws[s]["wall_p50_ms"], 1) for s in SYSTEMS])
        acc = {s: m(common, "correct", s) for s in SYSTEMS}
        n_acc = n_joint(common, "correct")
        qual_rows.append([suite, str(n_acc)] + [pct(acc[s]) for s in SYSTEMS])
        rec: dict[str, Any] = {"suite": suite, "aligned_decisions": n_acc, **{f"accuracy_{s}": acc[s] for s in SYSTEMS},
                               **{f"coverage_{s}": raws[s]["comparable"] / raws[s]["total"] for s in SYSTEMS},
                               **{f"statuses_{s}": raws[s]["statuses"] for s in SYSTEMS},
                               **{f"wall_p50_ms_{s}": raws[s]["wall_p50_ms"] for s in SYSTEMS}}
        if "calibration" in suite:
            prob_rows.append([suite, str(n_joint(common, "prob_abs_error")), "Binary exact-probability MAE"] + [num(m(common, "prob_abs_error", s)) for s in SYSTEMS])
            prob_rows.append([suite, str(n_joint(common, "distribution_tv")), "Mean TV to exact 3-class posterior"] + [num(m(common, "distribution_tv", s)) for s in SYSTEMS])
            for s in SYSTEMS:
                rec[f"prob_mae_{s}"] = m(common, "prob_abs_error", s)
                rec[f"tv_{s}"] = m(common, "distribution_tv", s)
        if suite == "semantic":
            sem_common = common
            sem_rows.append(["Accuracy"] + [pct(acc[s]) for s in SYSTEMS])
            sem_rows.append(["Mean TV distance to reference"] + [num(m(common, "distribution_tv", s)) for s in SYSTEMS])
            sem_rows.append(["Mean top probability (reference ≈ %s)" % num(m(common, "meta_reference_max_probability", "jev"), 3)]
                            + [num(m(common, "confidence_top", s), 3) for s in SYSTEMS])
            for s in SYSTEMS:
                rec[f"tv_{s}"] = m(common, "distribution_tv", s)
                rec[f"mean_top_prob_{s}"] = m(common, "confidence_top", s)
        summary.append(rec)

    domain_rows = []
    if not sem_common.empty:
        for dom, g in sem_common.groupby("meta_domain__jev"):
            domain_rows.append([str(dom), str(len(g))] + [pct(m(g, "correct", s)) for s in SYSTEMS])

    q_table, q_recs = scaling_table(study, args.laya_slug, args.semif_slug, "parallel_question_scaling", "question_count", "Questions per request")
    c_table, _ = scaling_table(study, args.laya_slug, args.semif_slug, "choice_cardinality_scaling", "cardinality", "Choices")
    x_table, _ = scaling_table(study, args.laya_slug, args.semif_slug, "context_length_position", "context_chars", "Context chars")

    head = ["Suite"] + [LABEL[s] for s in SYSTEMS]
    sections = [
        "# Jev vs Laya vs SemIf: Three-Way Comparison\n",
        "All three systems replay the same recorded `cases.jsonl` inputs. **Quality metrics use only decisions that Jev completed and that both Laya and "
        "SemIf consumed in full**, so the systems are scored on identical questions. Coverage and latency use every request. Jev is a hosted service; "
        "Laya (`aac6fef/laya-typed-decisions-mlx`, FP16) and SemIf (`Qwen/Qwen3.5-4B`, BF16, native MLX) run locally on Apple Silicon, so absolute latency is "
        "descriptive, not hardware-normalized. SemIf's `noul` and `score` results depend on this study's yes/no and ordinal option mapping (see the run manifests).\n",
        "## 1. Accuracy on identical decisions\n",
        table(["Suite", "Scored decisions"] + [LABEL[s] for s in SYSTEMS], qual_rows),
        "## 2. Probability quality (exact-probability suite)\n",
        table(["Suite", "Scored decisions", "Metric"] + [LABEL[s] for s in SYSTEMS], prob_rows),
        "## 3. Semantic suite\n",
        table(["Metric"] + [LABEL[s] for s in SYSTEMS], sem_rows),
        "Accuracy by domain:\n",
        table(["Domain", "n"] + [LABEL[s] for s in SYSTEMS], domain_rows),
        "## 4. Coverage: requests each system consumed in full (comparable / total)\n",
        table(head, cov_rows),
        "Laya's shortfall is context truncation and unsupported cardinality; SemIf's is unsupported cardinality only (more than 16 options). "
        "Jev completed every request.\n",
        "## 5. Scaling (server-side time; outcome shown when a request was not fully consumed)\n",
        "### Questions per request\n", q_table,
        "### Choice cardinality\n", c_table,
        "### Context length\n", x_table,
        "## 6. Median wall latency per request (ms)\n",
        table(head, latency_rows),
        "## Interpretation constraints\n",
        "- SemIf is an independent open-model reproduction of the typed-decision interface pattern, not of Jev's model.\n"
        "- Laya is a specialist typed-decisions checkpoint; SemIf uses a general 4B model with a prompted readout. Neither is a claim about Jev's architecture.\n"
        "- Latency: Jev is a remote service; Laya and SemIf are local. Scaling shape is more informative than absolute values.\n"
        "- Semantic references are authored adjudication distributions, not independent human-panel measurements.\n",
    ]
    (out / "THREE_WAY_REPORT.md").write_text("\n".join(sections), encoding="utf-8")
    (out / "three_way_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("statuses_")} for r in summary]).to_csv(out / "three_way_summary.csv", index=False)
    print(out / "THREE_WAY_REPORT.md")


if __name__ == "__main__":
    main()
