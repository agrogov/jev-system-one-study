#!/usr/bin/env python3
"""Recompute the integrated SemIf statistics used by the SemIf and comparison reports.

Writes ``report/semif_computed_statistics.json``. Paired bootstrap confidence intervals resample whole
cases (95%, 2,000 resamples, seed 260919), so multi-question cases are not treated as independent.
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
import compare_three_way as tw  # noqa: E402

SEED = 260919
RESAMPLES = 2000


def paired_bootstrap(df: pd.DataFrame, a: str, b: str) -> dict[str, float | int | None]:
    """Mean of (a - b) with a case-clustered percentile bootstrap CI. ``a``/``b`` are column names."""
    d = df[["case_id", a, b]].dropna()
    if d.empty:
        return {"n": 0, "diff": None, "ci_low": None, "ci_high": None}
    d = d.assign(diff=d[a].astype(float) - d[b].astype(float))
    g = d.groupby("case_id")["diff"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(g), size=(RESAMPLES, len(g)))
    sums, counts = g[idx, 0].sum(axis=1), g[idx, 1].sum(axis=1)
    boots = sums / counts
    return {"n": int(len(d)), "diff": float(d["diff"].mean()),
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5))}


def pair_common(jev: pd.DataFrame, other: pd.DataFrame, suffix: str) -> pd.DataFrame:
    keep = lambda df: df[[c for c in df.columns if c in tw.KEYS or c in tw.VALUE_COLS]]
    return keep(jev).merge(keep(other), on=tw.KEYS, how="inner", suffixes=("__jev", f"__{suffix}"))


def clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--laya-slug", default="typed-decisions_mlx")
    p.add_argument("--semif-slug", default="qwen3-5-4b_mlx")
    p.add_argument("--output", type=Path, help="Default: <study-root>/report/semif_computed_statistics.json")
    args = p.parse_args()
    study = args.study_root.expanduser().resolve()
    out_path = (args.output or study / "report" / "semif_computed_statistics.json")

    result: dict[str, Any] = {"seed": SEED, "resamples": RESAMPLES, "two_way": {}, "three_way": {}}
    for suite, source in tw.SUITES:
        dirs = tw.run_dirs(study, source, args.laya_slug, args.semif_slug)
        frames = {s: tw.load_side(s, d) for s, d in dirs.items()}
        raws = {s: tw.raw_summary(s, d) for s, d in dirs.items()}
        two = pair_common(frames["jev"], frames["semif"], "semif")
        three = tw.aligned(frames)
        rec2: dict[str, Any] = {
            "aligned_rows": int(len(two)),
            "coverage": {s: {"comparable": raws[s]["comparable"], "total": raws[s]["total"], "statuses": raws[s]["statuses"],
                             "wall_p50_ms": raws[s]["wall_p50_ms"]} for s in tw.SYSTEMS},
            "accuracy": paired_bootstrap(two, "correct__jev", "correct__semif"),
            "accuracy_jev": cl.mean_col(two.dropna(subset=["correct__jev", "correct__semif"]), "correct__jev"),
            "accuracy_semif": cl.mean_col(two.dropna(subset=["correct__jev", "correct__semif"]), "correct__semif"),
        }
        rec3: dict[str, Any] = {"aligned_rows": int(len(three))}
        if not three.empty:
            for name, (x, y) in {"jev_minus_laya": ("jev", "laya"), "jev_minus_semif": ("jev", "semif"), "semif_minus_laya": ("semif", "laya")}.items():
                j = tw.joint(three, "correct")
                rec3.setdefault("accuracy_diff", {})[name] = paired_bootstrap(j, f"correct__{x}", f"correct__{y}")
            rec3["accuracy"] = {s: tw.m(three, "correct", s) for s in tw.SYSTEMS}
            rec3["accuracy_n"] = tw.n_joint(three, "correct")

        if "calibration" in suite:
            mae = tw.joint(three, "prob_abs_error")
            rec2["binary_mae"] = {s: cl.mean_col(two.dropna(subset=["prob_abs_error__jev", "prob_abs_error__semif"]), f"prob_abs_error__{s}") for s in ("jev", "semif")}
            rec2["binary_mae_excess_semif_minus_jev"] = paired_bootstrap(two, "prob_abs_error__semif", "prob_abs_error__jev")
            rec3["binary_mae"] = {s: tw.m(three, "prob_abs_error", s) for s in tw.SYSTEMS}
            rec3["binary_mae_n"] = int(len(mae))
            rec3["binary_mae_diff"] = {n: paired_bootstrap(mae, f"prob_abs_error__{x}", f"prob_abs_error__{y}")
                                       for n, (x, y) in {"laya_minus_jev": ("laya", "jev"), "semif_minus_jev": ("semif", "jev"), "laya_minus_semif": ("laya", "semif")}.items()}
            tv = tw.joint(three, "distribution_tv")
            rec3["multiclass_tv"] = {s: tw.m(three, "distribution_tv", s) for s in tw.SYSTEMS}
            rec3["multiclass_tv_n"] = int(len(tv))
            rec3["multiclass_tv_diff"] = {n: paired_bootstrap(tv, f"distribution_tv__{x}", f"distribution_tv__{y}")
                                          for n, (x, y) in {"semif_minus_jev": ("semif", "jev"), "laya_minus_jev": ("laya", "jev")}.items()}
            fam = {}
            for exp, g in three.groupby("experiment__jev"):
                fam[str(exp)] = {"n": int(len(g)), **{s: cl.mean_col(g, f"prob_abs_error__{s}") for s in tw.SYSTEMS}}
            rec3["family_mae"] = fam
        if suite == "semantic":
            tv = tw.joint(three, "distribution_tv")
            rec2["tv"] = {s: cl.mean_col(tv, f"distribution_tv__{s}") for s in ("jev", "semif")}
            rec2["tv_excess_semif_minus_jev"] = paired_bootstrap(tv, "distribution_tv__semif", "distribution_tv__jev")
            rec3["tv_diff"] = {n: paired_bootstrap(tv, f"distribution_tv__{x}", f"distribution_tv__{y}")
                               for n, (x, y) in {"laya_minus_jev": ("laya", "jev"), "semif_minus_jev": ("semif", "jev"), "laya_minus_semif": ("laya", "semif")}.items()}
            rec3["tv"] = {s: cl.mean_col(tv, f"distribution_tv__{s}") for s in tw.SYSTEMS}
            rec3["mean_top_probability"] = {s: cl.mean_col(three, f"confidence_top__{s}") for s in tw.SYSTEMS}
            rec3["reference_mean_top_probability"] = cl.mean_col(three, "meta_reference_max_probability__jev")
            dom = {}
            for d, g in three.groupby("meta_domain__jev"):
                dom[str(d)] = {"n": int(len(g)), **{s: cl.mean_col(g, f"correct__{s}") for s in tw.SYSTEMS}}
            rec3["domains"] = dom
            half = {}
            for s in tw.SYSTEMS:
                srt = three.sort_values(f"confidence_top__{s}", ascending=False)
                k = len(srt) // 2
                half[s] = {"top_half_accuracy": cl.mean_col(srt.iloc[:k], f"correct__{s}"),
                           "bottom_half_accuracy": cl.mean_col(srt.iloc[k:], f"correct__{s}")}
            rec3["confidence_ranking"] = half
        result["two_way"][suite] = rec2
        result["three_way"][suite] = rec3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clean(result), indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
