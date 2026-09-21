#!/usr/bin/env python3
"""Compare the five recorded Jev runs with matching local Laya replay runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

SOURCE_RUNS = (
    "run_20260919T201050Z_core",
    "run_20260919T201739Z_core",
    "run_20260919T202139Z_full",
    "run_20260919T203758Z_calibration_full",
    "run_20260919T211437Z_semantic_full",
)


def slug(text: str) -> str:
    out = []
    for ch in text.lower():
        out.append(ch if ch.isalnum() else "-")
    return "-".join(filter(None, "".join(out).split("-")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        if math.isnan(float(value)):
            return "n/a"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def ensure_normalized(run_dir: Path, study_root: Path) -> None:
    if (run_dir / "normalized.csv").exists():
        return
    rows = read_jsonl(run_dir / "raw.jsonl")
    if not any(row.get("ok") for row in rows):
        return
    import sys
    src = study_root / "benchmark" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from jevbench.metrics import analyze
    from jevbench.report import generate_report
    analyze(run_dir)
    generate_report(run_dir)


def raw_stats(run_dir: Path, provider: str) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "raw.jsonl")
    ok = [r for r in rows if r.get("ok")]
    if provider == "laya":
        comparable = [r for r in rows if bool(r.get("comparable", r.get("ok") and r.get("compatibility_status") in {None, "full_input"}))]
    else:
        comparable = ok
    wall = [float(r["latency_ms"]) for r in ok if r.get("latency_ms") is not None]
    upstream: list[float] = []
    for row in ok:
        h = row.get("response_headers") or {}
        value = h.get("x-envoy-upstream-service-time")
        if value is None:
            value = h.get("x-laya-inference-time-ms")
        if value is not None:
            try:
                upstream.append(float(value))
            except (TypeError, ValueError):
                pass
    failures: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("compatibility_status") or ("full_input" if row.get("ok") else "backend_error"))
        statuses[status] = statuses.get(status, 0) + 1
        if not row.get("ok"):
            key = str(row.get("error_type") or "UnknownError")
            failures[key] = failures.get(key, 0) + 1
    return {
        "provider": provider,
        "total": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "coverage": len(ok) / len(rows) if rows else None,
        "comparable": len(comparable),
        "comparable_coverage": len(comparable) / len(rows) if rows else None,
        "compatibility_status": dict(sorted(statuses.items())),
        "wall_p50_ms": quantile(wall, 0.50),
        "wall_p95_ms": quantile(wall, 0.95),
        "upstream_p50_ms": quantile(upstream, 0.50),
        "upstream_p95_ms": quantile(upstream, 0.95),
        "failure_types": failures,
    }


def load_norm(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "normalized.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def comparable_case_ids(run_dir: Path, provider: str) -> set[str] | None:
    if provider != "laya":
        return None
    rows = read_jsonl(run_dir / "raw.jsonl")
    return {
        str(r["case_id"])
        for r in rows
        if r.get("case_id") is not None
        and bool(r.get("comparable", r.get("ok") and r.get("compatibility_status") in {None, "full_input"}))
    }


def filter_comparable(df: pd.DataFrame, case_ids: set[str] | None) -> pd.DataFrame:
    if case_ids is None or df.empty or "case_id" not in df.columns:
        return df
    return df[df["case_id"].astype(str).isin(case_ids)].copy()


def common_rows(jev: pd.DataFrame, laya: pd.DataFrame) -> pd.DataFrame:
    if jev.empty or laya.empty:
        return pd.DataFrame()
    keys = ["case_id", "question_id"]
    keep_j = [c for c in jev.columns if c in keys or c in {"experiment", "type", "label", "correct", "brier", "nll", "mae", "p_true", "p_label", "target_probability", "prob_abs_error", "prob_error", "prob_sq_error", "distribution_tv", "distribution_jsd", "target_kl", "soft_brier", "confidence_top", "reported_confidence", "target_distribution_json", "pred_distribution_json"}]
    keep_l = [c for c in laya.columns if c in keys or c in {"experiment", "type", "label", "correct", "brier", "nll", "mae", "p_true", "p_label", "target_probability", "prob_abs_error", "prob_error", "prob_sq_error", "distribution_tv", "distribution_jsd", "target_kl", "soft_brier", "confidence_top", "reported_confidence", "target_distribution_json", "pred_distribution_json"}]
    return jev[keep_j].merge(laya[keep_l], on=keys, how="inner", suffixes=("_jev", "_laya"))


def mean_col(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(x.mean()) if len(x) else None


def rmse_col(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(np.sqrt(x.mean())) if len(x) else None


def distribution_max(text: str) -> float:
    return max(float(v) for v in json.loads(text).values())


def metrics_for_pair(common: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"common_questions": len(common)}
    for side in ("jev", "laya"):
        out[f"accuracy_{side}"] = mean_col(common, f"correct_{side}")
        out[f"brier_{side}"] = mean_col(common, f"brier_{side}")
        out[f"nll_{side}"] = mean_col(common, f"nll_{side}")
        out[f"score_mae_{side}"] = mean_col(common, f"mae_{side}")
        out[f"prob_mae_{side}"] = mean_col(common, f"prob_abs_error_{side}")
        out[f"prob_rmse_{side}"] = rmse_col(common, f"prob_sq_error_{side}")
        out[f"tv_{side}"] = mean_col(common, f"distribution_tv_{side}")
        out[f"jsd_{side}"] = mean_col(common, f"distribution_jsd_{side}")
    return out


def semantic_sharpness(df: pd.DataFrame) -> dict[str, float | None]:
    needed = {"target_distribution_json", "pred_distribution_json"}
    if df.empty or not needed.issubset(df.columns):
        return {"reference_top": None, "model_top": None, "gap": None}
    d = df.dropna(subset=list(needed)).copy()
    if d.empty:
        return {"reference_top": None, "model_top": None, "gap": None}
    ref = d.target_distribution_json.map(distribution_max)
    pred = d.pred_distribution_json.map(distribution_max)
    return {"reference_top": float(ref.mean()), "model_top": float(pred.mean()), "gap": float((pred-ref).mean())}


def scaling_rows(run_dir: Path, provider: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_jsonl(run_dir / "raw.jsonl"):
        if row.get("experiment") not in {"parallel_question_scaling", "choice_cardinality_scaling", "context_length_position"}:
            continue
        m = row.get("metadata") or {}
        h = row.get("response_headers") or {}
        upstream = h.get("x-envoy-upstream-service-time")
        if upstream is None:
            upstream = h.get("x-laya-inference-time-ms")
        out.append(
            {
                "provider": provider,
                "experiment": row.get("experiment"),
                "ok": bool(row.get("ok")),
                "compatibility_status": row.get("compatibility_status"),
                "comparable": row.get("comparable"),
                "question_count": m.get("question_count"),
                "cardinality": m.get("cardinality"),
                "context_chars": m.get("context_chars"),
                "evidence_position": m.get("evidence_position"),
                "wall_ms": row.get("latency_ms"),
                "server_ms": float(upstream) if upstream is not None else None,
                "error_type": row.get("error_type"),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--laya-root", type=Path, help="Default: <study-root>/laya-results")
    p.add_argument("--checkpoint", default="typed-decisions")
    p.add_argument("--backend", default="mlx")
    p.add_argument("--output-dir", type=Path, help="Default: <study-root>/comparisons/laya_<checkpoint>_<backend>")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    study = args.study_root.expanduser().resolve()
    laya_root = (args.laya_root or (study / "laya-results")).expanduser().resolve()
    output = (args.output_dir or (study / "comparisons" / f"laya_{slug(args.checkpoint)}_{slug(args.backend)}")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scaling: list[dict[str, Any]] = []
    sections: list[str] = []
    sections.append(f"# Jev vs Laya ({args.checkpoint}, {args.backend})\n")
    sections.append("The comparison replays byte-identical `cases.jsonl` inputs from the five recorded Jev runs. Laya coverage failures are reported rather than silently dropping or mutating unsupported cases. Absolute latency is not hardware-equivalent: Jev is a remote service, whereas Laya runs locally; scaling shape and local latency are reported separately.\n")

    for source_name in SOURCE_RUNS:
        jev_dir = study / "results" / source_name
        laya_dir = laya_root / f"{source_name}__laya_{slug(args.checkpoint)}_{slug(args.backend)}"
        if not jev_dir.is_dir():
            raise FileNotFoundError(jev_dir)
        if not laya_dir.is_dir():
            raise FileNotFoundError(f"Missing Laya run: {laya_dir}")
        ensure_normalized(laya_dir, study)
        jev_stats = raw_stats(jev_dir, "jev")
        laya_stats = raw_stats(laya_dir, "laya")
        jev_norm = load_norm(jev_dir)
        laya_norm_all = load_norm(laya_dir)
        laya_ids = comparable_case_ids(laya_dir, "laya")
        laya_norm = filter_comparable(laya_norm_all, laya_ids)
        common = common_rows(jev_norm, laya_norm)
        metrics = metrics_for_pair(common)
        sharp_j = semantic_sharpness(jev_norm) if "semantic" in source_name else {}
        sharp_l = semantic_sharpness(laya_norm) if "semantic" in source_name else {}
        row = {
            "source_run": source_name,
            **{f"jev_{k}": v for k, v in jev_stats.items() if k != "provider"},
            **{f"laya_{k}": v for k, v in laya_stats.items() if k != "provider"},
            **metrics,
        }
        if sharp_j:
            row.update({f"semantic_jev_{k}": v for k, v in sharp_j.items()})
            row.update({f"semantic_laya_{k}": v for k, v in sharp_l.items()})
        rows.append(row)
        scaling.extend(scaling_rows(jev_dir, "jev"))
        scaling.extend(scaling_rows(laya_dir, "laya"))

        sections.append(f"## {source_name}\n")
        sections.append(
            f"- Laya completed requests: **{laya_stats['ok']}/{laya_stats['total']}** ({fmt((laya_stats['coverage'] or 0)*100,2)}%).\n"
            f"- Fully comparable/full-input requests: **{laya_stats['comparable']}/{laya_stats['total']}** ({fmt((laya_stats['comparable_coverage'] or 0)*100,2)}%).\n"
            f"- Compatibility status: `{json.dumps(laya_stats['compatibility_status'], sort_keys=True)}`.\n"
            f"- Common normalized decisions after compatibility filtering: **{metrics['common_questions']}**.\n"
            f"- Accuracy — Jev: **{fmt(metrics.get('accuracy_jev'))}**; Laya: **{fmt(metrics.get('accuracy_laya'))}**.\n"
            f"- Binary exact-probability MAE — Jev: **{fmt(metrics.get('prob_mae_jev'))}**; Laya: **{fmt(metrics.get('prob_mae_laya'))}**.\n"
            f"- Full-distribution TV — Jev: **{fmt(metrics.get('tv_jev'))}**; Laya: **{fmt(metrics.get('tv_laya'))}**.\n"
            f"- Local/remote wall p50 — Jev: **{fmt(jev_stats['wall_p50_ms'],1)} ms**; Laya: **{fmt(laya_stats['wall_p50_ms'],1)} ms**.\n"
        )
        if laya_stats["failure_types"]:
            sections.append("Failure types: `" + json.dumps(laya_stats["failure_types"], sort_keys=True) + "`.\n")
        if "semantic" in source_name:
            sections.append(
                f"Semantic top-probability gap versus reference — Jev: **{fmt(sharp_j.get('gap'))}**; Laya: **{fmt(sharp_l.get('gap'))}**.\n"
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output / "comparison_summary.csv", index=False)
    scaling_df = pd.DataFrame(scaling)
    scaling_df.to_csv(output / "scaling_rows.csv", index=False)

    sections.append("## Interpretation constraints\n")
    sections.append(
        "- The typed-decisions checkpoint is a specialist fine-tuned on four published workflows; results must be labelled as such.\n"
        "- Laya has a much smaller context/head budget than Jev. Truncated-input and unsupported-cardinality requests are reported as compatibility outcomes and excluded from full-input quality comparisons.\n"
        "- Jev upstream service time and Laya local end-to-end inference time are not measured on the same hardware.\n"
        "- Use the base checkpoint as an additional generalization control; do not silently mix routed/base/specialist results.\n"
    )
    (output / "JEV_VS_LAYA_REPORT.md").write_text("\n".join(sections), encoding="utf-8")
    (output / "comparison_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(output / "JEV_VS_LAYA_REPORT.md")


if __name__ == "__main__":
    main()
