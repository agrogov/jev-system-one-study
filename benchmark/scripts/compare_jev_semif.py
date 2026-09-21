#!/usr/bin/env python3
"""Compare the five recorded Jev runs with the matching SemIf replay runs.

Sibling of ``compare_jev_laya.py`` (same metrics, same aligned-decision logic); it reuses that
script's generic helpers and only supplies the SemIf naming, headers and compatibility rules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_jev_laya as cl  # noqa: E402

FAMILY = "semif"
SIDE = "semif"
METRIC_COLS = (
    "correct", "brier", "nll", "mae", "p_true", "p_label", "target_probability", "prob_abs_error",
    "prob_error", "prob_sq_error", "distribution_tv", "distribution_jsd", "target_kl", "soft_brier",
    "confidence_top", "reported_confidence", "target_distribution_json", "pred_distribution_json",
)
KEYS = ["case_id", "question_id"]
SCALING_EXPERIMENTS = {"parallel_question_scaling", "choice_cardinality_scaling", "context_length_position"}


def is_comparable(row: dict[str, Any]) -> bool:
    return bool(row.get("comparable", row.get("ok") and row.get("compatibility_status") in {None, "full_input"}))


def server_ms(row: dict[str, Any]) -> float | None:
    h = row.get("response_headers") or {}
    for name in ("x-envoy-upstream-service-time", f"x-{FAMILY}-inference-time-ms"):
        if h.get(name) is not None:
            try:
                return float(h[name])
            except (TypeError, ValueError):
                return None
    return None


def raw_stats(run_dir: Path, provider: str) -> dict[str, Any]:
    rows = cl.read_jsonl(run_dir / "raw.jsonl")
    ok = [r for r in rows if r.get("ok")]
    comparable = [r for r in rows if is_comparable(r)] if provider == FAMILY else ok
    wall = [float(r["latency_ms"]) for r in ok if r.get("latency_ms") is not None]
    upstream = [v for v in (server_ms(r) for r in ok) if v is not None]
    failures: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("compatibility_status") or ("full_input" if row.get("ok") else "backend_error"))
        statuses[status] = statuses.get(status, 0) + 1
        if not row.get("ok"):
            key = str(row.get("error_type") or "UnknownError")
            failures[key] = failures.get(key, 0) + 1
    n = len(rows)
    return {
        "total": n, "ok": len(ok), "failed": n - len(ok),
        "coverage": len(ok) / n if n else None,
        "comparable": len(comparable), "comparable_coverage": len(comparable) / n if n else None,
        "compatibility_status": dict(sorted(statuses.items())),
        "wall_p50_ms": cl.quantile(wall, 0.50), "wall_p95_ms": cl.quantile(wall, 0.95),
        "upstream_p50_ms": cl.quantile(upstream, 0.50), "upstream_p95_ms": cl.quantile(upstream, 0.95),
        "failure_types": failures,
    }


def comparable_case_ids(run_dir: Path) -> set[str]:
    return {str(r["case_id"]) for r in cl.read_jsonl(run_dir / "raw.jsonl") if r.get("case_id") is not None and is_comparable(r)}


def common_rows(jev: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    if jev.empty or other.empty:
        return pd.DataFrame()
    keep = lambda df: df[[c for c in df.columns if c in KEYS or c in {"experiment", "type", "label", *METRIC_COLS}]]
    return keep(jev).merge(keep(other), on=KEYS, how="inner", suffixes=("_jev", f"_{SIDE}"))


def metrics_for_pair(common: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"common_questions": len(common)}
    for side in ("jev", SIDE):
        out[f"accuracy_{side}"] = cl.mean_col(common, f"correct_{side}")
        out[f"brier_{side}"] = cl.mean_col(common, f"brier_{side}")
        out[f"nll_{side}"] = cl.mean_col(common, f"nll_{side}")
        out[f"score_mae_{side}"] = cl.mean_col(common, f"mae_{side}")
        out[f"prob_mae_{side}"] = cl.mean_col(common, f"prob_abs_error_{side}")
        out[f"prob_rmse_{side}"] = cl.rmse_col(common, f"prob_sq_error_{side}")
        out[f"tv_{side}"] = cl.mean_col(common, f"distribution_tv_{side}")
        out[f"jsd_{side}"] = cl.mean_col(common, f"distribution_jsd_{side}")
    return out


def scaling_rows(run_dir: Path, provider: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in cl.read_jsonl(run_dir / "raw.jsonl"):
        if row.get("experiment") not in SCALING_EXPERIMENTS:
            continue
        m = row.get("metadata") or {}
        out.append({
            "provider": provider, "experiment": row.get("experiment"), "ok": bool(row.get("ok")),
            "compatibility_status": row.get("compatibility_status"), "comparable": row.get("comparable"),
            "question_count": m.get("question_count"), "cardinality": m.get("cardinality"),
            "context_chars": m.get("context_chars"), "evidence_position": m.get("evidence_position"),
            "wall_ms": row.get("latency_ms"), "server_ms": server_ms(row), "error_type": row.get("error_type"),
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--semif-root", type=Path, help="Default: <study-root>/semif-results")
    p.add_argument("--checkpoint", default="qwen3.5-4b")
    p.add_argument("--backend", default="mlx")
    p.add_argument("--output-dir", type=Path, help="Default: <study-root>/comparisons/semif_<checkpoint>_<backend>")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    study = args.study_root.expanduser().resolve()
    semif_root = (args.semif_root or (study / "semif-results")).expanduser().resolve()
    tag = f"{cl.slug(args.checkpoint)}_{cl.slug(args.backend)}"
    output = (args.output_dir or (study / "comparisons" / f"semif_{tag}")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scaling: list[dict[str, Any]] = []
    sections = [f"# Jev vs SemIf ({args.checkpoint}, {args.backend})\n"]
    sections.append(
        "The comparison replays byte-identical `cases.jsonl` inputs from the five recorded Jev runs through "
        "SemIf's native MLX option readout (Qwen3.5-4B). SemIf coverage failures (more than 16 options, prompts "
        "over the input limit) are reported rather than silently dropped or repaired. Jev `noul` questions are "
        "replayed as a fixed yes/no decision with `noul = P(yes)`; that option wording is authored by this study, "
        "not by SemIf. Absolute latency is not hardware-equivalent: Jev is a remote service, SemIf runs locally.\n"
    )

    for source_name in cl.SOURCE_RUNS:
        jev_dir = study / "jev-results" / source_name
        semif_dir = semif_root / f"{source_name}__{FAMILY}_{tag}"
        if not jev_dir.is_dir():
            raise FileNotFoundError(jev_dir)
        if not semif_dir.is_dir():
            raise FileNotFoundError(f"Missing SemIf run: {semif_dir}")
        cl.ensure_normalized(semif_dir, study)
        jev_stats = raw_stats(jev_dir, "jev")
        sem_stats = raw_stats(semif_dir, FAMILY)
        jev_norm = cl.load_norm(jev_dir)
        sem_norm = cl.filter_comparable(cl.load_norm(semif_dir), comparable_case_ids(semif_dir))
        metrics = metrics_for_pair(common_rows(jev_norm, sem_norm))
        sharp_j = cl.semantic_sharpness(jev_norm) if "semantic" in source_name else {}
        sharp_s = cl.semantic_sharpness(sem_norm) if "semantic" in source_name else {}
        row = {
            "source_run": source_name,
            **{f"jev_{k}": v for k, v in jev_stats.items()},
            **{f"{SIDE}_{k}": v for k, v in sem_stats.items()},
            **metrics,
        }
        if sharp_j:
            row.update({f"semantic_jev_{k}": v for k, v in sharp_j.items()})
            row.update({f"semantic_{SIDE}_{k}": v for k, v in sharp_s.items()})
        rows.append(row)
        scaling.extend(scaling_rows(jev_dir, "jev"))
        scaling.extend(scaling_rows(semif_dir, FAMILY))

        f = cl.fmt
        sections.append(f"## {source_name}\n")
        sections.append(
            f"- SemIf completed requests: **{sem_stats['ok']}/{sem_stats['total']}** ({f((sem_stats['coverage'] or 0)*100,2)}%).\n"
            f"- Fully comparable/full-input requests: **{sem_stats['comparable']}/{sem_stats['total']}** ({f((sem_stats['comparable_coverage'] or 0)*100,2)}%).\n"
            f"- Compatibility status: `{json.dumps(sem_stats['compatibility_status'], sort_keys=True)}`.\n"
            f"- Common normalized decisions after compatibility filtering: **{metrics['common_questions']}**.\n"
            f"- Accuracy — Jev: **{f(metrics.get('accuracy_jev'))}**; SemIf: **{f(metrics.get(f'accuracy_{SIDE}'))}**.\n"
            f"- Binary exact-probability MAE — Jev: **{f(metrics.get('prob_mae_jev'))}**; SemIf: **{f(metrics.get(f'prob_mae_{SIDE}'))}**.\n"
            f"- Full-distribution TV — Jev: **{f(metrics.get('tv_jev'))}**; SemIf: **{f(metrics.get(f'tv_{SIDE}'))}**.\n"
            f"- Local/remote wall p50 — Jev: **{f(jev_stats['wall_p50_ms'],1)} ms**; SemIf: **{f(sem_stats['wall_p50_ms'],1)} ms**.\n"
        )
        if sem_stats["failure_types"]:
            sections.append("Failure types: `" + json.dumps(sem_stats["failure_types"], sort_keys=True) + "`.\n")
        if "semantic" in source_name:
            sections.append(f"Semantic top-probability gap versus reference — Jev: **{f(sharp_j.get('gap'))}**; SemIf: **{f(sharp_s.get('gap'))}**.\n")

    pd.DataFrame(rows).to_csv(output / "comparison_summary.csv", index=False)
    pd.DataFrame(scaling).to_csv(output / "scaling_rows.csv", index=False)
    sections.append("## Interpretation constraints\n")
    sections.append(
        "- SemIf is an open-model reproduction of the interface pattern, not of Jev's model; it is independent and not affiliated with TypeSafe.\n"
        "- SemIf accepts 2-16 options per decision. Larger Jev Choice sets are reported as `unsupported_cardinality`, never shortened.\n"
        "- SemIf never truncates: prompts beyond the configured input limit are `input_too_long`. The limit was raised from SemIf's 4,096-token default so long-context cases are actually attempted.\n"
        "- The `noul` and `score` mappings are this study's adaptation of SemIf's categorical interface (see the run manifests).\n"
        "- Jev upstream service time and SemIf local end-to-end inference time are not measured on the same hardware.\n"
    )
    (output / "JEV_VS_SEMIF_REPORT.md").write_text("\n".join(sections), encoding="utf-8")
    (output / "comparison_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(output / "JEV_VS_SEMIF_REPORT.md")


if __name__ == "__main__":
    main()
