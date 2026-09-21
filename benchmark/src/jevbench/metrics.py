from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _answer(row: dict[str, Any], qid: str = "target") -> dict[str, Any] | None:
    resp = row.get("response") or {}
    return (resp.get("answers") or {}).get(qid)


def _usage(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("response") or {}).get("usage") or {}


def _choice_probs(ans: dict[str, Any] | None) -> dict[str, float]:
    if not ans:
        return {}
    return {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}


def _score_probs(ans: dict[str, Any] | None) -> dict[str, float]:
    return _choice_probs(ans)


def _entropy(probs: list[float]) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    if not len(p):
        return float("nan")
    return float(-(p * np.log(p)).sum())


def _norm_entropy(probs: list[float]) -> float:
    if len(probs) <= 1:
        return 0.0
    return _entropy(probs) / math.log(len(probs))


def _jsd(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return float("nan")
    a = np.asarray([p.get(k, 0.0) for k in keys], dtype=float)
    b = np.asarray([q.get(k, 0.0) for k in keys], dtype=float)
    a = a / max(a.sum(), 1e-12)
    b = b / max(b.sum(), 1e-12)
    m = 0.5 * (a + b)
    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / np.clip(y[mask], 1e-12, None))))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    total = len(conf)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi if i == bins - 1 else conf < hi)
        if mask.any():
            ece += mask.sum() / total * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)




def _kl_target_pred(target: dict[str, float], pred: dict[str, float]) -> float:
    keys = sorted(set(target) | set(pred))
    total = 0.0
    for k in keys:
        q = float(target.get(k, 0.0))
        if q > 0:
            total += q * math.log(q / max(float(pred.get(k, 0.0)), 1e-12))
    return float(total)


def _tv_distance(target: dict[str, float], pred: dict[str, float]) -> float:
    keys = set(target) | set(pred)
    return 0.5 * sum(abs(float(target.get(k, 0.0)) - float(pred.get(k, 0.0))) for k in keys)


def _event_ece(prob: np.ndarray, outcome: np.ndarray, bins: int = 15) -> float:
    if len(prob) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    total = len(prob)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (prob >= lo) & (prob <= hi if i == bins - 1 else prob < hi)
        if mask.any():
            ece += mask.sum() / total * abs(float(outcome[mask].mean()) - float(prob[mask].mean()))
    return float(ece)


def analyze(run_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = Path(run_dir)
    rows = _read_jsonl(run_dir / "raw.jsonl")
    ok = [r for r in rows if r.get("ok")]
    normalized = []

    for r in ok:
        usage = _usage(r)
        base = {
            "case_id": r["case_id"],
            "experiment": r["experiment"],
            "latency_ms": float(r["latency_ms"]),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            **{f"meta_{k}": v for k, v in (r.get("metadata") or {}).items() if isinstance(v, (str, int, float, bool)) or v is None},
        }
        for qid, exp in (r.get("expected") or {}).items():
            ans = _answer(r, qid)
            if not ans:
                continue
            typ = exp.get("type")
            label_value = exp.get("label")
            rec = dict(base)
            rec.update({"question_id": qid, "type": typ, "label": label_value})
            if typ == "noul":
                p = float(ans.get("noul"))
                rec.update({"p_true": p, "confidence_top": max(p, 1-p)})
                if label_value is not None:
                    y = int(label_value)
                    rec.update({
                        "pred": int(p >= 0.5),
                        "correct": int((p >= 0.5) == bool(y)),
                        "brier": (p-y)**2,
                        "nll": -math.log(max(p if y else 1-p, 1e-12)),
                    })
                if exp.get("target_probability") is not None:
                    q = float(exp["target_probability"])
                    rec.update({
                        "target_probability": q,
                        "prob_error": p - q,
                        "prob_abs_error": abs(p - q),
                        "prob_sq_error": (p - q) ** 2,
                        "target_kl": q * math.log(max(q,1e-12)/max(p,1e-12)) + (1-q) * math.log(max(1-q,1e-12)/max(1-p,1e-12)),
                    })
            elif typ == "choice":
                probs = _choice_probs(ans)
                pred = str(ans.get("choice"))
                top = max(probs.values()) if probs else float("nan")
                rec.update({
                    "pred": pred,
                    "confidence_top": top,
                    "reported_confidence": ans.get("confidence"),
                    "entropy_norm": _norm_entropy(list(probs.values())) if probs else float("nan"),
                })
                if label_value is not None:
                    label = str(label_value)
                    p_label = probs.get(label, 0.0)
                    rec.update({
                        "correct": int(pred == label),
                        "p_label": p_label,
                        "nll": -math.log(max(p_label, 1e-12)),
                        "brier": sum((v - (1.0 if k == label else 0.0))**2 for k, v in probs.items()),
                    })
                if exp.get("target_distribution") is not None:
                    target = {str(k): float(v) for k, v in exp["target_distribution"].items()}
                    rec.update({
                        "target_distribution_json": json.dumps(target, sort_keys=True),
                        "pred_distribution_json": json.dumps(probs, sort_keys=True),
                        "distribution_tv": _tv_distance(target, probs),
                        "distribution_jsd": _jsd(target, probs),
                        "target_kl": _kl_target_pred(target, probs),
                        "soft_brier": sum((float(probs.get(k, 0.0)) - q) ** 2 for k, q in target.items()),
                    })
            elif typ == "score":
                probs = _score_probs(ans)
                score_value = float(ans.get("score"))
                pred_level = max(probs, key=probs.get) if probs else str(round(score_value))
                rec.update({
                    "pred": pred_level,
                    "score": score_value,
                    "confidence_top": max(probs.values()) if probs else float("nan"),
                    "reported_confidence": ans.get("confidence"),
                    "entropy_norm": _norm_entropy(list(probs.values())) if probs else float("nan"),
                })
                if label_value is not None:
                    label = int(label_value)
                    p_label = probs.get(str(label), probs.get(label, 0.0)) if probs else float("nan")
                    rec.update({
                        "correct": int(str(pred_level) == str(label)),
                        "mae": abs(score_value-label),
                        "p_label": p_label,
                        "nll": -math.log(max(float(p_label), 1e-12)) if not math.isnan(float(p_label)) else float("nan"),
                    })
            normalized.append(rec)

    df = pd.DataFrame(normalized)
    df.to_csv(run_dir / "normalized.csv", index=False)

    summary: dict[str, Any] = {
        "requests_total": len(rows),
        "requests_ok": len(ok),
        "requests_failed": len(rows) - len(ok),
        "latency_ms": {},
        "calibration": {},
        "metamorphic": {},
        "scaling": {},
    }
    if ok:
        lat = np.asarray([r["latency_ms"] for r in ok], dtype=float)
        summary["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99)), "mean": float(lat.mean())}

    # Calibration by primitive.
    for typ in ["noul", "choice", "score"]:
        d = df[df["type"] == typ] if not df.empty else pd.DataFrame()
        if d.empty:
            continue
        item = {"n": int(len(d))}
        if "correct" in d:
            item["accuracy"] = float(d["correct"].mean())
        if "nll" in d:
            item["nll_mean"] = float(d["nll"].dropna().mean())
        if "brier" in d:
            item["brier_mean"] = float(d["brier"].dropna().mean())
        if typ in ("noul", "choice") and "correct" in d.columns:
            dc = d.dropna(subset=["confidence_top", "correct"])
            if not dc.empty:
                conf = dc["confidence_top"].astype(float).to_numpy()
                corr = dc["correct"].astype(float).to_numpy()
                item["ece_15"] = _ece(conf, corr, 15)
        if typ == "score" and "mae" in d:
            item["mae_mean"] = float(d["mae"].mean())
        summary["calibration"][typ] = item

    # Accuracy/quality summary by experiment where ground truth exists.
    summary["by_experiment"] = {}
    if not df.empty:
        for exp, d in df.groupby("experiment"):
            item = {"n": int(len(d))}
            if "correct" in d.columns and d["correct"].notna().any():
                item["accuracy"] = float(d["correct"].dropna().mean())
            if "nll" in d.columns and d["nll"].notna().any():
                item["nll_mean"] = float(d["nll"].dropna().mean())
            if "brier" in d.columns and d["brier"].notna().any():
                item["brier_mean"] = float(d["brier"].dropna().mean())
            if "mae" in d.columns and d["mae"].notna().any():
                item["mae_mean"] = float(d["mae"].dropna().mean())
            if "prob_abs_error" in d.columns and d["prob_abs_error"].notna().any():
                item["prob_mae"] = float(d["prob_abs_error"].dropna().mean())
                item["prob_rmse"] = float(np.sqrt(d["prob_sq_error"].dropna().mean()))
                item["prob_bias"] = float(d["prob_error"].dropna().mean())
            if "distribution_tv" in d.columns and d["distribution_tv"].notna().any():
                item["distribution_tv_mean"] = float(d["distribution_tv"].dropna().mean())
                item["distribution_jsd_mean"] = float(d["distribution_jsd"].dropna().mean())
                item["target_kl_mean"] = float(d["target_kl"].dropna().mean())
            summary["by_experiment"][str(exp)] = item

    # Dedicated probabilistic calibration diagnostics.
    summary["probabilistic"] = {}
    if not df.empty and "target_probability" in df.columns:
        dprob = df[df["target_probability"].notna() & df["p_true"].notna()].copy() if "p_true" in df.columns else pd.DataFrame()
        if not dprob.empty:
            target = dprob["target_probability"].astype(float).to_numpy()
            pred = dprob["p_true"].astype(float).to_numpy()
            exact = {
                "n": int(len(dprob)),
                "mae": float(np.mean(np.abs(pred-target))),
                "rmse": float(np.sqrt(np.mean((pred-target)**2))),
                "bias": float(np.mean(pred-target)),
                "max_abs_error": float(np.max(np.abs(pred-target))),
                "pearson_r": float(np.corrcoef(target, pred)[0,1]) if len(dprob) > 1 and np.std(target) > 0 and np.std(pred) > 0 else None,
            }
            labelled = dprob[dprob["label"].notna()].copy()
            if not labelled.empty:
                yp = labelled["p_true"].astype(float).to_numpy()
                yy = labelled["label"].astype(float).to_numpy()
                exact["empirical_outcome_n"] = int(len(labelled))
                exact["event_ece_15"] = _event_ece(yp, yy, 15)
                exact["outcome_brier"] = float(np.mean((yp-yy)**2))
                exact["outcome_nll"] = float(np.mean(-(yy*np.log(np.clip(yp,1e-12,1)) + (1-yy)*np.log(np.clip(1-yp,1e-12,1)))))
            bins = []
            edges = np.linspace(0, 1, 11)
            for i in range(10):
                lo, hi = edges[i], edges[i+1]
                mask = (pred >= lo) & (pred <= hi if i == 9 else pred < hi)
                if mask.any():
                    row = {
                        "lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                        "mean_pred": float(pred[mask].mean()),
                        "mean_target": float(target[mask].mean()),
                    }
                    if "label" in dprob.columns:
                        labels = dprob.loc[mask, "label"].dropna()
                        if len(labels):
                            row["empirical_rate"] = float(labels.astype(float).mean())
                            row["empirical_n"] = int(len(labels))
                    bins.append(row)
            exact["bins"] = bins
            exact["unique_returned_probabilities"] = int(len(set(round(float(x), 8) for x in pred)))
            summary["probabilistic"]["binary_exact_probability"] = exact

    if not df.empty and "distribution_tv" in df.columns:
        dchoice = df[df["distribution_tv"].notna()].copy()
        if not dchoice.empty:
            summary["probabilistic"]["multiclass_exact_distribution"] = {
                "n": int(len(dchoice)),
                "tv_mean": float(dchoice["distribution_tv"].mean()),
                "tv_p95": float(np.percentile(dchoice["distribution_tv"], 95)),
                "jsd_mean": float(dchoice["distribution_jsd"].mean()),
                "kl_target_pred_mean": float(dchoice["target_kl"].mean()),
                "soft_brier_mean": float(dchoice["soft_brier"].mean()),
                "hard_accuracy": float(dchoice["correct"].dropna().mean()) if "correct" in dchoice.columns and dchoice["correct"].notna().any() else None,
            }


    # Semantic adjudication diagnostics: compare Jev's full Choice distribution with
    # an explicitly authored reference distribution that represents panel-style disagreement.
    if not df.empty and "target_distribution_json" in df.columns:
        dsem = df[df["experiment"].isin(["semantic_adjudication", "semantic_repeatability"]) & df["target_distribution_json"].notna()].copy()
        if not dsem.empty:
            target_max = []
            pred_max = []
            target_ent = []
            pred_ent = []
            overconf = []
            for _, row in dsem.iterrows():
                target = json.loads(row["target_distribution_json"])
                pred = json.loads(row["pred_distribution_json"])
                tm = max(target.values()) if target else float("nan")
                pm = max(pred.values()) if pred else float("nan")
                te = _norm_entropy(list(target.values())) if target else float("nan")
                pe = _norm_entropy(list(pred.values())) if pred else float("nan")
                target_max.append(tm); pred_max.append(pm); target_ent.append(te); pred_ent.append(pe); overconf.append(pm-tm)
            dsem = dsem.assign(reference_max=target_max, jev_max=pred_max, reference_entropy=target_ent, jev_entropy=pred_ent, overconfidence_gap=overconf)
            semantic = {
                "n": int(len(dsem)),
                "tv_mean": float(dsem["distribution_tv"].mean()),
                "tv_p95": float(np.percentile(dsem["distribution_tv"], 95)),
                "jsd_mean": float(dsem["distribution_jsd"].mean()),
                "kl_target_pred_mean": float(dsem["target_kl"].mean()),
                "soft_brier_mean": float(dsem["soft_brier"].mean()),
                "modal_accuracy": float(dsem["correct"].mean()) if dsem["correct"].notna().any() else None,
                "reference_max_mean": float(np.mean(target_max)),
                "jev_max_mean": float(np.mean(pred_max)),
                "overconfidence_gap_mean": float(np.mean(overconf)),
                "overconfidence_gap_p95": float(np.percentile(overconf, 95)),
                "reference_entropy_mean": float(np.mean(target_ent)),
                "jev_entropy_mean": float(np.mean(pred_ent)),
                "entropy_gap_mean": float(np.mean(np.asarray(pred_ent)-np.asarray(target_ent))),
            }
            domains = {}
            if "meta_domain" in dsem.columns:
                for dom, dd in dsem.groupby("meta_domain"):
                    domains[str(dom)] = {
                        "n": int(len(dd)),
                        "tv_mean": float(dd["distribution_tv"].mean()),
                        "jsd_mean": float(dd["distribution_jsd"].mean()),
                        "modal_accuracy": float(dd["correct"].mean()) if dd["correct"].notna().any() else None,
                        "overconfidence_gap_mean": float(dd["overconfidence_gap"].mean()),
                        "reference_entropy_mean": float(dd["reference_entropy"].mean()),
                        "jev_entropy_mean": float(dd["jev_entropy"].mean()),
                    }
            semantic["by_domain"] = domains
            summary["semantic"] = semantic

    # Pair/group analyses on raw responses.
    by_exp = defaultdict(list)
    for r in ok:
        by_exp[r["experiment"]].append(r)

    # Candidate-order invariance.
    vals = by_exp.get("candidate_order_invariance", [])
    if vals:
        basep = _choice_probs(_answer(vals[0]))
        js = [_jsd(basep, _choice_probs(_answer(r))) for r in vals[1:]]
        maxabs = []
        for r in vals[1:]:
            p = _choice_probs(_answer(r))
            keys = set(basep) | set(p)
            maxabs.append(max(abs(basep.get(k,0)-p.get(k,0)) for k in keys))
        summary["metamorphic"]["candidate_order_invariance"] = {"trials": len(vals), "jsd_mean": float(np.mean(js)) if js else 0.0, "jsd_max": float(np.max(js)) if js else 0.0, "max_abs_probability_delta": float(np.max(maxabs)) if maxabs else 0.0}

    # Paraphrase invariance.
    vals = by_exp.get("question_paraphrase_invariance", [])
    if vals:
        basep = _choice_probs(_answer(vals[0]))
        js = [_jsd(basep, _choice_probs(_answer(r))) for r in vals[1:]]
        summary["metamorphic"]["question_paraphrase_invariance"] = {"trials": len(vals), "jsd_mean": float(np.mean(js)) if js else 0.0, "jsd_max": float(np.max(js)) if js else 0.0}

    # Question independence.
    vals = sorted(by_exp.get("question_independence", []), key=lambda r: r["metadata"]["bundle_size"])
    if vals:
        basep = _choice_probs(_answer(vals[0]))
        deltas = []
        for r in vals:
            p = _choice_probs(_answer(r))
            keys = set(basep) | set(p)
            deltas.append({"bundle_size": r["metadata"]["bundle_size"], "jsd": _jsd(basep,p), "max_abs_delta": max(abs(basep.get(k,0)-p.get(k,0)) for k in keys)})
        summary["metamorphic"]["question_independence"] = deltas

    # Repeatability.
    vals = by_exp.get("repeatability", [])
    if vals:
        qids = ["outage", "severity", "route"]
        rep = {}
        for qid in qids:
            numeric_vectors = []
            for r in vals:
                a = _answer(r, qid)
                if not a:
                    continue
                if a.get("type") == "noul":
                    numeric_vectors.append([float(a["noul"])])
                else:
                    probs = a.get("probabilities") or {}
                    numeric_vectors.append([float(v) for k,v in sorted(probs.items())])
            arr = np.asarray(numeric_vectors, dtype=float)
            rep[qid] = {"repetitions": len(arr), "max_std_across_dimensions": float(arr.std(axis=0).max()) if len(arr) else float("nan"), "max_range_across_dimensions": float((arr.max(axis=0)-arr.min(axis=0)).max()) if len(arr) else float("nan")}
        summary["metamorphic"]["repeatability"] = rep

    # Irrelevant context robustness.
    vals = sorted(by_exp.get("irrelevant_context_robustness", []), key=lambda r: r["metadata"]["noise_units"])
    if vals:
        basep = _choice_probs(_answer(vals[0]))
        summary["metamorphic"]["irrelevant_context"] = [
            {"noise_units": r["metadata"]["noise_units"], "p_billing": _choice_probs(_answer(r)).get("billing"), "jsd_from_clean": _jsd(basep, _choice_probs(_answer(r)))} for r in vals
        ]

    # Evidence removal.
    vals = sorted(by_exp.get("evidence_removal", []), key=lambda r: r["metadata"]["removal_level"])
    if vals:
        summary["metamorphic"]["evidence_removal"] = [
            {"removal_level": r["metadata"]["removal_level"], "probabilities": _choice_probs(_answer(r)), "reported_confidence": (_answer(r) or {}).get("confidence"), "entropy_norm": _norm_entropy(list(_choice_probs(_answer(r)).values()))} for r in vals
        ]

    # Context length and evidence position.
    vals = sorted(by_exp.get("context_length_position", []), key=lambda r: (r["metadata"]["context_chars"], r["metadata"]["evidence_position"]))
    if vals:
        summary["scaling"]["context_length_position"] = [
            {
                "context_chars": r["metadata"]["context_chars"],
                "evidence_position": r["metadata"]["evidence_position"],
                "latency_ms": r["latency_ms"],
                "p_target": _choice_probs(_answer(r)).get("duplicate_charge"),
                "input_tokens": _usage(r).get("input_tokens"),
            } for r in vals
        ]

    # Probabilistic representation invariance.
    vals = by_exp.get("prob_representation_invariance", [])
    if vals:
        groups = defaultdict(list)
        for r in vals:
            groups[str(r["metadata"]["group"])].append(r)
        group_rows = []
        for g, rs in sorted(groups.items()):
            ps = [float((_answer(r) or {}).get("noul")) for r in rs if _answer(r) and (_answer(r) or {}).get("noul") is not None]
            if ps:
                group_rows.append({
                    "group": g,
                    "n": len(ps),
                    "target_probability": float(rs[0]["metadata"]["target_probability"]),
                    "p_mean": float(np.mean(ps)),
                    "p_std": float(np.std(ps)),
                    "p_range": float(max(ps)-min(ps)),
                })
        summary["metamorphic"]["prob_representation_invariance"] = {
            "groups": group_rows,
            "mean_range": float(np.mean([g["p_range"] for g in group_rows])) if group_rows else None,
            "max_range": float(np.max([g["p_range"] for g in group_rows])) if group_rows else None,
        }

    # Repeatability at deliberately non-degenerate probabilities.
    vals = by_exp.get("prob_repeatability", [])
    if vals:
        groups = defaultdict(list)
        for r in vals:
            groups[str(r["metadata"]["group"])].append(r)
        repeat = {}
        for g, rs in sorted(groups.items()):
            ps = [float((_answer(r) or {}).get("noul")) for r in rs if _answer(r) and (_answer(r) or {}).get("noul") is not None]
            if ps:
                repeat[g] = {
                    "n": len(ps),
                    "target_probability": float(rs[0]["metadata"]["target_probability"]),
                    "mean": float(np.mean(ps)),
                    "std": float(np.std(ps)),
                    "min": float(np.min(ps)),
                    "max": float(np.max(ps)),
                    "range": float(np.max(ps)-np.min(ps)),
                }
        summary["metamorphic"]["prob_repeatability"] = repeat

    # Scaling curves.
    for exp, meta_key in [("parallel_question_scaling", "question_count"), ("choice_cardinality_scaling", "cardinality")]:
        vals = sorted(by_exp.get(exp, []), key=lambda r: r["metadata"][meta_key])
        if vals:
            summary["scaling"][exp] = [
                {meta_key: r["metadata"][meta_key], "latency_ms": r["latency_ms"], "input_tokens": _usage(r).get("input_tokens"), "output_tokens": _usage(r).get("output_tokens")} for r in vals
            ]

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return df, summary
