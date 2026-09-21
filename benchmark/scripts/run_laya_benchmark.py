#!/usr/bin/env python3
"""Replay the exact Jev benchmark cases against a local Laya checkpoint.

The script is intentionally independent of the benchmark case generators: it reads the
recorded ``cases.jsonl`` files from the five Jev study runs. This guarantees byte-identical
state/question inputs even if the generator code changes later.

It writes the same raw-result schema used by jevbench, then invokes the existing
``jevbench.metrics`` and ``jevbench.report`` modules when they are available in the study
bundle.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import inspect
import json
import hashlib
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_SOURCE_RUNS = (
    "run_20260919T201050Z_core",
    "run_20260919T201739Z_core",
    "run_20260919T202139Z_full",
    "run_20260919T203758Z_calibration_full",
    "run_20260919T211437Z_semantic_full",
)

MODEL_REFS: dict[str, dict[str, str]] = {
    "torch": {
        "base": "convaiinnovations/laya",
        "typed-decisions": "convaiinnovations/laya-typed-decisions",
        "multilingual": "convaiinnovations/laya-multilingual",
    },
    "mlx": {
        "base": "aac6fef/laya-mlx",
        "typed-decisions": "aac6fef/laya-typed-decisions-mlx",
        "multilingual": "aac6fef/laya-multilingual-mlx",
    },
    "mock": {
        "base": "mock/laya-base",
        "typed-decisions": "mock/laya-typed-decisions",
        "multilingual": "mock/laya-multilingual",
    },
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slug(text: str) -> str:
    out = []
    for ch in text.lower():
        out.append(ch if ch.isalnum() else "-")
    return "-".join(filter(None, "".join(out).split("-")))


def json_safe(value: Any) -> Any:
    """Convert tensors/NumPy scalars/dataclasses into JSON-safe Python values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def call_supported(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a function with only supported, non-None keyword arguments.

    Some backend loaders expose ``**kwargs`` and then forward those values into a
    stricter constructor. Passing ``dtype=None`` or ``batch_size=None`` therefore
    bypasses the callee's defaults and can fail validation. Strip ``None`` before
    inspecting or forwarding keyword arguments.
    """
    non_none = {k: v for k, v in kwargs.items() if v is not None}
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **non_none)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return fn(*args, **non_none)
    accepted = {k: v for k, v in non_none.items() if k in signature.parameters}
    return fn(*args, **accepted)


@dataclass
class BackendInfo:
    name: str
    model_ref: str
    package_version: str | None
    device: str | None
    load_seconds: float
    dtype: str | None = None
    batch_size: int | None = None
    warmup_seconds: float | None = None


class LocalBackend:
    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def info(self) -> BackendInfo:
        raise NotImplementedError

    def apply_budget_overrides(self, max_len: int | None, head_max_len: int | None) -> dict[str, Any]:
        return {}

    def effective_limits(self) -> dict[str, int | None]:
        return {"max_len": None, "head_max_len": None}


class LayaBackend(LocalBackend):
    def __init__(
        self,
        backend: str,
        model_ref: str,
        device: str | None,
        hf_token: str | None,
        subfolder: str | None,
        dtype: str | None,
        batch_size: int | None,
    ) -> None:
        self.backend = backend
        self.model_ref = model_ref
        self.device = device
        self.hf_token = hf_token
        self.subfolder = subfolder
        # Pin the documented MLX defaults so the run manifest is explicit and the
        # loader never receives invalid ``None`` values. PyTorch keeps its own
        # defaults unless the caller supplies overrides.
        effective_dtype = dtype or ("float16" if backend == "mlx" else None)
        effective_batch_size = batch_size or (16 if backend == "mlx" else None)
        self.dtype = effective_dtype
        self.batch_size = effective_batch_size
        started = time.perf_counter()

        if backend == "mlx":
            module = importlib.import_module("laya_mlx")
            load_fn = getattr(module, "load")
            self.agent = call_supported(
                load_fn,
                model_ref,
                token=hf_token,
                subfolder=subfolder,
                device=device,
                dtype=effective_dtype,
                batch_size=effective_batch_size,
            )
        elif backend == "torch":
            # Laya documents that disabling TensorFlow avoids an import-time deadlock in
            # environments where TensorFlow/abseil are present.
            os.environ.setdefault("USE_TF", "0")
            module = importlib.import_module("laya")
            load_fn = getattr(module, "load")
            self.agent = call_supported(
                load_fn,
                model_ref,
                device=device,
                token=hf_token,
                subfolder=subfolder,
                dtype=effective_dtype,
                batch_size=effective_batch_size,
            )
        else:
            raise ValueError(f"Unsupported Laya backend: {backend}")

        version = getattr(module, "__version__", None)
        self._info = BackendInfo(
            name=backend,
            model_ref=model_ref,
            package_version=str(version) if version is not None else None,
            device=device,
            load_seconds=time.perf_counter() - started,
            dtype=effective_dtype,
            batch_size=effective_batch_size,
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def apply_budget_overrides(self, max_len: int | None, head_max_len: int | None) -> dict[str, Any]:
        changed: dict[str, Any] = {}
        cfg = getattr(self.agent, "cfg", None)
        if cfg is None:
            if max_len is not None or head_max_len is not None:
                raise RuntimeError("Installed Laya backend exposes no agent.cfg for budget overrides")
            return changed
        if max_len is not None:
            old = cfg.get("max_len") if isinstance(cfg, Mapping) else getattr(cfg, "max_len", None)
            if isinstance(cfg, Mapping):
                cfg["max_len"] = int(max_len)
            else:
                setattr(cfg, "max_len", int(max_len))
            changed["max_len"] = {"old": old, "new": int(max_len)}
        if head_max_len is not None:
            old = cfg.get("head_max_len") if isinstance(cfg, Mapping) else getattr(cfg, "head_max_len", None)
            if isinstance(cfg, Mapping):
                cfg["head_max_len"] = int(head_max_len)
            else:
                setattr(cfg, "head_max_len", int(head_max_len))
            changed["head_max_len"] = {"old": old, "new": int(head_max_len)}
        return changed

    def effective_limits(self) -> dict[str, int | None]:
        cfg = getattr(self.agent, "cfg", None)
        if cfg is None:
            return {"max_len": None, "head_max_len": None}
        def get_int(name: str) -> int | None:
            value = cfg.get(name) if isinstance(cfg, Mapping) else getattr(cfg, name, None)
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None
        return {"max_len": get_int("max_len"), "head_max_len": get_int("head_max_len")}

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self.agent, "predict", None) or getattr(self.agent, "system_one", None)
        if fn is None:
            raise RuntimeError("Loaded Laya agent exposes neither predict() nor system_one()")
        return json_safe(fn(state, questions))


class MockBackend(LocalBackend):
    """Deterministic backend used only to validate the replay/output pipeline offline."""

    def __init__(self, model_ref: str) -> None:
        self._info = BackendInfo(
            name="mock", model_ref=model_ref, package_version="test", device="cpu", load_seconds=0.0
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for qid, q in questions.items():
            qtype = q.get("type")
            if qtype == "noul":
                answers[qid] = {"type": "noul", "noul": 0.5}
            elif qtype == "choice":
                criteria = q.get("criteria") or {}
                keys = list(criteria)
                if not keys:
                    raise ValueError(f"Choice question {qid!r} has no criteria")
                p = 1.0 / len(keys)
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": {k: p for k in keys},
                    "confidence": p,
                }
            elif qtype == "score":
                criteria = q.get("criteria") or []
                if not criteria:
                    raise ValueError(f"Score question {qid!r} has no criteria")
                p = 1.0 / len(criteria)
                probs = {str(i): p for i in range(len(criteria))}
                answers[qid] = {
                    "type": "score",
                    "score": sum(i * p for i in range(len(criteria))),
                    "legend": {str(i): str(v) for i, v in enumerate(criteria)},
                    "probabilities": probs,
                    "confidence": p,
                }
            else:
                raise ValueError(f"Unsupported question type {qtype!r}")
        return {"model": self._info.model_ref, "answers": answers, "usage": {"input_tokens": None, "output_tokens": 0}}


def validate_and_normalize_result(
    result: Mapping[str, Any], questions: Mapping[str, Any], model_ref: str
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError(f"Laya returned {type(result).__name__}; expected mapping")
    answers = result.get("answers")
    if not isinstance(answers, Mapping):
        raise ValueError("Laya result has no mapping-valued 'answers'")

    normalized: dict[str, Any] = {}
    missing = set(questions) - set(answers)
    if missing:
        raise ValueError(f"Laya omitted answers for: {sorted(missing)}")

    for qid, q in questions.items():
        ans = json_safe(answers[qid])
        if not isinstance(ans, dict):
            raise TypeError(f"Answer {qid!r} is not an object")
        qtype = str(q.get("type"))
        ans.setdefault("type", qtype)
        if qtype == "noul":
            if "noul" not in ans:
                raise ValueError(f"Noul answer {qid!r} has no 'noul' value")
            ans["noul"] = float(ans["noul"])
        elif qtype in {"choice", "score"}:
            probs = ans.get("probabilities")
            if not isinstance(probs, Mapping) or not probs:
                raise ValueError(f"{qtype} answer {qid!r} has no probability mapping")
            ans["probabilities"] = {str(k): float(v) for k, v in probs.items()}
            if qtype == "choice":
                ans.setdefault("choice", max(ans["probabilities"], key=ans["probabilities"].get))
                ans.setdefault("confidence", max(ans["probabilities"].values()))
            else:
                if "score" not in ans:
                    ans["score"] = sum(float(k) * v for k, v in ans["probabilities"].items())
                ans.setdefault("confidence", max(ans["probabilities"].values()))
                criteria = q.get("criteria") or []
                ans.setdefault("legend", {str(i): str(v) for i, v in enumerate(criteria)})
        else:
            raise ValueError(f"Unsupported question type {qtype!r}")
        normalized[qid] = ans

    usage = result.get("usage") if isinstance(result.get("usage"), Mapping) else {}
    return {
        "model": str(result.get("model") or model_ref),
        "answers": normalized,
        "usage": {
            "input_tokens": json_safe(usage.get("input_tokens")),
            "output_tokens": json_safe(usage.get("output_tokens", 0)),
        },
        **({"routing": json_safe(result["routing"])} if "routing" in result else {}),
    }


def locate_study_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not (path / "results").is_dir():
        raise FileNotFoundError(f"Study root has no results/ directory: {path}")
    if not (path / "benchmark").is_dir():
        raise FileNotFoundError(f"Study root has no benchmark/ directory: {path}")
    return path


def resolve_source_runs(study_root: Path, names: Sequence[str] | None) -> list[Path]:
    wanted = tuple(names) if names else DEFAULT_SOURCE_RUNS
    paths: list[Path] = []
    for name in wanted:
        candidate = study_root / "results" / name
        if not candidate.is_dir():
            raise FileNotFoundError(f"Missing source run: {candidate}")
        if not (candidate / "cases.jsonl").is_file():
            raise FileNotFoundError(f"Missing cases.jsonl: {candidate}")
        paths.append(candidate)
    return paths


def install_benchmark_import(study_root: Path) -> None:
    src = study_root / "benchmark" / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def analyze_result(study_root: Path, out_dir: Path) -> None:
    install_benchmark_import(study_root)
    try:
        from jevbench.metrics import analyze
        from jevbench.report import generate_report
    except Exception as exc:
        print(f"WARNING: could not import jevbench analysis modules: {exc}", file=sys.stderr)
        return
    analyze(out_dir)
    generate_report(out_dir)


def warm_up(backend: LocalBackend) -> float:
    state = "Warm-up request: the invoice was charged twice and requires billing review."
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "payments, invoices, refunds, duplicate charges",
                "technical": "software bugs and product failures",
            },
        },
        "has_duplicate_charge": {
            "type": "noul",
            "instructions": "Does the state explicitly describe a duplicate charge?",
        },
    }
    started = time.perf_counter()
    validate_and_normalize_result(backend.predict(state, questions), questions, backend.info.model_ref)
    elapsed = time.perf_counter() - started
    backend.info.warmup_seconds = elapsed
    return elapsed


def upgrade_legacy_compatibility(raw_path: Path, limits: Mapping[str, int | None]) -> int:
    """Backfill v1.0.2 compatibility fields into existing v1.0.1 raw rows.

    This makes ``--resume`` safe across adapter versions: already-completed rows are
    reclassified from their stored request/response instead of being implicitly treated
    as full-input successes. Returns the number of rows changed.
    """
    if not raw_path.exists():
        return 0
    rows = read_jsonl(raw_path)
    changed = 0
    for row in rows:
        if row.get("compatibility_status") is not None and "comparable" in row:
            continue
        if row.get("ok"):
            request = row.get("request") or {}
            questions = request.get("questions") if isinstance(request, Mapping) else {}
            if not isinstance(questions, Mapping):
                questions = {}
            status, comparable, reason, details = classify_success_compatibility(
                row.get("response") if isinstance(row.get("response"), Mapping) else None,
                questions,
                limits,
            )
        else:
            status, comparable, reason = classify_error_compatibility(
                row.get("error"), row.get("error_type")
            )
            details = {**limits, "detection": "exception_classifier_legacy_backfill"}
        row["compatibility_status"] = status
        row["comparable"] = comparable
        row["compatibility_reason"] = reason
        row["compatibility_details"] = details
        changed += 1
    if changed:
        tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
        write_jsonl(tmp, rows)
        tmp.replace(raw_path)
    return changed


def completed_case_ids(raw_path: Path) -> set[str]:
    if not raw_path.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(raw_path):
        cid = row.get("case_id")
        if cid:
            ids.add(str(cid))
    return ids


def case_preflight(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    question_counts: list[int] = []
    choice_cardinalities: list[int] = []
    score_levels: list[int] = []
    state_chars: list[int] = []
    by_experiment: dict[str, int] = {}
    for case in cases:
        by_experiment[str(case.get("experiment"))] = by_experiment.get(str(case.get("experiment")), 0) + 1
        state_chars.append(len(json.dumps(case.get("state"), ensure_ascii=False)))
        questions = case.get("questions") or {}
        question_counts.append(len(questions))
        for q in questions.values():
            if q.get("type") == "choice":
                choice_cardinalities.append(len(q.get("criteria") or {}))
            elif q.get("type") == "score":
                score_levels.append(len(q.get("criteria") or []))
    def stats(values: Sequence[int]) -> dict[str, int | None]:
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return {
        "cases": len(cases),
        "by_experiment": dict(sorted(by_experiment.items())),
        "questions_per_request": stats(question_counts),
        "choice_cardinality": stats(choice_cardinalities),
        "score_levels": stats(score_levels),
        "state_characters": stats(state_chars),
    }


def classify_error_compatibility(error: str | None, error_type: str | None) -> tuple[str, bool, str | None]:
    text = (error or "").lower()
    if "too many options" in text and ("token budget" in text or "options" in text):
        return "unsupported_cardinality", False, error
    return "backend_error", False, f"{error_type or 'Error'}: {error}" if error else error_type


def classify_success_compatibility(
    response: Mapping[str, Any] | None,
    questions: Mapping[str, Any],
    limits: Mapping[str, int | None],
) -> tuple[str, bool, str | None, dict[str, Any]]:
    """Classify whether Laya consumed the complete request.

    Laya reports aggregate ``usage.input_tokens`` across independently encoded question
    rows. At the model context ceiling, each saturated row contributes ``max_len``
    tokens. A request whose aggregate usage reaches ``max_len * question_count`` is
    therefore treated as truncated. This is intentionally conservative and recorded as
    a heuristic in the row metadata.
    """
    usage = (response or {}).get("usage") if isinstance(response, Mapping) else None
    input_tokens = usage.get("input_tokens") if isinstance(usage, Mapping) else None
    max_len = limits.get("max_len")
    qcount = max(1, len(questions))
    details: dict[str, Any] = {
        "max_len": max_len,
        "head_max_len": limits.get("head_max_len"),
        "question_count": qcount,
        "reported_input_tokens": input_tokens,
        "detection": "usage_ceiling_heuristic",
    }
    try:
        used = int(input_tokens) if input_tokens is not None else None
    except (TypeError, ValueError):
        used = None
    if max_len is not None and used is not None:
        ceiling = int(max_len) * qcount
        details["aggregate_full_input_ceiling"] = ceiling
        if used >= ceiling:
            return (
                "truncated_input",
                False,
                f"reported input_tokens={used} reached the configured context ceiling "
                f"max_len={max_len} across {qcount} question row(s)",
                details,
            )
    return "full_input", True, None, details


def compatibility_summary(raw_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(raw_rows)
    by_exp: dict[str, dict[str, Any]] = {}
    errors: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        exp = str(row.get("experiment"))
        status = str(row.get("compatibility_status") or ("full_input" if row.get("ok") else "backend_error"))
        comparable = bool(row.get("comparable", row.get("ok") and status == "full_input"))
        statuses[status] = statuses.get(status, 0) + 1
        slot = by_exp.setdefault(
            exp,
            {"total": 0, "ok": 0, "failed": 0, "full_input": 0, "truncated_input": 0, "unsupported_cardinality": 0, "backend_error": 0, "comparable": 0},
        )
        slot["total"] += 1
        if row.get("ok"):
            slot["ok"] += 1
        else:
            slot["failed"] += 1
            key = str(row.get("error_type") or "UnknownError")
            errors[key] = errors.get(key, 0) + 1
        if status in slot:
            slot[status] += 1
        if comparable:
            slot["comparable"] += 1
    return {
        "requests_total": len(rows),
        "requests_ok": sum(1 for r in rows if r.get("ok")),
        "requests_failed": sum(1 for r in rows if not r.get("ok")),
        "requests_comparable": sum(1 for r in rows if bool(r.get("comparable", r.get("ok") and r.get("compatibility_status") in {None, "full_input"}))),
        "compatibility_status": dict(sorted(statuses.items())),
        "by_experiment": dict(sorted(by_exp.items())),
        "failure_types": dict(sorted(errors.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def run_source(
    study_root: Path,
    source_run: Path,
    backend: LocalBackend,
    checkpoint: str,
    output_root: Path,
    max_cases: int | None,
    resume: bool,
    fail_fast: bool,
    progress_every: int,
    budget_overrides: dict[str, Any],
    do_analyze: bool,
) -> Path:
    cases = read_jsonl(source_run / "cases.jsonl")
    if max_cases is not None:
        cases = cases[: max(0, max_cases)]

    run_slug = f"{source_run.name}__laya_{slug(checkpoint)}_{slug(backend.info.name)}"
    out_dir = output_root / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    if raw_path.exists() and not resume:
        raise FileExistsError(f"Output exists; use --resume or choose another output root: {out_dir}")
    if resume and raw_path.exists():
        upgraded = upgrade_legacy_compatibility(raw_path, backend.effective_limits())
        if upgraded:
            print(f"Backfilled compatibility metadata for {upgraded} existing row(s) in {raw_path}", flush=True)
    done = completed_case_ids(raw_path) if resume else set()

    selected_cases_path = out_dir / "cases.jsonl"
    write_jsonl(selected_cases_path, cases)
    preflight = case_preflight(cases)
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    source_manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "created_at_utc": utc_now(),
        "provider": "laya-local",
        "backend": backend.info.name,
        "checkpoint": checkpoint,
        "model": backend.info.model_ref,
        "package_version": backend.info.package_version,
        "device": backend.info.device,
        "dtype": backend.info.dtype,
        "batch_size": backend.info.batch_size,
        "model_load_seconds": backend.info.load_seconds,
        "warmup_seconds": backend.info.warmup_seconds,
        "budget_overrides": budget_overrides,
        "effective_limits": backend.effective_limits(),
        "source_run": source_run.name,
        "source_manifest": source_manifest,
        "source_cases_sha256": sha256_file(source_run / "cases.jsonl"),
        "selected_cases_sha256": sha256_file(selected_cases_path),
        "preflight": preflight,
        "case_count": len(cases),
        "python": sys.version,
        "platform": platform.platform(),
        "resume": resume,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    pending_total = sum(1 for case in cases if str(case["case_id"]) not in done)
    started_all = time.perf_counter()
    processed = 0
    for case in cases:
        cid = str(case["case_id"])
        if cid in done:
            continue
        started = time.perf_counter()
        response: dict[str, Any] | None = None
        ok = False
        error = None
        error_type = None
        tb = None
        compatibility_status = "backend_error"
        comparable = False
        compatibility_reason = None
        compatibility_details: dict[str, Any] = {}
        try:
            raw_result = backend.predict(case["state"], case["questions"])
            response = validate_and_normalize_result(raw_result, case["questions"], backend.info.model_ref)
            ok = True
            compatibility_status, comparable, compatibility_reason, compatibility_details = classify_success_compatibility(
                response, case["questions"], backend.effective_limits()
            )
        except Exception as exc:
            error = str(exc)
            error_type = type(exc).__name__
            compatibility_status, comparable, compatibility_reason = classify_error_compatibility(error, error_type)
            compatibility_details = {**backend.effective_limits(), "detection": "exception_classifier"}
            tb = traceback.format_exc(limit=12)
            if fail_fast:
                raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        row = {
            "case_id": cid,
            "experiment": case["experiment"],
            "metadata": case.get("metadata") or {},
            "expected": case.get("expected") or {},
            "request": {
                "state": case["state"],
                "questions": case["questions"],
                "model": backend.info.model_ref,
            },
            "ok": ok,
            "compatibility_status": compatibility_status,
            "comparable": comparable,
            "compatibility_reason": compatibility_reason,
            "compatibility_details": compatibility_details,
            "status_code": 200 if ok else None,
            "latency_ms": latency_ms,
            "attempts": 1,
            "response": response,
            "response_headers": {
                "x-local-provider": "laya",
                "x-laya-backend": backend.info.name,
                "x-laya-inference-time-ms": f"{latency_ms:.6f}",
            },
            "error": error,
            "error_type": error_type,
            "traceback": tb,
            "observed_at_utc": utc_now(),
        }
        append_jsonl(raw_path, row)
        processed += 1
        if processed % max(1, progress_every) == 0:
            print(
                f"[{source_run.name}] {processed}/{pending_total} "
                f"last={case['experiment']} ok={ok} {latency_ms:.1f}ms",
                flush=True,
            )

    rows = read_jsonl(raw_path)
    compat = compatibility_summary(rows)
    (out_dir / "compatibility.json").write_text(json.dumps(compat, indent=2), encoding="utf-8")
    manifest.update(
        {
            "finished_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started_all,
            "requests_total": compat["requests_total"],
            "requests_ok": compat["requests_ok"],
            "requests_failed": compat["requests_failed"],
            "requests_comparable": compat["requests_comparable"],
            "compatibility_status": compat["compatibility_status"],
        }
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if do_analyze and compat["requests_ok"]:
        analyze_result(study_root, out_dir)
    print(
        f"DONE {out_dir}: ok={compat['requests_ok']} comparable={compat['requests_comparable']} "
        f"failed={compat['requests_failed']} status={compat['compatibility_status']}",
        flush=True,
    )
    return out_dir


def auto_backend() -> str:
    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        try:
            importlib.import_module("laya_mlx")
            return "mlx"
        except Exception:
            pass
    return "torch"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", type=Path, required=True, help="Path containing benchmark/ and results/")
    p.add_argument("--backend", choices=["auto", "mlx", "torch", "mock"], default="auto")
    p.add_argument("--checkpoint", choices=["base", "typed-decisions", "multilingual"], default="typed-decisions")
    p.add_argument("--model-ref", help="Override the checkpoint repository or local model directory")
    p.add_argument("--subfolder", help="Optional Hugging Face subfolder for bundled checkpoints")
    p.add_argument("--device", help="Device override, e.g. gpu/cpu for MLX or mps/cuda/cpu for PyTorch")
    p.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        help="Runtime precision. MLX defaults explicitly to float16.",
    )
    p.add_argument("--batch-size", type=int, help="Optional Laya question batch size")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--source-run", action="append", help="Source run directory name; repeatable. Default: all five")
    p.add_argument("--output-root", type=Path, help="Default: <study-root>/laya-results")
    p.add_argument("--max-cases", type=int, help="Run only the first N cases from each source run")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--no-analyze", action="store_true")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--max-len", type=int, help="Optional non-baseline agent.cfg max_len override")
    p.add_argument("--head-max-len", type=int, help="Optional non-baseline agent.cfg head_max_len override")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    study_root = locate_study_root(args.study_root)
    backend_name = auto_backend() if args.backend == "auto" else args.backend
    model_ref = args.model_ref or MODEL_REFS[backend_name][args.checkpoint]
    output_root = (args.output_root or (study_root / "laya-results")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sources = resolve_source_runs(study_root, args.source_run)

    if backend_name == "mock":
        backend: LocalBackend = MockBackend(model_ref)
    else:
        backend = LayaBackend(
            backend=backend_name,
            model_ref=model_ref,
            device=args.device,
            hf_token=args.hf_token,
            subfolder=args.subfolder,
            dtype=args.dtype,
            batch_size=args.batch_size,
        )
    budget_overrides = backend.apply_budget_overrides(args.max_len, args.head_max_len)
    if not args.no_warmup:
        elapsed = warm_up(backend)
        print(f"Warm-up: {elapsed*1000:.1f} ms", flush=True)
    print(
        f"Loaded {model_ref} via {backend_name} in {backend.info.load_seconds:.2f}s; "
        f"source runs={len(sources)}",
        flush=True,
    )

    outputs: list[Path] = []
    for source in sources:
        outputs.append(
            run_source(
                study_root=study_root,
                source_run=source,
                backend=backend,
                checkpoint=args.checkpoint,
                output_root=output_root,
                max_cases=args.max_cases,
                resume=args.resume,
                fail_fast=args.fail_fast,
                progress_every=args.progress_every,
                budget_overrides=budget_overrides,
                do_analyze=not args.no_analyze,
            )
        )
    print("\nGenerated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
