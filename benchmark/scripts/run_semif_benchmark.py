#!/usr/bin/env python3
"""Replay the exact Jev benchmark cases against SemIf (Qwen3.5-4B, native MLX).

SemIf (https://github.com/TheoLeeCJ/SemIf) is an open-model reproduction of the Jev *interface
pattern*: a state, a runtime question and typed options go in, and per-option probabilities are
read straight from the model's answer-slot logits. There is no sampling and no decoding loop.

This script reuses the replay/normalization/compatibility machinery of ``run_laya_benchmark.py``
and only supplies the SemIf backend. It reads the recorded ``cases.jsonl`` files, so both systems
receive byte-identical states, questions and criteria, and writes the same raw-result schema so
that ``jevbench.metrics`` / ``jevbench.report`` work unchanged.

Primitive mapping (SemIf only exposes categorical options, so this mapping is part of the
experiment and is recorded in every manifest):

* ``choice`` -> options are the criteria keys/descriptions; probabilities are SemIf's softmax
  over the declared answer letters.
* ``score``  -> options are the ordinal criteria, keyed ``"0".."n-1"``; ``score`` is the expected
  index under those probabilities (the convention used by the Laya adapter).
* ``noul``   -> a fixed two-option ``yes``/``no`` decision; ``noul`` is P(yes). The wording of the
  two options is authored here, not by SemIf: see ``NOUL_OPTIONS`` and ``prompt_sha256`` in each row.

Execution mapping: a request with one question uses SemIf ``direct`` scoring; a request with
several questions over the same state uses ``shared`` scoring (one state prefill, parallel
question suffixes), which is SemIf's analogue of a multi-question Jev request.

SemIf limits are surfaced, never repaired: more than 16 options is recorded as
``unsupported_cardinality`` and an over-long prompt as ``input_too_long`` (SemIf never truncates).
"""
from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_laya_benchmark as rl  # noqa: E402  (shared replay machinery)

FAMILY = "semif"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_CHECKPOINT = "qwen3.5-4b"
# Qwen3.5-4B supports far more than Jev's ~21K-token probe; SemIf itself defaults to 4096 and
# refuses (never truncates) beyond the limit, so the limit is raised to keep the comparison fair.
DEFAULT_MAX_TOKENS = 65536

MAX_OPTIONS = 16  # SemIf answer slots are the letters A-P.

# Frozen, authored wording for the synthetic yes/no decision used for Jev "noul" questions.
NOUL_OPTIONS: tuple[dict[str, str], ...] = (
    {"id": "yes", "description": "Yes: the answer to the question is yes."},
    {"id": "no", "description": "No: the answer to the question is no."},
)


def build_rows(state: Any, questions: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Translate one Jev request into SemIf decision rows (one per question)."""
    rows: list[dict[str, Any]] = []
    for qid, q in questions.items():
        qtype = q.get("type")
        instructions = q.get("instructions")
        if qtype == "noul":
            options = [dict(o) for o in NOUL_OPTIONS]
        elif qtype == "choice":
            criteria = q.get("criteria") or {}
            if not criteria:
                raise ValueError(f"Choice question {qid!r} has no criteria")
            options = [{"id": str(k), "description": str(v)} for k, v in criteria.items()]
        elif qtype == "score":
            criteria = q.get("criteria") or []
            if not criteria:
                raise ValueError(f"Score question {qid!r} has no criteria")
            options = [{"id": str(i), "description": str(v)} for i, v in enumerate(criteria)]
        else:
            raise ValueError(f"Unsupported question type {qtype!r}")
        rows.append({"id": str(qid), "state": state, "question": str(instructions), "options": options})
    return rows


def assemble_answers(
    questions: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Convert SemIf option probabilities into Jev-style typed answers."""
    by_id = {str(r["id"]): r for r in results}
    answers: dict[str, dict[str, Any]] = {}
    for qid, q in questions.items():
        res = by_id[str(qid)]
        probs = {str(k): float(v) for k, v in zip(res["option_ids"], res["probabilities"])}
        qtype = q["type"]
        if qtype == "noul":
            answers[qid] = {"type": "noul", "noul": probs["yes"]}
        elif qtype == "choice":
            top = max(probs, key=probs.get)
            answers[qid] = {"type": "choice", "choice": top, "probabilities": probs, "confidence": probs[top]}
        else:  # score
            answers[qid] = {
                "type": "score",
                "score": sum(int(k) * v for k, v in probs.items()),
                "legend": {str(i): str(v) for i, v in enumerate(q.get("criteria") or [])},
                "probabilities": probs,
                "confidence": max(probs.values()),
            }
    return answers


Scorer = Callable[[list[dict[str, Any]]], "tuple[list[dict[str, Any]], dict[str, Any] | None, str]"]


class SemIfBackend(rl.LocalBackend):
    """SemIf decision backend; ``scorer`` is injectable so tests need no model weights."""

    def __init__(self, info: rl.BackendInfo, scorer: Scorer, max_tokens: int, extra: Mapping[str, Any]) -> None:
        self._info = info
        self._scorer = scorer
        self.max_tokens = max_tokens
        self.extra = dict(extra)

    @property
    def info(self) -> rl.BackendInfo:
        return self._info

    def effective_limits(self) -> dict[str, int | None]:
        # ``max_len`` stays None on purpose: SemIf raises instead of truncating, so the Laya
        # usage-ceiling truncation heuristic must never fire. The real limits are in the manifest.
        return {"max_len": None, "head_max_len": None}

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        rows = build_rows(state, questions)
        results, timing, mode = self._scorer(rows)
        answers = assemble_answers(questions, results)
        input_tokens = sum(int(r.get("input_tokens") or 0) for r in results)
        return {
            "model": self._info.model_ref,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "provider_meta": {
                "mode": mode,
                "shared_timing": timing,
                "prompt_sha256": {str(r["id"]): r.get("prompt_sha256") for r in results},
                "prompt_version": results[0].get("prompt_version") if results else None,
                "option_logits": {str(r["id"]): r.get("option_logits") for r in results},
            },
        }


def make_mlx_scorer(model: Any, tokenizer: Any, metadata: dict[str, Any], max_tokens: int) -> Scorer:
    from semif_phase1 import mlx_backend

    def scorer(rows: list[dict[str, Any]]):
        if len(rows) == 1:
            return [mlx_backend.score(model, tokenizer, rows[0], metadata, max_tokens)], None, "direct"
        results, timing = mlx_backend.score_shared(model, tokenizer, rows, metadata, max_tokens)
        return results, timing, "shared"

    return scorer


def load_backend(args: argparse.Namespace) -> SemIfBackend:
    if args.backend == "mock":
        return mock_backend(args)
    from semif_phase1 import mlx_backend

    started = time.perf_counter()
    model, tokenizer, metadata = mlx_backend.load_model(
        args.model, args.revision, args.mlx_bits, cache_limit_mib=args.mlx_cache_limit_mib
    )
    load_seconds = time.perf_counter() - started
    try:
        version = importlib_metadata.version("semif-phase1")
    except importlib_metadata.PackageNotFoundError:
        version = None
    info = rl.BackendInfo(
        name="mlx",
        model_ref=args.model,
        package_version=version,
        device="gpu",
        load_seconds=load_seconds,
        dtype=",".join(metadata.get("dtype") or []) or None,
        batch_size=None,
    )
    extra = {
        "revision": args.revision,
        "mlx_bits": args.mlx_bits,
        "max_input_tokens": args.max_tokens,
        "runtime": {k: metadata.get(k) for k in ("mlx_version", "mlx_lm_version", "transformers_version", "mlx_lm_source", "allocator_cache_limit_bytes", "quantization")},
        "source_artifact_sha256": metadata.get("source_artifact_sha256"),
        "primitive_mapping": {"noul": "yes/no options, noul=P(yes)", "choice": "criteria as options", "score": "ordinal criteria as options, score=E[index]"},
        "noul_options": list(NOUL_OPTIONS),
        "execution_mapping": {"1 question": "direct", ">1 questions": "shared (one state prefill, parallel suffixes)"},
    }
    return SemIfBackend(info, make_mlx_scorer(model, tokenizer, metadata, args.max_tokens), args.max_tokens, extra)


def mock_backend(args: argparse.Namespace) -> SemIfBackend:
    """Deterministic uniform scorer used only to validate the replay pipeline without weights."""

    def scorer(rows: list[dict[str, Any]]):
        results = []
        for row in rows:
            n = len(row["options"])
            if not 2 <= n <= MAX_OPTIONS:
                raise ValueError("options must contain 2-16 entries")
            results.append({
                "id": row["id"],
                "option_ids": [o["id"] for o in row["options"]],
                "probabilities": [1.0 / n] * n,
                "option_logits": [0.0] * n,
                "input_tokens": 1,
                "prompt_sha256": "mock",
                "prompt_version": "mock",
            })
        return results, None, "direct" if len(rows) == 1 else "shared"

    info = rl.BackendInfo(name="mock", model_ref="mock/semif", package_version="test", device="cpu", load_seconds=0.0)
    return SemIfBackend(info, scorer, args.max_tokens, {"max_input_tokens": args.max_tokens})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study-root", type=Path, required=True, help="Path containing benchmark/ and jev-results/")
    p.add_argument("--backend", choices=["mlx", "mock"], default="mlx")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Label used in result directory names")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo or local directory")
    p.add_argument("--revision", default=DEFAULT_REVISION, help="Immutable 40-character commit ID")
    p.add_argument("--mlx-bits", type=int, choices=(4, 8), help="In-memory affine quantization; default keeps source precision")
    p.add_argument("--mlx-cache-limit-mib", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Per-question input limit; SemIf never truncates")
    p.add_argument("--source-run", action="append", help="Source run directory name; repeatable. Default: all five")
    p.add_argument("--output-root", type=Path, help="Default: <study-root>/semif-results")
    p.add_argument("--max-cases", type=int, help="Run only the first N cases from each source run")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--no-analyze", action="store_true")
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    study_root = rl.locate_study_root(args.study_root)
    output_root = (args.output_root or (study_root / "semif-results")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sources = rl.resolve_source_runs(study_root, args.source_run)

    backend = load_backend(args)
    if not args.no_warmup:
        print(f"Warm-up: {rl.warm_up(backend) * 1000:.1f} ms", flush=True)
    print(f"Loaded {backend.info.model_ref}@{args.revision[:8]} via {backend.info.name} in "
          f"{backend.info.load_seconds:.2f}s; source runs={len(sources)}", flush=True)

    extra_manifest = {**backend.extra, "platform_semif": platform.platform()}
    outputs: list[Path] = []
    for source in sources:
        outputs.append(
            rl.run_source(
                study_root=study_root,
                source_run=source,
                backend=backend,
                checkpoint=args.checkpoint,
                output_root=output_root,
                max_cases=args.max_cases,
                resume=args.resume,
                fail_fast=args.fail_fast,
                progress_every=args.progress_every,
                budget_overrides={},
                do_analyze=not args.no_analyze,
                family=FAMILY,
                extra_manifest=extra_manifest,
            )
        )
    print("\nGenerated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
