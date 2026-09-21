# Replaying the Jev study against local Laya

This extension runs the **same recorded benchmark inputs** against a local Laya checkpoint and compares the result with the five preserved Jev runs.

It does not regenerate the benchmark. It replays the exact `cases.jsonl` files already stored under `results/`, which guarantees that both providers receive byte-identical states, question instructions, criteria, option order, target metadata, and case IDs.

## Why two Laya checkpoints matter

Use these runs as separate experiments:

1. **`typed-decisions` — primary specialist comparison.** It is the checkpoint explicitly fine-tuned for typed-decision workflows and is the closest public analogue to Jev's API behavior.
2. **`base` — generalization control.** It tests what the general English Laya checkpoint can do without typed-decisions specialization.

Do not merge their output or present a routed/best-of result as one model. The comparison script keeps each checkpoint/backend in a separate result namespace.

## Expected limitations are part of the benchmark

The exact Jev suite deliberately includes inputs beyond Laya's documented comfort zone:

- long states reaching about 21K tokens in the Jev run;
- up to 128 questions in one request;
- Choice cardinalities up to 255;
- Laya typed-decisions has a 1,024-token total context and its published guidance recommends keeping Choice under roughly 20 options.

The replay runner therefore records unsupported/overflow cases as failures instead of shortening states, deleting candidates, or silently changing the questions. Coverage is a first-class comparison metric.

Optional `--max-len` and `--head-max-len` switches exist only for explicitly labelled non-baseline experiments. Do not use them in the primary as-shipped comparison unless you clearly report the modification.

## Recommended platform for this Mac

On Apple Silicon, use the independent `laya-mlx` runtime and the published FP16 conversions. It runs locally and exposes the same `predict(state, questions)` contract.

Requirements:

- macOS 14+
- Apple Silicon
- Python 3.11+
- enough disk space for an approximately 843 MB checkpoint plus caches

The first load downloads model files; subsequent runs use the local cache.

## Install into the existing study bundle

From the downloaded extension directory:

```bash
./install_into_study.sh /Users/arogov/Downloads/jev-system-one-study-complete
```

The installer also detects an archive layout where the actual root is nested at:

```text
.../jev-system-one-study-complete/jev-system-one-study/
```

It copies only the Laya adapter/config/docs/tests and appends one marked README section. It does not alter the five Jev result folders.

## Environment setup

Use the benchmark's existing Python 3.12 virtual environment if available:

```bash
cd /Users/arogov/Downloads/jev-system-one-study-complete
source benchmark/.venv/bin/activate
```

If the bundle root is nested, `cd` to the directory containing `benchmark/` and `results/`.

Install the native runtime:

```bash
python -m pip install -r benchmark/requirements-laya-mlx.txt
```

Verify imports and adapter tests:

```bash
python -c 'import laya_mlx; print("laya_mlx import OK")'
python -m unittest benchmark/tests/test_laya_adapter.py -v
```

## Fast compatibility smoke test

Run ten cases from each of the five source runs without analysis:

```bash
python benchmark/scripts/run_laya_benchmark.py \
  --study-root "$PWD" \
  --backend mlx \
  --checkpoint typed-decisions \
  --max-cases 10 \
  --no-analyze \
  --output-root "$PWD/laya-smoke-results"
```

This verifies loading, schema compatibility, raw capture, and error recording. Delete `laya-smoke-results/` after checking it; it is intentionally not the full comparison dataset.

## Primary full run: typed-decisions checkpoint

```bash
./benchmark/scripts/run_laya_all.sh \
  --study-root "$PWD" \
  --backend mlx \
  --checkpoint typed-decisions
```

Equivalent explicit commands:

```bash
python benchmark/scripts/run_laya_benchmark.py \
  --study-root "$PWD" \
  --backend mlx \
  --checkpoint typed-decisions

python benchmark/scripts/compare_jev_laya.py \
  --study-root "$PWD" \
  --backend mlx \
  --checkpoint typed-decisions
```

The runner processes the five source runs sequentially with one resident model. Local inference timing excludes model loading and the one warm-up request but includes request formatting, tokenization, model execution, calibration, and output normalization performed by the runtime.

## Generalization control: base checkpoint

Run separately:

```bash
./benchmark/scripts/run_laya_all.sh \
  --study-root "$PWD" \
  --backend mlx \
  --checkpoint base
```

This writes different output directories and cannot overwrite the typed-decisions results.

## Optional runtime parameters

The MLX runtime defaults are the baseline. Useful explicitly reported variants include:

```bash
--batch-size 32
--dtype float32
--device gpu
```

FP16/default should be the primary Mac result. `float32` is useful as a numerical-fidelity sensitivity run but is not the same performance configuration.

To resume an interrupted run:

```bash
python benchmark/scripts/run_laya_benchmark.py ... --resume
```

A recorded failed case counts as completed so that documented compatibility failures are preserved. Use a fresh `--output-root` to retry with different budgets or runtime settings.

## Generated directories

```text
laya-results/
├── run_20260919T201050Z_core__laya_typed-decisions_mlx/
├── run_20260919T201739Z_core__laya_typed-decisions_mlx/
├── run_20260919T202139Z_full__laya_typed-decisions_mlx/
├── run_20260919T203758Z_calibration_full__laya_typed-decisions_mlx/
└── run_20260919T211437Z_semantic_full__laya_typed-decisions_mlx/

comparisons/
└── laya_typed-decisions_mlx/
    ├── JEV_VS_LAYA_REPORT.md
    ├── comparison_summary.csv
    ├── comparison_summary.json
    └── scaling_rows.csv
```

Each Laya run contains:

- `manifest.json` — checkpoint/backend/version/device, source manifest, case hashes, runtime settings;
- `preflight.json` — maximum questions, cardinality, state size and experiment counts before inference;
- `cases.jsonl` — exact selected source cases;
- `raw.jsonl` — one record per attempted Laya request, including failures and tracebacks;
- `compatibility.json` — success/failure counts by experiment and error type;
- `normalized.csv`, `summary.json`, `REPORT.md`, `figures/` — generated by the existing Jev analysis pipeline for successful decisions.

## Fairness rules

1. **Exact inputs first.** Do not truncate or rewrite cases in the primary run.
2. **Report coverage.** A model that cannot accept a case has not answered it.
3. **Compare common successful decisions for quality**, and report all-source coverage beside them.
4. **Do not compare absolute latency as same-hardware performance.** Jev is a remote API and Laya is local. Compare local Laya latency on its own terms; use scaling shape cautiously.
5. **Label the specialist.** `laya-typed-decisions` is trained on specific workflows and is not equivalent to a general zero-shot model.
6. **Keep checkpoints separate.** Base, multilingual and typed-decisions results are distinct experiments.
7. **Preserve failures.** Context/cardinality limitations are substantive findings, not rows to delete.

## PyTorch fallback

On a Linux/CUDA host or where the reference implementation is preferred:

```bash
python -m pip install -r benchmark/requirements-laya-torch.txt

./benchmark/scripts/run_laya_all.sh \
  --study-root "$PWD" \
  --backend torch \
  --checkpoint typed-decisions \
  --device cuda
```

For Apple Silicon, MLX is the recommended primary path; PyTorch/MPS can be used as a secondary fidelity check.

## Compatibility classification (v1.0.2)

Every Laya raw request now carries explicit compatibility metadata:

```text
compatibility_status: full_input | truncated_input | unsupported_cardinality | backend_error
comparable: true | false
compatibility_reason: ...
compatibility_details: ...
```

`full_input` is the only status included in apples-to-apples Jev-vs-Laya quality metrics. A completed Laya request that reaches the configured `max_len` input ceiling is tagged `truncated_input`; it remains in raw data and latency/coverage statistics but is excluded from full-input quality comparisons. The known high-cardinality token-budget exception is tagged `unsupported_cardinality` rather than a generic backend failure.

The truncation detector uses Laya's reported aggregate `usage.input_tokens` and the effective `agent.cfg.max_len`. For multi-question requests it compares against `max_len * question_count`. This definitively catches fully saturated request rows; partial per-question saturation may be conservative/undetected if aggregate usage remains below the ceiling, so the raw usage and effective limits are always preserved for audit.
