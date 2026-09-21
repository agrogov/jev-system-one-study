# Laya study index

This file indexes the Laya companion study. All Laya evidence lives at paths distinct from the original Jev results (`../jev-results`).

## Published reports

- `LAYA_OPEN_WEIGHT_REPORT.md` — Laya-only study, using the same five source benchmark suites as the Jev report.
- `JEV_VS_LAYA_COMPARISON_REPORT.md` — controlled aligned comparison against the recorded Jev results.
- `JEV_LAYA_SEMIF_COMPARISON_REPORT.md` — three-way comparison including the SemIf (Qwen3.5-4B) replay.

## Evidence

- `../laya-results` — five **primary full-run** result directories, one per Jev source run in `../jev-results`. These are the canonical Laya evidence.
- `../comparisons/laya_typed-decisions_mlx` — machine-readable comparison summary and scaling rows from the benchmark tooling.
- `laya_computed_statistics.json` — recomputed integrated statistics and confidence intervals used by the two reports.
- `../benchmark` — Laya replay adapter, configuration, dependencies, comparison script, wrapper, and regression tests.

## Primary model/runtime

- checkpoint: `aac6fef/laya-typed-decisions-mlx`
- backend: MLX / `laya-mlx` 0.1.0
- dtype: FP16
- batch size: 16
- effective max input: 1,024 tokens
- source benchmark seed: 260919
