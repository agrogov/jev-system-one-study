# Laya comparison publication overlay

This directory is designed to be copied into the existing `agrogov/jev-system-one-study` repository without replacing the original Jev results.

## Published reports

- `LAYA_SYSTEM_ONE_BLACKBOX_REPORT.md` — Laya-only study, using the same five source benchmark suites as the Jev report.
- `JEV_VS_LAYA_COMPARISON_REPORT.md` — controlled aligned comparison against the recorded Jev results.

## Evidence

- `../laya-results` — five **primary full-run** result directories. These are the canonical Laya evidence.
- `../comparisons/laya_typed-decisions_mlx` — machine-readable comparison summary and scaling rows from the benchmark tooling.
- `laya_computed_statistics.json` — recomputed integrated statistics and confidence intervals used by the two reports.
- `../benchmark` — Laya replay adapter, configuration, dependencies, comparison script, wrapper, and regression tests.

The smoke/validation data overlap the primary datasets and must **not** be pooled with the primary runs as independent samples.

## Primary model/runtime

- checkpoint: `aac6fef/laya-typed-decisions-mlx`
- backend: MLX / `laya-mlx` 0.1.0
- dtype: FP16
- batch size: 16
- effective max input: 1,024 tokens
- source benchmark seed: 260919

## Repository integration

Copy the contents of this overlay into the root of the Jev study repository. Existing Jev files remain untouched because all Laya evidence uses distinct paths.
