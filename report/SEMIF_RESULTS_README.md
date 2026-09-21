# SemIf study index

This file indexes the SemIf companion study. All SemIf evidence lives at paths distinct from the Jev (`../jev-results`) and Laya (`../laya-results`) results.

## Published reports

- `SEMIF_OPEN_MODEL_REPORT.md` — SemIf-only study, using the same five source benchmark suites as the Jev report.
- `JEV_VS_SEMIF_COMPARISON_REPORT.md` — controlled aligned comparison against the recorded Jev results.
- `JEV_LAYA_SEMIF_COMPARISON_REPORT.md` — Jev vs Laya vs SemIf on identical decisions, with bootstrap intervals.
- `../comparisons/semif_qwen3-5-4b_mlx/JEV_VS_SEMIF_REPORT.md` and `../comparisons/three_way/THREE_WAY_REPORT.md` — the generated tables behind those reports.

## Evidence

- `../semif-results` — five **primary full-run** result directories, one per Jev source run in `../jev-results`. These are the canonical SemIf evidence.
- `../comparisons/semif_qwen3-5-4b_mlx` — machine-readable Jev-vs-SemIf comparison summary and scaling rows.
- `../comparisons/three_way` — machine-readable three-way summary (`three_way_summary.json` / `.csv`).
- `semif_computed_statistics.json` — recomputed integrated statistics and paired bootstrap confidence intervals used by the reports (`../benchmark/scripts/compute_semif_statistics.py`).
- `../benchmark` — SemIf replay adapter, configuration, requirements, comparison scripts, wrapper, and regression tests.

## Primary model/runtime

- repository: `TheoLeeCJ/SemIf` at commit `ca3ba65f142967030ecb453346e94d6f476a69df`
- model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- backend: SemIf native MLX (`mlx` 0.32.2, `mlx-lm` 0.32.0)
- precision: source (BF16 with some FP32 parameters), no quantization
- input limit: 65,536 tokens per question (no truncation; SemIf refuses over-long input)
- option limit: 16 per decision (SemIf answer slots A-P)
- execution: `direct` for one question, `shared` (one state prefill) for several
- source benchmark seed: 260919
