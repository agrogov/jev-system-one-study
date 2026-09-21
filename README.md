# Jev System One Black-Box Study — Complete Reproducibility Bundle

This repository combines the complete research artifact in one self-contained tree: a black-box study of TypeSafe Jev / System One, plus controlled replays of the same benchmark against two open typed-decision systems: the Laya typed-decisions model and SemIf (Qwen3.5-4B).

```text
jev-system-one-study/
├── README.md
├── benchmark/       # unified benchmark v1.0.0 (single codebase) + Laya and SemIf replay adapters and tests
├── report/          # Jev, Laya and SemIf reports, comparisons, and independent analysis
├── jev-results/     # five recorded Jev experiment folders used by the reports
├── laya-results/    # five recorded Laya replay folders (same source cases as jev-results)
├── semif-results/   # five recorded SemIf replay folders (same source cases as jev-results)
└── comparisons/     # machine-readable Jev-vs-Laya, Jev-vs-SemIf and three-way comparisons
```

## Start here

1. Read `report/JEV_SYSTEM_ONE_BLACKBOX_REPORT.md` for the Jev findings and reverse-engineered architecture.
2. Read `report/LAYA_OPEN_WEIGHT_REPORT.md` for the Laya-only study on the same five suites.
3. Read `report/JEV_VS_LAYA_COMPARISON_REPORT.md` for the controlled, aligned Jev-vs-Laya comparison.
4. Read `report/SEMIF_OPEN_MODEL_REPORT.md` for the SemIf (Qwen3.5-4B) study on the same five suites.
5. Read `report/JEV_VS_SEMIF_COMPARISON_REPORT.md` for the controlled, aligned Jev-vs-SemIf comparison.
6. Read `report/JEV_LAYA_SEMIF_COMPARISON_REPORT.md` for Jev vs Laya vs SemIf on identical decisions.
7. Read `benchmark/README.md` for Python setup, build, installation, suite documentation, exact reproduction commands, and output formats.
8. Read `benchmark/LAYA_COMPARISON.md` and `benchmark/SEMIF_COMPARISON.md` to rerun the Laya and SemIf replays locally.
9. Inspect `jev-results/`, `laya-results/` and `semif-results/` for the complete raw evidence (`raw.jsonl`, `cases.jsonl`, manifests, normalized data, reports, and figures).

Per-side indexes: `report/JEV_RESULTS_README.md`, `report/LAYA_RESULTS_README.md` and `report/SEMIF_RESULTS_README.md`.

## Included Jev study runs (`jev-results/`)

| Jev run | Requests | Completed | Purpose |
|---|---:|---:|---|
| `run_20260919T201050Z_core` | 303 | 303 | concurrent architecture/robustness core |
| `run_20260919T201739Z_core` | 26 | 26 | isolated core architecture scaling |
| `run_20260919T202139Z_full` | 31 | 31 | isolated full boundary scaling |
| `run_20260919T203758Z_calibration_full` | 5,810 | 5,810 | exact probabilistic calibration |
| `run_20260919T211437Z_semantic_full` | 1,110 | 1,110 | semantic calibration |

## Included Laya study runs (`laya-results/`)

Laya replays the byte-identical `cases.jsonl` of each Jev run above; each folder is named `<jev run>__laya_typed-decisions_mlx`. Together they cover **7,280 requests**.

| Laya run | Requests | Completed | Full-input comparable |
|---|---:|---:|---:|
| `run_20260919T201050Z_core__laya_typed-decisions_mlx` | 303 | 303 | 297 |
| `run_20260919T201739Z_core__laya_typed-decisions_mlx` | 26 | 26 | 20 |
| `run_20260919T202139Z_full__laya_typed-decisions_mlx` | 31 | 30 | 21 |
| `run_20260919T203758Z_calibration_full__laya_typed-decisions_mlx` | 5,810 | 5,810 | 5,810 |
| `run_20260919T211437Z_semantic_full__laya_typed-decisions_mlx` | 1,110 | 1,110 | 1,110 |

Laya runtime: `aac6fef/laya-typed-decisions-mlx`, MLX (`laya-mlx` 0.1.0), FP16, question batch size 16, effective input limit 1,024 tokens, source benchmark seed 260919. Laya-specific incompatibilities (truncated input, unsupported cardinality) are recorded as outcomes, not repaired.

## Included SemIf study runs (`semif-results/`)

SemIf ([TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf), an independent open-model reproduction of the Jev interface pattern) replays the byte-identical `cases.jsonl` of each Jev run; each folder is named `<jev run>__semif_qwen3-5-4b_mlx`. Together they cover **7,280 requests**.

| SemIf run | Requests | Completed | Full-input comparable |
|---|---:|---:|---:|
| `run_20260919T201050Z_core__semif_qwen3-5-4b_mlx` | 303 | 300 | 300 |
| `run_20260919T201739Z_core__semif_qwen3-5-4b_mlx` | 26 | 23 | 23 |
| `run_20260919T202139Z_full__semif_qwen3-5-4b_mlx` | 31 | 27 | 27 |
| `run_20260919T203758Z_calibration_full__semif_qwen3-5-4b_mlx` | 5,810 | 5,810 | 5,810 |
| `run_20260919T211437Z_semantic_full__semif_qwen3-5-4b_mlx` | 1,110 | 1,110 | 1,110 |

SemIf runtime: `Qwen/Qwen3.5-4B` (revision `851bf6e8`), SemIf commit `ca3ba65`, native MLX (`mlx` 0.32.2, `mlx-lm` 0.32.0), source precision (BF16), 65,536-token input limit (SemIf never truncates), 16-option limit per decision, `direct` scoring for one question and `shared` scoring for several, source benchmark seed 260919. All ten requests that did not complete are Choice questions with more than 16 options (`unsupported_cardinality`); nothing was repaired. The `noul` and `score` mappings (yes/no options with P(yes), ordinal options with expected index) are this study's adaptation and are recorded in each manifest.

## Headline results: Jev vs Laya vs SemIf

All three systems are scored on **identical decisions** (Jev completed them and both local systems consumed the full request). Full analysis with bootstrap confidence intervals is in `report/JEV_LAYA_SEMIF_COMPARISON_REPORT.md`; the generated tables are in `comparisons/three_way/THREE_WAY_REPORT.md`.

Accuracy:

| Suite | Scored decisions | Jev | Laya | SemIf |
|---|---:|---:|---:|---:|
| architecture core | 438 | 98.86% | 86.07% | 100.00% |
| isolated core scaling | 137 | 100.00% | 77.37% | 100.00% |
| full boundary scaling | 265 | 100.00% | 66.42% | 92.45% |
| probability calibration | 4,600 | 80.63% | 59.41% | 74.13% |
| semantic | 1,110 | 90.36% | 44.05% | 69.10% |

Probability and confidence quality:

| Metric | Jev | Laya | SemIf |
|---|---:|---:|---:|
| Binary exact-probability MAE (4,810 cases, lower is better) | **0.0793** | 0.2892 | 0.2526 |
| Mean TV to exact 3-class posterior (1,000 cases, lower is better) | 0.3010 | 0.3427 | **0.1831** |
| Semantic mean TV to reference (lower is better) | **0.2407** | 0.3940 | 0.3012 |
| Semantic mean top probability (reference 0.677) | 0.882 | 0.417 | 0.851 |

Coverage and scaling:

| | Jev | Laya | SemIf |
|---|---|---|---|
| Requests consumed in full (architecture suites) | 360/360 | 338/360 | 350/360 |
| Choice options supported | up to 255 | up to 16 (32+ fail) | up to 16 (32+ rejected) |
| Longest context handled | ~21K tokens | truncated at 1,024 tokens | ~25K tokens, full accuracy |
| Server time, 1 → 128 questions | 99 → 167 ms (flat) | 81 → 9,697 ms (~120×) | 514 → 14,301 ms (~28×) |
| Median wall latency, semantic suite | 332 ms (remote) | 18 ms (local) | 209 ms (local) |

- **Paradigm:** both open systems reproduce the typed, non-autoregressive Choice/Noul/Score abstraction; neither reproduces Jev's serving behavior.
- **Semantics:** the general 4B SemIf model is far more accurate than the Laya specialist checkpoint (69.1% vs 44.05%) but well behind Jev (90.36%); Laya's probabilities are diffuse, SemIf's and Jev's are sharp.
- **Exact probabilities:** Jev is clearly best on binary probabilities; SemIf is over-confident toward "yes" (bias +0.20) but ranks cases well and is closest to the exact 3-class posterior.
- **Scaling:** Jev's backend time stays nearly flat with question count and context; both local systems grow with them, SemIf far more gently than Laya thanks to shared-state execution.

Latency is measured locally for Laya and SemIf and remotely for Jev, so absolute latency is descriptive rather than hardware-normalized. The Laya numbers are for the typed-decisions specialist checkpoint and the SemIf numbers for one model, one prompt version and one runtime only.

## Recorded-run metadata

Each run directory under `jev-results/`, `laya-results/` and `semif-results/` contains its own `manifest.json` with the exact configuration used for that run (Laya and SemIf manifests also store SHA-256 digests of the replayed `cases.jsonl`). The raw evidence is preserved directly in each run's `raw.jsonl`; no duplicate legacy result archives are included.

The `benchmark/` directory supersedes all earlier development benchmark versions and contains every experiment family used by these runs. A prebuilt pure-Python wheel is also included under `benchmark/dist/`; source remains authoritative.
