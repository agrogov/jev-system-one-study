# Jev System One Black-Box Study — Complete Reproducibility Bundle

This repository combines the complete research artifact in one self-contained tree: a black-box study of TypeSafe Jev / System One, plus a controlled replay of the same benchmark against the open Laya typed-decisions model.

```text
jev-system-one-study/
├── README.md
├── benchmark/       # unified benchmark v1.0.0 (single codebase) + Laya replay adapter and tests
├── report/          # Jev report, Laya report, Jev-vs-Laya comparison, and independent analysis
├── jev-results/     # five recorded Jev experiment folders used by the reports
├── laya-results/    # five recorded Laya replay folders (same source cases as jev-results)
└── comparisons/     # machine-readable Jev-vs-Laya comparison summary and scaling rows
```

## Start here

1. Read `report/JEV_SYSTEM_ONE_BLACKBOX_REPORT.md` for the Jev findings and reverse-engineered architecture.
2. Read `report/LAYA_OPEN_WEIGHT_REPORT.md` for the Laya-only study on the same five suites.
3. Read `report/JEV_VS_LAYA_COMPARISON_REPORT.md` for the controlled, aligned Jev-vs-Laya comparison.
4. Read `benchmark/README.md` for Python setup, build, installation, suite documentation, exact reproduction commands, and output formats.
5. Read `benchmark/LAYA_COMPARISON.md` to rerun the Laya replay locally.
6. Inspect `jev-results/` and `laya-results/` for the complete raw evidence (`raw.jsonl`, `cases.jsonl`, manifests, normalized data, reports, and figures).

Per-side indexes: `report/JEV_RESULTS_README.md` and `report/LAYA_RESULTS_README.md`.

## Included Jev study runs (`jev-results/`)

- `run_20260919T201050Z_core` — 303-request concurrent architecture/robustness core run.
- `run_20260919T201739Z_core` — 26-request isolated core architecture scaling run.
- `run_20260919T202139Z_full` — 31-request isolated full boundary scaling run.
- `run_20260919T203758Z_calibration_full` — 5,810-request exact probabilistic calibration run.
- `run_20260919T211437Z_semantic_full` — 1,110-request semantic calibration run.

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

## Headline Jev-vs-Laya results

Accuracy on aligned, full-input decisions (details in `report/JEV_VS_LAYA_COMPARISON_REPORT.md`):

| Suite | Aligned decisions | Jev | Laya |
|---|---:|---:|---:|
| architecture core | 441 | 98.87% | 85.49% |
| isolated core scaling | 140 | 100.00% | 75.71% |
| full boundary scaling | 268 | 100.00% | 65.67% |
| probability calibration | 4,600 | 80.63% | 59.41% |
| semantic | 1,110 | 90.36% | 44.05% |

- **Paradigm:** Laya reproduces the typed, non-autoregressive Choice/Noul/Score abstraction and is very fast for short local calls (p50 ≈ 13–18 ms on Apple Silicon).
- **Exact probabilities:** on 4,810 binary cases, MAE is 0.0793 for Jev versus 0.2892 for Laya.
- **Multi-question scaling:** Jev's backend time stays nearly flat from 1 to 64 questions; Laya's latency grows roughly linearly (84 ms at 1 question, about 9.7 s at 128).
- **Choice cardinality:** Laya is correct through 16 options, fails at 32–128, and rejects 255; Jev solved the task through 255.
- **Long context:** Laya's 1,024-token ceiling truncates long states; Jev recovered evidence across the ~21K-token probe.
- **Semantic calibration:** Laya is under-confident (mean top probability 0.4165 vs 0.6770 reference), whereas Jev is sharper than the references.

Laya latency is measured locally and Jev is a remote service, so absolute latency is descriptive rather than hardware-normalized. The Laya numbers are for the typed-decisions specialist checkpoint only.

## Recorded-run metadata

Each run directory under `jev-results/` and `laya-results/` contains its own `manifest.json` with the exact configuration used for that run (Laya manifests also store SHA-256 digests of the replayed `cases.jsonl`). The raw evidence is preserved directly in each run's `raw.jsonl`; no duplicate legacy result archives are included.

The `benchmark/` directory supersedes all earlier development benchmark versions and contains every experiment family used by these runs. A prebuilt pure-Python wheel is also included under `benchmark/dist/`; source remains authoritative.
