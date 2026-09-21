# Jev System One Black-Box Study — Complete Reproducibility Bundle

This archive combines the complete research artifact in one self-contained tree:

```text
jev-system-one-study/
├── README.md
├── benchmark/       # unified benchmark v1.0.0 (single codebase)
├── report/          # integrated research report and independent analysis
└── results/         # five self-contained recorded experiment folders used by the report
```

## Start here

1. Read `report/JEV_SYSTEM_ONE_BLACKBOX_REPORT.md` for the scientific findings and reverse-engineered architecture.
2. Read `benchmark/README.md` for Python setup, build, installation, suite documentation, exact reproduction commands, and output formats.
3. Inspect `jev-results` for the complete raw API evidence (`raw.jsonl`, `cases.jsonl`, manifests, normalized data, reports, and figures).

## Included study runs

- `jev-results` — 303-request concurrent architecture/robustness core run.
- `jev-results` — 26-request isolated core architecture scaling run.
- `jev-results` — 31-request isolated full boundary scaling run.
- `jev-results` — 5,810-request exact probabilistic calibration run.
- `jev-results` — 1,110-request semantic calibration run.

The `benchmark/` directory supersedes all earlier development benchmark versions and contains every experiment family used by these runs.

## Recorded-run metadata

Each directory under `jev-results` contains its own `manifest.json` with the exact benchmark configuration used for that run. The raw API evidence is preserved directly in each run's `raw.jsonl`; no duplicate legacy result archives are included.

A prebuilt pure-Python wheel is also included under `benchmark/dist/`; source remains authoritative.
