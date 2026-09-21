# Jev System One research report

This directory contains the integrated black-box study of TypeSafe Jev/System One.

## Contents

- `JEV_SYSTEM_ONE_BLACKBOX_REPORT.md` — integrated research paper.
- `computed_statistics.json` — machine-readable recomputed metrics and confidence intervals.
- `reproduce_analysis.py` — analysis/report script, now using bundle-relative paths.
- `requirements-analysis.txt` — dependencies required by the integrated analysis script.
- `figures/` — publication figures generated from the five recorded result folders.

The five self-contained recorded run directories are available at `../results/`. They are the sole experimental inputs used by `reproduce_analysis.py`; no duplicate legacy result archives are required.

## Recompute the integrated analysis

From the bundle root, create/activate a Python environment and install:

```bash
python -m pip install -r report/requirements-analysis.txt
python report/reproduce_analysis.py
```

The script reads the five directories under `results/` and rewrites the report statistics/figures in this directory.

For the benchmark itself, installation and exact study-run reproduction commands are documented in `../benchmark/README.md`.
