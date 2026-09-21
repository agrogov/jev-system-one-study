# Replaying the Jev study against SemIf (Qwen3.5-4B)

`scripts/run_semif_benchmark.py` runs the **same recorded benchmark inputs** through
[SemIf](https://github.com/TheoLeeCJ/SemIf)'s native MLX backend and writes the same raw-result schema as
`jevbench`, so the existing analysis and report modules work unchanged. It replays the exact `cases.jsonl`
files under `jev-results/`; nothing is regenerated.

SemIf is an independent open-model reproduction of the Jev *interface pattern* (typed options read straight
from answer-slot logits). Jev primitives are mapped as `choice` -> criteria as options, `score` -> ordinal
criteria as options with `score` = expected index, `noul` -> a fixed yes/no decision with `noul` = P(yes). The
mapping and the authored yes/no wording are recorded in every manifest (`primitive_mapping`, `noul_options`).

## Requirements

Apple Silicon Mac with Metal, Python 3.10+, Git, and about 9 GB free disk for the checkpoint.

```bash
cd benchmark
python3 -m venv .venv-semif
source .venv-semif/bin/activate
pip install -r requirements-semif-mlx.txt
python -m unittest discover -s tests -v
```

The first run downloads `Qwen/Qwen3.5-4B` at the pinned revision into the Hugging Face cache.

## Fast smoke test (10 cases per source run)

```bash
cd ..   # study root, the directory containing benchmark/ and jev-results/
python benchmark/scripts/run_semif_benchmark.py \
  --study-root "$PWD" --max-cases 10 --no-analyze \
  --output-root /tmp/semif-smoke
```

## Full run and comparison

```bash
benchmark/scripts/run_semif_all.sh --study-root "$PWD"
```

This writes `semif-results/` (about 30 minutes on an M4 Max), then
`comparisons/semif_qwen3-5-4b_mlx/JEV_VS_SEMIF_REPORT.md`. Use `--resume` to continue an interrupted run.

Build the Jev / Laya / SemIf table (requires `laya-results/` as well):

```bash
python benchmark/scripts/compare_three_way.py --study-root "$PWD"
```

## Limits are reported, never repaired

- More than 16 options -> `unsupported_cardinality` (Choice ladder at 32/64/128/255).
- SemIf never truncates. The per-question input limit defaults to 65,536 tokens (`--max-tokens`) rather than SemIf's
  4,096 so that Jev's ~21K-token probe is actually attempted; anything beyond it would be `input_too_long`.
- `--mlx-bits 4|8` runs an in-memory quantized variant and must be labelled as a separate experiment.
