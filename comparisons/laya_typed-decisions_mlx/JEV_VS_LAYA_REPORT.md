# Jev vs Laya (typed-decisions, mlx)

The comparison replays byte-identical `cases.jsonl` inputs from the five recorded Jev runs. Laya coverage failures are reported rather than silently dropping or mutating unsupported cases. Absolute latency is not hardware-equivalent: Jev is a remote service, whereas Laya runs locally; scaling shape and local latency are reported separately.

## run_20260919T201050Z_core

- Laya completed requests: **303/303** (100.00%).
- Fully comparable/full-input requests: **297/303** (98.02%).
- Compatibility status: `{"full_input": 297, "truncated_input": 6}`.
- Common normalized decisions after compatibility filtering: **441**.
- Accuracy — Jev: **0.9887**; Laya: **0.8549**.
- Binary exact-probability MAE — Jev: **n/a**; Laya: **n/a**.
- Full-distribution TV — Jev: **n/a**; Laya: **n/a**.
- Local/remote wall p50 — Jev: **332.5 ms**; Laya: **12.8 ms**.

## run_20260919T201739Z_core

- Laya completed requests: **26/26** (100.00%).
- Fully comparable/full-input requests: **20/26** (76.92%).
- Compatibility status: `{"full_input": 20, "truncated_input": 6}`.
- Common normalized decisions after compatibility filtering: **140**.
- Accuracy — Jev: **1.0000**; Laya: **0.7571**.
- Binary exact-probability MAE — Jev: **n/a**; Laya: **n/a**.
- Full-distribution TV — Jev: **n/a**; Laya: **n/a**.
- Local/remote wall p50 — Jev: **323.8 ms**; Laya: **66.5 ms**.

## run_20260919T202139Z_full

- Laya completed requests: **30/31** (96.77%).
- Fully comparable/full-input requests: **21/31** (67.74%).
- Compatibility status: `{"full_input": 21, "truncated_input": 9, "unsupported_cardinality": 1}`.
- Common normalized decisions after compatibility filtering: **268**.
- Accuracy — Jev: **1.0000**; Laya: **0.6567**.
- Binary exact-probability MAE — Jev: **n/a**; Laya: **n/a**.
- Full-distribution TV — Jev: **n/a**; Laya: **n/a**.
- Local/remote wall p50 — Jev: **329.6 ms**; Laya: **88.0 ms**.

Failure types: `{"ValueError": 1}`.

## run_20260919T203758Z_calibration_full

- Laya completed requests: **5810/5810** (100.00%).
- Fully comparable/full-input requests: **5810/5810** (100.00%).
- Compatibility status: `{"full_input": 5810}`.
- Common normalized decisions after compatibility filtering: **5810**.
- Accuracy — Jev: **0.8063**; Laya: **0.5941**.
- Binary exact-probability MAE — Jev: **0.0793**; Laya: **0.2892**.
- Full-distribution TV — Jev: **0.3010**; Laya: **0.3427**.
- Local/remote wall p50 — Jev: **321.4 ms**; Laya: **13.6 ms**.

## run_20260919T211437Z_semantic_full

- Laya completed requests: **1110/1110** (100.00%).
- Fully comparable/full-input requests: **1110/1110** (100.00%).
- Compatibility status: `{"full_input": 1110}`.
- Common normalized decisions after compatibility filtering: **1110**.
- Accuracy — Jev: **0.9036**; Laya: **0.4405**.
- Binary exact-probability MAE — Jev: **n/a**; Laya: **n/a**.
- Full-distribution TV — Jev: **0.2407**; Laya: **0.3940**.
- Local/remote wall p50 — Jev: **332.3 ms**; Laya: **18.3 ms**.

Semantic top-probability gap versus reference — Jev: **0.2051**; Laya: **-0.2605**.

## Interpretation constraints

- The typed-decisions checkpoint is a specialist fine-tuned on four published workflows; results must be labelled as such.
- Laya has a much smaller context/head budget than Jev. Truncated-input and unsupported-cardinality requests are reported as compatibility outcomes and excluded from full-input quality comparisons.
- Jev upstream service time and Laya local end-to-end inference time are not measured on the same hardware.
- Use the base checkpoint as an additional generalization control; do not silently mix routed/base/specialist results.
