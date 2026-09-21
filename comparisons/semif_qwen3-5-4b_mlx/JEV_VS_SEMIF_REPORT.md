# Jev vs SemIf (qwen3.5-4b, mlx)

The comparison replays byte-identical `cases.jsonl` inputs from the five recorded Jev runs through SemIf's native MLX option readout (Qwen3.5-4B). SemIf coverage failures (more than 16 options, prompts over the input limit) are reported rather than silently dropped or repaired. Jev `noul` questions are replayed as a fixed yes/no decision with `noul = P(yes)`; that option wording is authored by this study, not by SemIf. Absolute latency is not hardware-equivalent: Jev is a remote service, SemIf runs locally.

## run_20260919T201050Z_core

- SemIf completed requests: **300/303** (99.01%).
- Fully comparable/full-input requests: **300/303** (99.01%).
- Compatibility status: `{"full_input": 300, "unsupported_cardinality": 3}`.
- Common normalized decisions after compatibility filtering: **444**.
- Accuracy — Jev: **0.9887**; SemIf: **1.0000**.
- Binary exact-probability MAE — Jev: **n/a**; SemIf: **n/a**.
- Full-distribution TV — Jev: **n/a**; SemIf: **n/a**.
- Local/remote wall p50 — Jev: **332.5 ms**; SemIf: **163.9 ms**.

Failure types: `{"ValueError": 3}`.

## run_20260919T201739Z_core

- SemIf completed requests: **23/26** (88.46%).
- Fully comparable/full-input requests: **23/26** (88.46%).
- Compatibility status: `{"full_input": 23, "unsupported_cardinality": 3}`.
- Common normalized decisions after compatibility filtering: **143**.
- Accuracy — Jev: **1.0000**; SemIf: **1.0000**.
- Binary exact-probability MAE — Jev: **n/a**; SemIf: **n/a**.
- Full-distribution TV — Jev: **n/a**; SemIf: **n/a**.
- Local/remote wall p50 — Jev: **323.8 ms**; SemIf: **554.0 ms**.

Failure types: `{"ValueError": 3}`.

## run_20260919T202139Z_full

- SemIf completed requests: **27/31** (87.10%).
- Fully comparable/full-input requests: **27/31** (87.10%).
- Compatibility status: `{"full_input": 27, "unsupported_cardinality": 4}`.
- Common normalized decisions after compatibility filtering: **274**.
- Accuracy — Jev: **1.0000**; SemIf: **0.9270**.
- Binary exact-probability MAE — Jev: **n/a**; SemIf: **n/a**.
- Full-distribution TV — Jev: **n/a**; SemIf: **n/a**.
- Local/remote wall p50 — Jev: **329.6 ms**; SemIf: **1504.9 ms**.

Failure types: `{"ValueError": 4}`.

## run_20260919T203758Z_calibration_full

- SemIf completed requests: **5810/5810** (100.00%).
- Fully comparable/full-input requests: **5810/5810** (100.00%).
- Compatibility status: `{"full_input": 5810}`.
- Common normalized decisions after compatibility filtering: **5810**.
- Accuracy — Jev: **0.8063**; SemIf: **0.7413**.
- Binary exact-probability MAE — Jev: **0.0793**; SemIf: **0.2526**.
- Full-distribution TV — Jev: **0.3010**; SemIf: **0.1831**.
- Local/remote wall p50 — Jev: **321.4 ms**; SemIf: **206.1 ms**.

## run_20260919T211437Z_semantic_full

- SemIf completed requests: **1110/1110** (100.00%).
- Fully comparable/full-input requests: **1110/1110** (100.00%).
- Compatibility status: `{"full_input": 1110}`.
- Common normalized decisions after compatibility filtering: **1110**.
- Accuracy — Jev: **0.9036**; SemIf: **0.6910**.
- Binary exact-probability MAE — Jev: **n/a**; SemIf: **n/a**.
- Full-distribution TV — Jev: **0.2407**; SemIf: **0.3012**.
- Local/remote wall p50 — Jev: **332.3 ms**; SemIf: **208.5 ms**.

Semantic top-probability gap versus reference — Jev: **0.2051**; SemIf: **0.1744**.

## Interpretation constraints

- SemIf is an open-model reproduction of the interface pattern, not of Jev's model; it is independent and not affiliated with TypeSafe.
- SemIf accepts 2-16 options per decision. Larger Jev Choice sets are reported as `unsupported_cardinality`, never shortened.
- SemIf never truncates: prompts beyond the configured input limit are `input_too_long`. The limit was raised from SemIf's 4,096-token default so long-context cases are actually attempted.
- The `noul` and `score` mappings are this study's adaptation of SemIf's categorical interface (see the run manifests).
- Jev upstream service time and SemIf local end-to-end inference time are not measured on the same hardware.
