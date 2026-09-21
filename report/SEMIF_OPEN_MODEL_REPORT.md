# SemIf (Qwen3.5-4B) Black-Box Study

## Abstract

This report evaluates **SemIf** (`TheoLeeCJ/SemIf`, commit `ca3ba65`) running `Qwen/Qwen3.5-4B` (revision `851bf6e8`) through its native Apple Silicon MLX backend, on the same recorded benchmark cases used in the Jev System One study. SemIf is an independent open-model reproduction of the *interface pattern* behind Jev: a state, a runtime question and typed options go in, and option probabilities are read directly from the model's answer-slot logits with no generated text. The primary run contains **7,280 requests** across five suites: architecture/robustness, isolated scaling, full boundary scaling, exact probabilistic calibration, and six-domain semantic calibration. Inputs were replayed unchanged; SemIf-specific incompatibilities were recorded, not repaired.

The central finding is that a general-purpose 4B model with a prompted typed readout gets surprisingly far: it is perfect on the architecture core primitives, handles 25K-token states and 128-question requests, and is much more semantically accurate than the Laya specialist checkpoint (69.1% vs 44.05%). It still trails Jev clearly on semantic accuracy (90.36%), exact-probability recovery (MAE 0.2526 vs 0.0793), and scaling (multi-question latency grows with question count), and it cannot express Choice sets larger than 16 options.

## 1. Runtime and experimental provenance

- Model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, source precision (BF16 with some FP32 parameters), no quantization
- Backend: SemIf native MLX (`mlx` 0.32.2, `mlx-lm` 0.32.0 at commit `a63e24c`)
- Platform: Apple M4 Max, macOS, MLX allocator cache limit 256 MiB
- Input limit: 65,536 tokens per question (SemIf's own default of 4,096 was raised; SemIf refuses rather than truncates, so no request was shortened)
- All five source `cases.jsonl` files were replayed unchanged; their SHA-256 digests are stored in each run manifest.
- Model load 6.3 s; the full replay took about 29 minutes.

### Primitive and execution mapping

SemIf only exposes categorical options, so the Jev primitives are mapped as follows (recorded in every manifest):

| Jev primitive | SemIf decision |
|---|---|
| `choice` | criteria are the options; probabilities are SemIf's softmax over the answer letters |
| `score` | ordinal criteria are the options; `score` = expected index |
| `noul` | fixed two-option yes/no decision; `noul` = P(yes) |

A request with one question uses SemIf `direct` scoring; a request with several questions over one state uses `shared` scoring (one state prefill, parallel question suffixes). **The yes/no option wording for `noul` is authored by this study, not by SemIf**; because `noul` makes up roughly two thirds of all questions, the calibration results depend on it. The frozen wording and per-question `prompt_sha256` are stored with the results.

### Primary recorded runs

| Run | Requests | Completed | Full-input comparable | Purpose |
|---|---:|---:|---:|---|
| `run_20260919T201050Z_core` | 303 | 300 | 300 | architecture/robustness core |
| `run_20260919T201739Z_core` | 26 | 23 | 23 | isolated architecture core |
| `run_20260919T202139Z_full` | 31 | 27 | 27 | isolated full boundary |
| `run_20260919T203758Z_calibration_full` | 5810 | 5810 | 5810 | exact probabilistic calibration |
| `run_20260919T211437Z_semantic_full` | 1110 | 1110 | 1110 | semantic calibration |

All ten failed requests are the same thing: a Choice question with 32, 64, 128 or 255 options, which SemIf rejects (`unsupported_cardinality`, 2–16 options allowed). No request was truncated or failed for any other reason.

## 2. Typed primitive behavior

On the architecture-core decisions SemIf was correct on every one:

| Primitive | Decisions | Accuracy | Additional metric |
|---|---:|---:|---|
| noul | 241 | 100.00% | — |
| choice | 141 | 100.00% | — |
| score | 62 | 100.00% | mean absolute score error 0.040 |

## 3. Architecture and scaling

### 3.1 Multi-question scaling

| Questions | SemIf latency | Decision accuracy |
|---:|---:|---:|
| 1 | 514 ms | 100.00% |
| 2 | 556 ms | 100.00% |
| 4 | 685 ms | 75.00% |
| 8 | 896 ms | 62.50% |
| 16 | 1,377 ms | 75.00% |
| 32 | 2,369 ms | 87.50% |
| 64 | 4,382 ms | 93.75% |
| 128 | 14,301 ms | 96.88% |

Shared-state execution makes growth much gentler than Laya's (roughly 28× slower at 128 questions, versus about 120× for Laya), but time still rises steadily with question count, whereas Jev's backend time is nearly flat. Accuracy across this experiment in the full boundary run is 92.2% (Jev 100%, Laya 65.1%); the dip at 4–16 questions rests on very few decisions per level.

### 3.2 Choice cardinality

Correct through 16 options; 32, 64, 128 and 255 are rejected as unsupported. Jev solved the task through 255 choices.

### 3.3 Context length

SemIf completed every context-length probe up to 100,000 characters (about 25K tokens) with 100% accuracy, taking 25 s for the longest. Latency grows roughly linearly with context (0.3 s at 500 characters, 5.9 s at 32,000), compared with Jev's near-flat 100–200 ms.

## 4. Local latency

| Suite | Median wall latency per request |
|---|---:|
| architecture core | 164 ms |
| isolated core scaling | 554 ms |
| full boundary scaling | 1,505 ms |
| probability calibration | 206 ms |
| semantic | 209 ms |

These are local Apple Silicon measurements and are not hardware-equivalent to the hosted Jev service.

## 5. Exact probabilistic calibration

Across the **4,810** binary cases with analytically known probabilities, SemIf achieved MAE **0.2526** (Jev 0.0793, Laya 0.2892), RMSE **0.3067**, mean bias **+0.2010**, and Pearson correlation **0.750**. The model ranks cases well (correlation 0.75, above Laya's 0.451) but is systematically over-confident toward "yes".

| Family | n | SemIf MAE | Jev MAE | Laya MAE |
|---|---:|---:|---:|---:|
| `prob_explicit_randomness` | 760 | 0.3609 | 0.0286 | 0.2342 |
| `prob_bayes_single_signal` | 1500 | 0.2277 | 0.0853 | 0.3857 |
| `prob_bayes_two_signals` | 1500 | 0.1878 | 0.0859 | 0.2446 |
| `prob_base_rate_stress` | 600 | 0.3208 | 0.1108 | 0.2662 |
| `prob_representation_invariance` | 300 | 0.1829 | 0.1135 | 0.2498 |
| `prob_repeatability` | 150 | 0.4660 | 0.0167 | 0.2202 |

On the 1,000 three-class Bayes cases, hard accuracy was **60.90%**, mean TV distance **0.1831** (Jev 0.3010, Laya 0.3427), JSD **0.0454**, and soft Brier **0.0883**. SemIf is closer to the exact 3-class posterior than either Jev or Laya on this measure even though its binary probabilities are worse than Jev's.

## 6. Semantic calibration

Across the 1,110 six-domain semantic cases SemIf modal accuracy was **69.10%** (Wilson 95% CI **66.32%–71.75%**). Mean TV distance from the authored reference distribution was **0.3012**, JSD **0.0966**, and soft Brier **0.1989**.

Unlike Laya (diffuse) and like Jev, SemIf is sharper than the authored references: reference mean top probability was 0.677, SemIf 0.851, Jev 0.882, Laya 0.417. Mean normalized entropy was 0.294 versus reference entropy 0.541.

Confidence ranking is useful but weaker than Jev's: SemIf's most-confident half of cases had 88.3% accuracy versus 49.9% for the rest (Jev: 100% vs 80.7%).

### 6.1 Domain breakdown

| Domain | n | SemIf | Jev | Laya |
|---|---:|---:|---:|---:|
| code_review_severity | 210 | 76.67% | 98.10% | 28.10% |
| hardware_fault | 150 | 92.67% | 80.00% | 58.00% |
| incident_triage | 210 | 81.43% | 100.00% | 39.05% |
| policy_compliance | 150 | 74.67% | 93.33% | 43.33% |
| security_classification | 180 | 52.22% | 100.00% | 76.67% |
| support_routing | 210 | 42.86% | 70.00% | 27.62% |

SemIf beats Jev on hardware_fault but is weak on security_classification and support_routing.

## 7. Interpretation

- **The typed-decision paradigm is reproducible with a general open model.** No decision-specific fine-tuning was used; a prompted answer-slot readout on Qwen3.5-4B already gives perfect primitive-level behavior and long-context capability.
- **The remaining gaps are calibration, semantic breadth and serving.** Exact-probability recovery and multi-question scaling are where Jev is still clearly ahead, along with unbounded Choice sets.
- **SemIf and Laya fail differently.** Laya is limited by a 1,024-token window and diffuse probabilities; SemIf is limited by a 16-option ceiling, biased binary probabilities, and compute that scales with context and question count.

## 8. Limitations

- Only one model (Qwen3.5-4B BF16, one prompt version) and one runtime were tested; SemIf's quantized and Torch/CUDA paths were not run.
- The `noul`/`score` mapping and its yes/no wording are this study's adaptation; different wording could change the probability results.
- Semantic references are authored adjudication distributions, not independent human-panel measurements.
- Local Apple Silicon latency cannot identify hardware-normalized efficiency relative to the hosted Jev service.
- SemIf is independent of and unaffiliated with TypeSafe; black-box behavior does not identify either system's hidden training or architecture.

## 9. Reproducibility and preserved evidence

The repository preserves every primary run under `../semif-results/` with `manifest.json`, `cases.jsonl`, `raw.jsonl`, `normalized.csv`, `summary.json`, compatibility metadata, per-run reports, and figures. The adapter, configuration and tests are under `../benchmark` (see `../benchmark/SEMIF_COMPARISON.md`), and the machine-readable comparisons are under `../comparisons/semif_qwen3-5-4b_mlx` and `../comparisons/three_way`.
