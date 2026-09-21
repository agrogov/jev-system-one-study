# Laya Typed-Decisions Black-Box Study

## Abstract

This report evaluates **Laya typed-decisions** (`aac6fef/laya-typed-decisions-mlx`) with the same recorded benchmark cases used in the Jev System One study. Laya was executed locally through MLX in FP16 with question batch size 16. The primary run contains **7,280 requests** across five suites: architecture/robustness, isolated scaling, full boundary scaling, exact probabilistic calibration, and six-domain semantic calibration. The exact inputs were preserved from the Jev study; Laya-specific context/cardinality incompatibilities were recorded rather than hidden or repaired.

The central finding is that Laya successfully implements the typed, non-autoregressive decision paradigm, but the tested checkpoint has a substantially narrower capability envelope than Jev: a 1,024-token effective input limit, a high-cardinality Choice breakdown beyond roughly 16–32 options, nearly linear multi-question cost, weak exact-probability recovery, and much lower semantic accuracy on the authored six-domain suite. For short local decisions, however, Laya is extremely fast and deterministic.

## 1. Runtime and experimental provenance

- Model: `aac6fef/laya-typed-decisions-mlx`
- Backend: MLX (`laya-mlx` 0.1.0)
- Dtype: `float16`
- Question batch size: `16`
- Platform: `macOS-26.6.2-arm64-arm-64bit`
- Effective model limits observed/reported by runner: `max_len=1024`, `head_max_len=256`
- Benchmark seed inherited from Jev source runs: `260919`
- All five primary source `cases.jsonl` files were replayed unchanged; their SHA-256 digests are stored in each run manifest.

### Primary recorded runs

| Run | Requests | Completed | Full-input comparable | Purpose |
|---|---:|---:|---:|---|
| `run_20260919T201050Z_core` | 303 | 303 | 297 | architecture/robustness core |
| `run_20260919T201739Z_core` | 26 | 26 | 20 | isolated architecture core |
| `run_20260919T202139Z_full` | 31 | 30 | 21 | isolated full boundary |
| `run_20260919T203758Z_calibration_full` | 5810 | 5810 | 5810 | exact probabilistic calibration |
| `run_20260919T211437Z_semantic_full` | 1110 | 1110 | 1110 | semantic calibration |

Earlier smoke and validation runs were used during development to verify adapter correctness and boundary handling, but they are not part of the published evidence set and are not included in aggregate metrics. The published Laya study uses only the five primary runs under `../laya-results/`.

## 2. Typed primitive behavior

Laya returned all three benchmark primitives (`noul`, `choice`, `score`) through the adapter. On the fully comparable architecture-core decisions:

| Primitive | Decisions | Accuracy | Additional metric |
|---|---:|---:|---|
| noul | 241 | 86.72% | — |
| choice | 138 | 92.03% | — |
| score | 62 | 66.13% | mean absolute score error 0.493 |

The primitive contract is therefore functional, but primitive-level correctness should not be confused with posterior calibration; Sections 5–6 show that Laya probabilities often behave as task scores rather than accurate general-purpose probabilities.

## 3. Architecture and scaling

### 3.1 Parallel-question scaling

| Questions | Laya latency | Input tokens | Decision accuracy |
|---:|---:|---:|---:|
| 1 | 84.0 ms | 931 | 100.00% |
| 2 | 186.6 ms | 1,862 | 100.00% |
| 4 | 484.1 ms | 3,724 | 100.00% |
| 8 | 790.3 ms | 7,448 | 100.00% |
| 16 | 1739.5 ms | 14,896 | 93.75% |
| 32 | 2407.4 ms | 29,792 | 75.00% |
| 64 | 4813.9 ms | 59,584 | 62.50% |
| 128 | 9697.2 ms | 119,168 | 56.25% |

Input accounting grows approximately linearly with question count, showing that this Laya implementation constructs/encodes a state-question row per question rather than exposing Jev-like shared-state scaling. Previously-present question answers remain nearly invariant as more questions are added, so the principal issue is compute reuse/retrieval rather than cross-question contamination.

### 3.2 Choice cardinality

| Choices | Correct | P(correct label) | Latency |
|---:|:---:|---:|---:|
| 2 | yes | 0.8238 | 20.1 ms |
| 4 | yes | 0.7111 | 24.9 ms |
| 8 | yes | 0.8603 | 36.5 ms |
| 16 | yes | 1.0000 | 60.1 ms |
| 32 | no | 0.0001 | 51.6 ms |
| 64 | no | 0.0000 | 58.3 ms |
| 128 | no | 0.0000 | 72.2 ms |
| 255 | unsupported | — | — |

The model is strong through 16 candidates in this synthetic lookup, then collapses sharply at 32+ candidates. The 255-candidate case is rejected as unsupported by the option/head token budget.

### 3.3 Context length and truncation

Laya completed over-length requests by consuming at most 1,024 input tokens. In the full boundary run this creates a position-dependent failure pattern: beginning evidence can survive truncation while middle/end evidence disappears. These requests are tagged `truncated_input` and excluded from apples-to-apples quality comparisons.

Compatibility in the primary architecture runs:

| Run | Full input | Truncated | Unsupported cardinality |
|---|---:|---:|---:|
| core | 297 | 6 | 0 |
| iso | 20 | 6 | 0 |
| full | 21 | 9 | 1 |

## 4. Local latency

Absolute Laya latency is measured locally and is therefore not directly hardware-equivalent to remote Jev wall latency. It is nevertheless useful for characterizing Laya itself:

| Suite | p50 | p95 | Mean |
|---|---:|---:|---:|
| architecture core | 12.8 ms | 49.7 ms | 30.2 ms |
| probability calibration | 13.6 ms | 21.7 ms | 15.7 ms |
| semantic | 18.3 ms | 39.6 ms | 24.8 ms |

For short one-question decisions the checkpoint is therefore very fast on Apple Silicon. The scaling curve is the important caveat: large question bundles multiply work almost linearly.

## 5. Exact probabilistic calibration

Across the **4,810** binary cases with analytically known probabilities, Laya achieved MAE **0.2892**, RMSE **0.3252**, bias **+0.0900**, and Pearson correlation **0.451**. On the 3,600 sampled outcome cases, empirical ECE15 was **0.1641** and Brier score **0.2061**.

### 5.1 By probability family

| Family | n | MAE | Accuracy (where defined) |
|---|---:|---:|---:|
| `prob_explicit_randomness` | 760 | 0.2342 | — |
| `prob_bayes_single_signal` | 1500 | 0.3857 | 48.33% |
| `prob_bayes_two_signals` | 1500 | 0.2446 | 84.33% |
| `prob_base_rate_stress` | 600 | 0.2662 | 59.67% |
| `prob_representation_invariance` | 300 | 0.2498 | — |
| `prob_repeatability` | 150 | 0.2202 | — |

Repeatability itself is very high: the repeated 0.2/0.5/0.8 cases returned effectively identical values within each condition. The problem is not stochastic variance; it is systematic probability compression/miscalibration. For example, the 0.2, 0.5, and 0.8 repeated targets centered around approximately 0.554, 0.547, and 0.540 respectively.

### 5.2 Multiclass exact posterior recovery

On 1,000 three-class Bayes cases, hard accuracy was **38.50%**, mean TV distance **0.3427**, JSD **0.1002**, and soft Brier **0.2446**.

## 6. Semantic calibration

Across the 1,110 six-domain semantic cases, Laya modal accuracy was **44.05%** (Wilson 95% CI **41.16%–46.99%**). Mean TV distance from the authored reference distribution was **0.3940**, JSD **0.1311**, and soft Brier **0.2473**.

Unlike Jev, which was sharper than the authored references, Laya was strongly **under-confident/diffuse** on this suite: reference mean top probability was 0.6770 while Laya mean top probability was 0.4165. Mean normalized entropy was 0.9227 versus reference entropy 0.5408, close to a near-uniform four-way distribution.

### 6.1 Domain breakdown

| Domain | n | Accuracy | 95% CI | Mean TV |
|---|---:|---:|---:|---:|
| code_review_severity | 210 | 28.10% | 22.45%–34.53% | 0.4183 |
| hardware_fault | 150 | 58.00% | 50.00%–65.60% | 0.3828 |
| incident_triage | 210 | 39.05% | 32.70%–45.79% | 0.3859 |
| policy_compliance | 150 | 43.33% | 35.67%–51.33% | 0.4937 |
| security_classification | 180 | 76.67% | 69.97%–82.25% | 0.3368 |
| support_routing | 210 | 27.62% | 22.01%–34.03% | 0.3637 |

### 6.2 Confidence as a selective-risk signal

Laya top probability is not a strong correctness-ranking signal on this semantic suite. Selecting only its most-confident half does not produce the low-risk subset observed with Jev. Consequently, the checkpoint should not be deployed with a generic confidence threshold without task-specific validation/calibration.

## 7. What this establishes about Laya

High-confidence findings from the black-box evidence:

- Laya implements a genuine non-generative typed-decision interface with full Choice distributions and deterministic local inference.
- Short individual decisions are fast on the tested Apple Silicon/MLX setup.
- Multi-question execution does not expose Jev-like shared-state scaling; input accounting and latency increase approximately with question count.
- The tested typed-decisions checkpoint has an effective 1,024-token context limit and silently truncates long inputs.
- Choice quality has a sharp high-cardinality boundary: the synthetic task is solved through 16 candidates, fails badly at 32–128, and 255 is unsupported.
- Exact-probability recovery is weak across direct randomness, Bayesian, base-rate, representation, and repeatability families.
- Semantic outputs are generally too diffuse relative to the authored reference distributions, with substantially lower top-1 accuracy than Jev on this suite.

## 8. Limitations

- The semantic reference distributions are benchmark-authored adjudication distributions, not measured human-panel frequencies.
- Only the `typed-decisions` MLX checkpoint is analyzed here; these findings must not be generalized to every Laya checkpoint.
- Absolute latency is hardware-specific and should not be compared directly with hosted API wall latency as a model-efficiency verdict.
- The high-cardinality and context tests intentionally exceed Laya’s documented capability envelope; failures are capability/compatibility observations, not implementation bugs.
- The benchmark evaluates the public model as used, not its training corpus or hidden internal representations.

## 9. Reproducibility and preserved evidence

The repository preserves every primary run with `manifest.json`, `cases.jsonl`, `raw.jsonl`, `normalized.csv`, `summary.json`, compatibility metadata, per-run reports, and figures. The adapter/configuration used for local Laya replay is included under `../benchmark` so the study can be rerun against the same source cases.

## Conclusion

Laya validates the broader System-One-style engineering idea: many useful semantic tasks can be expressed as typed, direct probability decisions without autoregressive text generation. In the tested specialist checkpoint, however, the interface abstraction is more mature than the capability envelope. The model is compelling for fast, small local decisions, while the controlled benchmark exposes significant limitations in long context, large candidate sets, probability reasoning, semantic generalization, and scalable multi-question execution.
