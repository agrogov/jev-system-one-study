# Jev vs Laya: Controlled Typed-Decision Comparison

## Abstract

This report compares TypeSafe Jev and the local Laya typed-decisions MLX checkpoint on **byte-identical benchmark cases**. The comparison uses the five recorded Jev suites as the source of truth. Laya-specific incompatibilities are retained as outcomes: truncated-input requests and unsupported cardinalities are not silently modified, and capability metrics are computed on aligned full-input decisions where appropriate.

The result separates the **typed-decision paradigm** from the **implementation/capability envelope**. Laya reproduces the core non-autoregressive Choice/Noul/Score abstraction and is much faster for short local calls, but Jev is substantially stronger in semantic accuracy, exact-probability recovery, large dynamic Choice sets, long-context behavior, confidence ranking, and multi-question scaling.

## 1. Fairness and scope

- Same `cases.jsonl` source inputs and case IDs are used for both systems.
- Laya `truncated_input` and `unsupported_cardinality` cases are reported, not repaired.
- Quality comparisons exclude non-full-input Laya cases where the original input was not actually consumed.
- Jev is hosted/remote; Laya is local MLX. Absolute latency is therefore descriptive, not a hardware-normalized contest.
- Laya model: `aac6fef/laya-typed-decisions-mlx`, FP16, batch size 16.

## 2. Overall aligned results

| Suite | Aligned decisions | Jev accuracy | Laya accuracy | Jev − Laya |
|---|---:|---:|---:|---:|
| architecture core | 441 | 98.87% | 85.49% | +13.38 pp |
| isolated core scaling | 140 | 100.00% | 75.71% | +24.29 pp |
| full boundary scaling | 268 | 100.00% | 65.67% | +34.33 pp |
| probability calibration | 4,600 | 80.63% | 59.41% | +21.22 pp |
| semantic | 1,110 | 90.36% | 44.05% | +46.31 pp |

For the semantic suite, the paired Jev accuracy advantage is **+46.31 percentage points** with a bootstrap 95% CI of **+42.97 to +49.64 pp**.

## 3. Architecture: where the implementations diverge

### 3.1 Multi-question scaling

Jev’s isolated backend timing remained nearly flat while question count increased from 1 to 64 and rose modestly at 128; Laya increased approximately linearly and its input accounting multiplied with question count. This is strong evidence that Laya reprocesses state/question rows while Jev performs substantial shared computation or an equivalent fused operation.

| Questions | Jev upstream service time | Laya local latency |
|---:|---:|---:|
| 1 | 99 ms | 84.0 ms |
| 2 | 85 ms | 186.6 ms |
| 4 | 101 ms | 484.1 ms |
| 8 | 143 ms | 790.3 ms |
| 16 | 87 ms | 1739.5 ms |
| 32 | 80 ms | 2407.4 ms |
| 64 | 87 ms | 4813.9 ms |
| 128 | 167 ms | 9697.2 ms |

### 3.2 High-cardinality Choice

Laya solves the synthetic candidate lookup through 16 choices, fails at 32–128, and rejects 255. Jev solved the corresponding task through 255 choices with no obvious latency cliff. This is a direct capability difference, not merely a calibration issue.

### 3.3 Long context

Laya’s effective 1,024-token input ceiling causes silent truncation and position-dependent failures on long states. Jev correctly recovered beginning, middle, and end evidence in the approximately 21K-token full probe. These truncated Laya cases are excluded from aligned quality metrics but retained as compatibility evidence.

## 4. Exact probability recovery

On 4,810 aligned binary cases, Jev MAE is **0.0793** versus Laya **0.2892**. Laya therefore adds **0.2099** absolute probability error on average (bootstrap 95% CI **0.2055–0.2142**).

| Family | Jev MAE | Laya MAE |
|---|---:|---:|
| `prob_explicit_randomness` | 0.0286 | 0.2342 |
| `prob_bayes_single_signal` | 0.0853 | 0.3857 |
| `prob_bayes_two_signals` | 0.0859 | 0.2446 |
| `prob_base_rate_stress` | 0.1108 | 0.2662 |
| `prob_representation_invariance` | 0.1135 | 0.2498 |
| `prob_repeatability` | 0.0167 | 0.2202 |

On the 1,000 multiclass exact-posterior cases, mean TV distance is **0.3010 for Jev** versus **0.3427 for Laya**.

The two systems fail differently: Jev tends to be sharper than exact/reference distributions in several Choice settings, whereas Laya’s typed-decisions checkpoint often compresses Noul probabilities toward the middle and produces diffuse semantic Choice distributions.

## 5. Semantic comparison

Across 1,110 aligned semantic cases, Jev accuracy is **90.36%** versus Laya **44.05%**. Mean TV distance is **0.2407** versus **0.3940**; Laya’s excess TV distance is **0.1533** (bootstrap 95% CI **0.1372–0.1697**).

### 5.1 Domain accuracy

| Domain | n | Jev | Laya | Jev − Laya |
|---|---:|---:|---:|---:|
| code_review_severity | 210 | 98.10% | 28.10% | +70.00 pp |
| hardware_fault | 150 | 80.00% | 58.00% | +22.00 pp |
| incident_triage | 210 | 100.00% | 39.05% | +60.95 pp |
| policy_compliance | 150 | 93.33% | 43.33% | +50.00 pp |
| security_classification | 180 | 100.00% | 76.67% | +23.33 pp |
| support_routing | 210 | 70.00% | 27.62% | +42.38 pp |

### 5.2 Distribution shape

The authored semantic references have mean top probability about **0.677**. Jev’s mean top probability is about **0.882** (too sharp relative to those references); Laya’s is about **0.417** (too diffuse). Their normalized entropy moves in opposite directions: Jev ≈ **0.191**, reference ≈ **0.541**, Laya ≈ **0.923**. Thus “lower confidence” is not equivalent to better calibration here—Laya often approaches a near-uniform four-way distribution.

### 5.3 Selective confidence

Jev’s confidence/top-probability ranking produces a useful low-risk subset: the most-confident half of the semantic cases had zero error in the recorded run. Laya confidence did not show comparable selective-risk behavior; its most-confident half remained high-error. Raw risk-coverage artifacts are preserved in the run outputs.

## 6. Latency interpretation

Laya is dramatically faster in absolute wall time for short one-question tasks on the local Apple Silicon setup (e.g. semantic p50 ≈ **18.3 ms**). Jev’s recorded remote wall p50 is roughly **332 ms** on the same semantic source run. This is **not** an apples-to-apples hardware/model efficiency comparison: one is local inference and one is a hosted network service. The scientifically stronger latency result is scaling shape: Jev’s backend time is nearly flat across large question bundles while Laya’s local time grows with question count.

## 7. Interpretation

The controlled evidence supports a two-layer view of “Jev-like” systems:

1. **Reproducible paradigm:** typed Choice/Noul/Score decisions, direct distributions, schema-safe non-generative inference. Laya demonstrates that this layer is open and practical.
2. **Capability/runtime layer:** semantic/world knowledge, shared-state execution, long-context handling, large dynamic candidate sets, probability reasoning, and confidence ranking. On this benchmark Jev is substantially stronger across this layer.

Accordingly, Laya should be described as reproducing the **typed-decision abstraction and non-autoregressive paradigm**, not as reproducing the full observable Jev serving architecture or capability envelope.

## 8. Limitations

- Only Laya typed-decisions MLX is tested; base/multilingual checkpoints may behave differently.
- Semantic references are authored adjudication distributions, not independent human-panel measurements.
- Absolute local-vs-hosted latency cannot identify intrinsic hardware-normalized efficiency.
- Some architecture cases intentionally exceed Laya’s documented model limits; they are retained as compatibility findings and excluded where full-input comparability is required.
- Black-box behavior does not uniquely identify either system’s hidden model architecture or training data.

## 9. Preserved evidence

The repository includes all five primary Laya runs, the generated machine-readable comparison files, the earlier smoke/validation runs, and the exact adapter/configuration used for replay. No duplicate ZIP archives are needed: extracted raw result directories are the canonical evidence.

## Conclusion

The Laya control experiment sharpens the Jev study considerably. A small open typed-decision model can duplicate the API-level paradigm and deliver very fast local inference, but that alone does not reproduce Jev’s observed multi-question execution, large-choice behavior, context capability, probability recovery, or semantic reliability. The likely engineering moat is therefore less about the existence of typed decision heads and more about representation quality, decision-native training, candidate/context machinery, and serving architecture.
