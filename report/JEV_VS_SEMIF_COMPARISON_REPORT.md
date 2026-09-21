# Jev vs SemIf: Controlled Typed-Decision Comparison

## Abstract

This report compares TypeSafe Jev and **SemIf** (`TheoLeeCJ/SemIf`, commit `ca3ba65`) running `Qwen/Qwen3.5-4B` on Apple Silicon (native MLX), on **byte-identical benchmark cases**. The comparison uses the five recorded Jev suites as the source of truth. SemIf-specific incompatibilities (Choice sets larger than 16 options) are retained as outcomes rather than repaired, and quality metrics are computed on aligned decisions that both systems completed.

The result separates the **typed-decision paradigm** from the **capability/serving envelope**. SemIf shows that a general-purpose 4B open model, with no decision-specific training, reproduces the direct-probability Choice/Noul/Score interface and matches Jev on the architecture primitives, long-context retrieval and 128-question requests. Jev remains stronger on semantic accuracy (+21.3 pp), exact-probability recovery (MAE 0.079 vs 0.253), unbounded Choice sets (255 vs 16 options) and multi-question and long-context scaling. SemIf is better than Jev on one distribution-level measure: distance to the exact three-class posterior.

## 1. Fairness and scope

- Same `cases.jsonl` source inputs and case IDs are used for both systems.
- SemIf never truncates input. Its per-question limit was raised from the default 4,096 to 65,536 tokens so that Jev's ~21K-token probe is actually attempted; no request exceeded it.
- SemIf `unsupported_cardinality` cases (more than 16 options) are reported, not repaired.
- SemIf only exposes categorical options, so the Jev primitives are adapted (`choice` → criteria as options, `score` → ordinal options with expected index, `noul` → fixed yes/no options with `noul = P(yes)`). **The `noul` option wording is authored by this study** and `noul` is roughly two thirds of all questions, so the probability results depend on it.
- Quality comparisons use decisions both systems completed. SemIf coverage is 350/360 architecture requests and 100% of calibration and semantic requests.
- Jev is hosted/remote; SemIf is local MLX. Absolute latency is descriptive, not a hardware-normalized contest.
- SemIf model: `Qwen/Qwen3.5-4B` revision `851bf6e8`, source precision (BF16), one prompt version. SemIf is independent of and unaffiliated with TypeSafe.

## 2. Overall aligned results

| Suite | Aligned decisions | Jev accuracy | SemIf accuracy | Jev − SemIf |
|---|---:|---:|---:|---:|
| architecture core | 444 | 98.87% | 100.00% | −1.13 pp |
| isolated core scaling | 143 | 100.00% | 100.00% | 0.00 pp |
| full boundary scaling | 274 | 100.00% | 92.70% | +7.30 pp |
| probability calibration | 4,600 | 80.63% | 74.13% | +6.50 pp |
| semantic | 1,110 | 90.36% | 69.10% | +21.26 pp |

Paired case-clustered bootstrap 95% CIs (2,000 resamples): architecture core −2.36 to −0.24 pp (SemIf is significantly better, though on very few disagreements), full boundary +4.02 to +17.34 pp, probability calibration +5.37 to +7.63 pp, semantic **+18.65 to +24.05 pp**.

Primitive-level accuracy on the aligned architecture-core decisions:

| Primitive | Decisions | Jev | SemIf |
|---|---:|---:|---:|
| noul | 241 | 100.00% | 100.00% |
| choice | 135 | 100.00% | 100.00% |
| score | 62 | 91.94% | 100.00% |

## 3. Architecture: where the implementations diverge

### 3.1 Multi-question scaling

Jev's isolated backend timing stays nearly flat as the question count grows from 1 to 128. SemIf's shared-state mode prefills the state once and scores the questions as parallel suffixes, which keeps growth far gentler than a per-question loop, but time still rises steadily with question count.

| Questions | Jev upstream service time | SemIf local latency | SemIf accuracy |
|---:|---:|---:|---:|
| 1 | 99 ms | 514 ms | 100.00% |
| 2 | 103 ms | 556 ms | 100.00% |
| 4 | 101 ms | 685 ms | 75.00% |
| 8 | 107 ms | 896 ms | 62.50% |
| 16 | 88 ms | 1,377 ms | 75.00% |
| 32 | 131 ms | 2,369 ms | 87.50% |
| 64 | 96 ms | 4,382 ms | 93.75% |
| 128 | 167 ms | 14,301 ms | 96.88% |

Jev answered every question correctly at every level. SemIf's accuracy on this experiment is 92.2% overall in the full boundary run; the dip at 4–16 questions rests on very few decisions per level. From 1 to 128 questions Jev's time grows about 1.7× and SemIf's about 28×.

### 3.2 High-cardinality Choice

SemIf solves the synthetic candidate lookup through 16 choices and rejects 32, 64, 128 and 255 as unsupported (its answer slots are the letters A–P). Jev solved the corresponding task through 255 choices with no latency cliff. This is a hard interface limit, not a calibration issue.

### 3.3 Long context

SemIf completed every context-length probe up to 100,000 characters (about 25K tokens) with 100% accuracy, matching Jev's recovery of beginning, middle and end evidence in the ~21K-token full probe. The cost differs: Jev's server time stayed at 94–208 ms, whereas SemIf's grew roughly linearly with context (0.3 s at 500 characters, 5.9 s at 32,000, 25 s at 100,000).

| Context chars | Jev server time | SemIf local latency |
|---:|---:|---:|
| 500 | 119 ms | 275 ms |
| 2,000 | 100 ms | 528 ms |
| 8,000 | 94 ms | 1,528 ms |
| 32,000 | 113 ms | 5,938 ms |
| 100,000 | 208 ms | 25,106 ms |

## 4. Exact probability recovery

On 4,810 aligned binary cases, Jev MAE is **0.0793** versus SemIf **0.2526**. SemIf therefore adds **0.1732** absolute probability error on average (bootstrap 95% CI **0.1681–0.1786**). SemIf's probabilities correlate well with the exact values (Pearson 0.750) but are biased toward "yes" (mean bias +0.201) and often too extreme.

| Family | Jev MAE | SemIf MAE |
|---|---:|---:|
| `prob_explicit_randomness` | 0.0286 | 0.3609 |
| `prob_bayes_single_signal` | 0.0853 | 0.2277 |
| `prob_bayes_two_signals` | 0.0859 | 0.1878 |
| `prob_base_rate_stress` | 0.1108 | 0.3208 |
| `prob_representation_invariance` | 0.1135 | 0.1829 |
| `prob_repeatability` | 0.0167 | 0.4660 |

Jev is far better wherever the state describes explicit randomness (a known draw probability): SemIf turns "the ball is red with probability 5%" into a near-certain answer instead of encoding the stated uncertainty.

On the 1,000 multiclass exact-posterior cases the ordering reverses: mean TV distance is **0.3010 for Jev** versus **0.1831 for SemIf**, while hard accuracy is 62.5% (Jev) versus 60.9% (SemIf). SemIf's full three-class distributions are closer to the exact posterior even though its binary probabilities are worse.

## 5. Semantic comparison

Across 1,110 aligned semantic cases, Jev accuracy is **90.36%** versus SemIf **69.10%**. Mean TV distance to the authored reference is **0.2407** versus **0.3012**; SemIf's excess TV distance is **0.0605** (bootstrap 95% CI **0.0507–0.0711**).

### 5.1 Domain accuracy

| Domain | n | Jev | SemIf | Jev − SemIf |
|---|---:|---:|---:|---:|
| code_review_severity | 210 | 98.10% | 76.67% | +21.43 pp |
| hardware_fault | 150 | 80.00% | 92.67% | −12.67 pp |
| incident_triage | 210 | 100.00% | 81.43% | +18.57 pp |
| policy_compliance | 150 | 93.33% | 74.67% | +18.66 pp |
| security_classification | 180 | 100.00% | 52.22% | +47.78 pp |
| support_routing | 210 | 70.00% | 42.86% | +27.14 pp |

SemIf beats Jev on hardware_fault but is much weaker on security_classification and support_routing.

### 5.2 Distribution shape

The authored semantic references have mean top probability about **0.677**. Jev's mean top probability is **0.882** and SemIf's **0.851**: both are sharper than the references, unlike Laya, which is diffuse. Normalized entropy is Jev ≈ **0.191**, SemIf ≈ **0.294**, reference ≈ **0.541**. SemIf's sharpness is not matched by accuracy, so its sharpness is a larger calibration error than Jev's.

### 5.3 Selective confidence

Jev's confidence ranking produces a clean low-risk subset: the most-confident half of the semantic cases had 100% accuracy versus 80.7% for the rest. SemIf's ranking is informative but weaker: 88.3% accuracy in its most-confident half versus 49.9% in the rest.

## 6. Latency interpretation

SemIf is faster in absolute wall time for short single-question tasks on the local Apple Silicon setup (median 164 ms for architecture core, 206 ms for calibration and 209 ms for semantic versus Jev's ≈ 320–333 ms), but slower on the multi-question and long-context suites (554 ms and 1,505 ms medians for isolated and full scaling versus Jev's ≈ 324–330 ms). This is **not** an apples-to-apples hardware/model efficiency comparison: one is local inference and one is a hosted network service. The stronger latency result is scaling shape: Jev's backend time is nearly flat across large question bundles and long contexts, whereas SemIf's grows with both.

## 7. Interpretation

The controlled evidence supports a two-layer view:

1. **Reproducible paradigm:** typed Choice/Noul/Score decisions, direct distributions and schema-safe non-generative inference. SemIf shows this layer works with a general open 4B model and a prompted readout, with no decision-specific training.
2. **Capability/runtime layer:** semantic breadth, probability calibration, unbounded candidate sets and constant-time shared-state execution. On this benchmark Jev is stronger across this layer, although SemIf narrows the gap substantially compared with a specialist small model.

Accordingly SemIf should be described as reproducing the **typed-decision interface and non-autoregressive readout**, not Jev's model, calibration or serving architecture.

## 8. Limitations

- One model (Qwen3.5-4B, BF16), one prompt version and one runtime were tested; quantized and Torch/CUDA paths were not run.
- The `noul`/`score` mapping and its yes/no wording are this study's adaptation; different wording could change the probability results.
- Semantic references are authored adjudication distributions, not independent human-panel measurements.
- Some architecture cases (more than 16 options) exceed SemIf's interface; they are retained as compatibility findings and excluded where aligned comparison is required.
- Local-vs-hosted latency cannot identify hardware-normalized efficiency.
- Black-box behavior does not uniquely identify either system's hidden model architecture or training data.

## 9. Preserved evidence

The repository includes all five primary SemIf runs under `../semif-results/`, the generated machine-readable comparison files under `../comparisons/semif_qwen3-5-4b_mlx/`, integrated statistics in `semif_computed_statistics.json`, and the exact adapter and configuration used for replay under `../benchmark/`. Extracted raw result directories are the canonical evidence.

## Conclusion

The SemIf control experiment shows that the typed-decision paradigm does not require a specialist or proprietary model: a general 4B open model with a prompted answer-slot readout is perfect on the architecture primitives, handles 25K-token states and 128-question requests, and gives sharper, more accurate semantic distributions than the Laya specialist. It does not reproduce Jev's semantic accuracy, exact-probability recovery, Choice sets beyond 16 options, or flat-cost scaling with questions and context. The likely engineering moat therefore lies less in the readout mechanism and more in representation quality, calibration training, candidate/context machinery and serving architecture.
