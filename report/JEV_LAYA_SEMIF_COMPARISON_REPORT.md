# Jev vs Laya vs SemIf: Three-Way Typed-Decision Comparison

## Abstract

This report scores three systems on **byte-identical benchmark cases** and, for quality, on the **same decisions**: the hosted TypeSafe Jev service, the local Laya typed-decisions MLX checkpoint (`aac6fef/laya-typed-decisions-mlx`, FP16, a specialist small model) and SemIf (`TheoLeeCJ/SemIf`) running the general-purpose `Qwen/Qwen3.5-4B` (BF16, native MLX). A decision is scored only if Jev completed it and both local systems consumed the full request; incompatibilities (Laya truncation, unsupported Choice sizes) are reported separately as coverage.

Jev is best on nearly everything that measures capability: semantic accuracy, exact binary probabilities, unbounded Choice sets, and flat-cost scaling. The two open systems reproduce the typed-decision **interface** but not that capability envelope, and they fail in different ways. SemIf, a general 4B model with a prompted readout, is far more accurate than the Laya specialist (semantic 69.1% vs 44.1%) and handles long context and large question bundles that Laya truncates or scales linearly. Laya is much faster for short local calls. SemIf is the closest system to the exact three-class posterior, ahead even of Jev on that one measure.

## 1. Fairness and scope

- All three systems use the same `cases.jsonl` inputs and case IDs.
- **Aligned quality set:** (case, question) pairs completed by Jev and fully consumed by Laya and SemIf. Each metric is averaged over rows where all three systems have a value, so every system is scored on identical rows.
- Coverage, compatibility outcomes and latency use every request.
- Laya: 1,024-token effective input window, roughly 16-option Choice limit; truncated and unsupported requests are reported, not repaired.
- SemIf: 2–16 options per decision, no truncation, input limit raised to 65,536 tokens; unsupported requests are reported, not repaired. SemIf's `noul`/`score` results depend on this study's yes/no and ordinal option mapping.
- Jev is hosted; Laya and SemIf run locally on Apple Silicon. Absolute latency is descriptive, not hardware-normalized.
- Only one Laya checkpoint (typed-decisions specialist) and one SemIf model/prompt/runtime were tested. SemIf is independent of and unaffiliated with TypeSafe.

## 2. Overall aligned results

| Suite | Scored decisions | Jev | Laya | SemIf |
|---|---:|---:|---:|---:|
| architecture core | 438 | 98.86% | 86.07% | 100.00% |
| isolated core scaling | 137 | 100.00% | 77.37% | 100.00% |
| full boundary scaling | 265 | 100.00% | 66.42% | 92.45% |
| probability calibration | 4,600 | 80.63% | 59.41% | 74.13% |
| semantic | 1,110 | 90.36% | 44.05% | 69.10% |

Paired case-clustered bootstrap differences in accuracy (percentage points, 95% CI, 2,000 resamples):

| Suite | Jev − Laya | Jev − SemIf | SemIf − Laya |
|---|---:|---:|---:|
| architecture core | +12.79 (6.12 to 19.12) | −1.14 (−2.39 to −0.24) | +13.93 (7.56 to 20.11) |
| isolated core scaling | +22.63 (0.00 to 31.22) | 0.00 (0.00 to 0.00) | +22.63 (0.00 to 31.22) |
| full boundary scaling | +33.58 (3.39 to 40.38) | +7.55 (4.02 to 19.45) | +26.04 (−14.29 to 36.18) |
| probability calibration | +21.22 (19.67 to 22.85) | +6.50 (5.37 to 7.63) | +14.72 (13.13 to 16.33) |
| semantic | +46.31 (43.15 to 49.55) | +21.26 (18.65 to 24.05) | +25.05 (21.44 to 28.83) |

Ordering on every suite with a real difference is Jev ≥ SemIf > Laya (SemIf slightly exceeds Jev on architecture core). The full-boundary SemIf-vs-Laya interval includes zero because that suite has only a few dozen independent cases.

Primitive-level accuracy on the aligned architecture-core decisions:

| Primitive | Decisions | Jev | Laya | SemIf |
|---|---:|---:|---:|---:|
| noul | 241 | 100.00% | 86.72% | 100.00% |
| choice | 135 | 100.00% | 94.07% | 100.00% |
| score | 62 | 91.94% | 66.13% | 100.00% |

## 3. Coverage: what each system could consume in full

| Suite | Jev | Laya | SemIf |
|---|---:|---:|---:|
| architecture core | 303/303 | 297/303 | 300/303 |
| isolated core scaling | 26/26 | 20/26 | 23/26 |
| full boundary scaling | 31/31 | 21/31 | 27/31 |
| probability calibration | 5,810/5,810 | 5,810/5,810 | 5,810/5,810 |
| semantic | 1,110/1,110 | 1,110/1,110 | 1,110/1,110 |

Across the architecture suites Jev consumed 360/360 requests, SemIf 350/360 and Laya 338/360. Laya's shortfall is context truncation (1,024-token window) plus unsupported Choice sizes; SemIf's shortfall is exclusively Choice questions with more than 16 options.

## 4. Architecture and scaling

### 4.1 Multi-question scaling

| Questions | Jev server time | Laya server time | SemIf server time |
|---:|---:|---:|---:|
| 1 | 99 ms | 81 ms | 514 ms |
| 2 | 103 ms | 89 ms | 556 ms |
| 4 | 101 ms | 159 ms | 685 ms |
| 8 | 107 ms | 300 ms | 896 ms |
| 16 | 88 ms | 610 ms | 1,377 ms |
| 32 | 131 ms | 1,177 ms | 2,369 ms |
| 64 | 96 ms | 2,368 ms | 4,382 ms |
| 128 | 167 ms | 9,697 ms | 14,301 ms |

Jev stays essentially flat. Laya's time grows about 120× from 1 to 128 questions because its input accounting multiplies with question count (it re-encodes state per question). SemIf grows about 28×: shared-state execution prefills once and scores the question suffixes in parallel, so it is much closer to the intended shape than Laya, but its per-question cost is higher in absolute terms, and time still increases steadily. Accuracy on this experiment in the full boundary run: Jev 100%, SemIf 92.2%, Laya 65.1%.

### 4.2 Choice cardinality

| Choices | Jev | Laya | SemIf |
|---:|---:|---:|---:|
| 2–16 | solved | solved | solved |
| 32–128 | solved | fails (wrong answer) | rejected (unsupported) |
| 255 | solved | rejected (unsupported) | rejected (unsupported) |

Server time (Jev / Laya / SemIf, ms): 2 choices 72 / 15 / 162; 16 choices 76 / 30 / 423. Jev showed no latency cliff up to 255 choices. Laya returned confidently wrong answers at 32–128; SemIf explicitly refused to answer beyond 16, which is safer behavior but the same functional limit.

### 4.3 Context length

| Context chars | Jev server time | Laya server time | SemIf server time |
|---:|---:|---:|---:|
| 500 | 119 ms | 19 ms | 275 ms |
| 2,000 | 100 ms | 43 ms | 528 ms |
| 8,000 | 94 ms | 87 ms (truncated) | 1,528 ms |
| 32,000 | 113 ms | 90 ms (truncated) | 5,938 ms |
| 100,000 | 208 ms | 99 ms (truncated) | 25,106 ms |

Laya's 1,024-token window silently truncates from 8,000 characters up. Jev and SemIf both recover evidence at every length up to 100,000 characters (about 25K tokens), but Jev does so at near-constant cost while SemIf's cost grows roughly linearly with context.

## 5. Exact probability recovery

On 4,810 aligned binary cases (bootstrap 95% CIs on differences):

| System | Binary exact-probability MAE | Excess over Jev |
|---|---:|---:|
| Jev | **0.0793** | — |
| SemIf | 0.2526 | +0.1732 (0.1681–0.1786) |
| Laya | 0.2892 | +0.2099 (0.2055–0.2141) |

SemIf's error is 0.0366 lower than Laya's (0.0304–0.0425). Per family:

| Family | Jev | Laya | SemIf |
|---|---:|---:|---:|
| `prob_explicit_randomness` | 0.0286 | 0.2342 | 0.3609 |
| `prob_bayes_single_signal` | 0.0853 | 0.3857 | 0.2277 |
| `prob_bayes_two_signals` | 0.0859 | 0.2446 | 0.1878 |
| `prob_base_rate_stress` | 0.1108 | 0.2662 | 0.3208 |
| `prob_representation_invariance` | 0.1135 | 0.2498 | 0.1829 |
| `prob_repeatability` | 0.0167 | 0.2202 | 0.4660 |

Jev is best on every family. The open systems fail differently: Laya compresses probabilities toward the middle (centered near 0.55 regardless of the true value); SemIf tracks the exact values much better (Pearson 0.750 vs Laya 0.451) but is biased toward "yes" (+0.201) and too extreme on explicit-randomness and repeatability items.

On the 1,000 three-class exact-posterior cases:

| System | Hard accuracy | Mean TV to exact posterior | JSD |
|---|---:|---:|---:|
| Jev | 62.5% | 0.3010 | 0.1113 |
| Laya | 38.5% | 0.3427 | 0.1002 |
| SemIf | 60.9% | **0.1831** | 0.0454 |

SemIf's full distribution is closer to the exact posterior than Jev's (TV difference −0.1179, 95% CI −0.1290 to −0.1072), even though Jev's binary probabilities are far better and its hard accuracy is slightly higher.

## 6. Semantic comparison

| Metric | Jev | Laya | SemIf |
|---|---:|---:|---:|
| Accuracy | **90.36%** | 44.05% | 69.10% |
| Mean TV to reference | **0.2407** | 0.3940 | 0.3012 |
| JSD | **0.0736** | 0.1311 | 0.0966 |
| Soft Brier | **0.1215** | 0.2473 | 0.1989 |
| Mean top probability (reference 0.677) | 0.882 | 0.417 | 0.851 |
| Mean normalized entropy (reference 0.541) | 0.191 | 0.923 | 0.294 |

TV differences: Laya − Jev +0.1533 (0.1373–0.1699); SemIf − Jev +0.0605 (0.0507–0.0711); Laya − SemIf +0.0929 (0.0757–0.1095).

### 6.1 Domain accuracy

| Domain | n | Jev | Laya | SemIf |
|---|---:|---:|---:|---:|
| code_review_severity | 210 | 98.10% | 28.10% | 76.67% |
| hardware_fault | 150 | 80.00% | 58.00% | 92.67% |
| incident_triage | 210 | 100.00% | 39.05% | 81.43% |
| policy_compliance | 150 | 93.33% | 43.33% | 74.67% |
| security_classification | 180 | 100.00% | 76.67% | 52.22% |
| support_routing | 210 | 70.00% | 27.62% | 42.86% |

SemIf beats Jev only on hardware_fault, beats Laya on five of six domains, and is worse than Laya on security_classification (52.2% vs 76.7%).

### 6.2 Distribution shape and selective confidence

Laya is diffuse (near-uniform four-way distributions); Jev and SemIf are both sharper than the authored references, and SemIf's sharpness is not backed by accuracy. Confidence ranking (accuracy in the most-confident half of cases vs the rest):

| System | Most-confident half | Least-confident half |
|---|---:|---:|
| Jev | 100.0% | 80.7% |
| SemIf | 88.3% | 49.9% |
| Laya | 41.3% | 46.8% |

Jev's and SemIf's confidence usefully separates reliable from unreliable decisions; Laya's does not.

## 7. Latency interpretation

Median wall latency per request (ms):

| Suite | Jev (remote) | Laya (local) | SemIf (local) |
|---|---:|---:|---:|
| architecture core | 332.5 | 12.8 | 163.9 |
| isolated core scaling | 323.8 | 66.5 | 554.0 |
| full boundary scaling | 329.6 | 88.0 | 1,504.9 |
| probability calibration | 321.4 | 13.6 | 206.1 |
| semantic | 332.3 | 18.3 | 208.5 |

For short one-question decisions both local systems are faster than the hosted service, Laya by an order of magnitude. This is **not** a hardware- or model-normalized efficiency comparison. The scientifically stronger latency result is scaling shape: Jev is nearly flat in question count and context length, while both local systems grow with them (Laya with question count, SemIf with both).

## 8. Interpretation

1. **The interface is reproducible; the capability is not (yet).** Both open systems return typed Choice/Noul/Score distributions without generating text. Neither matches Jev's semantic accuracy, exact-probability recovery, unbounded Choice sets or flat-cost scaling.
2. **General model + prompted readout beats a small specialist here.** SemIf (Qwen3.5-4B, no decision-specific training) outperforms the Laya typed-decisions specialist on every quality suite and on coverage, showing that model scale/pretraining matters more than a dedicated typed-decision head for semantic breadth and long context. Laya keeps a large speed advantage for short local calls and a native 3-primitive API.
3. **Failure modes differ and matter operationally.** Laya truncates silently, answers wrongly at 32–128 choices and returns diffuse probabilities. SemIf refuses out-of-range requests explicitly, but is over-confident and biased on binary probabilities.
4. **Where Jev's advantage is architectural.** Flat server time with question count and context, 255-way Choice sets and calibrated binary probabilities are not matched by either open system; these are the aspects most plausibly attributable to representation quality, calibration training, candidate/context machinery and serving design rather than the readout mechanism.

## 9. Limitations

- One Laya checkpoint (typed-decisions specialist) and one SemIf model, prompt and runtime; base/multilingual Laya and quantized or larger SemIf models may behave differently.
- SemIf's `noul`/`score` mapping and yes/no wording are this study's adaptation.
- Semantic references are authored adjudication distributions, not independent human-panel measurements.
- Some architecture cases intentionally exceed one or both local systems' documented limits; they are retained as compatibility findings and excluded from aligned quality metrics.
- Absolute latency compares a remote service with local Apple Silicon inference.
- Black-box behavior does not uniquely identify any system's hidden architecture or training data.

## 10. Preserved evidence

The repository includes the five primary runs for each system (`../jev-results/`, `../laya-results/`, `../semif-results/`), the generated machine-readable comparisons (`../comparisons/laya_typed-decisions_mlx/`, `../comparisons/semif_qwen3-5-4b_mlx/`, `../comparisons/three_way/`), integrated statistics (`laya_computed_statistics.json`, `semif_computed_statistics.json`), and the exact adapters and configuration used for replay under `../benchmark/`. Pairwise comparisons with bootstrap intervals are in `JEV_VS_LAYA_COMPARISON_REPORT.md` and `JEV_VS_SEMIF_COMPARISON_REPORT.md`.

## Conclusion

Placing an open general-purpose model next to an open specialist and the hosted service separates the layers of a "Jev-like" system. The typed-decision readout is a solved, open, cheap layer that even a general 4B model implements with no special training. Semantic accuracy, calibration, very large candidate sets and constant-cost shared-state execution are the layers where Jev remains clearly ahead, and where neither a small specialist (Laya) nor a general small model with a prompted readout (SemIf) closes the gap.
