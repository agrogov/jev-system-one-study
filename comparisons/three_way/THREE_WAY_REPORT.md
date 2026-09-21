# Jev vs Laya vs SemIf: Three-Way Comparison

All three systems replay the same recorded `cases.jsonl` inputs. **Quality metrics use only decisions that Jev completed and that both Laya and SemIf consumed in full**, so the systems are scored on identical questions. Coverage and latency use every request. Jev is a hosted service; Laya (`aac6fef/laya-typed-decisions-mlx`, FP16) and SemIf (`Qwen/Qwen3.5-4B`, BF16, native MLX) run locally on Apple Silicon, so absolute latency is descriptive, not hardware-normalized. SemIf's `noul` and `score` results depend on this study's yes/no and ordinal option mapping (see the run manifests).

## 1. Accuracy on identical decisions

| Suite | Scored decisions | Jev | Laya | SemIf |
|---|---:|---:|---:|---:|
| architecture core | 438 | 98.86% | 86.07% | 100.00% |
| isolated core scaling | 137 | 100.00% | 77.37% | 100.00% |
| full boundary scaling | 265 | 100.00% | 66.42% | 92.45% |
| probability calibration | 4600 | 80.63% | 59.41% | 74.13% |
| semantic | 1110 | 90.36% | 44.05% | 69.10% |

## 2. Probability quality (exact-probability suite)

| Suite | Scored decisions | Metric | Jev | Laya | SemIf |
|---|---:|---:|---:|---:|---:|
| probability calibration | 4810 | Binary exact-probability MAE | 0.0793 | 0.2892 | 0.2526 |
| probability calibration | 1000 | Mean TV to exact 3-class posterior | 0.3010 | 0.3427 | 0.1831 |

## 3. Semantic suite

| Metric | Jev | Laya | SemIf |
|---|---:|---:|---:|
| Accuracy | 90.36% | 44.05% | 69.10% |
| Mean TV distance to reference | 0.2407 | 0.3940 | 0.3012 |
| Mean top probability (reference ≈ 0.677) | 0.882 | 0.417 | 0.851 |

Accuracy by domain:

| Domain | n | Jev | Laya | SemIf |
|---|---:|---:|---:|---:|
| code_review_severity | 210 | 98.10% | 28.10% | 76.67% |
| hardware_fault | 150 | 80.00% | 58.00% | 92.67% |
| incident_triage | 210 | 100.00% | 39.05% | 81.43% |
| policy_compliance | 150 | 93.33% | 43.33% | 74.67% |
| security_classification | 180 | 100.00% | 76.67% | 52.22% |
| support_routing | 210 | 70.00% | 27.62% | 42.86% |

## 4. Coverage: requests each system consumed in full (comparable / total)

| Suite | Jev | Laya | SemIf |
|---|---:|---:|---:|
| architecture core | 303/303 | 297/303 | 300/303 |
| isolated core scaling | 26/26 | 20/26 | 23/26 |
| full boundary scaling | 31/31 | 21/31 | 27/31 |
| probability calibration | 5810/5810 | 5810/5810 | 5810/5810 |
| semantic | 1110/1110 | 1110/1110 | 1110/1110 |

Laya's shortfall is context truncation and unsupported cardinality; SemIf's is unsupported cardinality only (more than 16 options). Jev completed every request.

## 5. Scaling (server-side time; outcome shown when a request was not fully consumed)

### Questions per request

| Questions per request | Jev server time | Laya server time | SemIf server time |
|---|---:|---:|---:|
| 1 | 99 ms | 81 ms | 514 ms |
| 2 | 103 ms | 89 ms | 556 ms |
| 4 | 101 ms | 159 ms | 685 ms |
| 8 | 107 ms | 300 ms | 896 ms |
| 16 | 88 ms | 610 ms | 1377 ms |
| 32 | 131 ms | 1177 ms | 2369 ms |
| 64 | 96 ms | 2368 ms | 4382 ms |
| 128 | 167 ms | 9697 ms | 14301 ms |

### Choice cardinality

| Choices | Jev server time | Laya server time | SemIf server time |
|---|---:|---:|---:|
| 2 | 72 ms | 15 ms | 162 ms |
| 4 | 77 ms | 16 ms | 214 ms |
| 8 | 92 ms | 21 ms | 276 ms |
| 16 | 76 ms | 30 ms | 423 ms |
| 32 | 121 ms | 29 ms | (unsupported_cardinality) |
| 64 | 70 ms | 32 ms | (unsupported_cardinality) |
| 128 | 104 ms | 58 ms | (unsupported_cardinality) |
| 255 | 113 ms | (unsupported_cardinality) | (unsupported_cardinality) |

### Context length

| Context chars | Jev server time | Laya server time | SemIf server time |
|---|---:|---:|---:|
| 500 | 119 ms | 19 ms | 275 ms |
| 2000 | 100 ms | 43 ms | 528 ms |
| 8000 | 94 ms | 87 ms (truncated_input) | 1528 ms |
| 32000 | 113 ms | 90 ms (truncated_input) | 5938 ms |
| 100000 | 208 ms | 99 ms (truncated_input) | 25106 ms |

## 6. Median wall latency per request (ms)

| Suite | Jev | Laya | SemIf |
|---|---:|---:|---:|
| architecture core | 332.5 | 12.8 | 163.9 |
| isolated core scaling | 323.8 | 66.5 | 554.0 |
| full boundary scaling | 329.6 | 88.0 | 1504.9 |
| probability calibration | 321.4 | 13.6 | 206.1 |
| semantic | 332.3 | 18.3 | 208.5 |

## Interpretation constraints

- SemIf is an independent open-model reproduction of the typed-decision interface pattern, not of Jev's model.
- Laya is a specialist typed-decisions checkpoint; SemIf uses a general 4B model with a prompted readout. Neither is a claim about Jev's architecture.
- Latency: Jev is a remote service; Laya and SemIf are local. Scaling shape is more informative than absolute values.
- Semantic references are authored adjudication distributions, not independent human-panel measurements.
