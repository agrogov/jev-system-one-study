# Changelog

## 1.0.0 — unified research release

- Consolidates architecture, probabilistic-calibration, and semantic-calibration suites into one codebase.
- Adds `config/architecture.yaml` as the canonical architecture config; `config/default.yaml` is retained for backward compatibility.
- Adds CLI `--profile`, `--concurrency`, `--model`, and `--output-dir` overrides for reproducible runs without editing YAML.
- Includes the HTTP/2 dependency via `httpx[http2]`.
- Documents `uv`-managed Python setup that does not modify macOS system Python.
- Adds exact commands for reproducing the five research-report runs.
- Adds installation/build verification and reproduction helper scripts.

## 0.3.x — semantic calibration development

- Added semantic ambiguity/adjudication suite and semantic report metrics.

## 0.2.x — probabilistic calibration development

- Added exact-probability, Bayesian, base-rate, representation-invariance, and multiclass calibration suites.

## 0.1.x — architecture/robustness development

- Initial API harness, raw capture, invariance, robustness, latency, question scaling, candidate scaling, and context scaling.
