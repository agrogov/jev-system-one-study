# Jev Black-Box Benchmark — Unified Research Harness

This repository is the **single authoritative codebase** used for the Jev/System One black-box experiments in the accompanying research report.

It consolidates the earlier benchmark iterations into one package and reproduces all experiment families used in the study:

- architecture / robustness / latency scaling;
- exact probabilistic calibration and Bayesian posterior recovery;
- semantic calibration against authored adjudication distributions.

The benchmark calls the public TypeSafe System One API directly, records immutable raw responses, normalizes typed decisions, computes metrics, and generates per-run Markdown reports and figures.

> The harness is for behavioral evaluation. It does not attempt to recover weights, training data, or proprietary implementation details, and it does not implement model distillation.

---

## 1. Repository layout

```text
benchmark/
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── .env.example
├── config/
│   ├── architecture.yaml     # architecture, robustness, scaling
│   ├── default.yaml          # compatibility copy of architecture.yaml
│   ├── calibration.yaml      # exact probabilistic/Bayesian suite
│   └── semantic.yaml         # semantic ambiguity/calibration suite
├── data/
│   └── semantic_reference.jsonl
├── scripts/
│   ├── check_install.sh
│   └── reproduce_five_runs.sh
├── src/jevbench/
│   ├── api.py
│   ├── cases.py
│   ├── cli.py
│   ├── metrics.py
│   ├── probabilistic.py
│   ├── report.py
│   ├── runner.py
│   ├── semantic.py
│   └── util.py
└── results/
    └── .gitkeep
```

`data/semantic_reference.jsonl` is an auditable snapshot of the semantic reference scenarios. The semantic suite is generated from the same authored scenario definitions in `src/jevbench/semantic.py`; these are **not independently collected human-panel votes**.

---

## 2. Requirements

- macOS or Linux
- Python **3.10+**; Python **3.12** is recommended because the recorded study runs used Python 3.12.13
- TypeSafe API key in `TYPESAFE_API_KEY`
- network access to `https://api.typesafe.ai`

The study machine used a user-managed Python 3.12 environment. There is no reason to replace or upgrade the operating system's Python installation.

---

## 3. Recommended installation with `uv`

This is the safest setup on macOS because `uv` can manage a separate Python interpreter without modifying the system Python.

### 3.1 Install `uv`

If `uv` is already installed, skip this step.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new shell, or source the shell profile that the installer updated.

### 3.2 Install a user-managed Python 3.12

```bash
uv python install 3.12
```

This installs a separate Python under `uv`'s user-managed directory; it does not replace `/usr/bin/python3`.

### 3.3 Create the virtual environment

From the repository root:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
python --version
```

Expected:

```text
Python 3.12.x
```

`--seed` installs `pip` into the virtual environment.

### 3.4 Install the benchmark

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The package declares `httpx[http2]`, so the HTTP/2 `h2` dependency is installed automatically.

Verify:

```bash
jevbench --help
./scripts/check_install.sh
```

---

## 4. Alternative installation with an existing Python 3.10+

If you already have a suitable non-system Python interpreter:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

A virtual environment uses the interpreter with which it was created; it does not itself upgrade Python.

---

## 5. Build a wheel and source distribution

Activate the virtual environment and install the optional build dependency:

```bash
python -m pip install -e '.[dev]'
python -m build
```

Artifacts are written to:

```text
dist/
├── jev_blackbox_benchmark-1.0.0-py3-none-any.whl
└── jev_blackbox_benchmark-1.0.0.tar.gz
```

With `uv`, the equivalent build command is:

```bash
uv build
```

Install the built wheel with:

```bash
python -m pip install dist/jev_blackbox_benchmark-1.0.0-py3-none-any.whl
```

---

## 6. Configure credentials

Export the API key in the shell:

```bash
export TYPESAFE_API_KEY='YOUR_KEY'
```

Do not commit API keys. `.env` is ignored, but the harness intentionally does not load it automatically; use shell environment variables or a secret manager.

Optional verification:

```bash
test -n "$TYPESAFE_API_KEY" && echo 'TYPESAFE_API_KEY is set'
```

---

## 7. CLI overview

The package installs the `jevbench` command.

```bash
jevbench --help
```

The main commands are:

```text
jevbench ... generate    build/inspect cases without API calls
jevbench ... run         call Jev, record raw data, analyze, report
jevbench ... analyze     recompute metrics for an existing run directory
jevbench ... report      regenerate Markdown report/figures for a run
```

### Runtime overrides

You do **not** need to edit YAML for common changes:

```text
--profile PROFILE
--concurrency N
--model MODEL
--output-dir PATH
```

`--only EXPERIMENT` can be repeated to execute a subset of experiments.

---

## 8. Inspect suites without spending API calls

Architecture core:

```bash
jevbench --config config/architecture.yaml generate --profile core --list
```

Expected total:

```text
profile=core cases=303
```

Architecture full:

```bash
jevbench --config config/architecture.yaml generate --profile full --list
```

Expected total:

```text
profile=full cases=1048
```

Probabilistic calibration full:

```bash
jevbench --config config/calibration.yaml generate --profile calibration_full --list
```

Expected total:

```text
profile=calibration_full cases=5810
```

Semantic full:

```bash
jevbench --config config/semantic.yaml generate --profile semantic_full --list
```

Expected total:

```text
profile=semantic_full cases=1110
```

---

## 9. The five study runs reproduced by the report

The research report is based on the following five result directories. The final research bundle includes all five raw folders under `../jev-results/`.

### Run 1 — architecture/robustness core, concurrent

Recorded result:

```text
run_20260919T201050Z_core
303 requests
concurrency = 4
```

Reproduce:

```bash
jevbench --config config/architecture.yaml run \
  --profile core \
  --concurrency 4
```

Purpose: correctness, invariance, robustness, repeatability, cross-domain behavior, initial scaling, and throughput-oriented timing.

### Run 2 — isolated architecture core probes

Recorded result:

```text
run_20260919T201739Z_core
26 requests
concurrency = 1
```

Reproduce:

```bash
jevbench --config config/architecture.yaml run \
  --profile core \
  --concurrency 1 \
  --only parallel_question_scaling \
  --only choice_cardinality_scaling \
  --only context_length_position
```

Purpose: remove client-side concurrency as a confounder in architectural latency inference.

### Run 3 — isolated architecture full boundary probes

Recorded result:

```text
run_20260919T202139Z_full
31 requests
concurrency = 1
```

Reproduce:

```bash
jevbench --config config/architecture.yaml run \
  --profile full \
  --concurrency 1 \
  --only parallel_question_scaling \
  --only choice_cardinality_scaling \
  --only context_length_position
```

Purpose: extend the architecture probe to 128 questions, 255 candidates, and ~100k-character state construction.

### Run 4 — exact probabilistic calibration full

Recorded result:

```text
run_20260919T203758Z_calibration_full
5810 requests
concurrency = 4
```

Reproduce:

```bash
jevbench --config config/calibration.yaml run \
  --profile calibration_full \
  --concurrency 4
```

Purpose: direct probability recovery, exact Bayesian posterior recovery, base-rate stress, representation invariance, multiclass posterior recovery, and repeatability.

### Run 5 — semantic calibration full

Recorded result:

```text
run_20260919T211437Z_semantic_full
1110 requests
concurrency = 4
```

Reproduce:

```bash
jevbench --config config/semantic.yaml run \
  --profile semantic_full \
  --concurrency 4
```

Purpose: ambiguity-sensitive semantic decisions across incident triage, hardware attribution, security classification, support routing, code review, and policy compliance.

A convenience script prints and optionally executes these same commands:

```bash
./scripts/reproduce_five_runs.sh --print
./scripts/reproduce_five_runs.sh --run
```

Running all five consumes thousands of API requests. Use `--print` first.

---

## 10. Architecture / robustness suite

Configuration:

```text
config/architecture.yaml
```

Profiles:

```text
smoke
core
full
```

Experiment families include:

- deterministic Noul correctness;
- negation handling;
- Choice routing;
- Score behavior;
- candidate-order invariance;
- opaque candidate-key invariance;
- question paraphrase invariance;
- question independence;
- irrelevant-context robustness;
- evidence removal;
- contradiction sensitivity;
- state prompt-injection resilience;
- repeatability;
- parallel-question scaling;
- Choice-cardinality scaling;
- cross-domain semantic cases;
- context length / evidence-position scaling.

For architectural latency conclusions use `--concurrency 1` and prefer the response header:

```text
x-envoy-upstream-service-time
```

over client wall-clock latency, because client timing includes network/TLS/proxy overhead.

---

## 11. Exact probabilistic calibration suite

Configuration:

```text
config/calibration.yaml
```

Profiles:

```text
calibration_smoke
calibration_core
calibration_full
```

Experiment families:

### `prob_explicit_randomness`

Direct aleatoric probability expressed in auditable random processes.

### `prob_bayes_single_signal`

Known prior plus one noisy detector. The harness stores both the hidden sampled outcome and analytically exact Bayesian posterior.

### `prob_bayes_two_signals`

Two conditionally independent signals, including conflicting evidence.

### `prob_base_rate_stress`

Rare priors with apparently strong positive evidence, designed to reveal base-rate neglect.

### `prob_representation_invariance`

The same probability problem expressed using percentages, decimal probabilities, or natural frequencies.

### `prob_multiclass_bayes`

Three-class Choice task with exact posterior distribution known analytically. Full returned distributions are evaluated using TV, JSD, KL, and soft Brier error.

### `prob_repeatability`

Repeated non-degenerate 0.2/0.5/0.8 probability cases to measure stochastic variation away from saturated 0/1 outputs.

---

## 12. Semantic calibration suite

Configuration:

```text
config/semantic.yaml
```

Profiles:

```text
semantic_smoke
semantic_core
semantic_full
```

Domains:

- incident triage;
- hardware fault attribution;
- security classification;
- support routing;
- code-review severity;
- policy compliance.

Each scenario has an authored reference disagreement distribution. These distributions are intentionally explicit and reproducible but must **not** be described as real human-panel frequencies.

The suite varies:

- option order;
- instruction paraphrase;
- irrelevant context;
- repeated ambiguous cases.

It measures full-distribution distance, top-class accuracy, entropy/sharpness, overconfidence relative to the authored reference distribution, and selective risk/coverage.

---

## 13. Output directory format

Every `run` creates a timestamped directory under `results/`:

```text
results/run_<UTC timestamp>_<profile>/
├── manifest.json
├── cases.jsonl
├── raw.jsonl
├── normalized.csv
├── summary.json
├── REPORT.md
└── figures/
```

### `manifest.json`

Contains the exact configuration, model selector, profile, seed, Python/platform information, case count, and elapsed time.

### `cases.jsonl`

Generated benchmark inputs and independently defined expected/reference data.

### `raw.jsonl`

The most important provenance file. One record per HTTP request, including exact request body, raw Jev response, usage accounting, selected response headers, wall-clock latency, retries, and errors.

Preserve this file even if a run is interrupted.

### `normalized.csv`

One normalized row per scored question/decision, suitable for independent analysis.

### `summary.json`

Machine-readable per-run metrics.

### `REPORT.md` and `figures/`

Compact automatically generated run report and figures. The integrated research paper in the outer bundle recomputes important metrics independently from these raw results.

---

## 14. Analyze or regenerate an existing run

```bash
jevbench --config config/architecture.yaml analyze results/run_...
jevbench --config config/architecture.yaml report results/run_...
```

The config is loaded by the CLI, but analysis/reporting primarily use the files inside the specified run directory.

---

## 15. Reproducibility notes

- All study runs used seed `260919`.
- The observed served model version in raw responses was `jev-1.13.0`, although the API request used `jev-latest`.
- `jev-latest` can change over time. A future rerun may therefore test a newer model even with identical benchmark code.
- Preserve `raw.jsonl`, `manifest.json`, and response headers for longitudinal comparisons.
- Timing is inherently noisy. Architectural timing claims should be based on isolated `concurrency=1` runs and multiple repetitions when possible.
- Returned probabilities are rounded in the public response representation; derived scores/confidence may reflect finer internal precision.

---

## 16. Interpreting metrics

### Calibration is not accuracy

A model can be accurate but poorly calibrated, or calibrated but uninformative. The benchmark reports both decision correctness and probability/distribution metrics.

### Binary probability recovery

Where an exact target probability `q` is known, the suite measures quantities such as:

```text
MAE   = mean(|p - q|)
RMSE  = sqrt(mean((p - q)^2))
```

as well as empirical Brier/NLL and reliability metrics when sampled outcomes are available.

### Choice distribution recovery

Where the entire reference distribution is known, the suite evaluates:

- total variation distance;
- Jensen-Shannon divergence;
- KL divergence;
- soft Brier error;
- entropy/sharpness;
- top-class accuracy.

### Selective prediction

Risk/coverage asks whether rejecting lower-confidence decisions reduces error. A confidence score can be operationally useful for routing even if its numeric value is not a universally calibrated posterior probability.

---

## 17. Legal / publication caution

Review your TypeSafe agreement before publishing benchmark results or using API outputs for any training or imitation purpose. This repository records outputs for private behavioral evaluation and reproducibility; it does not implement distillation.

---

## 18. Version history

This unified `1.0.0` repository supersedes the three working benchmark packages used during development:

- initial architecture/robustness harness;
- v0.2 probabilistic-calibration extension;
- v0.3 semantic-calibration extension.

All experiment generators, metrics, reports, and configs are now present in this single tree. `config/default.yaml` remains only for compatibility and is identical to `config/architecture.yaml`.

<!-- LAYA-COMPARISON-EXTENSION -->

## Laya replay comparison

A local Laya provider/replay adapter is available in this study tree. It replays the exact five recorded Jev `cases.jsonl` inputs and generates directly comparable raw and normalized results.

See [`LAYA_COMPARISON.md`](LAYA_COMPARISON.md) for Apple Silicon/MLX installation, exact commands, fairness constraints, output structure, specialist-vs-base checkpoint guidance, and comparison reporting.

See [`SEMIF_COMPARISON.md`](SEMIF_COMPARISON.md) for the SemIf (Qwen3.5-4B, MLX) replay: installation, commands, primitive mapping, and reported limits.
