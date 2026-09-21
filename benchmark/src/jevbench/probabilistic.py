from __future__ import annotations

import math
import random
from typing import Any

from .cases import Case, choice, noul


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _posterior_binary(prior: float, likelihood_if_true: float, likelihood_if_false: float) -> float:
    num = prior * likelihood_if_true
    den = num + (1.0 - prior) * likelihood_if_false
    if den <= 0:
        return prior
    return num / den


def _sample_bool(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def _sample_categorical(rng: random.Random, probs: dict[str, float]) -> str:
    x = rng.random()
    acc = 0.0
    last = next(reversed(probs))
    for key, p in probs.items():
        acc += p
        if x <= acc:
            return key
    return last


def _normalize(d: dict[str, float]) -> dict[str, float]:
    s = sum(d.values())
    if s <= 0:
        return {k: 1.0 / len(d) for k in d}
    return {k: v / s for k, v in d.items()}


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def explicit_randomness_cases(rng: random.Random, n: int) -> list[Case]:
    """Aleatoric uncertainty where the true probability is explicitly defined by the state.

    These cases do not have an observed binary outcome because the draw has not happened yet.
    The benchmark compares Jev's p(true) directly with the mathematically correct probability.
    """
    templates = [
        (
            "bag",
            lambda good, total: (
                f"A sealed bag contains {good} red balls and {total-good} blue balls. "
                "One ball will be drawn uniformly at random without looking. The draw has not happened yet.",
                "Before the draw occurs, is the randomly drawn ball red? Return the probability implied by the state, not a deterministic guess.",
            ),
        ),
        (
            "tickets",
            lambda good, total: (
                f"A raffle has exactly {total} equally likely tickets. {good} tickets are marked WIN and {total-good} are marked LOSE. "
                "One ticket will be selected uniformly at random. Selection has not occurred yet.",
                "Before selection, will the selected ticket be marked WIN? Express the uncertainty through the Noul probability.",
            ),
        ),
        (
            "quality_lot",
            lambda good, total: (
                f"A lot contains exactly {total} sealed components, of which {good} are certified grade A and {total-good} are grade B. "
                "One component will be sampled uniformly at random; no sample has been drawn yet.",
                "Is the randomly sampled component grade A? Use the probability warranted by the stated composition.",
            ),
        ),
        (
            "random_record",
            lambda good, total: (
                f"A dataset has {total} records and a uniformly random record will be selected. Exactly {good} records have flag=true and {total-good} have flag=false. "
                "The record has not yet been selected.",
                "Will the selected record have flag=true? Report the probability implied by the data-generating process.",
            ),
        ),
    ]
    # Deliberately dense coverage of 0.05..0.95 rather than only easy endpoints.
    probs = [i / 20 for i in range(1, 20)]
    out: list[Case] = []
    for i in range(n):
        q = probs[i % len(probs)]
        total = rng.choice([20, 40, 100])
        good = int(round(q * total))
        q = good / total
        family, render = templates[i % len(templates)]
        state, instructions = render(good, total)
        out.append(Case(
            "prob_explicit_randomness",
            state,
            {"target": noul(instructions)},
            {"target": {"type": "noul", "target_probability": q}},
            {
                "family": family,
                "target_probability": q,
                "probability_bin": round(q, 2),
                "index": i,
            },
        ).finalize())
    return out


def _binary_world_template(domain: str, prior: float, tpr: float, fpr: float, signal: bool) -> tuple[str, str]:
    signal_text = "POSITIVE" if signal else "NEGATIVE"
    if domain == "machine_fault":
        state = (
            f"In this synthetic fleet, before reading the diagnostic sensor, {_pct(prior)} of machines have latent fault X. "
            f"For a machine with fault X, the sensor is POSITIVE with probability {_pct(tpr)}. "
            f"For a machine without fault X, the sensor is POSITIVE with probability {_pct(fpr)}. "
            f"For the machine being evaluated, the observed sensor result is {signal_text}. "
            "Assume these rates are exact and the machine was sampled from this fleet."
        )
        question = "Given only the stated base rate and observed sensor result, does this machine have latent fault X? Return the posterior probability, not merely the most likely class."
    elif domain == "fraud":
        state = (
            f"In this synthetic transaction population, {_pct(prior)} of transactions are truly fraudulent before the detector output is known. "
            f"The detector outputs POSITIVE for a fraudulent transaction with probability {_pct(tpr)}, and outputs POSITIVE for a legitimate transaction with probability {_pct(fpr)}. "
            f"The current transaction's detector output is {signal_text}. Treat all probabilities as exact."
        )
        question = "After observing the detector output, is this transaction truly fraudulent? Encode the Bayesian posterior in the Noul probability."
    elif domain == "quality":
        state = (
            f"A factory's synthetic population has a defect base rate of {_pct(prior)}. "
            f"An inspection test is POSITIVE on a defective unit with probability {_pct(tpr)} and POSITIVE on a non-defective unit with probability {_pct(fpr)}. "
            f"The inspected unit produced a {signal_text} result. These rates are exact and stable."
        )
        question = "Given the test result and base rate, is the unit actually defective? Return the posterior probability rather than a hard decision."
    elif domain == "network_incident":
        state = (
            f"In a synthetic operations dataset, {_pct(prior)} of sampled intervals contain latent incident Z. "
            f"Alert A is POSITIVE in an interval with incident Z with probability {_pct(tpr)}, and is POSITIVE without incident Z with probability {_pct(fpr)}. "
            f"For this interval Alert A is {signal_text}. All stated rates are exact."
        )
        question = "Given the observed alert and the base rate, is latent incident Z present in this interval? Use the posterior probability."
    else:
        state = (
            f"A hidden binary condition Y has prior probability {_pct(prior)}. "
            f"Signal S is POSITIVE with probability {_pct(tpr)} when Y is true and {_pct(fpr)} when Y is false. "
            f"The observed signal is {signal_text}. Treat these probabilities as exact."
        )
        question = "Given the prior and observed signal, is Y true? Return the posterior probability."
    return state, question


def bayes_single_signal_cases(rng: random.Random, n: int) -> list[Case]:
    """Sample hidden worlds and one noisy signal; retain the exact Bayesian posterior."""
    priors = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
    sensors = [
        (0.60, 0.40),
        (0.70, 0.25),
        (0.80, 0.20),
        (0.90, 0.10),
        (0.95, 0.05),
        (0.99, 0.10),
    ]
    domains = ["machine_fault", "fraud", "quality", "network_incident", "abstract"]
    out: list[Case] = []
    for i in range(n):
        prior = priors[i % len(priors)]
        tpr, fpr = sensors[(i // len(priors)) % len(sensors)]
        y = _sample_bool(rng, prior)
        signal = _sample_bool(rng, tpr if y else fpr)
        if signal:
            post = _posterior_binary(prior, tpr, fpr)
        else:
            post = _posterior_binary(prior, 1 - tpr, 1 - fpr)
        domain = domains[i % len(domains)]
        state, question = _binary_world_template(domain, prior, tpr, fpr, signal)
        out.append(Case(
            "prob_bayes_single_signal",
            state,
            {"target": noul(question)},
            {"target": {"type": "noul", "label": int(y), "target_probability": post}},
            {
                "family": domain,
                "prior": prior,
                "tpr": tpr,
                "fpr": fpr,
                "signal_positive": signal,
                "target_probability": post,
                "hidden_label": int(y),
                "index": i,
            },
        ).finalize())
    return out


def bayes_two_signal_cases(rng: random.Random, n: int) -> list[Case]:
    """Two conditionally independent sensors, including conflicting evidence."""
    priors = [0.03, 0.08, 0.15, 0.30, 0.50, 0.70]
    sensor_pairs = [
        ((0.85, 0.15), (0.75, 0.25)),
        ((0.95, 0.10), (0.65, 0.20)),
        ((0.80, 0.30), (0.90, 0.10)),
        ((0.70, 0.20), (0.70, 0.20)),
    ]
    out: list[Case] = []
    for i in range(n):
        prior = priors[i % len(priors)]
        (tpr_a, fpr_a), (tpr_b, fpr_b) = sensor_pairs[(i // len(priors)) % len(sensor_pairs)]
        y = _sample_bool(rng, prior)
        a = _sample_bool(rng, tpr_a if y else fpr_a)
        b = _sample_bool(rng, tpr_b if y else fpr_b)
        like_t = (tpr_a if a else 1 - tpr_a) * (tpr_b if b else 1 - tpr_b)
        like_f = (fpr_a if a else 1 - fpr_a) * (fpr_b if b else 1 - fpr_b)
        post = _posterior_binary(prior, like_t, like_f)
        state = (
            f"A synthetic system has hidden condition Y with prior probability {_pct(prior)}. "
            f"Conditional on Y, sensors A and B are independent. Sensor A is POSITIVE with probability {_pct(tpr_a)} when Y is true and {_pct(fpr_a)} when Y is false. "
            f"Sensor B is POSITIVE with probability {_pct(tpr_b)} when Y is true and {_pct(fpr_b)} when Y is false. "
            f"Observed results: sensor A={'POSITIVE' if a else 'NEGATIVE'}; sensor B={'POSITIVE' if b else 'NEGATIVE'}. "
            "The stated probabilities are exact."
        )
        out.append(Case(
            "prob_bayes_two_signals",
            state,
            {"target": noul("Given the prior and both observed conditionally independent sensor results, is hidden condition Y true? Encode the Bayesian posterior in the Noul probability.")},
            {"target": {"type": "noul", "label": int(y), "target_probability": post}},
            {
                "prior": prior,
                "a_positive": a,
                "b_positive": b,
                "conflicting": a != b,
                "target_probability": post,
                "hidden_label": int(y),
                "index": i,
            },
        ).finalize())
    return out


def base_rate_stress_cases(rng: random.Random, n: int) -> list[Case]:
    """Low-prevalence/high-specificity cases designed to expose base-rate neglect."""
    priors = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    sensor = [(0.95, 0.01), (0.99, 0.02), (0.90, 0.005), (0.995, 0.05)]
    out: list[Case] = []
    for i in range(n):
        prior = priors[i % len(priors)]
        tpr, fpr = sensor[(i // len(priors)) % len(sensor)]
        # Force a positive signal to specifically probe posterior reasoning under rare priors.
        post = _posterior_binary(prior, tpr, fpr)
        # A hidden outcome is still sampled from the mathematically correct posterior so empirical
        # calibration can be evaluated without revealing the outcome to the model.
        y = _sample_bool(rng, post)
        state = (
            f"Consider a synthetic screening problem. Before the test, condition R is present in {_pct(prior)} of cases. "
            f"The test is POSITIVE in {_pct(tpr)} of cases with R and also POSITIVE in {_pct(fpr)} of cases without R. "
            "This case's test result is POSITIVE. Assume the rates are exact and there is no other evidence."
        )
        out.append(Case(
            "prob_base_rate_stress",
            state,
            {"target": noul("Given the base rate and positive test, is condition R present? Return the posterior probability, not the test sensitivity.")},
            {"target": {"type": "noul", "label": int(y), "target_probability": post}},
            {
                "prior": prior,
                "tpr": tpr,
                "fpr": fpr,
                "target_probability": post,
                "hidden_label": int(y),
                "index": i,
            },
        ).finalize())
    return out


def probability_representation_cases(rng: random.Random, groups: int) -> list[Case]:
    """Same Bayesian problem rendered in several mathematically equivalent surface forms."""
    out: list[Case] = []
    for g in range(groups):
        prior = rng.choice([0.1, 0.2, 0.35, 0.5, 0.7])
        tpr, fpr = rng.choice([(0.8, 0.2), (0.9, 0.1), (0.75, 0.25), (0.95, 0.05)])
        signal = rng.choice([True, False])
        post = _posterior_binary(prior, tpr if signal else 1-tpr, fpr if signal else 1-fpr)
        signal_text = "POSITIVE" if signal else "NEGATIVE"
        variants = [
            (
                "percent",
                f"Condition Y has prior probability {_pct(prior)}. A test is POSITIVE with probability {_pct(tpr)} when Y is true and {_pct(fpr)} when Y is false. The observed test is {signal_text}.",
            ),
            (
                "decimal",
                f"Condition Y has prior probability {prior:.4f}. P(test=POSITIVE | Y=true)={tpr:.4f}; P(test=POSITIVE | Y=false)={fpr:.4f}. The observed test is {signal_text}.",
            ),
            (
                "frequency",
                f"Imagine 10,000 exchangeable cases: about {round(prior*10000)} have Y. Among Y cases, about {round(tpr*10000)} per 10,000 tests are POSITIVE; among non-Y cases, about {round(fpr*10000)} per 10,000 tests are POSITIVE. This case's test is {signal_text}.",
            ),
        ]
        for variant, state in variants:
            out.append(Case(
                "prob_representation_invariance",
                state + " Treat the stated rates as exact.",
                {"target": noul("Given the prior and observed test, is Y true? Return the posterior probability.")},
                {"target": {"type": "noul", "target_probability": post}},
                {
                    "group": f"representation_{g:04d}",
                    "variant": variant,
                    "target_probability": post,
                    "index": g,
                },
            ).finalize())
    return out


def multinomial_bayes_cases(rng: random.Random, n: int) -> list[Case]:
    """Three-class Bayesian inference with a known posterior distribution."""
    classes = ["cause_a", "cause_b", "cause_c"]
    signals = ["RED", "AMBER", "GREEN"]
    configs = [
        (
            {"cause_a": 0.20, "cause_b": 0.50, "cause_c": 0.30},
            {
                "cause_a": {"RED": 0.70, "AMBER": 0.20, "GREEN": 0.10},
                "cause_b": {"RED": 0.20, "AMBER": 0.60, "GREEN": 0.20},
                "cause_c": {"RED": 0.10, "AMBER": 0.30, "GREEN": 0.60},
            },
        ),
        (
            {"cause_a": 0.60, "cause_b": 0.25, "cause_c": 0.15},
            {
                "cause_a": {"RED": 0.55, "AMBER": 0.30, "GREEN": 0.15},
                "cause_b": {"RED": 0.20, "AMBER": 0.50, "GREEN": 0.30},
                "cause_c": {"RED": 0.15, "AMBER": 0.25, "GREEN": 0.60},
            },
        ),
        (
            {"cause_a": 0.10, "cause_b": 0.20, "cause_c": 0.70},
            {
                "cause_a": {"RED": 0.80, "AMBER": 0.15, "GREEN": 0.05},
                "cause_b": {"RED": 0.30, "AMBER": 0.50, "GREEN": 0.20},
                "cause_c": {"RED": 0.05, "AMBER": 0.20, "GREEN": 0.75},
            },
        ),
    ]
    criteria = {
        "cause_a": "The hidden state is cause A.",
        "cause_b": "The hidden state is cause B.",
        "cause_c": "The hidden state is cause C.",
    }
    out: list[Case] = []
    for i in range(n):
        prior, matrix = configs[i % len(configs)]
        hidden = _sample_categorical(rng, prior)
        observed = _sample_categorical(rng, matrix[hidden])
        unnorm = {c: prior[c] * matrix[c][observed] for c in classes}
        posterior = _normalize(unnorm)
        state = (
            "A synthetic system has exactly one hidden cause: A, B, or C. "
            f"Prior probabilities are A={_pct(prior['cause_a'])}, B={_pct(prior['cause_b'])}, C={_pct(prior['cause_c'])}. "
            "A sensor emits one of RED, AMBER, GREEN. "
            f"If cause A: P(RED)={_pct(matrix['cause_a']['RED'])}, P(AMBER)={_pct(matrix['cause_a']['AMBER'])}, P(GREEN)={_pct(matrix['cause_a']['GREEN'])}. "
            f"If cause B: P(RED)={_pct(matrix['cause_b']['RED'])}, P(AMBER)={_pct(matrix['cause_b']['AMBER'])}, P(GREEN)={_pct(matrix['cause_b']['GREEN'])}. "
            f"If cause C: P(RED)={_pct(matrix['cause_c']['RED'])}, P(AMBER)={_pct(matrix['cause_c']['AMBER'])}, P(GREEN)={_pct(matrix['cause_c']['GREEN'])}. "
            f"The observed sensor output is {observed}. All probabilities are exact."
        )
        out.append(Case(
            "prob_multiclass_bayes",
            state,
            {"target": choice("Given the prior and observed sensor output, which hidden cause is present? The Choice probabilities should represent the Bayesian posterior over all three causes.", criteria)},
            {"target": {"type": "choice", "label": hidden, "target_distribution": posterior}},
            {
                "observed_signal": observed,
                "hidden_label": hidden,
                "target_entropy": -sum(p * math.log(max(p, 1e-12)) for p in posterior.values()),
                "index": i,
            },
        ).finalize())
    return out


def probabilistic_repeatability_cases(repetitions: int) -> list[Case]:
    """Repeat deliberately non-degenerate probabilities to measure stochastic variance."""
    specs = [
        ("p20", 0.2),
        ("p50", 0.5),
        ("p80", 0.8),
    ]
    out: list[Case] = []
    for name, p in specs:
        for rep in range(repetitions):
            state = (
                f"A box contains exactly 100 equally likely sealed cards. {round(p*100)} cards say YES and {100-round(p*100)} say NO. "
                "One card will be drawn uniformly at random; it has not been drawn yet."
            )
            out.append(Case(
                "prob_repeatability",
                state,
                {"target": noul("Before the draw, will the selected card say YES? Encode the known uncertainty in the probability.")},
                {"target": {"type": "noul", "target_probability": p}},
                {"group": name, "target_probability": p, "repetition": rep},
            ).finalize())
    return out


def build_probabilistic_suite(cfg: dict[str, Any]) -> list[Case]:
    seed = int(cfg["run"]["seed"])
    rng = random.Random(seed)
    profile = cfg["profiles"][cfg["run"]["profile"]]
    cases: list[Case] = []
    cases += explicit_randomness_cases(rng, int(profile.get("explicit_randomness_cases", 0)))
    cases += bayes_single_signal_cases(rng, int(profile.get("bayes_single_signal_cases", 0)))
    cases += bayes_two_signal_cases(rng, int(profile.get("bayes_two_signal_cases", 0)))
    cases += base_rate_stress_cases(rng, int(profile.get("base_rate_stress_cases", 0)))
    cases += probability_representation_cases(rng, int(profile.get("representation_groups", 0)))
    cases += multinomial_bayes_cases(rng, int(profile.get("multiclass_bayes_cases", 0)))
    cases += probabilistic_repeatability_cases(int(profile.get("repeatability_repetitions", 0)))
    return cases
