from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from .cases import Case, choice


# IMPORTANT: These are benchmark-authored adjudication distributions, not human-study data.
# They are intentionally explicit and auditable. Replace them with independent panel votes
# through the same target_distribution schema for publication-grade claims.

COMMON_NOISE = [
    "The dashboard theme was changed last week.",
    "A documentation cleanup is scheduled for next month.",
    "The office network maintenance window is on Saturday.",
    "A separate team is reviewing travel expenses.",
    "The weekly status meeting was moved by thirty minutes.",
    "The customer success newsletter was sent yesterday.",
]


def _dist(votes: dict[str, int]) -> dict[str, float]:
    total = sum(votes.values())
    return {k: v / total for k, v in votes.items()}


def _mode(dist: dict[str, float]) -> str:
    return max(dist, key=dist.get)


def _entropy_norm(dist: dict[str, float]) -> float:
    vals = [v for v in dist.values() if v > 0]
    if len(vals) <= 1:
        return 0.0
    return -sum(v * math.log(v) for v in vals) / math.log(len(dist))


def _spec(
    domain: str,
    scenario_id: str,
    state: str,
    instructions: str,
    criteria: dict[str, str],
    votes: dict[str, int],
    rationale: str,
) -> dict[str, Any]:
    d = _dist(votes)
    return {
        "domain": domain,
        "scenario_id": scenario_id,
        "state": state,
        "instructions": instructions,
        "criteria": criteria,
        "votes": votes,
        "distribution": d,
        "label": _mode(d),
        "reference_entropy_norm": _entropy_norm(d),
        "rationale": rationale,
    }


INCIDENT_CRITERIA = {
    "sev1": "Critical incident: broad production outage, severe safety impact, or immediate major loss; immediate incident command is warranted.",
    "sev2": "Major incident: material customer or production impact with significant degradation, but not a broad total outage or immediate catastrophic loss.",
    "sev3": "Minor incident: limited degradation or a workaround exists; normal incident handling is sufficient.",
    "insufficient": "The evidence is insufficient or too contradictory to assign an incident severity reliably.",
}

HARDWARE_CRITERIA = {
    "pcie_transport": "PCIe link, signal-integrity, slot, riser, or transport instability is the primary fault domain.",
    "power_delivery": "Power-supply, cable, connector, or transient power delivery is the primary fault domain.",
    "device_fault": "The accelerator/device itself, including device memory or silicon, is the primary fault domain.",
    "insufficient": "The supplied evidence does not distinguish the listed fault domains reliably.",
}

SECURITY_CRITERIA = {
    "credential_attack": "Credential stuffing, password spraying, brute-force login abuse, or account takeover attempt.",
    "malware": "Malware execution, persistence, or malicious payload activity on an endpoint.",
    "data_exfiltration": "Unauthorized bulk or targeted extraction of data is the primary observed pattern.",
    "insufficient": "The evidence is insufficient or conflicting for a reliable classification.",
}

SUPPORT_CRITERIA = {
    "billing": "Payments, invoices, refunds, renewals, duplicate charges, or subscription billing.",
    "technical": "Software malfunction, API/integration failure, crash, connectivity, or product defect.",
    "account": "Login, identity, permissions, ownership, profile, or account access.",
    "insufficient": "The request lacks enough information to route reliably.",
}

CODE_CRITERIA = {
    "low": "Low severity: style/readability issue, minor maintainability concern, or non-impacting cleanup.",
    "medium": "Medium severity: plausible defect or reliability problem with bounded impact and an available workaround.",
    "high": "High severity: likely production defect, security/reliability risk, data loss/corruption risk, or broad user impact.",
    "insufficient": "The snippet/context is insufficient to assess severity reliably.",
}

POLICY_CRITERIA = {
    "allowed": "The action is clearly allowed by the stated policy.",
    "review": "The action may be allowed but requires documented approval, exception review, or additional conditions.",
    "prohibited": "The action clearly violates the stated policy.",
    "insufficient": "The facts or policy text are insufficient to decide reliably.",
}


def _base_specs() -> list[dict[str, Any]]:
    s: list[dict[str, Any]] = []

    # Incident triage: clear -> ambiguous -> conflicting -> insufficient.
    s += [
        _spec("incident_triage", "inc_clear_sev1",
              "Monitoring confirms that all public API traffic in every region is failing. There is no working customer workaround and transaction processing has stopped globally.",
              "Assign the incident severity best supported by the state.", INCIDENT_CRITERIA,
              {"sev1": 19, "sev2": 1, "sev3": 0, "insufficient": 0},
              "Broad total production outage with no workaround strongly supports SEV1."),
        _spec("incident_triage", "inc_border_12",
              "One of three production regions is unavailable. About 38% of customers are failing requests; traffic can be manually shifted to the other regions, but capacity headroom is low and some customers continue to fail.",
              "Assign the incident severity best supported by the state.", INCIDENT_CRITERIA,
              {"sev1": 7, "sev2": 12, "sev3": 1, "insufficient": 0},
              "Material impact suggests SEV2, but low headroom and broad customer failures make SEV1 plausible."),
        _spec("incident_triage", "inc_border_23",
              "A background export feature is failing for roughly 12% of users. Core transactions are unaffected. A manual export workaround is available, but it takes several extra minutes per customer.",
              "Assign the incident severity best supported by the state.", INCIDENT_CRITERIA,
              {"sev1": 0, "sev2": 7, "sev3": 13, "insufficient": 0},
              "Limited impact and workaround lean SEV3, though some organizations might classify repeated customer impact as SEV2."),
        _spec("incident_triage", "inc_insufficient",
              "A customer reports that 'the service is broken.' Internal monitoring shows no global alert. No affected feature, region, error rate, duration, or workaround information is available yet.",
              "Assign the incident severity best supported by the state.", INCIDENT_CRITERIA,
              {"sev1": 1, "sev2": 2, "sev3": 4, "insufficient": 13},
              "There is too little impact evidence to assign a reliable severity."),
    ]

    s += [
        _spec("hardware_fault", "hw_pcie_clear",
              "Under sustained GPU load the device disappears from lspci. The kernel logs repeated PCIe AER BadDLLP and Replay Timer Timeout events immediately before removal. The same GPU passes memory tests in another machine and the PSU rails remain stable.",
              "Which fault domain is best supported by the evidence?", HARDWARE_CRITERIA,
              {"pcie_transport": 19, "power_delivery": 0, "device_fault": 1, "insufficient": 0},
              "Transport errors plus cross-system device stability strongly implicate PCIe transport."),
        _spec("hardware_fault", "hw_power_border",
              "The GPU resets only during sudden load transitions. No PCIe AER messages are recorded. The system uses a marginally rated PSU and an adapter cable that becomes warm; reducing the GPU power limit by 25% stops the resets.",
              "Which fault domain is best supported by the evidence?", HARDWARE_CRITERIA,
              {"pcie_transport": 2, "power_delivery": 15, "device_fault": 2, "insufficient": 1},
              "Power sensitivity strongly favors delivery, though device or transport issues remain possible."),
        _spec("hardware_fault", "hw_conflict",
              "The GPU reports intermittent ECC-like memory errors under load, but the same crashes are preceded by PCIe link retraining events. Moving the card to another slot reduces link errors but does not eliminate the memory errors. A different PSU makes no change.",
              "Which fault domain is best supported by the evidence?", HARDWARE_CRITERIA,
              {"pcie_transport": 8, "power_delivery": 0, "device_fault": 8, "insufficient": 4},
              "Evidence points to both transport and device-level problems; a single primary cause is unclear."),
        _spec("hardware_fault", "hw_insufficient",
              "The workstation sometimes reboots while a GPU workload is running. No kernel, PCIe, PSU telemetry, thermal data, memory diagnostics, or cross-machine tests are available.",
              "Which fault domain is best supported by the evidence?", HARDWARE_CRITERIA,
              {"pcie_transport": 2, "power_delivery": 4, "device_fault": 2, "insufficient": 12},
              "The symptom is compatible with multiple fault domains and evidence is insufficient."),
    ]

    s += [
        _spec("security_classification", "sec_credential_clear",
              "Authentication logs show 14,000 failed password attempts against 2,300 accounts from a rotating set of IPs, followed by 27 successful logins using passwords. There are no password reset events and no endpoint execution alerts.",
              "Which security pattern is best supported?", SECURITY_CRITERIA,
              {"credential_attack": 19, "malware": 0, "data_exfiltration": 0, "insufficient": 1},
              "High-volume distributed login failures followed by successes strongly indicate credential abuse."),
        _spec("security_classification", "sec_exfil_border",
              "A service account downloaded 18 GB of customer exports overnight, far above its historical baseline. The account had valid credentials and there is no confirmed compromise. The exports were sent to an approved corporate storage endpoint.",
              "Which security pattern is best supported?", SECURITY_CRITERIA,
              {"credential_attack": 2, "malware": 0, "data_exfiltration": 8, "insufficient": 10},
              "The volume is suspicious, but destination and valid access make confirmed exfiltration uncertain."),
        _spec("security_classification", "sec_malware_border",
              "An endpoint spawned PowerShell from a document viewer and contacted a newly registered domain. EDR blocked the child process before payload execution. No persistence or data transfer is observed.",
              "Which security pattern is best supported?", SECURITY_CRITERIA,
              {"credential_attack": 0, "malware": 15, "data_exfiltration": 1, "insufficient": 4},
              "Behavior is strongly malware-like, but execution was blocked before full confirmation."),
        _spec("security_classification", "sec_insufficient",
              "A user reports that their laptop was 'acting strangely' after opening an email. No EDR telemetry, process tree, network logs, authentication anomalies, or file samples are available.",
              "Which security pattern is best supported?", SECURITY_CRITERIA,
              {"credential_attack": 2, "malware": 4, "data_exfiltration": 0, "insufficient": 14},
              "The report is too vague for a reliable security classification."),
    ]

    s += [
        _spec("support_routing", "sup_billing_clear",
              "I cancelled last week but my card was charged for another month. Please reverse the renewal charge.",
              "Which team should own this request?", SUPPORT_CRITERIA,
              {"billing": 20, "technical": 0, "account": 0, "insufficient": 0},
              "Explicit renewal charge/refund request is billing."),
        _spec("support_routing", "sup_account_technical_border",
              "I can log in on the website, but the desktop app says my organization membership cannot be verified and refuses to open the workspace. Reinstalling did not help.",
              "Which team should own this request?", SUPPORT_CRITERIA,
              {"billing": 0, "technical": 9, "account": 10, "insufficient": 1},
              "The symptom crosses product malfunction and account/membership verification."),
        _spec("support_routing", "sup_billing_account_border",
              "My subscription belongs to an old company email I can no longer access. I need to move the paid plan to my new account without losing the remaining subscription period.",
              "Which team should own this request?", SUPPORT_CRITERIA,
              {"billing": 9, "technical": 0, "account": 10, "insufficient": 1},
              "Ownership transfer mixes account access with subscription/billing concerns."),
        _spec("support_routing", "sup_insufficient",
              "Nothing works and I need help urgently. Please fix my account.",
              "Which team should own this request?", SUPPORT_CRITERIA,
              {"billing": 2, "technical": 3, "account": 5, "insufficient": 10},
              "The word account gives a weak clue, but there is not enough concrete information."),
    ]

    s += [
        _spec("code_review_severity", "code_high_clear",
              "The payment handler writes the debit to the database and then calls the external gateway. If the gateway call times out, the retry path executes the debit again because there is no idempotency key or uniqueness guard.",
              "Assess the severity of the code issue described.", CODE_CRITERIA,
              {"low": 0, "medium": 1, "high": 19, "insufficient": 0},
              "Duplicate financial debits are a high-severity correctness/data-integrity issue."),
        _spec("code_review_severity", "code_medium_border",
              "A cache lookup is performed without a timeout. In normal conditions it returns in milliseconds; if the cache stalls, the request thread can block until the TCP stack times out. The endpoint is non-critical and callers retry automatically.",
              "Assess the severity of the code issue described.", CODE_CRITERIA,
              {"low": 2, "medium": 14, "high": 3, "insufficient": 1},
              "Reliability defect is real but impact is bounded and retries exist."),
        _spec("code_review_severity", "code_low_medium",
              "A helper duplicates about 25 lines of validation logic used by two endpoints. Current behavior is correct and tests cover both copies, but future changes could diverge.",
              "Assess the severity of the code issue described.", CODE_CRITERIA,
              {"low": 12, "medium": 7, "high": 0, "insufficient": 1},
              "Primarily maintainability risk, with some chance of future correctness drift."),
        _spec("code_review_severity", "code_insufficient",
              "The review note says: 'This function looks dangerous and could maybe corrupt data.' No code, call path, tests, inputs, or failure example is provided.",
              "Assess the severity of the code issue described.", CODE_CRITERIA,
              {"low": 1, "medium": 3, "high": 3, "insufficient": 13},
              "Severity cannot be assessed from an unsupported assertion."),
    ]

    s += [
        _spec("policy_compliance", "policy_allowed_clear",
              "Policy: production logs may be retained for up to 30 days in the approved logging platform. The team plans to retain production logs for 14 days in that approved platform.",
              "Classify the proposed action under the stated policy.", POLICY_CRITERIA,
              {"allowed": 20, "review": 0, "prohibited": 0, "insufficient": 0},
              "The action is directly within the stated allowance."),
        _spec("policy_compliance", "policy_review_border",
              "Policy: customer data may not be exported to external systems unless the Security team approves a documented exception. A team wants to export a sanitized customer sample to a new external analytics vendor; no exception has yet been approved.",
              "Classify the proposed action under the stated policy.", POLICY_CRITERIA,
              {"allowed": 0, "review": 17, "prohibited": 3, "insufficient": 0},
              "The policy provides an exception path, but approval is required before proceeding."),
        _spec("policy_compliance", "policy_prohibited_clear",
              "Policy: API credentials must never be committed to source control, including private repositories. The proposal is to place a production API key directly in a private Git repository so deployment scripts can read it.",
              "Classify the proposed action under the stated policy.", POLICY_CRITERIA,
              {"allowed": 0, "review": 0, "prohibited": 20, "insufficient": 0},
              "The proposal directly violates an explicit prohibition."),
        _spec("policy_compliance", "policy_insufficient",
              "Policy excerpt: sensitive information requires appropriate safeguards. Proposal: move a dataset to another internal system. The excerpt does not define whether the dataset is sensitive or what safeguards the destination provides.",
              "Classify the proposed action under the stated policy.", POLICY_CRITERIA,
              {"allowed": 2, "review": 5, "prohibited": 1, "insufficient": 12},
              "Both classification of the data and safeguards are unspecified."),
    ]
    return s


PARAPHRASES = [
    "Use only the evidence in the state and select the best-supported option.",
    "Choose the option that is most justified by the supplied facts.",
    "Classify this case using the criteria below; do not assume facts that are not stated.",
    "Based strictly on the available evidence, which option is most appropriate?",
]


def build_semantic_suite(cfg: dict[str, Any]) -> list[Case]:
    seed = int(cfg["run"]["seed"])
    rng = random.Random(seed)
    profile = cfg["profiles"][cfg["run"]["profile"]]
    variants_per_scenario = int(profile.get("variants_per_scenario", 6))
    noise_variants = list(profile.get("noise_units", [0, 2, 6]))
    include_key_permutations = bool(profile.get("permute_option_order", True))

    specs = _base_specs()
    cases: list[Case] = []
    for spec in specs:
        for i in range(variants_per_scenario):
            criteria_items = list(spec["criteria"].items())
            if include_key_permutations:
                rng.shuffle(criteria_items)
            criteria = dict(criteria_items)
            noise_n = noise_variants[i % len(noise_variants)] if noise_variants else 0
            state = spec["state"]
            if noise_n:
                state += "\n\nUnrelated operational notes:\n" + " ".join(rng.choice(COMMON_NOISE) for _ in range(noise_n))
            instruction = spec["instructions"] + " " + PARAPHRASES[i % len(PARAPHRASES)]
            dist = dict(spec["distribution"])
            label = spec["label"]
            cases.append(Case(
                "semantic_adjudication",
                state,
                {"target": choice(instruction, criteria)},
                {"target": {"type": "choice", "label": label, "target_distribution": dist}},
                {
                    "domain": spec["domain"],
                    "scenario_id": spec["scenario_id"],
                    "variant": i,
                    "noise_units": noise_n,
                    "reference_entropy_norm": spec["reference_entropy_norm"],
                    "reference_max_probability": max(dist.values()),
                    "panel_votes": ",".join(f"{k}:{v}" for k, v in spec["votes"].items()),
                    "reference_source": "benchmark_authored_adjudication",
                    "rationale": spec["rationale"],
                },
            ).finalize())

    # Add deliberately semantically equivalent duplicates to measure stability around ambiguous cases.
    repeats = int(profile.get("ambiguous_repeat_repetitions", 0))
    ambiguous_specs = [x for x in specs if 0.45 <= max(x["distribution"].values()) <= 0.70]
    for spec in ambiguous_specs:
        for j in range(repeats):
            cases.append(Case(
                "semantic_repeatability",
                spec["state"],
                {"target": choice(spec["instructions"] + " Use only the stated evidence.", spec["criteria"])},
                {"target": {"type": "choice", "label": spec["label"], "target_distribution": spec["distribution"]}},
                {
                    "domain": spec["domain"],
                    "scenario_id": spec["scenario_id"],
                    "repetition": j,
                    "reference_entropy_norm": spec["reference_entropy_norm"],
                    "reference_max_probability": max(spec["distribution"].values()),
                    "panel_votes": ",".join(f"{k}:{v}" for k, v in spec["votes"].items()),
                    "reference_source": "benchmark_authored_adjudication",
                },
            ).finalize())

    return cases
