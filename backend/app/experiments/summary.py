"""
Experiment summary

Computed in Python from results.jsonl. Every interval is a seeded guest-level
bootstrap: a guest's replicates are averaged first and guests are resampled,
because replicates of one guest are not independent observations. Results are
relative comparisons between variants on the same frozen guests, useful for
picking A/B candidates, not validated forecasts of production engagement.
"""

import itertools
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional

from .calibration import STATUS_NOT_CALIBRATED, logit
from .schema import DECISION_ERROR, DECISION_RESPOND, ExperimentSpec
from .snapshot import REPORTING_COHORTS, reporting_cohort

BOOTSTRAP_SAMPLES = 2000

ValueFn = Callable[[Dict[str, Any]], Optional[float]]


def _reply(row):
    return None if row["decision"] == DECISION_ERROR else float(row["decision"] == DECISION_RESPOND)


def _propensity(row):
    return row.get("propensity_raw")


def _p_calibrated(row):
    return row.get("p_calibrated")


def _log_odds_raw(row):
    return logit(row["p_raw"]) if row.get("p_raw") is not None else None


def _actionable(row):
    """Per message sent: an actionable reply counts 1, an ignore or non-actionable reply 0."""
    if row["decision"] == DECISION_ERROR:
        return None
    return float(row["decision"] == DECISION_RESPOND
                 and bool((row.get("coded") or {}).get("actionable")))


def _unsupported(row):
    evaluation = row.get("evaluation")
    if row["decision"] != DECISION_RESPOND or not evaluation:
        return None
    return float(bool(evaluation.get("unsupported_incident")))


def guest_means(rows: List[Dict[str, Any]], value: ValueFn) -> Dict[str, float]:
    values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        v = value(row)
        if v is not None:
            values[row["guest_id"]].append(v)
    return {guest: sum(v) / len(v) for guest, v in values.items()}


def _bootstrap_ci(values: List[float], key: str, samples: int):
    rng = random.Random(key)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return (round(means[int(0.025 * samples)], 4),
            round(means[min(samples - 1, int(0.975 * samples))], 4))


def guest_level(rows: List[Dict[str, Any]], value: ValueFn, key: str,
                samples: int = BOOTSTRAP_SAMPLES) -> Dict[str, Any]:
    """Mean of guest means with a guest-level bootstrap interval."""
    means = guest_means(rows, value)
    result = {"estimate": None, "ci95": None,
              "instances": sum(value(r) is not None for r in rows),
              "unique_guests": len(means)}
    if means:
        values = [means[g] for g in sorted(means)]
        result["estimate"] = round(sum(values) / len(values), 4)
        result["ci95"] = _bootstrap_ci(values, key, samples)
    return result


def paired_difference(
    rows: List[Dict[str, Any]],
    variant_a: str,
    variant_b: str,
    seed: int,
    value: ValueFn = _reply,
    metric: str = "reply_rate",
    samples: int = BOOTSTRAP_SAMPLES,
) -> Dict[str, Any]:
    """Mean difference a - b over guests with valid rows in both variants."""
    a = guest_means([r for r in rows if r["variant_id"] == variant_a], value)
    b = guest_means([r for r in rows if r["variant_id"] == variant_b], value)
    guests = sorted(set(a) & set(b))
    result: Dict[str, Any] = {"difference": None, "ci95": None, "paired_guests": len(guests),
                              "instances_a": sum(value(r) is not None for r in rows
                                                 if r["variant_id"] == variant_a),
                              "instances_b": sum(value(r) is not None for r in rows
                                                 if r["variant_id"] == variant_b)}
    if not guests:
        return result
    diffs = [a[g] - b[g] for g in guests]
    result["difference"] = round(sum(diffs) / len(diffs), 4)
    result["ci95"] = _bootstrap_ci(diffs, f"{seed}:{metric}:{variant_a}:{variant_b}", samples)
    return result


COMPARISON_METRICS = (
    ("reply_rate", _reply),
    ("p_calibrated", _p_calibrated),
    ("actionable_yield", _actionable),
    ("log_odds_raw", _log_odds_raw),
)


def _coding_complete(rows: List[Dict[str, Any]]) -> bool:
    return all(r.get("coded") and not r["coded"].get("error")
               for r in rows if r["decision"] == DECISION_RESPOND)


def _split_by_cohort(rows: List[Dict[str, Any]],
                     snapshot_by_id: Dict[str, Dict]) -> Dict[str, List[Dict[str, Any]]]:
    """Rows grouped by reporting cohort, in REPORTING_COHORTS order (empty cohorts omitted)."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[reporting_cohort(snapshot_by_id.get(row["guest_id"]))].append(row)
    order = list(REPORTING_COHORTS) + sorted(set(groups) - set(REPORTING_COHORTS))
    return {cohort: groups[cohort] for cohort in order if groups.get(cohort)}


def _cohort_metrics(rows: List[Dict[str, Any]], key: str, coding_complete: bool) -> Dict[str, Any]:
    return {
        "mean_propensity_raw": guest_level(rows, _propensity, f"{key}:propensity"),
        "mean_p_calibrated": guest_level(rows, _p_calibrated, f"{key}:p_calibrated"),
        "simulated_reply_rate": guest_level(rows, _reply, f"{key}:reply"),
        "actionable_yield_per_message": guest_level(rows, _actionable, f"{key}:actionable")
        if coding_complete else None,
    }


def _paired_metrics(rows, a: str, b: str, seed, coding_complete: bool) -> Dict[str, Any]:
    result = {}
    for metric, value in COMPARISON_METRICS:
        if metric == "actionable_yield" and not coding_complete:
            result[metric] = None
            continue
        result[metric] = paired_difference(rows, a, b, seed, value=value, metric=metric)
    return result


def _variant_summary(variant, rows: List[Dict[str, Any]], snapshot_by_id: Dict[str, Dict],
                     seed: int) -> Dict[str, Any]:
    key = f"{seed}:{variant.variant_id}"
    decisions = Counter(r["decision"] for r in rows)
    replies = [r for r in rows if r["decision"] == DECISION_RESPOND]
    coding_complete = _coding_complete(rows)

    summary: Dict[str, Any] = {
        "variant_id": variant.variant_id,
        "name": variant.name,
        "rows": len(rows),
        "decisions": dict(decisions),
        "errors": decisions[DECISION_ERROR],
        "errors_by_stage": dict(Counter(r.get("error_stage") or "unknown"
                                        for r in rows if r["decision"] == DECISION_ERROR)),
        "mean_propensity_raw": guest_level(rows, _propensity, f"{key}:propensity"),
        "mean_p_calibrated": guest_level(rows, _p_calibrated, f"{key}:p_calibrated"),
        "simulated_reply_rate": guest_level(rows, _reply, f"{key}:reply"),
        "by_cohort": {
            cohort: _cohort_metrics(cohort_rows, f"{key}:{cohort}", coding_complete)
            for cohort, cohort_rows in _split_by_cohort(rows, snapshot_by_id).items()
        },
        "median_reply_chars": statistics.median(len(r["reply_text"]) for r in replies) if replies else None,
        "actionable_yield_per_message": guest_level(rows, _actionable, f"{key}:actionable")
        if coding_complete else None,
    }

    if variant.score_scale:
        scores = [r["score"] for r in replies if r.get("score") is not None]
        summary["scores"] = {
            "scale": variant.score_scale,
            "replies_with_score": len(scores),
            "unique_guests_with_score": len({r["guest_id"] for r in replies
                                             if r.get("score") is not None}),
            "share_of_replies_with_score": round(len(scores) / len(replies), 4) if replies else None,
            "distribution": {str(v): scores.count(v) for v in range(1, variant.score_scale + 1)},
            "mean": round(sum(scores) / len(scores), 3) if scores else None,
        }

    coded = [r["coded"] for r in replies if r.get("coded") and not r["coded"].get("error")]
    summary["coding"] = {
        "replies": len(replies),
        "coded": len(coded),
        "complete": coding_complete,
        "feedback_types": dict(Counter(c["feedback_type"] for c in coded)),
        "replies_mentioning_issue": sum(bool(c["mentions_issue"]) for c in coded),
        "actionable_replies": sum(bool(c.get("actionable")) for c in coded),
        "issue_categories": dict(Counter(cat for c in coded for cat in c["issue_categories"])),
    }

    valid = [r for r in rows if r["decision"] != DECISION_ERROR]
    evaluated = all(r.get("evaluation") is not None for r in valid)
    assigned_rows = [r for r in valid
                     if (snapshot_by_id.get(r["guest_id"]) or {}).get("assigned_need")]
    unsupported = sum(bool((r.get("evaluation") or {}).get("unsupported_incident"))
                      for r in replies)
    summary["issue_discovery"] = {
        "evaluated": evaluated,
        "need_discovery": guest_level(
            assigned_rows, lambda r: float(bool((r.get("evaluation") or {}).get("need_detected"))),
            f"{key}:discovery") if evaluated else None,
        "unassigned_issue_rows": sum(bool((r.get("evaluation") or {}).get("unassigned_issue"))
                                     for r in valid) if evaluated else None,
    }
    summary["unsupported_incidents"] = {
        "count": unsupported,
        "rate_per_reply": guest_level(replies, _unsupported, f"{key}:unsupported"),
        "note": "Incidents the experience state does not establish; not proof of fabrication.",
    } if evaluated else None
    return summary


def calibration_block(spec: ExperimentSpec, fit: Optional[Dict[str, Any]],
                      variants: List[Dict[str, Any]]) -> Dict[str, Any]:
    resolved = spec.response_calibration or {}
    fit = fit or {}
    status = fit.get("status") or resolved.get("status") or STATUS_NOT_CALIBRATED
    block: Dict[str, Any] = {
        "status": status,
        "method": resolved.get("method"),
        "score_calibration": "deferred_v2",
    }
    if status == STATUS_NOT_CALIBRATED:
        block["warning"] = (fit.get("warning") or resolved.get("warning")
                            or "Probabilities were not calibrated to a target reply rate.")
        return block
    baseline_id = resolved.get("baseline_variant_id")
    baseline = next((v for v in variants if v["variant_id"] == baseline_id), None)
    block.update({k: resolved.get(k) for k in (
        "baseline_variant_id", "target_reply_rate", "provisional", "source",
        "observed_responses", "observed_checks", "origin")})
    block.update({k: fit.get(k) for k in (
        "intercept", "baseline_instances", "baseline_unique_guests",
        "baseline_mean_p_raw", "baseline_mean_p_calibrated")})
    block["baseline_simulated_reply_rate"] = (baseline or {}).get("simulated_reply_rate")
    block["note"] = resolved.get("note")
    return block


def summarize(
    spec: ExperimentSpec,
    rows: List[Dict[str, Any]],
    snapshot: List[Dict[str, Any]],
    snapshot_sha256: str,
    calibration_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot_by_id = {g["guest_id"]: g for g in snapshot}
    variants = [
        _variant_summary(v, [r for r in rows if r["variant_id"] == v.variant_id],
                         snapshot_by_id, spec.seed)
        for v in spec.variants
    ]
    coding_complete = _coding_complete(rows)
    cohorts = _split_by_cohort(rows, snapshot_by_id)
    comparisons = []
    for a, b in itertools.combinations(spec.variants, 2):
        comparison = {"variant_a": a.variant_id, "variant_b": b.variant_id,
                      **_paired_metrics(rows, a.variant_id, b.variant_id, spec.seed,
                                        coding_complete)}
        comparison["within_cohort"] = {
            cohort: _paired_metrics(cohort_rows, a.variant_id, b.variant_id,
                                    f"{spec.seed}:{cohort}", coding_complete)
            for cohort, cohort_rows in cohorts.items()
        }
        comparisons.append(comparison)
    expected = len(snapshot) * len(spec.variants) * spec.replicates
    return {
        "experiment_id": spec.experiment_id,
        "population_id": spec.population_id,
        "engine": spec.engine,
        "persona_source": spec.persona_source,
        "prompt_version": spec.prompt_version,
        "snapshot_sha256": snapshot_sha256,
        "guests": len(snapshot),
        "guests_by_cohort": dict(Counter(reporting_cohort(g) for g in snapshot)),
        "replicates": spec.replicates,
        "rows": len(rows),
        "expected_rows": expected,
        "complete": len(rows) == expected,
        "variants": variants,
        "comparisons": comparisons,
        "interpretation": {
            "type": "exploratory_relative_comparison",
            "note": ("Use these results to pick promising A/B candidates, not as validated "
                     "treatment effects. Intervals are guest-level bootstraps over the frozen "
                     "guests. Every variant reuses the same random draw per guest and "
                     "replicate, so reply differences come only from the message's effect "
                     "on the stated propensity."),
        },
        "calibration": calibration_block(spec, calibration_fit, variants),
    }
