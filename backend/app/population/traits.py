"""
Synthetic trait assignment

Python, not the LLM, decides each guest's synthetic traits, from fixed
dimensions with seeded sampling. The LLM only puts the assigned values into
words. Responsiveness, helpfulness and willingness to give feedback are
deliberately not dimensions: whether a guest replies is the outcome the
simulation measures.
"""

import random
from collections import Counter
from typing import Any, Dict, List, Optional

PROVENANCE_OBSERVED = "observed"
PROVENANCE_SYNTHETIC = "synthetic_uncalibrated"

TRAIT_DIMENSIONS: Dict[str, List[str]] = {
    "formality": ["casual", "neutral", "formal"],
    "tone": ["warm", "matter_of_fact", "reserved", "impatient"],
    "detail": ["brief", "moderate", "detailed"],
    "priority": ["speed", "comfort", "value", "quiet", "convenience"],
}

# The pipeline's observed_writing_style maps onto the detail dimension.
OBSERVED_DETAIL = {"brief": "brief", "medium": "moderate", "detailed": "detailed"}


def validate_weights(weights: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Normalize per-dimension weights; unspecified dimensions are uniform."""
    resolved = {}
    weights = weights or {}
    unknown = set(weights) - set(TRAIT_DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown trait dimensions: {sorted(unknown)}")
    for dimension, values in TRAIT_DIMENSIONS.items():
        given = weights.get(dimension) or {}
        bad = set(given) - set(values)
        if bad:
            raise ValueError(f"Unknown values for {dimension}: {sorted(bad)}")
        raw = {value: float(given.get(value, 0 if given else 1)) for value in values}
        if any(w < 0 for w in raw.values()) or sum(raw.values()) <= 0:
            raise ValueError(f"Weights for {dimension} must be non-negative with a positive sum")
        total = sum(raw.values())
        resolved[dimension] = {value: round(w / total, 6) for value, w in raw.items()}
    return resolved


def assign_traits(
    record: Dict[str, Any],
    index: int,
    seed: int,
    weights: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, str]]:
    """Traits for one guest; the same seed and index always give the same values."""
    rng = random.Random(f"{seed}:{index}")
    attributes = record.get("attributes") or {}
    traits = {}
    for dimension, values in TRAIT_DIMENSIONS.items():
        dimension_weights = [weights[dimension][value] for value in values]
        sampled = rng.choices(values, weights=dimension_weights, k=1)[0]
        traits[dimension] = {"value": sampled, "provenance": PROVENANCE_SYNTHETIC}

    observed = OBSERVED_DETAIL.get(str(attributes.get("observed_writing_style") or ""))
    if observed:
        traits["detail"] = {"value": observed, "provenance": PROVENANCE_OBSERVED}
    return traits


def audit(
    records: List[Dict[str, Any]],
    personas: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Population-level view of trait mix and enrichment outcomes."""
    trait_counts = {d: {PROVENANCE_OBSERVED: Counter(), PROVENANCE_SYNTHETIC: Counter()}
                    for d in TRAIT_DIMENSIONS}
    statuses: Counter = Counter()
    failures: Counter = Counter()
    cohorts: Counter = Counter()
    cooperative_flags = 0

    for record in records:
        cohorts[str((record.get("stay_context") or {}).get("cohort") or "unknown")] += 1
        persona = personas.get(str(record["guest_id"]))
        if not persona:
            statuses["not_enriched"] += 1
            continue
        statuses[persona.get("status", "unknown")] += 1
        for failure in persona.get("check_failures", []):
            failures[failure.split(":", 1)[0]] += 1
            if failure.startswith("cooperative_language"):
                cooperative_flags += 1
        for dimension, trait in (persona.get("traits") or {}).items():
            if dimension in trait_counts:
                trait_counts[dimension][trait.get("provenance", PROVENANCE_SYNTHETIC)][trait["value"]] += 1

    total = len(records)
    return {
        "guest_count": total,
        "cohort_counts": dict(cohorts),
        "persona_status_counts": dict(statuses),
        "check_failure_counts": dict(failures),
        "cooperative_language_flag_rate": round(cooperative_flags / total, 4) if total else 0.0,
        "trait_counts": {
            dimension: {provenance: dict(counter) for provenance, counter in by_source.items()}
            for dimension, by_source in trait_counts.items()
        },
    }
