"""
Response-probability calibration for the two-stage gate

The guest LLM states a raw reply propensity (0-100). One logit intercept b is
fitted so the baseline variant's mean calibrated probability matches a target
reply rate, and that same b is applied to every variant:

    p_calibrated = sigmoid(logit(p_raw) + b)

This keeps each guest's ordering and the odds ratios between variants, but not
absolute probability differences. It anchors overall reply frequency only; it
is not a validated model of how much a message changes response.
"""

import math
import random
from typing import Any, Dict, Iterable, List, Optional

METHOD = "baseline_logit_intercept"
DEFAULT_BASELINE_VARIANT = "rating"
P_MIN, P_MAX = 0.01, 0.99
INTERCEPT_BOUNDS = (-20.0, 20.0)

STATUS_PROVISIONAL = "provisional_target"
STATUS_CALIBRATED = "calibrated"
STATUS_NOT_CALIBRATED = "not_calibrated"

NOTE = ("The intercept anchors overall reply frequency on the baseline variant only. It is "
        "not a validated population response model, and it does not establish how much a "
        "different message changes response probability.")
NO_TARGET_WARNING = ("No target reply rate was provided by the request or the population's "
                     "simulation_assumptions.json, so probabilities are the LLM's raw estimates.")


class CalibrationError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


def p_from_propensity(propensity: float) -> float:
    return min(P_MAX, max(P_MIN, propensity / 100.0))


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def calibrate(p_raw: float, intercept: Optional[float]) -> float:
    return p_raw if intercept is None else sigmoid(logit(p_raw) + intercept)


def fit_intercept(p_raws: Iterable[float], target: float, iterations: int = 100) -> float:
    """Bisection for b with mean(sigmoid(logit(p) + b)) == target; the mean rises with b."""
    logits = [logit(p) for p in p_raws]
    if not logits:
        raise ValueError("no baseline probabilities to fit")
    low, high = INTERCEPT_BOUNDS
    for _ in range(iterations):
        mid = (low + high) / 2
        if sum(sigmoid(x + mid) for x in logits) / len(logits) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def shared_draw(seed: int, guest_id: str, replicate: int) -> float:
    """One uniform draw per guest and replicate, identical for every variant."""
    return random.Random(f"{seed}:{guest_id}:{replicate}").random()


def resolve_calibration(
    request_block: Any,
    assumptions_block: Any,
    variant_ids: List[str],
) -> Dict[str, Any]:
    """Request fields override the population's assumptions; no target means no calibration."""
    if request_block is not None and not isinstance(request_block, dict):
        raise CalibrationError("invalid", "response_calibration must be an object")
    request_block = request_block or {}
    assumptions_block = assumptions_block if isinstance(assumptions_block, dict) else {}
    merged = {**assumptions_block, **request_block}

    target = merged.get("target_reply_rate")
    if target is None:
        return {"method": None, "status": STATUS_NOT_CALIBRATED, "target_reply_rate": None,
                "origin": None, "warning": NO_TARGET_WARNING}
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not 0 < target < 1:
        raise CalibrationError("invalid", "target_reply_rate must be a number between 0 and 1")

    baseline = merged.get("baseline_variant_id") or DEFAULT_BASELINE_VARIANT
    if baseline not in variant_ids:
        raise CalibrationError("unknown_baseline", str(baseline))

    if "provisional" in request_block:
        provisional = request_block["provisional"]
    elif "target_reply_rate" in request_block:
        provisional = True
    else:
        provisional = assumptions_block.get("provisional", True)
    if not isinstance(provisional, bool):
        raise CalibrationError("invalid", "provisional must be true or false")

    target_overridden = "target_reply_rate" in request_block
    source_block = request_block if target_overridden else merged
    origin = ("request" if not assumptions_block else
              "request+assumptions" if request_block else "assumptions")
    return {
        "method": METHOD,
        "status": STATUS_PROVISIONAL if provisional else STATUS_CALIBRATED,
        "provisional": provisional,
        "baseline_variant_id": baseline,
        "target_reply_rate": float(target),
        "source": source_block.get("source") or ("request_override" if target_overridden else None),
        "observed_responses": source_block.get("observed_responses"),
        "observed_checks": source_block.get("observed_checks"),
        "origin": origin,
        "note": NOTE,
    }


def fit_from_propensities(rows: List[Dict[str, Any]], resolved: Dict[str, Any]) -> Dict[str, Any]:
    """Fit the intercept on the baseline variant's valid stage-1 rows only."""
    if resolved.get("method") != METHOD:
        return {"intercept": None, "status": STATUS_NOT_CALIBRATED,
                "warning": resolved.get("warning")}
    baseline = [r for r in rows if r["variant_id"] == resolved["baseline_variant_id"]
                and r.get("propensity_raw") is not None]
    if not baseline:
        return {"intercept": None, "status": STATUS_NOT_CALIBRATED,
                "warning": "No valid baseline propensities to fit the intercept."}
    p_raws = [p_from_propensity(r["propensity_raw"]) for r in baseline]
    intercept = fit_intercept(p_raws, resolved["target_reply_rate"])
    return {
        "intercept": round(intercept, 6),
        "status": resolved["status"],
        "baseline_variant_id": resolved["baseline_variant_id"],
        "baseline_instances": len(baseline),
        "baseline_unique_guests": len({r["guest_id"] for r in baseline}),
        "baseline_mean_p_raw": round(sum(p_raws) / len(p_raws), 4),
        "baseline_mean_p_calibrated": round(
            sum(calibrate(p, intercept) for p in p_raws) / len(p_raws), 4),
    }
