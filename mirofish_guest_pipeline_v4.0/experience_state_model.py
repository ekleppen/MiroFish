"""Construct explicitly assumed guest experiences from sampled ticket-detection needs.

The ticket data supports the *mix of reported needs*, not the prevalence,
severity, resolution, or true satisfaction effect of those needs.

This module supplies a controllable experimental scenario for each selected
need. Numeric pressure is a sensitivity-test index (not a predicted rating),
and must never be used as the guest's required survey response.
"""

# Assumption knobs for sensitivity testing, not fitted effects from warehouse data.
PROBLEM_BASE_PRESSURE = {"mild": -0.10, "moderate": -0.25, "severe": -0.40}
IMPACT_MULTIPLIER = {"weak": 0.6, "moderate": 1.0, "strong": 1.4}
SEVERITY_DESCRIPTION = {
    "mild": "It is a minor inconvenience with limited disruption to your stay.",
    "moderate": "It noticeably interferes with your comfort or plans.",
    "severe": "It substantially disrupts your comfort or ability to use an expected service.",
}


def need_kind(event):
    """Distinguish problems from normal service requests without assuming all requests are failures."""
    item = str(event.get("item") or "").lower()
    category = str(event.get("category") or "").lower()
    service_type = str(event.get("service_type") or "").lower()
    return ("problem" if service_type == "repair" or "complaint" in item
            or "complaint" in category else "request")


def build_experience_state(event, severity="moderate", resolution="unresolved",
                           impact_strength="moderate"):
    """Build an agent-readable scenario plus non-agent-facing experimental metadata."""
    if severity not in PROBLEM_BASE_PRESSURE:
        raise ValueError(f"Unknown scenario severity: {severity}")
    if resolution not in {"unresolved", "resolved"}:
        raise ValueError(f"Unknown scenario resolution: {resolution}")
    if impact_strength not in IMPACT_MULTIPLIER:
        raise ValueError(f"Unknown impact strength: {impact_strength}")

    if event is None:
        return {
            "assignment": "no_controlled_event",
            "events": [],
            "prior_experience": "not_specified",
            "relative_satisfaction_pressure": None,
            "pressure_scale": "experimental_relative_index_not_a_predicted_rating",
            "guest_context": (
                "No particular stay experience has been provided for this experiment. "
                "This does not establish that the stay is excellent or problem-free. "
                "Do not fabricate specific incidents."
            ),
            "provenance": "no_synthetic_event_assigned",
        }

    kind = need_kind(event)
    category = event.get("category") or "Other"
    label = event.get("item") or category
    if kind == "problem":
        guest_context = (
            f"During this stay, you experienced a problem involving {label} ({category}). "
            f"{SEVERITY_DESCRIPTION[severity]} "
            + ("The issue has been resolved before this message. " if resolution == "resolved"
               else "The issue has not been resolved when this message arrives. ")
            + "No details about compensation or the staff's intentions have been specified."
        )
        # A resolved problem may still leave residual inconvenience. The values
        # below are hypothetical sensitivity parameters, not empirical impacts.
        remaining_fraction = 0.30 if resolution == "resolved" else 1.0
        pressure = round(
            PROBLEM_BASE_PRESSURE[severity] * IMPACT_MULTIPLIER[impact_strength]
            * remaining_fraction, 3
        )
        experience_level = "modeled_problem"
    else:
        guest_context = (
            f"During this stay, you wanted or needed {label} ({category}). "
            + ("The request has been fulfilled before this message. " if resolution == "resolved"
               else "The request has not yet been fulfilled when this message arrives. ")
            + "This is a service need, not necessarily a hotel failure or a complaint. "
            + "No staff failure or negative emotional reaction has been assumed."
        )
        # A routine request, even if not yet fulfilled, is NOT automatically a
        # negative experience. Do not manufacture a satisfaction penalty.
        pressure = None
        experience_level = "modeled_request_no_failure_assumed"

    return {
        "assignment": "controlled_synthetic_event",
        "events": [{
            "service_type": event["service_type"],
            "category": category,
            "item": event.get("item"),
            "need_kind": kind,
            "severity_assumption": severity if kind == "problem" else "not_applicable",
            "resolution_assumption": resolution,
            "occurs_before_engagement_message": True,
            "staff_has_been_notified": "unknown",
        }],
        "prior_experience": experience_level,
        "relative_satisfaction_pressure": pressure,
        "pressure_scale": "experimental_relative_index_not_a_predicted_rating",
        "impact_strength_assumption": impact_strength if kind == "problem" else "not_applicable",
        "guest_context": guest_context,
        "provenance": "ticket_type_empirical_severity_resolution_and_impact_assumed",
    }
