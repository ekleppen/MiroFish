#!/usr/bin/env python3
"""Sample historical stay profiles and attach data-derived controlled need types.

Creates persona INPUT records for MiroFish adaptation, not a MiroFish import.
Runs after 00_build_guest_profiles_redshift.sql builds the table and
01_export_guest_profiles.sql exports it to CSV.

Example:
 python 02_generate_synthetic_guests.py --csv guest_profiles.csv \
  --ticket-csv ticket_mix.csv --n 1000 --seed 42 \
  --scenario-fraction 0.20 --out-dir ./out
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ticket_event_model import TicketEventModel
from experience_state_model import build_experience_state

REQUIRED = [
    "company_id", "stay_nights", "prior_guest_messages", "prior_staff_messages",
    "prior_welcome_messages", "prior_welcome_responses", "prior_guest_text_messages",
    "avg_guest_message_chars",
    "status_check_score", "status_check_scale",
]


def observed_writing_style(row):
    if int(row["prior_guest_text_messages"]) < 2 or pd.isna(row["avg_guest_message_chars"]):
        return "not_observed"
    length = float(row["avg_guest_message_chars"])
    return "brief" if length <= 45 else "detailed" if length >= 140 else "medium"


def make_persona(row, style):
    nights = int(row["stay_nights"])
    sub_brand = str(row.get("sub_brand", "")).strip()
    where = f" at a {sub_brand} hotel" if sub_brand and sub_brand.lower() != "nan" else " at a hotel"
    opening = f"You are a guest on a {nights}-night stay{where}. "
    prehistory = (
        f"Prior to this point in the stay, you sent {int(row['prior_guest_messages'])} "
        f"message(s) and the hotel sent {int(row['prior_staff_messages'])} staff message(s). "
    )
    if int(row["prior_welcome_messages"]):
        prehistory += (
            "You replied to an earlier welcome message. "
            if int(row["prior_welcome_responses"]) else
            "You had not replied to an earlier welcome message. "
        )
    if style != "not_observed":
        prehistory += f"Your earlier written messages were generally {style} in length. "
    return (
        opening + prehistory +
        "You may ignore a message, reply briefly, ask questions, or raise concerns. "
        "Your decisions should depend on your situation and the incoming message, "
        "not on a predetermined required response. Do not invent past events."
    )


def response_calibration_block(responses, checks, baseline_variant_id):
    """Provisional reply-rate target for the experiments' probability gate.

    Valid only if `responses` counts guest replies to the `checks` delivered
    status checks (not just valid numerical scores); keep provisional until verified.
    """
    if checks <= 0 or not 0 < responses < checks:
        raise ValueError("--calibration-responses must be between 0 and --calibration-checks")
    return {
        "method": "baseline_logit_intercept",
        "baseline_variant_id": baseline_variant_id,
        "target_reply_rate": round(responses / checks, 4),
        "source": "historical_status_check_response_rate",
        "observed_responses": responses,
        "observed_checks": checks,
        "provisional": True,
        "note": "Provisional: confirm the numerator counts guest replies to delivered "
                "status checks before treating this as the official target.",
    }


def run(csv_file, ticket_csv, n, seed, scenario_fraction, out_dir, default_message_rate,
        property_rules=None, scenario_severity="moderate", scenario_resolution="unresolved",
        impact_strength="moderate", calibration_responses=3128, calibration_checks=17593,
        calibration_baseline="rating"):
    if n <= 0 or not (0 <= scenario_fraction <= 1):
        raise ValueError("--n must be positive; --scenario-fraction in [0, 1]")
    response_calibration = response_calibration_block(
        calibration_responses, calibration_checks, calibration_baseline)
    df = pd.read_csv(csv_file)
    missing = [name for name in REQUIRED if name not in df.columns]
    if missing:
        raise ValueError(f"Missing guest-profile columns: {missing}")
    for name in REQUIRED:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df[df["company_id"].notna() & df["stay_nights"].ge(1)].copy()
    for name in ["prior_guest_messages", "prior_staff_messages", "prior_welcome_messages",
                 "prior_welcome_responses", "prior_guest_text_messages"]:
        df[name] = df[name].fillna(0)
    # Scores are observed OUTCOMES, not persona attributes. Preserve scale.
    score_present = df["status_check_score"].notna() | df["status_check_scale"].notna()
    df["score_pair_valid"] = (
        df["status_check_scale"].isin([5, 10])
        & df["status_check_score"].ge(1)
        & df["status_check_score"].le(df["status_check_scale"])
        & (df["status_check_score"] % 1).eq(0)
    )
    df["score_pair_invalid"] = score_present & ~df["score_pair_valid"]
    # Count full source dataset, not just resampled guest instances.
    source_valid_scores = df.loc[df["score_pair_valid"],
        ["status_check_scale", "status_check_score"]].copy()
    invalid_count = int(df["score_pair_invalid"].sum())
    valid_count = int(df["score_pair_valid"].sum())
    unscored_count = int((~score_present).sum())
    if df.empty:
        raise ValueError("No valid guest profiles")

    tickets = TicketEventModel(ticket_csv, default_message_rate, property_rules)
    # Whole-row sampling preserves the joint distribution either way; without
    # replacement, no source row is duplicated into identical agent prompts.
    sampled_with_replacement = len(df) < n
    profiles = df.sample(n=n, replace=sampled_with_replacement,
                         random_state=seed).reset_index(drop=True)
    rng = random.Random(seed)
    controlled_ids = set(rng.sample(range(n), round(n * scenario_fraction)))
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Empirical score distribution is CONDITIONED on a scored response, not all stays.
    score_dist = (source_valid_scores
        .groupby(["status_check_scale", "status_check_score"])
        .size().reset_index(name="observations"))
    if len(score_dist):
        score_dist["pct_within_scale"] = (100 * score_dist["observations"] /
            score_dist.groupby("status_check_scale")["observations"].transform("sum"))
        score_dist["pct_within_scale"] = score_dist["pct_within_scale"].round(2)
    score_dist.to_csv(output / "status_check_score_distribution.csv", index=False)
    summary = []
    calibration_rows = []
    assigned_count = 0
    # synthetic_guests.jsonl is SAFE to give to an agent: historical ratings are
    # deliberately only written to a separate calibration file.
    with (output / "synthetic_guests.jsonl").open("w", encoding="utf-8") as f:
        for i, row in profiles.iterrows():
            company_id = int(row["company_id"])
            style = observed_writing_style(row)
            event = tickets.draw_type(company_id, rng) if i in controlled_ids else None
            experience = build_experience_state(
                event, severity=scenario_severity,
                resolution=scenario_resolution, impact_strength=impact_strength,
            )
            cohort = "controlled_scenario" if event else "unspecified_experience"
            if event:
                assigned_count += 1
            organic_message_rate, settings = tickets.property_settings(company_id)
            attrs = {
                "company_id": company_id,
                "stay_nights": int(row["stay_nights"]),
                "prior_guest_messages": int(row["prior_guest_messages"]),
                "prior_staff_messages": int(row["prior_staff_messages"]),
                "prior_welcome_messages": int(row["prior_welcome_messages"]),
                "prior_welcome_responses": int(row["prior_welcome_responses"]),
                "prior_guest_text_messages": int(row["prior_guest_text_messages"]),
                "avg_guest_message_chars": None if pd.isna(row["avg_guest_message_chars"])
                    else float(row["avg_guest_message_chars"]),
                "observed_writing_style": style,
            }
            if "sub_brand" in row and pd.notna(row["sub_brand"]):
                attrs["sub_brand"] = str(row["sub_brand"])
            if bool(row["score_pair_valid"]):
                observed_status = {
                    "availability": "observed_scored_response",
                    "status_check_score": int(row["status_check_score"]),
                    "status_check_scale": int(row["status_check_scale"]),
                }
            elif bool(row["score_pair_invalid"]):
                observed_status = {
                    "availability": "invalid_score_scale_pair_excluded",
                    "status_check_score": None,
                    "status_check_scale": None,
                }
            else:
                observed_status = {
                    "availability": "unscored_stay",
                    "status_check_score": None,
                    "status_check_scale": None,
                }
            guest_id = f"guest_{i + 1:05d}"
            persona_text = make_persona(row, style)
            record = {
                "guest_id": guest_id,
                "attributes": attrs,
                "persona_text": persona_text,
                "experience_state": experience,
                # This is what to supply as the agent's initial situational context.
                # It contains NO observed 1-5/1-10 ratings.
                "agent_context": persona_text + "\n\nCURRENT STAY CONTEXT: " + experience["guest_context"],
                "stay_context": {
                    "cohort": cohort,
                    "controlled_need": event,
                    "description": experience["guest_context"],
                    "status": "synthetic_experimental_assumption" if event else "unobserved",
                },
                "message_event_calibration": {
                    "organic_guest_message_ticket_detection_rate": organic_message_rate,
                    "source": "user_supplied_property_override" if "organic_message_ticket_rate" in settings
                        else "user_estimate_three_property_average",
                    "unit": "eligible_guest_message_not_stay",
                    "applies_to": "future_organic_guest_messages_only",
                },
                "provenance": {
                    "profile_features": "resampled_historical_stay_record",
                    "style": "message_length_inferred_if_2_or_more_text_messages",
                    "controlled_need_type": "weighted_sample_from_three_property_ticket_taxonomy" if event else "none",
                    "controlled_need_occurrence": "experimenter_defined_scenario_fraction_not_observed_incidence",
                    "personality_trait": "none_generated",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            calibration_rows.append({
                "guest_id": guest_id,
                "historical_score": observed_status["status_check_score"],
                "historical_scale": observed_status["status_check_scale"],
                "historical_outcome_status": observed_status["availability"],
            })
            summary.append({
                "guest_id": record["guest_id"],
                "company_id": company_id,
                "stay_nights": attrs["stay_nights"],
                "prior_guest_messages": attrs["prior_guest_messages"],
                "observed_writing_style": style,
                "cohort": cohort,
                "service_type": event["service_type"] if event else "",
                "category": event["category"] if event else "",
                "item": event["item"] if event else "",
                "organic_message_ticket_rate": organic_message_rate,
                "assumed_severity": scenario_severity if event else "",
                "assumed_resolution": scenario_resolution if event else "",
                "relative_satisfaction_pressure": experience["relative_satisfaction_pressure"],
            })
    pd.DataFrame(summary).to_csv(output / "synthetic_guests_summary.csv", index=False)
    # Keep historical outcomes completely out of agent-facing JSONL.
    pd.DataFrame(calibration_rows).to_csv(
        output / "historical_outcomes_for_calibration.csv", index=False
    )
    (output / "simulation_assumptions.json").write_text(json.dumps({
        "synthetic_stay_count": n,
        "random_seed": seed,
        "source_profile_rows": len(df),
        "sampled_with_replacement": sampled_with_replacement,
        "controlled_stay_scenario_fraction": scenario_fraction,
        "controlled_scenarios_selected": len(controlled_ids),
        "controlled_scenarios_assigned": assigned_count,
        "controlled_scenarios_without_eligible_ticket_type": len(controlled_ids) - assigned_count,
        "controlled_scenario_fraction_is_not_empirical_prevalence": True,
        "global_organic_guest_message_detection_rate": default_message_rate,
        "organic_message_rate_unit": "eligible_guest_message",
        "ticket_type_distribution": "pooled_three_property_recommendation_counts",
        "property_rules_provided": property_rules is not None,
        "critical_note": "Never force this rate on status-check replies in randomized experiments.",
        "historical_scored_stays": valid_count,
        "historical_invalid_score_scale_pairs_excluded": invalid_count,
        "historical_stays_with_no_score": unscored_count,
        "score_sampling_warning": "Distribution is conditional on a scored status-check response; do not assign it to all guests or use it as a required response in the agent persona.",
        "score_scale_policy": "Keep original 1-5 and 1-10 scales separate; exclude score outside [1, scale] or other scales.",
        "experience_state_model": "ticket-taxonomy type + explicitly assumed severity/resolution/impact",
        "scenario_severity": scenario_severity,
        "scenario_resolution": scenario_resolution,
        "impact_strength": impact_strength,
        "impact_index_is_not_a_predicted_rating": True,
        "historical_outcomes_separated_from_agent_input": True,
        "mirofish_adapter": "backend/scripts/build_guest_simulation.py (agent_context only)",
        "response_calibration": response_calibration,
    }, indent=2), encoding="utf-8")
    print(f"Generated {n:,} guests; {assigned_count:,} assigned controlled ticket-taxonomy needs")
    if assigned_count < len(controlled_ids):
        print(f"Warning: {len(controlled_ids) - assigned_count:,} selected guests had no eligible "
              "ticket type after property rules and were left unspecified")
    if sampled_with_replacement:
        print(f"Warning: only {len(df):,} source rows for {n:,} guests; sampled with replacement")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path, help="Redshift-exported guest profiles")
    p.add_argument("--ticket-csv", required=True, type=Path, help="Ticket detection aggregate (provided)")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario-fraction", type=float, default=0.2,
                   help="Share with controlled hypothetical need (NOT the 18%% per-message detection rate)")
    p.add_argument("--organic-message-rate", type=float, default=0.18,
                   help="Calibrated per eligible organic guest message, not per stay")
    p.add_argument("--property-rules", type=Path, default=None,
                   help="Optional per-company rates and offering filters JSON")
    p.add_argument("--scenario-severity", choices=["mild", "moderate", "severe"],
                   default="moderate", help="Assumed severity for controlled events (not observed)")
    p.add_argument("--scenario-resolution", choices=["unresolved", "resolved"],
                   default="unresolved", help="Assumed resolution of controlled events")
    p.add_argument("--impact-strength", choices=["weak", "moderate", "strong"],
                   default="moderate", help="Sensitivity assumption, NOT a calibrated rating")
    p.add_argument("--calibration-responses", type=int, default=3128,
                   help="Historical guest replies to status checks (provisional target numerator)")
    p.add_argument("--calibration-checks", type=int, default=17593,
                   help="Historical stays that received a status check (target denominator)")
    p.add_argument("--calibration-baseline", default="rating",
                   help="Experiment variant_id the reply-rate target is anchored to")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    a = p.parse_args()
    run(a.csv, a.ticket_csv, a.n, a.seed, a.scenario_fraction,
        a.out_dir, a.organic_message_rate, a.property_rules,
        a.scenario_severity, a.scenario_resolution, a.impact_strength,
        a.calibration_responses, a.calibration_checks, a.calibration_baseline)
