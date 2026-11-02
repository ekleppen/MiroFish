"""
Synthetic guest adapter

Turns the guest pipeline's synthetic_guests.jsonl into a ready-to-run
simulation folder, bypassing ontology generation and Zep entity extraction.

Only each record's agent_context reaches the agent's LLM. The full record is
kept in population.json for analysis, and historical outcome files from the
pipeline are never read.
"""

import csv
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

GUEST_ENTITY_TYPE = "Guest"
PERSONA_SOURCE_TEMPLATE = "template"
PERSONA_SOURCE_ENRICHED = "enriched"
SYNTHETIC_TRAITS_LABEL = "ADDITIONAL TRAITS (synthetic):"
STAY_CONTEXT_LABEL = "CURRENT STAY CONTEXT:"
DEFAULT_ACTIVITY_LEVEL = 0.5
ALL_HOURS = list(range(24))
TWITTER_HEADERS = ["user_id", "name", "username", "user_char", "description"]
ASSUMPTIONS_FILENAME = "simulation_assumptions.json"

# OASIS's Reddit prompt states gender, age, MBTI and country. The guest data has
# none of these, so placeholders are written and Reddit is disabled for guests.
REDDIT_PLACEHOLDERS = {
    "age": "unknown",
    "gender": "other",
    "mbti": "unknown",
    "country": "unknown",
}


class GuestRecordError(ValueError):
    """A synthetic guest record is missing required fields or is duplicated."""


@dataclass
class TimeSettings:
    total_hours: int = 24
    minutes_per_round: int = 60
    agents_per_hour_min: int = 10
    agents_per_hour_max: int = 25

    def validate(self) -> None:
        if self.total_hours <= 0 or self.minutes_per_round <= 0:
            raise ValueError("total_hours and minutes_per_round must be positive")
        if not 0 < self.agents_per_hour_min <= self.agents_per_hour_max:
            raise ValueError("Need 0 < agents_per_hour_min <= agents_per_hour_max")


def flatten_text(text: str) -> str:
    """Collapse all whitespace, including newlines, to single spaces."""
    return " ".join(str(text).split())


def load_guest_records(path: str) -> List[Dict[str, Any]]:
    """Read and validate synthetic_guests.jsonl, preserving file order."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_guest_records(f, source=path)


def parse_guest_records(lines: Iterable[str], source: str = "JSONL") -> List[Dict[str, Any]]:
    """Validate JSONL lines: every record needs a unique guest_id and an agent_context."""
    records = []
    seen_ids = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GuestRecordError(f"Line {line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise GuestRecordError(f"Line {line_number}: expected a JSON object")

        guest_id = str(record.get("guest_id") or "").strip()
        if not guest_id:
            raise GuestRecordError(f"Line {line_number}: missing guest_id")
        if not flatten_text(record.get("agent_context") or ""):
            raise GuestRecordError(f"Line {line_number} ({guest_id}): missing agent_context")
        if guest_id in seen_ids:
            raise GuestRecordError(f"Line {line_number}: duplicate guest_id {guest_id}")
        seen_ids.add(guest_id)
        records.append(record)

    if not records:
        raise GuestRecordError(f"No guest records found in {source}")
    return records


def guest_context(record: Dict[str, Any]) -> str:
    """The controlled stay experience, exactly as the pipeline wrote it."""
    experience = record.get("experience_state") or {}
    stay = record.get("stay_context") or {}
    return flatten_text(experience.get("guest_context") or stay.get("description") or "")


def guest_cohort(record: Dict[str, Any]) -> str:
    return str((record.get("stay_context") or {}).get("cohort") or "unknown")


def compose_agent_text(
    record: Dict[str, Any],
    persona: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Return (agent text, persona source).

    Enriched text keeps the observed persona and the experience verbatim; only
    the middle paragraph is LLM-written. Anything missing falls back to the
    pipeline's template agent_context.
    """
    persona_text = flatten_text(record.get("persona_text") or "")
    context = guest_context(record)
    paragraph = flatten_text((persona or {}).get("traits_paragraph") or "")
    if persona and persona.get("status") == "enriched" and paragraph and persona_text and context:
        text = (f"{persona_text} {SYNTHETIC_TRAITS_LABEL} {paragraph} "
                f"{STAY_CONTEXT_LABEL} {context}")
        return flatten_text(text), PERSONA_SOURCE_ENRICHED
    return flatten_text(record["agent_context"]), PERSONA_SOURCE_TEMPLATE


def guest_handles(guest_id: str) -> Dict[str, str]:
    """Neutral display name and username; OASIS puts these in the system prompt."""
    username = guest_id.strip().lower().replace(" ", "_")
    suffix = username[len("guest_"):] if username.startswith("guest_") else username
    return {"name": f"Guest {suffix}", "username": username}


def guest_bio(record: Dict[str, Any]) -> str:
    attributes = record.get("attributes") or {}
    parts = ["Hotel guest"]
    nights = attributes.get("stay_nights")
    if isinstance(nights, (int, float)) and nights >= 1:
        nights = int(nights)
        parts.append(f"{nights}-night stay")
    sub_brand = str(attributes.get("sub_brand") or "").strip()
    if sub_brand and sub_brand.lower() != "nan":
        parts.append(sub_brand)
    return ", ".join(parts) + "."


def build_time_config(settings: TimeSettings) -> Dict[str, Any]:
    """Uniform activity across the day: no hour bands, every multiplier 1.0."""
    return {
        "total_simulation_hours": settings.total_hours,
        "minutes_per_round": settings.minutes_per_round,
        "agents_per_hour_min": settings.agents_per_hour_min,
        "agents_per_hour_max": settings.agents_per_hour_max,
        "peak_hours": [],
        "peak_activity_multiplier": 1.0,
        "off_peak_hours": [],
        "off_peak_activity_multiplier": 1.0,
        "morning_hours": [],
        "morning_activity_multiplier": 1.0,
        "work_hours": [],
        "work_activity_multiplier": 1.0,
    }


def new_simulation_id() -> str:
    return f"pop_{uuid.uuid4().hex[:12]}"


def build_guest_simulation(
    records: List[Dict[str, Any]],
    simulation_dir: str,
    simulation_id: str,
    settings: Optional[TimeSettings] = None,
    assumptions_path: Optional[str] = None,
    source_path: Optional[str] = None,
    personas: Optional[Dict[str, Dict[str, Any]]] = None,
    population_id: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Write every file the OASIS run scripts and the readiness check need.

    personas maps guest_id to an enrichment entry; guests without a usable
    enriched entry get their template agent_context.
    """
    settings = settings or TimeSettings()
    settings.validate()
    os.makedirs(simulation_dir, exist_ok=True)
    now = datetime.now().isoformat()

    agents = []
    for agent_id, record in enumerate(records):
        guest_id = str(record["guest_id"])
        handles = guest_handles(guest_id)
        persona = (personas or {}).get(guest_id)
        user_char, persona_source = compose_agent_text(record, persona)
        agents.append({
            "agent_id": agent_id,
            "guest_id": guest_id,
            "name": handles["name"],
            "username": handles["username"],
            "user_char": user_char,
            "bio": guest_bio(record),
            "persona_source": persona_source,
            "persona": persona,
        })

    # OASIS assigns agent ids from CSV row position, so rows stay in agent_id order.
    with open(os.path.join(simulation_dir, "twitter_profiles.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TWITTER_HEADERS)
        for agent in agents:
            writer.writerow([agent["agent_id"], agent["name"], agent["username"],
                             agent["user_char"], agent["bio"]])

    reddit_profiles = [{
        "user_id": agent["agent_id"],
        "username": agent["username"],
        "name": agent["name"],
        "bio": agent["bio"][:150],
        "persona": agent["user_char"],
        "karma": 1000,
        "created_at": now[:10],
        **REDDIT_PLACEHOLDERS,
    } for agent in agents]
    _write_json(os.path.join(simulation_dir, "reddit_profiles.json"), reddit_profiles)

    config = {
        "simulation_id": simulation_id,
        "project_id": "",
        "graph_id": "",
        "simulation_requirement": "",
        "population_source": "synthetic_guest_pipeline",
        "population_id": population_id,
        "random_seed": random_seed,
        "time_config": build_time_config(settings),
        "agent_configs": [{
            "agent_id": agent["agent_id"],
            "entity_uuid": agent["guest_id"],
            "entity_name": agent["name"],
            "entity_type": GUEST_ENTITY_TYPE,
            "activity_level": DEFAULT_ACTIVITY_LEVEL,
            "active_hours": list(ALL_HOURS),
        } for agent in agents],
        "event_config": {"initial_posts": [], "scheduled_events": [], "hot_topics": []},
        "generated_at": now,
    }
    _write_json(os.path.join(simulation_dir, "simulation_config.json"), config)

    _write_json(os.path.join(simulation_dir, "population.json"), {
        "population_id": population_id,
        "random_seed": random_seed,
        "agents": {
            str(agent["agent_id"]): {
                "record": record,
                "persona_source": agent["persona_source"],
                "persona": agent["persona"],
            }
            for agent, record in zip(agents, records)
        },
    })

    copied_assumptions = False
    if assumptions_path and os.path.exists(assumptions_path):
        shutil.copyfile(assumptions_path, os.path.join(simulation_dir, ASSUMPTIONS_FILENAME))
        copied_assumptions = True

    state = {
        "simulation_id": simulation_id,
        "project_id": "",
        "graph_id": "",
        "enable_twitter": True,
        "enable_reddit": False,
        "status": "ready",
        "entities_count": len(agents),
        "profiles_count": len(agents),
        "entity_types": [GUEST_ENTITY_TYPE],
        "profiles_generated": True,
        "config_generated": True,
        "population_id": population_id,
        "config_reasoning": "Synthetic guest population loaded from "
                            f"{os.path.basename(source_path) if source_path else 'JSONL'}; "
                            "time settings supplied by the operator.",
        "current_round": 0,
        "twitter_status": "not_started",
        "reddit_status": "not_started",
        "created_at": now,
        "updated_at": now,
        "error": None,
    }
    _write_json(os.path.join(simulation_dir, "state.json"), state)

    return {
        "simulation_id": simulation_id,
        "simulation_dir": os.path.abspath(simulation_dir),
        "agents": len(agents),
        "distinct_prompts": len({agent["user_char"] for agent in agents}),
        "enriched_agents": sum(a["persona_source"] == PERSONA_SOURCE_ENRICHED for a in agents),
        "assumptions_copied": copied_assumptions,
    }


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
