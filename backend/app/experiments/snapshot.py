"""
Experiment snapshots

Each experiment copies the exact agent texts and experiences it uses into its
own snapshot.jsonl before any LLM call. Runs, coding and evaluation read only
from the snapshot, so results stay reproducible if population storage or the
persona code changes later.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from ..population.mirofish_adapter import compose_agent_text, guest_cohort
from ..population.persona_enricher import observed_facts


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assigned_need(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The controlled need the pipeline assigned, or None for unspecified guests."""
    events = (record.get("experience_state") or {}).get("events") or []
    if not events:
        return None
    event = events[0]
    category = str(event.get("category") or "").strip()
    if not category:
        return None
    return {
        "category": category,
        "item": event.get("item"),
        "need_kind": event.get("need_kind"),
        "resolution_assumption": event.get("resolution_assumption"),
    }


COHORT_UNSPECIFIED = "unspecified"
COHORT_SERVICE_REQUEST = "service_request"
COHORT_REPAIR_PROBLEM = "repair_problem"
COHORT_UNCLASSIFIED = "unclassified_need"
REPORTING_COHORTS = (COHORT_UNSPECIFIED, COHORT_SERVICE_REQUEST, COHORT_REPAIR_PROBLEM)


def reporting_cohort(guest: Optional[Dict[str, Any]]) -> str:
    """Unspecified, service request, or repair/problem, from the assigned need's need_kind."""
    need = (guest or {}).get("assigned_need")
    if not need:
        return COHORT_UNSPECIFIED
    return {"request": COHORT_SERVICE_REQUEST,
            "problem": COHORT_REPAIR_PROBLEM}.get(need.get("need_kind"), COHORT_UNCLASSIFIED)


def build_snapshot(
    records: List[Dict[str, Any]],
    personas: Optional[Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        guest_id = str(record["guest_id"])
        agent_text, source = compose_agent_text(record, (personas or {}).get(guest_id))
        rows.append({
            "guest_id": guest_id,
            "cohort": guest_cohort(record),
            "brand": observed_facts(record).get("sub_brand"),
            "agent_text": agent_text,
            "agent_text_source": source,
            "agent_text_sha256": sha256_text(agent_text),
            "experience_state": record.get("experience_state") or {},
            "assigned_need": assigned_need(record),
        })
    return rows


def snapshot_lines(rows: List[Dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def issue_categories(rows: List[Dict[str, Any]]) -> List[str]:
    """Fixed coder vocabulary: every assigned category in the snapshot, plus 'other'.

    The list is identical for every reply, so it reveals nothing about any one guest.
    """
    categories = {r["assigned_need"]["category"] for r in rows if r.get("assigned_need")}
    return sorted(categories) + ["other"]
