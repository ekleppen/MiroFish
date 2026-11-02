"""
Issue evaluation

Compares the blind coder's categories with each guest's assigned need from
the snapshot. No LLM is involved, so it can be rerun at any time.

An incident the reply reports outside the assigned need is flagged
`unsupported_incident`: the experience state doesn't establish it. That is not
proof it was fabricated, because the persona text may imply other events.
"""

from typing import Any, Dict, List, Optional


def evaluate_row(row: Dict[str, Any], guest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """None when the row's reply still needs coding; otherwise the evaluation."""
    need = guest.get("assigned_need")
    assigned = need["category"] if need else None
    if row.get("decision") != "respond":
        return {"assigned_category": assigned,
                "need_detected": False if assigned else None,
                "unassigned_issue": False,
                "unsupported_incident": False}
    coded = row.get("coded")
    if not coded or coded.get("error"):
        return None
    categories = coded.get("issue_categories") or []
    incidents = coded.get("incident_categories") or []
    return {
        "assigned_category": assigned,
        "need_detected": (assigned in categories) if assigned else None,
        "unassigned_issue": any(c != assigned for c in categories),
        "unsupported_incident": bool(coded.get("claims_incident"))
        and any(c != assigned for c in incidents),
    }


def evaluate_rows(rows: List[Dict[str, Any]], snapshot: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    guests = {g["guest_id"]: g for g in snapshot}
    for row in rows:
        guest = guests.get(row["guest_id"])
        row["evaluation"] = evaluate_row(row, guest) if guest else None
    return rows
