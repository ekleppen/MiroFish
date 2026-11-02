"""
Blind reply coder

Codes what a guest's reply says. The coder sees only the reply text and a
category list that is identical for every reply: never the guest, persona,
experience, assigned need or message variant. Comparing coded issues with the
assigned needs happens afterwards in evaluate.py.
"""

import concurrent.futures
from typing import Any, Callable, Dict, List, Optional

from ..population.mirofish_adapter import flatten_text
from ..utils.logger import get_logger

logger = get_logger("mirofish.experiments")

CODER_PROMPT_VERSION = "reply-coder-v2"
FEEDBACK_TYPES = ("complaint", "request", "question", "praise", "acknowledgement", "other")
MAX_ATTEMPTS = 2
MAX_TOKENS = 1500
MAX_PARALLEL = 10
REASONING_EFFORT = "low"

SYSTEM_PROMPT = (
    "You code text messages that hotel guests sent to their hotel, for research. "
    "Code only what the message explicitly says. Do not infer problems that are not "
    "stated. Return valid JSON only."
)


def build_prompt(reply_text: str, categories: List[str]) -> str:
    return (
        "Guest message:\n"
        f"\"\"\"{reply_text}\"\"\"\n\n"
        f"feedback_type: one of {', '.join(FEEDBACK_TYPES)} (the main purpose of the message).\n"
        "mentions_issue: true only if the message states a problem, unmet need or "
        "something the guest wants fixed or provided.\n"
        "issue_categories: the categories of the stated issues, chosen only from this "
        f"list: {', '.join(categories)}. Use [] when mentions_issue is false.\n"
        "issue_summary: a short neutral phrase naming the stated issue, or null.\n"
        "actionable: true only if hotel staff could act on the message as written: it "
        "names a specific problem, request or question they could fix, provide or answer.\n"
        "claims_incident: true only if the message reports a specific thing that happened "
        "or went wrong during the stay (not a general opinion, preference or question).\n"
        "incident_categories: the categories of the reported incidents, chosen only from "
        "the same list. Use [] when claims_incident is false.\n"
        "summary: one short neutral sentence describing the message.\n\n"
        'Return JSON: {"feedback_type": "...", "mentions_issue": false, '
        '"issue_categories": [], "issue_summary": null, "actionable": false, '
        '"claims_incident": false, "incident_categories": [], "summary": "..."}'
    )


def _categories(raw: Any, allowed: Dict[str, str], present: bool) -> List[str]:
    """Allowed categories only; a stated issue with no matching category becomes 'other'."""
    if not present:
        return []
    if not isinstance(raw, list):
        raw = [raw] if raw else []
    coded = []
    for value in raw:
        category = allowed.get(str(value).strip().lower())
        if category and category not in coded:
            coded.append(category)
    return coded or ["other"]


def parse_coding(result: Any, categories: List[str]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("coder response is not a JSON object")
    feedback_type = str(result.get("feedback_type") or "").strip().lower()
    if feedback_type not in FEEDBACK_TYPES:
        feedback_type = "other"
    mentions_issue = result.get("mentions_issue") is True
    claims_incident = result.get("claims_incident") is True
    allowed = {c.lower(): c for c in categories}
    issue_summary = result.get("issue_summary")
    return {
        "feedback_type": feedback_type,
        "mentions_issue": mentions_issue,
        "issue_categories": _categories(result.get("issue_categories"), allowed, mentions_issue),
        "issue_summary": flatten_text(issue_summary) if mentions_issue and isinstance(issue_summary, str) else None,
        "actionable": result.get("actionable") is True,
        "claims_incident": claims_incident,
        "incident_categories": _categories(result.get("incident_categories"), allowed,
                                           claims_incident),
        "summary": flatten_text(result.get("summary") or ""),
        "coder_prompt_version": CODER_PROMPT_VERSION,
    }


class ReplyCoder:
    def __init__(self, llm_client: Any = None):
        if llm_client is None:
            from ..utils.llm_client import LLMClient
            llm_client = LLMClient()
        self.llm = llm_client

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm, "model", "") or "unknown")

    def code_one(self, reply_text: str, categories: List[str]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(reply_text, categories)},
        ]
        error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                coding = parse_coding(
                    self.llm.chat_json(messages=messages, temperature=0.0,
                                       max_tokens=MAX_TOKENS if attempt == 1 else None,
                                       reasoning_effort=REASONING_EFFORT),
                    categories)
                coding["model"] = self.model_name
                return coding
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                logger.warning(f"Reply coding failed (attempt {attempt}): {error}")
        return {"error": error, "coder_prompt_version": CODER_PROMPT_VERSION,
                "model": self.model_name}

    def code_rows(
        self,
        rows: List[Dict[str, Any]],
        categories: List[str],
        parallel: int = 5,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Code every reply that isn't coded yet; returns the updated rows."""
        pending = [
            i for i, row in enumerate(rows)
            if row.get("decision") == "respond" and row.get("reply_text")
            and (force or not row.get("coded") or row["coded"].get("error"))
        ]
        total = len(pending)
        if progress_callback:
            progress_callback(0, total)
        workers = max(1, min(int(parallel), MAX_PARALLEL))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.code_one, rows[i]["reply_text"], categories): i
                       for i in pending}
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                rows[futures[future]]["coded"] = future.result()
                if progress_callback:
                    progress_callback(done, total)
        return rows
