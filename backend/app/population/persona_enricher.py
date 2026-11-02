"""
Bounded LLM persona enrichment for synthetic guests

The LLM writes one short paragraph that puts Python-assigned traits into
words. It never sees the guest's stay experience, historical scores, property
id or satisfaction-pressure metadata, and it never rewrites the observed
persona. Paragraphs that fail the checks fall back to the template persona.
"""

import concurrent.futures
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..utils.logger import get_logger
from .mirofish_adapter import flatten_text
from .traits import TRAIT_DIMENSIONS, assign_traits, validate_weights

logger = get_logger("mirofish.population")

PROMPT_VERSION = "guest-persona-v2"
STATUS_ENRICHED = "enriched"
STATUS_FALLBACK = "fallback_template"
MAX_ATTEMPTS = 2
MIN_WORDS = 30
MAX_WORDS = 150
MAX_PARALLEL = 10
REASONING_EFFORT = "low"

TRAIT_GLOSSES = {
    "formality": {
        "casual": "relaxed, informal wording",
        "neutral": "plain, ordinary wording",
        "formal": "polite, formal wording",
    },
    "tone": {
        "warm": "friendly and personable",
        "matter_of_fact": "direct and unemotional",
        "reserved": "private and sparing with personal detail",
        "impatient": "dislikes waiting and wants things handled quickly",
    },
    "detail": {
        "brief": "keeps messages short",
        "moderate": "writes messages of moderate length",
        "detailed": "explains things thoroughly in writing",
    },
    "priority": {
        "speed": "getting things done quickly",
        "comfort": "physical comfort of the room",
        "value": "getting fair value for money",
        "quiet": "peace and quiet",
        "convenience": "convenience and ease",
    },
}

# Each pattern describes a failure; the name before ":" is what the audit counts.
CHECK_PATTERNS = {
    "rating_language": r"\b(scores?|ratings?|rated|rate\b|stars?|out of (5|10|five|ten))\b",
    "compensation": r"\b(refunds?|compensat\w*|vouchers?|discounts?|free nights?)\b",
    "demographics": r"\b(mbti|years old|aged? \d+|nationality|profession|occupation)\b",
    "response_propensity": r"\b(repl(y|ies|ied|ying)|respond\w*|responses?|feedback|surveys?)\b",
    "cooperative_language": (
        r"\b(happy to help|eager\w*|always (respond|answer|engage)\w*|enjoys? giving|"
        r"glad to share|likes? to (reply|respond|share)|cooperative|accommodating|responsive|"
        r"helpful to (staff|the hotel)|engaged guest|loves? (to )?(chat|communicat\w*))\b"
    ),
}
NIGHTS_PATTERN = re.compile(r"\b(\d+)[- ]nights?\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You write short persona paragraphs for synthetic hotel guests in a research "
    "simulation. Follow every rule exactly and return valid JSON only."
)


def observed_facts(record: Dict[str, Any]) -> Dict[str, Any]:
    attributes = record.get("attributes") or {}
    facts: Dict[str, Any] = {}
    nights = attributes.get("stay_nights")
    if isinstance(nights, (int, float)) and nights >= 1:
        facts["stay_nights"] = int(nights)
    sub_brand = str(attributes.get("sub_brand") or "").strip()
    if sub_brand and sub_brand.lower() != "nan":
        facts["sub_brand"] = sub_brand
    return facts


def build_prompt(traits: Dict[str, Dict[str, str]], facts: Dict[str, Any]) -> str:
    trait_lines = []
    for dimension in TRAIT_DIMENSIONS:
        trait = traits[dimension]
        note = " (observed from the guest's real messages)" if trait["provenance"] == "observed" else ""
        gloss = TRAIT_GLOSSES[dimension][trait["value"]]
        label = "main priority during a stay" if dimension == "priority" else dimension
        trait_lines.append(f"- {label}: {trait['value']} - {gloss}{note}")

    fact_lines = []
    if "stay_nights" in facts:
        fact_lines.append(f"- Stay length: {facts['stay_nights']} night(s)")
    fact_lines.append(f"- Hotel sub-brand: {facts.get('sub_brand', 'not specified')}")

    return (
        "Write one paragraph of 40-120 words in second person (\"You ...\") that "
        "expresses ONLY these assigned traits:\n"
        + "\n".join(trait_lines)
        + "\n\nObserved facts about this guest (read-only; do not change or contradict "
          "them, and you do not need to restate them):\n"
        + "\n".join(fact_lines)
        + "\n\nRules:\n"
          "- Describe how this person writes and what they care about during a stay, "
          "in plain language.\n"
          "- Do NOT say how likely they are to reply, respond or engage, how cooperative "
          "or helpful they are, or how they feel about giving feedback or surveys. Do not "
          "use the words reply, respond, feedback or survey.\n"
          "- Do NOT add facts: no names, age, gender, nationality, profession, travel "
          "purpose, family, past stays, events, problems, complaints or compensation.\n"
          "- Do NOT mention ratings, scores or stars.\n"
          "- Stay neutral about the hotel: no praise or criticism of it.\n\n"
          'Return JSON: {"traits_paragraph": "..."}'
    )


def check_paragraph(
    paragraph: str,
    facts: Dict[str, Any],
    known_brands: Iterable[str] = (),
) -> List[str]:
    """Deterministic checks; an empty list means the paragraph is usable."""
    failures = []
    words = len(paragraph.split())
    if words < MIN_WORDS or words > MAX_WORDS:
        failures.append(f"length: {words} words")
    lowered = paragraph.lower()
    for name, pattern in CHECK_PATTERNS.items():
        match = re.search(pattern, lowered)
        if match:
            failures.append(f"{name}: {match.group(0)}")
    for match in NIGHTS_PATTERN.finditer(paragraph):
        if facts.get("stay_nights") != int(match.group(1)):
            failures.append(f"stay_nights: {match.group(0)}")
    own_brand = (facts.get("sub_brand") or "").lower()
    for brand in known_brands:
        if brand and brand.lower() != own_brand and brand.lower() in lowered:
            failures.append(f"sub_brand: {brand}")
    return failures


class GuestPersonaEnricher:
    def __init__(self, llm_client: Any = None, temperature: float = 0.7):
        if llm_client is None:
            from ..utils.llm_client import LLMClient
            llm_client = LLMClient()
        self.llm = llm_client
        self.temperature = temperature

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm, "model", "") or "unknown")

    def enrich_one(
        self,
        record: Dict[str, Any],
        traits: Dict[str, Dict[str, str]],
        known_brands: Iterable[str] = (),
    ) -> Dict[str, Any]:
        facts = observed_facts(record)
        prompt = build_prompt(traits, facts)
        failures: List[str] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = self.llm.chat_json(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=1500,
                    reasoning_effort=REASONING_EFFORT,
                )
                paragraph = flatten_text((result or {}).get("traits_paragraph") or "")
                failures = check_paragraph(paragraph, facts, known_brands)
                if not failures:
                    return self._entry(traits, paragraph, STATUS_ENRICHED, [], attempt)
            except Exception as exc:
                failures = [f"llm_error: {type(exc).__name__}"]
                logger.warning(f"Persona enrichment failed for {record.get('guest_id')} "
                               f"(attempt {attempt}): {str(exc)[:120]}")
        return self._entry(traits, None, STATUS_FALLBACK, failures, MAX_ATTEMPTS)

    def _entry(self, traits, paragraph, status, failures, attempts) -> Dict[str, Any]:
        return {
            "traits": traits,
            "traits_paragraph": paragraph,
            "status": status,
            "check_failures": failures,
            "attempts": attempts,
            "model": self.model_name,
            "prompt_version": PROMPT_VERSION,
            "reasoning_effort": REASONING_EFFORT,
        }

    def enrich_all(
        self,
        records: List[Dict[str, Any]],
        seed: int,
        weights: Optional[Dict[str, Dict[str, float]]] = None,
        existing: Optional[Dict[str, Dict[str, Any]]] = None,
        force: bool = False,
        parallel: int = 5,
        progress_callback: Optional[Callable[[int, int, Dict[str, int]], None]] = None,
        on_result: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Enrich every guest that isn't enriched yet (all guests with force)."""
        resolved = validate_weights(weights)
        personas = dict(existing or {})
        known_brands = sorted({observed_facts(r).get("sub_brand", "") for r in records} - {""})

        pending = []
        for index, record in enumerate(records):
            guest_id = str(record["guest_id"])
            if not force and personas.get(guest_id, {}).get("status") == STATUS_ENRICHED:
                continue
            pending.append((guest_id, record, assign_traits(record, index, seed, resolved)))

        counts = {STATUS_ENRICHED: 0, STATUS_FALLBACK: 0}
        total = len(pending)
        if progress_callback:
            progress_callback(0, total, dict(counts))

        workers = max(1, min(int(parallel), MAX_PARALLEL))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.enrich_one, record, traits, known_brands): guest_id
                for guest_id, record, traits in pending
            }
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                guest_id = futures[future]
                entry = future.result()
                personas[guest_id] = entry
                counts[entry["status"]] += 1
                if on_result:
                    on_result(guest_id, entry)
                if progress_callback:
                    progress_callback(done, total, dict(counts))
        return personas
