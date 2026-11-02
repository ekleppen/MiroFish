"""
Experiment specification and the engine-neutral result row.
"""

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

EXPERIMENT_ID_PATTERN = re.compile(r"^ex_[0-9a-f]{12}$")
VARIANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
ENGINE_DIRECT = "direct"
ENGINE_OASIS_SOCIAL = "oasis_social"
KNOWN_ENGINES = (ENGINE_DIRECT, ENGINE_OASIS_SOCIAL)
SCORE_SCALES = (5, 10)
MAX_VARIANTS = 5
MAX_REPLICATES = 10
MAX_MESSAGE_CHARS = 1000
BRAND_PLACEHOLDER = "{brand}"
DEFAULT_BRAND = "the hotel"

DECISION_RESPOND = "respond"
DECISION_IGNORE = "ignore"
DECISION_ERROR = "error"


class SpecError(ValueError):
    pass


def new_experiment_id() -> str:
    return f"ex_{uuid.uuid4().hex[:12]}"


@dataclass
class Variant:
    variant_id: str
    name: str
    text: str
    score_scale: Optional[int] = None

    def message_for(self, brand: Optional[str]) -> str:
        """The exact text a guest receives; {brand} becomes the guest's sub-brand."""
        return self.text.replace(BRAND_PLACEHOLDER, brand or DEFAULT_BRAND)


@dataclass
class ExperimentSpec:
    experiment_id: str
    population_id: str
    persona_source: str
    engine: str
    variants: List[Variant]
    replicates: int
    seed: int
    prompt_version: str = ""
    coder_prompt_version: str = ""
    model: str = ""
    reasoning_effort: str = ""
    coder_reasoning_effort: str = ""
    response_calibration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentSpec":
        values = dict(data)
        values["variants"] = [Variant(**v) for v in data.get("variants") or []]
        return cls(**values)


def parse_variants(raw: Any) -> List[Variant]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_VARIANTS:
        raise SpecError(f"variants must be a list of 1-{MAX_VARIANTS} objects")
    variants, seen = [], set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SpecError(f"variant {index} must be an object")
        variant_id = str(item.get("variant_id") or "").strip()
        if not VARIANT_ID_PATTERN.match(variant_id):
            raise SpecError(f"variant {index}: variant_id must be 1-40 letters, digits, '_' or '-'")
        if variant_id in seen:
            raise SpecError(f"duplicate variant_id: {variant_id}")
        seen.add(variant_id)
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SpecError(f"variant {variant_id}: text is required")
        if len(text) > MAX_MESSAGE_CHARS:
            raise SpecError(f"variant {variant_id}: text is longer than {MAX_MESSAGE_CHARS} characters")
        scale = item.get("score_scale")
        if scale is not None and (isinstance(scale, bool) or scale not in SCORE_SCALES):
            raise SpecError(f"variant {variant_id}: score_scale must be 5, 10 or null")
        name = str(item.get("name") or variant_id).strip()
        variants.append(Variant(variant_id=variant_id, name=name, text=text.strip(),
                                score_scale=scale))
    return variants


@dataclass
class ResultRow:
    """One guest x variant x replicate outcome, shared by every engine."""
    experiment_id: str
    engine: str
    variant_id: str
    replicate: int
    guest_id: str
    cohort: str
    decision: str
    reply_text: Optional[str]
    score: Optional[int]
    score_scale: Optional[int]
    attempts: int
    model: str
    prompt_version: str
    error: Optional[str] = None
    error_stage: Optional[str] = None
    # Two-stage gate audit trail: the LLM's stated propensity, the calibrated
    # probability, and the shared seeded draw that decided respond vs ignore.
    propensity_raw: Optional[int] = None
    p_raw: Optional[float] = None
    p_calibrated: Optional[float] = None
    calibration_intercept: Optional[float] = None
    random_draw: Optional[float] = None
    coded: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def row_key(row: Dict[str, Any]) -> tuple:
    return (row["variant_id"], int(row["replicate"]), row["guest_id"])
