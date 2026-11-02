"""
Direct private-message engine (two-stage gate)

Stage 1 asks each guest only how likely they would be to reply to the hotel's
text message, with no reply text, so the model can't drift toward answering
just because it can write something. Python then calibrates that propensity
and makes a seeded draw shared across variants; only guests drawn to respond
get a stage-2 call that writes the reply. There is no feed and no other guest,
so variants differ only in the message.
"""

import concurrent.futures
import random
from typing import Any, Callable, Dict, List, Optional, Set

from ...population.mirofish_adapter import flatten_text
from ...utils.logger import get_logger
from ..calibration import calibrate, p_from_propensity, shared_draw
from ..schema import (
    DECISION_ERROR,
    DECISION_IGNORE,
    DECISION_RESPOND,
    ENGINE_DIRECT,
    ExperimentSpec,
    ResultRow,
    Variant,
    row_key,
)
from ..scoring import parse_score
from .base import ProgressCallback, ResultCallback

logger = get_logger("mirofish.experiments")

PROMPT_VERSION = "direct-message-v2-gated"
MAX_ATTEMPTS = 2
MAX_TOKENS = 1500
MAX_PARALLEL = 10
# Default thinking effort for guest calls; recorded in the spec because it can shape behavior
REASONING_EFFORT = "low"
REASONING_EFFORTS = ("low", "medium", "high")

STAGE_PROPENSITY = "propensity"
STAGE_WRITING = "writing"

ARRIVAL = (
    "SITUATION: You are partway through this stay. A text message from the hotel has "
    "just arrived on your phone. Nothing requires you to answer it."
)

PROPENSITY_INSTRUCTIONS = (
    "Do not write a reply. Estimate how likely you, as this specific guest, would be to "
    "send any reply to this message at all, given your stay so far and the message itself. "
    'Return JSON only: {"reply_likelihood": an integer from 0 (you certainly would not '
    "reply) to 100 (you certainly would reply)}."
)

WRITING_INSTRUCTIONS = (
    "You have decided to reply to this message. "
    'Return JSON only: {"reply_text": "..."}. reply_text is the exact text message you '
    "send back: in your own voice, as long or as short as you would really write, and only "
    "about things in your stay context. Do not invent events that are not in your stay context."
)


def propensity_prompt(agent_text: str) -> str:
    return f"{agent_text}\n\n{ARRIVAL}\n\n{PROPENSITY_INSTRUCTIONS}"


def writing_prompt(agent_text: str) -> str:
    return f"{agent_text}\n\n{ARRIVAL}\n\n{WRITING_INSTRUCTIONS}"


def user_prompt(message: str) -> str:
    return f"Text message from the hotel:\n{message}"


def parse_propensity(result: Any) -> int:
    """An integer 0-100; raises ValueError when the call must be retried."""
    if not isinstance(result, dict):
        raise ValueError("response is not a JSON object")
    value = result.get("reply_likelihood")
    if isinstance(value, bool):
        raise ValueError("reply_likelihood is not a number")
    try:
        number = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        raise ValueError(f"reply_likelihood is not a number: {value!r}")
    if not 0 <= number <= 100:
        raise ValueError(f"reply_likelihood out of range: {number}")
    return int(round(number))


def parse_reply(result: Any) -> str:
    if not isinstance(result, dict):
        raise ValueError("response is not a JSON object")
    reply = result.get("reply_text")
    reply = flatten_text(reply) if isinstance(reply, str) else ""
    if not reply:
        raise ValueError("empty reply_text")
    return reply


class DirectMessageRunner:
    engine = ENGINE_DIRECT
    prompt_version = PROMPT_VERSION

    def __init__(self, llm_client: Any = None, temperature: float = 0.7,
                 reasoning_effort: str = REASONING_EFFORT):
        if llm_client is None:
            from ...utils.llm_client import LLMClient
            llm_client = LLMClient()
        self.llm = llm_client
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm, "model", "") or "unknown")

    def _call(self, system: str, message: str, parse: Callable[[Any], Any], label: str):
        """Returns (parsed value or None, attempts, error)."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user_prompt(message)}]
        error = None
        attempts = 0
        for attempts in range(1, MAX_ATTEMPTS + 1):
            try:
                result = self.llm.chat_json(messages=messages, temperature=self.temperature,
                                            max_tokens=MAX_TOKENS if attempts == 1 else None,
                                            reasoning_effort=self.reasoning_effort)
                return parse(result), attempts, None
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                logger.warning(f"Direct message {label} failed (attempt {attempts}): {error}")
        return None, attempts, error

    def propensity_one(self, spec: ExperimentSpec, variant: Variant, replicate: int,
                       guest: Dict[str, Any]) -> Dict[str, Any]:
        value, attempts, error = self._call(
            propensity_prompt(guest["agent_text"]), variant.message_for(guest.get("brand")),
            parse_propensity, f"propensity for {guest['guest_id']} / {variant.variant_id}")
        return {
            "variant_id": variant.variant_id,
            "replicate": replicate,
            "guest_id": guest["guest_id"],
            "cohort": guest["cohort"],
            "propensity_raw": value,
            "attempts": attempts,
            "error": error,
            "model": self.model_name,
            "prompt_version": self.prompt_version,
        }

    def write_one(self, variant: Variant, guest: Dict[str, Any]):
        return self._call(
            writing_prompt(guest["agent_text"]), variant.message_for(guest.get("brand")),
            parse_reply, f"reply for {guest['guest_id']} / {variant.variant_id}")

    def _final_row(self, spec, variant, stage1, intercept, guest) -> Dict[str, Any]:
        row = ResultRow(
            experiment_id=spec.experiment_id,
            engine=self.engine,
            variant_id=variant.variant_id,
            replicate=stage1["replicate"],
            guest_id=guest["guest_id"],
            cohort=guest["cohort"],
            decision=DECISION_ERROR,
            reply_text=None,
            score=None,
            score_scale=variant.score_scale,
            attempts=stage1["attempts"],
            model=self.model_name,
            prompt_version=self.prompt_version,
            calibration_intercept=intercept,
        )
        if stage1.get("propensity_raw") is None:
            row.error = stage1.get("error") or "missing propensity"
            row.error_stage = STAGE_PROPENSITY
            return row.to_dict()

        row.propensity_raw = stage1["propensity_raw"]
        row.p_raw = p_from_propensity(row.propensity_raw)
        row.p_calibrated = round(calibrate(row.p_raw, intercept), 6)
        row.random_draw = round(shared_draw(spec.seed, guest["guest_id"], row.replicate), 6)
        if row.random_draw >= row.p_calibrated:
            row.decision = DECISION_IGNORE
            return row.to_dict()

        reply, attempts, error = self.write_one(variant, guest)
        row.attempts += attempts
        if reply is None:
            row.error = error
            row.error_stage = STAGE_WRITING
        else:
            row.decision = DECISION_RESPOND
            row.reply_text = reply
            row.score = parse_score(reply, variant.score_scale)
        return row.to_dict()

    def run(
        self,
        spec: ExperimentSpec,
        guests: List[Dict[str, Any]],
        on_result: ResultCallback,
        progress_callback: Optional[ProgressCallback] = None,
        completed_keys: Optional[Set[tuple]] = None,
        parallel: int = 5,
        existing_propensities: Optional[List[Dict[str, Any]]] = None,
        on_propensity: Optional[ResultCallback] = None,
        calibrator: Optional[Callable[[List[Dict[str, Any]]], Optional[float]]] = None,
    ) -> None:
        """Stage 1 for every job, then one calibration fit, then gating and stage 2."""
        workers = max(1, min(int(parallel), MAX_PARALLEL))
        variants = {v.variant_id: v for v in spec.variants}
        guests_by_id = {g["guest_id"]: g for g in guests}
        all_jobs = [(variant, replicate, guest)
                    for replicate in range(1, spec.replicates + 1)
                    for variant in spec.variants
                    for guest in guests]

        stage1 = {row_key(r): r for r in existing_propensities or []}
        pending = [job for job in all_jobs
                   if (job[0].variant_id, job[1], job[2]["guest_id"]) not in stage1]
        random.Random(f"{spec.seed}:{STAGE_PROPENSITY}").shuffle(pending)

        def keep_propensity(row):
            stage1[row_key(row)] = row
            if on_propensity:
                on_propensity(row)

        self._run_jobs(pending, lambda job: self.propensity_one(spec, *job), workers,
                       STAGE_PROPENSITY, progress_callback, keep_propensity)

        intercept = calibrator(list(stage1.values())) if calibrator else None

        done_keys = completed_keys or set()
        final_jobs = [stage1[key] for key in stage1 if key not in done_keys
                      and key[0] in variants and key[2] in guests_by_id]
        random.Random(f"{spec.seed}:{STAGE_WRITING}").shuffle(final_jobs)
        self._run_jobs(
            final_jobs,
            lambda s1: self._final_row(spec, variants[s1["variant_id"]], s1, intercept,
                                       guests_by_id[s1["guest_id"]]),
            workers, STAGE_WRITING, progress_callback, on_result)

    @staticmethod
    def _run_jobs(jobs, fn, workers, stage, progress_callback, on_done) -> None:
        total = len(jobs)
        if progress_callback:
            progress_callback(stage, 0, total)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fn, job) for job in jobs]
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                on_done(future.result())
                if progress_callback:
                    progress_callback(stage, done, total)
