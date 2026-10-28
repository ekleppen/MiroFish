"""
Round schedule shared by the simulation scripts and the round preview API

Maps simulation rounds to simulated clock times and picks the Agents that are
active in each round. Standard library only, so the OASIS subprocesses and the
Flask backend can both import it.
"""

import random
from typing import Any, Dict, List, Optional

DEFAULT_START_HOUR = 8
MINUTES_PER_DAY = 24 * 60

DEFAULT_ACTIVE_HOURS = list(range(8, 23))
DEFAULT_PEAK_HOURS = [19, 20, 21, 22]
DEFAULT_OFF_PEAK_HOURS = [0, 1, 2, 3, 4, 5]
DEFAULT_MORNING_HOURS = [6, 7, 8]
DEFAULT_WORK_HOURS = list(range(9, 19))


def _hour_list(value: Any, default: List[int]) -> List[int]:
    if not isinstance(value, (list, tuple)):
        return list(default)
    hours = []
    for hour in value:
        try:
            hours.append(int(hour) % 24)
        except (TypeError, ValueError):
            continue
    return hours


def _active_hours(agent_config: Dict[str, Any]) -> List[int]:
    return _hour_list(agent_config.get("active_hours"), DEFAULT_ACTIVE_HOURS)


def full_round_count(time_config: Dict[str, Any]) -> int:
    """Number of rounds the configured simulated duration needs."""
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = max(1, int(time_config.get("minutes_per_round", 30)))
    return int(total_hours * 60) // minutes_per_round


def schedule_start_hour(
    time_config: Dict[str, Any],
    agent_configs: List[Dict[str, Any]],
) -> int:
    """Earliest peak hour, else the earliest Agent active hour, else 08:00."""
    peak_hours = _hour_list(time_config.get("peak_hours"), [])
    if peak_hours:
        return min(peak_hours)
    agent_hours = [hour for cfg in agent_configs for hour in _active_hours(cfg)]
    if agent_hours:
        return min(agent_hours)
    return DEFAULT_START_HOUR


def build_round_schedule(
    time_config: Dict[str, Any],
    agent_configs: List[Dict[str, Any]],
    max_rounds: Optional[int] = None,
) -> List[Dict[str, int]]:
    """Return one slot per round with its simulated clock time.

    When max_rounds truncates the run, the rounds are spread over up to one
    simulated day instead of covering only its first hours.
    """
    minutes_per_round = max(1, int(time_config.get("minutes_per_round", 30)))
    total_rounds = full_round_count(time_config)
    step_minutes = minutes_per_round

    if max_rounds is not None and 0 < max_rounds < total_rounds:
        total_minutes = int(time_config.get("total_simulation_hours", 72) * 60)
        span_minutes = min(total_minutes, MINUTES_PER_DAY)
        step_minutes = max(minutes_per_round, span_minutes // max_rounds)
        total_rounds = max_rounds

    start_minutes = schedule_start_hour(time_config, agent_configs) * 60
    schedule = []
    for index in range(total_rounds):
        simulated_minutes = index * step_minutes
        clock_minutes = start_minutes + simulated_minutes
        schedule.append({
            "round_num": index + 1,
            "simulated_minutes": simulated_minutes,
            "step_minutes": step_minutes,
            "day": clock_minutes // MINUTES_PER_DAY + 1,
            "hour": (clock_minutes // 60) % 24,
        })
    return schedule


def activity_multiplier(time_config: Dict[str, Any], hour: int) -> float:
    """Activity multiplier for an hour of the simulated day."""
    bands = (
        ("peak_hours", DEFAULT_PEAK_HOURS, "peak_activity_multiplier", 1.5),
        ("off_peak_hours", DEFAULT_OFF_PEAK_HOURS, "off_peak_activity_multiplier", 0.05),
        ("morning_hours", DEFAULT_MORNING_HOURS, "morning_activity_multiplier", 0.4),
        ("work_hours", DEFAULT_WORK_HOURS, "work_activity_multiplier", 0.7),
    )
    for hours_key, default_hours, multiplier_key, default_multiplier in bands:
        if hour in _hour_list(time_config.get(hours_key), default_hours):
            return float(time_config.get(multiplier_key, default_multiplier))
    return 1.0


def eligible_agent_ids(agent_configs: List[Dict[str, Any]], hour: int) -> List[int]:
    """Agents whose active hours include this hour."""
    return [
        cfg.get("agent_id", 0)
        for cfg in agent_configs
        if hour in _active_hours(cfg)
    ]


def select_active_agent_ids(
    time_config: Dict[str, Any],
    agent_configs: List[Dict[str, Any]],
    hour: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Pick the Agents that act this round.

    At least one Agent acts whenever any Agent is awake at this hour, so a
    short run never spends its rounds with nobody posting.
    """
    rng = rng or random
    eligible = [cfg for cfg in agent_configs if hour in _active_hours(cfg)]
    if not eligible:
        return []

    base_min = time_config.get("agents_per_hour_min", 5)
    base_max = time_config.get("agents_per_hour_max", 20)
    target_count = max(
        1,
        int(rng.uniform(base_min, base_max) * activity_multiplier(time_config, hour)),
    )

    candidates = [
        cfg.get("agent_id", 0)
        for cfg in eligible
        if rng.random() < cfg.get("activity_level", 0.5)
    ]
    if not candidates:
        most_active = max(eligible, key=lambda cfg: cfg.get("activity_level", 0.5))
        candidates = [most_active.get("agent_id", 0)]

    return rng.sample(candidates, min(target_count, len(candidates)))


def preview_coverage(
    config: Dict[str, Any],
    max_rounds: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize which simulated hours a run covers and which Agents can act."""
    time_config = config.get("time_config", {}) or {}
    agent_configs = config.get("agent_configs", []) or []
    schedule = build_round_schedule(time_config, agent_configs, max_rounds)
    covered_hours = sorted({slot["hour"] for slot in schedule})

    empty_rounds = [
        slot["round_num"]
        for slot in schedule
        if not eligible_agent_ids(agent_configs, slot["hour"])
    ]
    never_active = [
        cfg.get("agent_id", 0)
        for cfg in agent_configs
        if not set(_active_hours(cfg)) & set(covered_hours)
    ]
    step_minutes = (
        schedule[0]["step_minutes"]
        if schedule
        else int(time_config.get("minutes_per_round", 30))
    )

    return {
        "total_rounds": len(schedule),
        "full_rounds": full_round_count(time_config),
        "start_hour": schedule_start_hour(time_config, agent_configs),
        "step_minutes": step_minutes,
        "covered_hours": covered_hours,
        "rounds_without_agents": empty_rounds,
        "agents_total": len(agent_configs),
        "agents_never_active": never_active,
        "warning": bool(agent_configs) and (
            len(empty_rounds) > 0 or len(never_active) > 0
        ),
    }
