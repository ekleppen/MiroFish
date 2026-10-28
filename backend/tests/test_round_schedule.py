import random
from types import SimpleNamespace

from flask import Flask

from app.api import simulation as simulation_api
from scripts.round_schedule import (
    activity_multiplier,
    build_round_schedule,
    preview_coverage,
    schedule_start_hour,
    select_active_agent_ids,
)


TIME_CONFIG = {
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 2,
    "agents_per_hour_max": 4,
    "peak_hours": [19, 20, 21, 22],
    "peak_activity_multiplier": 1.5,
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "off_peak_activity_multiplier": 0.05,
    "morning_hours": [6, 7, 8],
    "morning_activity_multiplier": 0.4,
    "work_hours": list(range(9, 19)),
    "work_activity_multiplier": 0.7,
}


def _agents(*hour_lists):
    return [
        {"agent_id": index, "active_hours": hours, "activity_level": 0.5}
        for index, hours in enumerate(hour_lists)
    ]


def test_start_hour_prefers_peak_then_agent_hours_then_default():
    assert schedule_start_hour(TIME_CONFIG, []) == 19
    assert schedule_start_hour({"peak_hours": []}, _agents([12, 13], [10])) == 10
    assert schedule_start_hour({"peak_hours": []}, []) == 8


def test_full_schedule_uses_configured_step_from_start_hour():
    schedule = build_round_schedule(TIME_CONFIG, [])

    assert len(schedule) == 72
    assert schedule[0] == {
        "round_num": 1, "simulated_minutes": 0, "step_minutes": 60, "day": 1, "hour": 19,
    }
    assert schedule[5] == {
        "round_num": 6, "simulated_minutes": 300, "step_minutes": 60, "day": 2, "hour": 0,
    }


def test_truncated_schedule_spreads_rounds_over_one_day():
    schedule = build_round_schedule(TIME_CONFIG, [], max_rounds=10)

    assert len(schedule) == 10
    assert [slot["simulated_minutes"] for slot in schedule[:3]] == [0, 144, 288]
    assert schedule[-1]["simulated_minutes"] == 9 * 144
    assert len({slot["hour"] for slot in schedule}) == 10


def test_truncation_never_steps_below_minutes_per_round():
    schedule = build_round_schedule(
        {"total_simulation_hours": 4, "minutes_per_round": 60, "peak_hours": [9]},
        [],
        max_rounds=2,
    )

    assert [slot["simulated_minutes"] for slot in schedule] == [0, 120]


def test_activity_multiplier_covers_every_band():
    assert activity_multiplier(TIME_CONFIG, 20) == 1.5
    assert activity_multiplier(TIME_CONFIG, 3) == 0.05
    assert activity_multiplier(TIME_CONFIG, 7) == 0.4
    assert activity_multiplier(TIME_CONFIG, 10) == 0.7
    assert activity_multiplier(TIME_CONFIG, 23) == 1.0


def test_selection_keeps_at_least_one_awake_agent():
    agents = [
        {"agent_id": 1, "active_hours": [3], "activity_level": 0.0},
        {"agent_id": 2, "active_hours": [3], "activity_level": 0.01},
    ]

    selected = select_active_agent_ids(TIME_CONFIG, agents, 3, rng=random.Random(0))

    assert selected == [2]


def test_selection_is_empty_when_nobody_is_awake():
    agents = _agents([9, 10])

    assert select_active_agent_ids(TIME_CONFIG, agents, 3, rng=random.Random(0)) == []


def test_preview_flags_rounds_and_agents_outside_covered_hours():
    config = {
        "time_config": {**TIME_CONFIG, "peak_hours": [9]},
        "agent_configs": _agents([9, 10], [21]),
    }

    preview = preview_coverage(config, max_rounds=2)

    assert preview["covered_hours"] == [9, 21]
    assert preview["agents_never_active"] == []
    assert preview["warning"] is False

    narrow = preview_coverage(
        {"time_config": {**config["time_config"], "total_simulation_hours": 2},
         "agent_configs": config["agent_configs"]},
        max_rounds=2,
    )

    assert narrow["covered_hours"] == [9, 10]
    assert narrow["agents_never_active"] == [1]
    assert narrow["warning"] is True


def test_round_preview_endpoint_reports_coverage(monkeypatch):
    configs = {
        "sim-1": {
            "time_config": {**TIME_CONFIG, "total_simulation_hours": 2, "peak_hours": [9]},
            "agent_configs": _agents([9, 10], [21]),
        }
    }
    monkeypatch.setattr(
        simulation_api,
        "SimulationManager",
        lambda: SimpleNamespace(get_simulation_config=configs.get),
    )

    app = Flask(__name__)
    with app.test_request_context("/api/simulation/sim-1/round-preview?max_rounds=2"):
        response = simulation_api.get_round_preview("sim-1")

    data = response.get_json()["data"]
    assert data["total_rounds"] == 2
    assert data["agents_never_active"] == [1]
    assert data["warning"] is True

    with app.test_request_context("/api/simulation/missing/round-preview"):
        _response, status = simulation_api.get_round_preview("missing")

    assert status == 404
