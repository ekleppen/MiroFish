import csv
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from app.api import simulation as simulation_api
from app.population.mirofish_adapter import (
    GuestRecordError,
    TimeSettings,
    build_guest_simulation,
    flatten_text,
    load_guest_records,
)

PIPELINE_DIR = Path(__file__).resolve().parents[2] / "mirofish_guest_pipeline_v4.0"
FORBIDDEN_PROMPT_TERMS = (
    "status_check", "score", "scale", "pressure", "satisfaction", "calibration",
)


def _record(index, controlled=False, sub_brand="Brand A"):
    persona = (
        f"You are a guest on a {index % 5 + 1}-night stay at a {sub_brand} hotel. "
        f"Prior to this point in the stay, you sent {index} message(s)."
    )
    context = (
        "During this stay, you experienced a problem involving A/C Unit (HVAC). "
        "The issue has not been resolved when this message arrives."
        if controlled else
        "No particular stay experience has been provided for this experiment."
    )
    return {
        "guest_id": f"guest_{index + 1:05d}",
        "attributes": {"company_id": 100 + index, "stay_nights": index % 5 + 1,
                       "sub_brand": sub_brand},
        "persona_text": persona,
        "experience_state": {
            "relative_satisfaction_pressure": -0.25 if controlled else None,
            "pressure_scale": "experimental_relative_index_not_a_predicted_rating",
            "guest_context": context,
        },
        "agent_context": persona + "\n\nCURRENT STAY CONTEXT: " + context,
        "stay_context": {"cohort": "controlled_scenario" if controlled else "unspecified_experience"},
        "message_event_calibration": {"organic_guest_message_ticket_detection_rate": 0.18},
        "provenance": {"profile_features": "resampled_historical_stay_record"},
    }


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _build(tmp_path, records, **kwargs):
    sim_dir = tmp_path / "simulations" / "pop_test"
    summary = build_guest_simulation(records, str(sim_dir), "pop_test", **kwargs)
    return sim_dir, summary


def _read_csv(sim_dir):
    with open(sim_dir / "twitter_profiles.csv", newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_twitter_csv_matches_oasis_contract(tmp_path):
    records = [_record(i, controlled=i % 4 == 0) for i in range(12)]
    sim_dir, summary = _build(tmp_path, records)

    rows = _read_csv(sim_dir)
    assert rows[0] == ["user_id", "name", "username", "user_char", "description"]
    assert [int(row[0]) for row in rows[1:]] == list(range(12))
    assert rows[1][1:3] == ["Guest 00001", "guest_00001"]
    assert rows[1][4] == "Hotel guest, 1-night stay, Brand A."
    assert summary["agents"] == 12
    assert summary["distinct_prompts"] == 12


def test_prompt_is_flattened_agent_context_only(tmp_path):
    records = [_record(i, controlled=i % 2 == 0) for i in range(6)]
    sim_dir, _ = _build(tmp_path, records)

    for row, record in zip(_read_csv(sim_dir)[1:], records):
        user_char = row[3]
        assert user_char == flatten_text(record["agent_context"])
        assert "\n" not in user_char and "\r" not in user_char
        lowered = user_char.lower()
        for term in FORBIDDEN_PROMPT_TERMS:
            assert term not in lowered

    controlled_prompts = [row[3] for row in _read_csv(sim_dir)[1::2]]
    assert all("A/C Unit" in prompt for prompt in controlled_prompts)


def test_agent_configs_align_with_profiles(tmp_path):
    records = [_record(i) for i in range(5)]
    sim_dir, _ = _build(tmp_path, records, settings=TimeSettings(
        total_hours=12, minutes_per_round=30, agents_per_hour_min=2, agents_per_hour_max=4,
    ))

    config = json.loads((sim_dir / "simulation_config.json").read_text())
    reddit = json.loads((sim_dir / "reddit_profiles.json").read_text())
    csv_names = [row[1] for row in _read_csv(sim_dir)[1:]]

    assert [a["agent_id"] for a in config["agent_configs"]] == list(range(5))
    assert [p["user_id"] for p in reddit] == list(range(5))
    assert [a["entity_name"] for a in config["agent_configs"]] == csv_names
    assert {a["entity_type"] for a in config["agent_configs"]} == {"Guest"}
    assert config["agent_configs"][0]["entity_uuid"] == "guest_00001"
    assert config["agent_configs"][0]["active_hours"] == list(range(24))
    assert config["event_config"]["initial_posts"] == []
    assert config["time_config"]["total_simulation_hours"] == 12
    assert config["time_config"]["minutes_per_round"] == 30
    assert config["time_config"]["agents_per_hour_max"] == 4
    assert config["time_config"]["peak_hours"] == []


def test_reddit_profiles_use_placeholders_not_invented_demographics(tmp_path):
    sim_dir, _ = _build(tmp_path, [_record(0)])
    profile = json.loads((sim_dir / "reddit_profiles.json").read_text())[0]
    assert profile["gender"] == "other"
    assert profile["age"] == profile["mbti"] == profile["country"] == "unknown"
    state = json.loads((sim_dir / "state.json").read_text())
    assert state["enable_reddit"] is False


def test_population_keeps_full_records_and_assumptions(tmp_path):
    records = [_record(i, controlled=True) for i in range(3)]
    assumptions = tmp_path / "simulation_assumptions.json"
    assumptions.write_text(json.dumps({"random_seed": 42}))
    sim_dir, summary = _build(tmp_path, records, assumptions_path=str(assumptions))

    population = json.loads((sim_dir / "population.json").read_text())["agents"]
    assert population["2"]["record"] == records[2]
    assert population["2"]["persona_source"] == "template"
    assert population["0"]["record"]["experience_state"]["relative_satisfaction_pressure"] == -0.25
    assert summary["assumptions_copied"] is True
    assert json.loads((sim_dir / "simulation_assumptions.json").read_text()) == {"random_seed": 42}


def test_missing_sub_brand_gives_generic_bio(tmp_path):
    record = _record(0)
    record["attributes"].pop("sub_brand")
    sim_dir, _ = _build(tmp_path, [record])
    assert _read_csv(sim_dir)[1][4] == "Hotel guest, 1-night stay."


@pytest.mark.parametrize("bad, message", [
    ({"agent_context": "x"}, "missing guest_id"),
    ({"guest_id": "guest_00001"}, "missing agent_context"),
    ({"guest_id": "guest_00001", "agent_context": "   \n "}, "missing agent_context"),
])
def test_load_rejects_incomplete_records(tmp_path, bad, message):
    path = _write_jsonl(tmp_path / "guests.jsonl", [bad])
    with pytest.raises(GuestRecordError, match=message):
        load_guest_records(str(path))


def test_load_rejects_duplicate_guest_ids(tmp_path):
    path = _write_jsonl(tmp_path / "guests.jsonl", [_record(0), _record(0)])
    with pytest.raises(GuestRecordError, match="duplicate guest_id"):
        load_guest_records(str(path))


def test_load_preserves_order(tmp_path):
    records = [_record(i) for i in (3, 1, 2)]
    path = _write_jsonl(tmp_path / "guests.jsonl", records)
    assert [r["guest_id"] for r in load_guest_records(str(path))] == [
        "guest_00004", "guest_00002", "guest_00003",
    ]


def test_generated_folder_passes_readiness_check(tmp_path, monkeypatch):
    sim_dir, _ = _build(tmp_path, [_record(i) for i in range(4)])
    monkeypatch.setattr(simulation_api.Config, "OASIS_SIMULATION_DATA_DIR",
                        str(sim_dir.parent))

    prepared, info = simulation_api._check_simulation_prepared("pop_test")
    assert prepared is True, info


@pytest.fixture
def pipeline_generator(monkeypatch):
    if not (PIPELINE_DIR / "02_generate_synthetic_guests.py").exists():
        pytest.skip("guest pipeline folder not present")
    monkeypatch.syspath_prepend(str(PIPELINE_DIR))
    module = importlib.import_module("02_generate_synthetic_guests")
    yield module
    sys.modules.pop("02_generate_synthetic_guests", None)


def _write_fake_pipeline_inputs(tmp_path, rows):
    profiles = tmp_path / "guest_profiles.csv"
    header = ("source_profile_key,company_id,sub_brand,stay_nights,prior_guest_messages,"
              "prior_staff_messages,prior_welcome_messages,prior_welcome_responses,"
              "prior_guest_text_messages,avg_guest_message_chars,"
              "status_check_score,status_check_scale\n")
    lines = [
        f"k{i},{10 + i % 3},Brand {'AB'[i % 2]},{i % 4 + 1},{i},{i + 1},1,{i % 2},"
        f"{i % 3},{40 + i * 10},{(i % 5) + 1},5\n"
        for i in range(rows)
    ]
    profiles.write_text(header + "".join(lines), encoding="utf-8")
    tickets = tmp_path / "ticket_mix.csv"
    tickets.write_text(
        "service_type,category,item,recommendations\n"
        "Repair,HVAC,A/C Unit,10\n"
        "Request,Towels & Terry,Bath Towel,20\n",
        encoding="utf-8",
    )
    return profiles, tickets


def test_pipeline_output_loads_without_leaking_scores(tmp_path, pipeline_generator):
    profiles, tickets = _write_fake_pipeline_inputs(tmp_path, rows=30)
    out_dir = tmp_path / "out"
    pipeline_generator.run(profiles, tickets, 20, 7, 0.25, out_dir, 0.18)

    assumptions = json.loads((out_dir / "simulation_assumptions.json").read_text())
    assert assumptions["sampled_with_replacement"] is False
    assert assumptions["controlled_scenarios_assigned"] == 5
    calibration = assumptions["response_calibration"]
    assert calibration["target_reply_rate"] == 0.1778
    assert calibration["baseline_variant_id"] == "rating"
    assert calibration["provisional"] is True
    assert (calibration["observed_responses"], calibration["observed_checks"]) == (3128, 17593)

    records = load_guest_records(str(out_dir / "synthetic_guests.jsonl"))
    sim_dir, summary = _build(
        tmp_path, records, assumptions_path=str(out_dir / "simulation_assumptions.json"),
    )
    assert summary["agents"] == 20
    assert summary["distinct_prompts"] == 20
    for row in _read_csv(sim_dir)[1:]:
        lowered = row[3].lower()
        for term in FORBIDDEN_PROMPT_TERMS:
            assert term not in lowered


def test_pipeline_samples_with_replacement_only_when_short(tmp_path, pipeline_generator):
    profiles, tickets = _write_fake_pipeline_inputs(tmp_path, rows=5)
    out_dir = tmp_path / "out"
    pipeline_generator.run(profiles, tickets, 10, 7, 0.0, out_dir, 0.18)
    assumptions = json.loads((out_dir / "simulation_assumptions.json").read_text())
    assert assumptions["sampled_with_replacement"] is True
    assert assumptions["controlled_scenarios_assigned"] == 0


def test_pipeline_calibration_overrides(pipeline_generator):
    block = pipeline_generator.response_calibration_block(200, 1000, "open")
    assert block["target_reply_rate"] == 0.2 and block["baseline_variant_id"] == "open"
    for responses, checks in ((0, 100), (100, 100), (5, 0)):
        with pytest.raises(ValueError):
            pipeline_generator.response_calibration_block(responses, checks, "rating")
