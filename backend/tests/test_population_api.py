import io
import json
import re
import time
from collections import Counter

import pytest
from flask import Flask

from app.api import population as population_api
from app.api import population_bp
from app.api import simulation as simulation_api
from app.config import Config
from app.population.mirofish_adapter import flatten_text
from app.population.persona_enricher import (
    GuestPersonaEnricher,
    build_prompt,
    check_paragraph,
    observed_facts,
)
from app.population.traits import TRAIT_DIMENSIONS, assign_traits, validate_weights
from scripts.round_schedule import make_activation_rng, select_active_agent_ids

GOOD_PARAGRAPH = (
    "You write in plain, ordinary language and keep things practical. You notice whether "
    "the room is comfortable and whether small tasks get handled without fuss. You prefer "
    "straightforward exchanges, you rarely add personal detail, and you judge a situation "
    "by what actually happens rather than by promises."
)


def _record(index, controlled=False, sub_brand="Brand A", style="not_observed"):
    persona = (
        f"You are a guest on a {index % 4 + 1}-night stay at a {sub_brand} hotel. "
        "Prior to this point in the stay, you sent 0 message(s) and the hotel sent 1 "
        "staff message(s). Do not invent past events."
    )
    context = (
        "During this stay, you experienced a problem involving A/C Unit (HVAC). "
        "It noticeably interferes with your comfort or plans. "
        "The issue has not been resolved when this message arrives."
        if controlled else
        "No particular stay experience has been provided for this experiment."
    )
    return {
        "guest_id": f"guest_{index + 1:05d}",
        "attributes": {"company_id": 4242, "stay_nights": index % 4 + 1,
                       "sub_brand": sub_brand, "observed_writing_style": style},
        "persona_text": persona,
        "experience_state": {"guest_context": context,
                             "relative_satisfaction_pressure": -0.25 if controlled else None},
        "agent_context": persona + "\n\nCURRENT STAY CONTEXT: " + context,
        "stay_context": {"cohort": "controlled_scenario" if controlled else "unspecified_experience",
                         "description": context},
        "message_event_calibration": {"organic_guest_message_ticket_detection_rate": 0.18},
    }


def _jsonl(records):
    return "".join(json.dumps(r) + "\n" for r in records).encode()


class FakeLLM:
    model = "fake-model"

    def __init__(self, paragraph=GOOD_PARAGRAPH, fail=False):
        self.paragraph = paragraph
        self.fail = fail
        self.prompts = []
        self.efforts = []

    def chat_json(self, messages, temperature=0.3, max_tokens=None, reasoning_effort=None):
        self.prompts.append(messages[-1]["content"])
        self.efforts.append(reasoning_effort)
        if self.fail:
            raise RuntimeError("provider down")
        paragraph = self.paragraph(messages[-1]["content"]) if callable(self.paragraph) else self.paragraph
        return {"traits_paragraph": paragraph}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path / "simulations"))
    fake = FakeLLM()
    monkeypatch.setattr(population_api, "enricher_factory",
                        lambda: GuestPersonaEnricher(llm_client=fake))
    app = Flask(__name__)
    app.register_blueprint(population_bp, url_prefix="/api/population")
    return {"client": app.test_client(), "fake": fake, "tmp": tmp_path}


def _upload(client, records, name="test pop"):
    response = client.post("/api/population/upload", data={
        "guests": (io.BytesIO(_jsonl(records)), "synthetic_guests.jsonl"),
        "assumptions": (io.BytesIO(json.dumps({"random_seed": 42}).encode()),
                        "simulation_assumptions.json"),
        "name": name,
    }, content_type="multipart/form-data")
    return response


def _wait(client, task_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = client.get(f"/api/population/task/{task_id}").get_json()["data"]
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.02)
    raise AssertionError("enrichment task did not finish")


def _enrich(client, population_id, **body):
    response = client.post(f"/api/population/{population_id}/enrich", json=body)
    assert response.status_code == 200, response.get_json()
    return _wait(client, response.get_json()["data"]["task_id"])


def _profiles_csv(env, simulation_id):
    return (env["tmp"] / "simulations" / simulation_id / "twitter_profiles.csv").read_bytes()


# Upload and browsing

def test_upload_creates_population(env):
    records = [_record(i, controlled=i < 2) for i in range(6)]
    response = _upload(env["client"], records)
    data = response.get_json()["data"]

    assert response.status_code == 200
    assert data["guest_count"] == 6
    assert data["cohort_counts"] == {"controlled_scenario": 2, "unspecified_experience": 4}
    assert data["default_seed"] == 42

    listed = env["client"].get("/api/population/list").get_json()["data"]
    assert [p["population_id"] for p in listed] == [data["population_id"]]
    meta = env["client"].get(f"/api/population/{data['population_id']}").get_json()["data"]
    assert meta["frozen"] is False and meta["enrichment"]["status"] == "none"


@pytest.mark.parametrize("payload, filename", [
    (b'{"guest_id": "guest_00001"}\n', "g.jsonl"),
    (b"not json\n", "g.jsonl"),
    (_jsonl([_record(0), _record(0)]), "g.jsonl"),
    (_jsonl([_record(0)]), "g.csv"),
])
def test_upload_rejects_bad_files(env, payload, filename):
    response = env["client"].post("/api/population/upload", data={
        "guests": (io.BytesIO(payload), filename),
    }, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_unknown_population_is_404(env):
    assert env["client"].get("/api/population/pp_000000000000").status_code == 404
    assert env["client"].get("/api/population/../../etc").status_code == 404


def test_guests_view_filters_by_cohort(env):
    records = [_record(i, controlled=i % 2 == 0) for i in range(6)]
    population_id = _upload(env["client"], records).get_json()["data"]["population_id"]
    view = env["client"].get(
        f"/api/population/{population_id}/guests?cohort=controlled_scenario&limit=2"
    ).get_json()["data"]
    assert view["total"] == 3 and len(view["guests"]) == 2
    assert view["guests"][0]["persona_status"] == "not_enriched"
    assert view["guests"][0]["agent_text"] == flatten_text(records[0]["agent_context"])


# Traits

def test_traits_are_deterministic_and_seed_dependent():
    weights = validate_weights(None)
    record = _record(0)
    assert assign_traits(record, 3, 42, weights) == assign_traits(record, 3, 42, weights)
    population = [_record(i) for i in range(40)]
    first = [assign_traits(r, i, 42, weights) for i, r in enumerate(population)]
    second = [assign_traits(r, i, 7, weights) for i, r in enumerate(population)]
    assert first != second


def test_observed_writing_style_overrides_detail():
    traits = assign_traits(_record(0, style="detailed"), 0, 42, validate_weights(None))
    assert traits["detail"] == {"value": "detailed", "provenance": "observed"}
    assert traits["tone"]["provenance"] == "synthetic_uncalibrated"


def test_trait_weights_are_respected():
    weights = validate_weights({"tone": {"impatient": 3, "warm": 1}})
    assert weights["tone"]["reserved"] == 0
    counts = Counter(assign_traits(_record(0), i, 1, weights)["tone"]["value"] for i in range(4000))
    assert set(counts) == {"impatient", "warm"}
    assert 0.7 < counts["impatient"] / 4000 < 0.8


def test_invalid_trait_weights_rejected():
    with pytest.raises(ValueError):
        validate_weights({"responsiveness": {"high": 1}})
    with pytest.raises(ValueError):
        validate_weights({"tone": {"eager": 1}})


# Enrichment prompt and checks

def test_prompt_excludes_experience_and_outcome_data():
    record = _record(0, controlled=True)
    prompt = build_prompt(assign_traits(record, 0, 42, validate_weights(None)),
                          observed_facts(record))
    lowered = prompt.lower()
    assert "a/c unit" not in lowered and "not been resolved" not in lowered
    assert "4242" not in prompt
    for term in ("pressure", "score", "calibration", "satisfaction"):
        assert term not in lowered.replace("scores or stars", "")
    assert "Brand A" in prompt and "1 night" in prompt
    for dimension in TRAIT_DIMENSIONS:
        assert dimension in lowered or dimension == "priority"


@pytest.mark.parametrize("paragraph, failure", [
    (GOOD_PARAGRAPH + " You are always happy to help the staff.", "cooperative_language"),
    (GOOD_PARAGRAPH + " You tend to reply quickly.", "response_propensity"),
    (GOOD_PARAGRAPH + " On this 5-night stay you settle in.", "stay_nights"),
    (GOOD_PARAGRAPH + " You would rate things fairly.", "rating_language"),
    (GOOD_PARAGRAPH + " You expect a refund when things go wrong.", "compensation"),
    (GOOD_PARAGRAPH + " You are 40 years old.", "demographics"),
    (GOOD_PARAGRAPH + " You prefer Brand B hotels.", "sub_brand"),
    ("You are brief.", "length"),
])
def test_checks_flag_forbidden_content(paragraph, failure):
    facts = {"stay_nights": 3, "sub_brand": "Brand A"}
    failures = check_paragraph(paragraph, facts, known_brands=["Brand A", "Brand B"])
    assert any(f.startswith(failure) for f in failures), failures


def test_good_paragraph_passes_checks():
    assert check_paragraph(GOOD_PARAGRAPH, {"stay_nights": 3, "sub_brand": "Brand A"},
                           ["Brand A", "Brand B"]) == []


def test_failed_checks_fall_back_after_two_attempts():
    fake = FakeLLM(paragraph=GOOD_PARAGRAPH + " You are eager to share your thoughts.")
    entry = GuestPersonaEnricher(llm_client=fake).enrich_one(
        _record(0), assign_traits(_record(0), 0, 1, validate_weights(None)))
    assert entry["status"] == "fallback_template"
    assert entry["traits_paragraph"] is None
    assert len(fake.prompts) == 2
    assert fake.efforts == ["low", "low"] and entry["reasoning_effort"] == "low"
    assert entry["check_failures"][0].startswith("cooperative_language")


def test_llm_errors_fall_back():
    entry = GuestPersonaEnricher(llm_client=FakeLLM(fail=True)).enrich_one(
        _record(0), assign_traits(_record(0), 0, 1, validate_weights(None)))
    assert entry["status"] == "fallback_template"
    assert entry["check_failures"] == ["llm_error: RuntimeError"]


# Enrichment through the API

def test_enrich_endpoint_completes_and_audits(env):
    records = [_record(i, controlled=i == 0, style="brief" if i == 1 else "not_observed")
               for i in range(5)]
    population_id = _upload(env["client"], records).get_json()["data"]["population_id"]
    task = _enrich(env["client"], population_id)

    assert task["status"] == "completed"
    assert task["result"]["enriched"] == 5
    meta = env["client"].get(f"/api/population/{population_id}").get_json()["data"]
    assert meta["enrichment"]["status"] == "completed"
    assert meta["enrichment"]["model"] == "fake-model"
    assert meta["enrichment"]["seed"] == 42
    assert len(meta["enrichment"]["personas_sha256"]) == 64

    report = env["client"].get(f"/api/population/{population_id}/audit").get_json()["data"]
    assert report["persona_status_counts"] == {"enriched": 5}
    assert report["trait_counts"]["detail"]["observed"] == {"brief": 1}
    assert report["cooperative_language_flag_rate"] == 0.0


def test_enrich_resumes_only_unenriched_guests(env):
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        return "too short" if calls["n"] <= 2 else GOOD_PARAGRAPH

    env["fake"].paragraph = flaky
    records = [_record(i) for i in range(3)]
    population_id = _upload(env["client"], records).get_json()["data"]["population_id"]
    first = _enrich(env["client"], population_id, parallel=1)
    assert first["result"] == {"population_id": population_id, "enriched": 2,
                               "fallback_template": 1}

    env["fake"].prompts.clear()
    second = _enrich(env["client"], population_id)
    assert len(env["fake"].prompts) == 1
    assert second["result"]["enriched"] == 3


def test_enriched_simulation_text_and_freeze(env):
    records = [_record(i, controlled=i % 2 == 0) for i in range(4)]
    population_id = _upload(env["client"], records).get_json()["data"]["population_id"]

    early = env["client"].post(f"/api/population/{population_id}/simulations",
                               json={"persona_source": "enriched"})
    assert early.status_code == 400

    _enrich(env["client"], population_id)
    created = env["client"].post(f"/api/population/{population_id}/simulations",
                                 json={"persona_source": "enriched"}).get_json()["data"]
    assert created["enriched_agents"] == 4 and created["random_seed"] == 42

    rows = _profiles_csv(env, created["simulation_id"]).decode().splitlines()
    first_text = rows[1]
    expected = flatten_text(
        records[0]["persona_text"] + " ADDITIONAL TRAITS (synthetic): " + GOOD_PARAGRAPH
        + " CURRENT STAY CONTEXT: " + records[0]["experience_state"]["guest_context"]
    )
    assert expected in first_text

    meta = env["client"].get(f"/api/population/{population_id}").get_json()["data"]
    assert meta["frozen"] is True
    blocked = env["client"].post(f"/api/population/{population_id}/enrich", json={})
    assert blocked.status_code == 409


def test_simulations_share_profiles_and_experiences(env):
    records = [_record(i, controlled=i < 2) for i in range(4)]
    population_id = _upload(env["client"], records).get_json()["data"]["population_id"]
    _enrich(env["client"], population_id)

    first = env["client"].post(f"/api/population/{population_id}/simulations",
                               json={"persona_source": "enriched"}).get_json()["data"]
    second = env["client"].post(f"/api/population/{population_id}/simulations",
                                json={"persona_source": "enriched"}).get_json()["data"]
    assert first["simulation_id"] != second["simulation_id"]
    assert _profiles_csv(env, first["simulation_id"]) == _profiles_csv(env, second["simulation_id"])

    def experiences(simulation_id):
        path = env["tmp"] / "simulations" / simulation_id / "population.json"
        agents = json.loads(path.read_text())["agents"]
        return [a["record"]["experience_state"] for a in agents.values()]

    assert experiences(first["simulation_id"]) == experiences(second["simulation_id"])


def test_template_simulation_and_readiness(env):
    records = [_record(i) for i in range(3)]
    population_id = _upload(env["client"], records).get_json()["data"]["population_id"]
    created = env["client"].post(f"/api/population/{population_id}/simulations", json={
        "persona_source": "template", "random_seed": 7, "hours": 6,
    }).get_json()["data"]

    sim_dir = env["tmp"] / "simulations" / created["simulation_id"]
    config = json.loads((sim_dir / "simulation_config.json").read_text())
    assert config["random_seed"] == 7
    assert config["population_id"] == population_id
    assert config["time_config"]["total_simulation_hours"] == 6
    assert created["enriched_agents"] == 0
    assert flatten_text(records[0]["agent_context"]) in (sim_dir / "twitter_profiles.csv").read_text()

    prepared, info = simulation_api._check_simulation_prepared(created["simulation_id"])
    assert prepared is True, info


@pytest.mark.parametrize("body", [
    {"persona_source": "llm"},
    {"hours": "many"},
    {"agents_per_hour_min": 5, "agents_per_hour_max": 2},
])
def test_simulation_rejects_bad_options(env, body):
    population_id = _upload(env["client"], [_record(0)]).get_json()["data"]["population_id"]
    response = env["client"].post(f"/api/population/{population_id}/simulations", json=body)
    assert response.status_code == 400


# Seeded activation

def test_seeded_activation_is_repeatable():
    agents = [{"agent_id": i, "activity_level": 0.5, "active_hours": list(range(24))}
              for i in range(100)]
    time_config = {"agents_per_hour_min": 10, "agents_per_hour_max": 25, "peak_hours": [],
                   "off_peak_hours": [], "morning_hours": [], "work_hours": []}

    def run(seed):
        rng = make_activation_rng({"random_seed": seed})
        return [select_active_agent_ids(time_config, agents, hour % 24, rng=rng)
                for hour in range(12)]

    assert run(42) == run(42)
    assert run(42) != run(43)
    assert make_activation_rng({}) is None
    assert make_activation_rng({"random_seed": True}) is None


def test_run_scripts_pass_the_seeded_rng():
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("run_parallel_simulation.py", "run_twitter_simulation.py",
                 "run_reddit_simulation.py"):
        source = (scripts / name).read_text()
        assert "make_activation_rng(" in source, name
        calls = re.findall(r"select_active_agent_ids\((.*?)\)\n", source, re.S)
        assert calls and all("rng=" in call for call in calls), name
