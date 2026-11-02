import io
import json
import math
import time

import pytest
from flask import Flask

from app.api import experiments as experiments_api
from app.api import experiments_bp, population_bp
from app.config import Config
from app.experiments.calibration import (
    CalibrationError,
    calibrate,
    fit_from_propensities,
    fit_intercept,
    p_from_propensity,
    resolve_calibration,
    shared_draw,
)
from app.experiments.coder import SYSTEM_PROMPT as CODER_SYSTEM_PROMPT
from app.experiments.coder import ReplyCoder, parse_coding
from app.experiments.evaluate import evaluate_row
from app.experiments.runners import EngineNotImplemented, get_runner
from app.experiments.runners.direct import (
    PROPENSITY_INSTRUCTIONS,
    WRITING_INSTRUCTIONS,
    DirectMessageRunner,
    parse_propensity,
    parse_reply,
)
from app.experiments.schema import ExperimentSpec, SpecError, Variant, parse_variants, row_key
from app.experiments.scoring import parse_score
from app.experiments.snapshot import build_snapshot, issue_categories, reporting_cohort
from app.experiments.summary import guest_level, paired_difference, summarize
from app.services import simulation_runner as runner_module

VARIANTS = [
    {"variant_id": "short", "text": "Hi from {brand}! How is your stay going?"},
    {"variant_id": "rating", "text": "How would you rate your stay so far, 1-5?", "score_scale": 5},
    {"variant_id": "open", "text": "Is there anything we can do to make your stay better?"},
]
CALIBRATION = {"method": "baseline_logit_intercept", "baseline_variant_id": "rating",
               "target_reply_rate": 0.1778, "source": "historical_status_check_response_rate",
               "observed_responses": 3128, "observed_checks": 17593, "provisional": True}


def _record(index, controlled=False):
    persona = (f"You are a guest on a {index % 3 + 1}-night stay at a Brand A hotel. "
               "Do not invent past events.")
    context = ("During this stay, you experienced a problem involving A/C Unit (HVAC). "
               "The issue has not been resolved when this message arrives."
               if controlled else
               "No particular stay experience has been provided for this experiment.")
    events = [{"service_type": "repair", "category": "HVAC", "item": "A/C Unit",
               "need_kind": "problem", "resolution_assumption": "unresolved"}] if controlled else []
    return {
        "guest_id": f"guest_{index + 1:05d}",
        "attributes": {"stay_nights": index % 3 + 1, "sub_brand": "Brand A"},
        "persona_text": persona,
        "experience_state": {"guest_context": context, "events": events},
        "agent_context": persona + " CURRENT STAY CONTEXT: " + context,
        "stay_context": {"cohort": "controlled_scenario" if controlled else "unspecified_experience",
                         "description": context},
    }


def _hvac(system):
    return "HVAC" in system or "A/C" in system


def _default_propensity(system, user):
    if _hvac(system):
        return {"reply_likelihood": 90}
    return {"reply_likelihood": 60 if "rate" in user else 20}


def _default_write(system, user):
    if _hvac(system):
        return {"reply_text": "The AC still isn't working, 2/5."}
    return {"reply_text": "4" if "rate" in user else "All good, thanks."}


def _default_code(user):
    reply = user.split('"""')[1]
    if "AC" in reply:
        return {"feedback_type": "complaint", "mentions_issue": True, "issue_categories": ["HVAC"],
                "issue_summary": "AC broken", "actionable": True, "claims_incident": True,
                "incident_categories": ["HVAC"], "summary": "AC complaint"}
    if "towel" in reply:
        return {"feedback_type": "complaint", "mentions_issue": True,
                "issue_categories": ["other"], "issue_summary": "No towels", "actionable": True,
                "claims_incident": True, "incident_categories": ["other"], "summary": "Towels"}
    return {"feedback_type": "acknowledgement", "mentions_issue": False, "issue_categories": [],
            "issue_summary": None, "actionable": False, "claims_incident": False,
            "incident_categories": [], "summary": "A short reply"}


class FakeLLM:
    model = "fake-model"

    def __init__(self, propensity=_default_propensity, write=_default_write, code=_default_code):
        self.propensity = propensity
        self.write = write
        self.code = code
        self.propensity_calls = []
        self.writing_calls = []
        self.coder_prompts = []
        self.max_tokens = []
        self.efforts = []

    def chat_json(self, messages, temperature=0.3, max_tokens=None, reasoning_effort=None):
        self.max_tokens.append(max_tokens)
        self.efforts.append(reasoning_effort)
        system, user = messages[0]["content"], messages[-1]["content"]
        if system == CODER_SYSTEM_PROMPT:
            self.coder_prompts.append(user)
            return self.code(user)
        if PROPENSITY_INSTRUCTIONS in system:
            self.propensity_calls.append((system, user))
            return self.propensity(system, user)
        assert WRITING_INSTRUCTIONS in system
        self.writing_calls.append((system, user))
        return self.write(system, user)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    fake = FakeLLM()
    monkeypatch.setattr(experiments_api, "runner_factory",
                        lambda engine, reasoning_effort=None: get_runner(
                            engine, llm_client=fake, reasoning_effort=reasoning_effort))
    monkeypatch.setattr(experiments_api, "coder_factory", lambda: ReplyCoder(llm_client=fake))
    app = Flask(__name__)
    app.register_blueprint(population_bp, url_prefix="/api/population")
    app.register_blueprint(experiments_bp, url_prefix="/api/experiments")
    return {"client": app.test_client(), "fake": fake, "tmp": tmp_path}


def _population(client, n=4, controlled=(0,), calibration=CALIBRATION):
    records = [_record(i, controlled=i in controlled) for i in range(n)]
    payload = "".join(json.dumps(r) + "\n" for r in records).encode()
    assumptions = {"random_seed": 42, "historical_scored_stays": 2445, "source_profile_rows": 92655}
    if calibration is not None:
        assumptions["response_calibration"] = calibration
    response = client.post("/api/population/upload", data={
        "guests": (io.BytesIO(payload), "synthetic_guests.jsonl"),
        "assumptions": (io.BytesIO(json.dumps(assumptions).encode()), "simulation_assumptions.json"),
    }, content_type="multipart/form-data")
    return response.get_json()["data"]["population_id"], records


def _wait(client, task_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = client.get(f"/api/experiments/task/{task_id}").get_json()["data"]
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.02)
    raise AssertionError("experiment task did not finish")


def _run(client, population_id, **body):
    response = client.post("/api/experiments", json={
        "population_id": population_id, "variants": VARIANTS, **body})
    assert response.status_code == 200, response.get_json()
    data = response.get_json()["data"]
    task = _wait(client, data["task_id"])
    assert task["status"] == "completed", task
    return data


def _spec(variants=None, replicates=1, seed=1):
    return ExperimentSpec(experiment_id="ex_000000000000", population_id="pp_000000000000",
                          persona_source="template", engine="direct",
                          variants=variants or parse_variants(VARIANTS),
                          replicates=replicates, seed=seed)


def _snapshot(n=3, controlled=(0,)):
    return build_snapshot([_record(i, controlled=i in controlled) for i in range(n)], None)


def _gated_run(fake, n=3, replicates=1, seed=1, calibrator=None, **kwargs):
    rows, stage1 = [], []
    DirectMessageRunner(llm_client=fake).run(
        _spec(replicates=replicates, seed=seed), _snapshot(n), rows.append,
        on_propensity=stage1.append, calibrator=calibrator, **kwargs)
    return rows, stage1


# Calibration math

def test_fit_intercept_hits_target_and_preserves_ordering_and_odds_ratios():
    p_raws = [p_from_propensity(x) for x in (5, 30, 60, 90, 100)]
    b = fit_intercept(p_raws, 0.1778)
    calibrated = [calibrate(p, b) for p in p_raws]
    assert sum(calibrated) / len(calibrated) == pytest.approx(0.1778, abs=1e-6)
    assert calibrated == sorted(calibrated)

    def odds(p):
        return p / (1 - p)

    assert odds(calibrated[3]) / odds(calibrated[1]) == pytest.approx(
        odds(p_raws[3]) / odds(p_raws[1]))


def test_propensity_is_clipped_and_uncalibrated_passes_through():
    assert p_from_propensity(0) == 0.01 and p_from_propensity(100) == 0.99
    assert p_from_propensity(42) == 0.42
    assert calibrate(0.3, None) == 0.3


def test_shared_draw_is_keyed_by_seed_guest_and_replicate():
    assert shared_draw(1, "g1", 1) == shared_draw(1, "g1", 1)
    assert len({shared_draw(1, "g1", 1), shared_draw(1, "g1", 2),
                shared_draw(1, "g2", 1), shared_draw(2, "g1", 1)}) == 4


def test_resolve_uses_assumptions_then_request_field_by_field():
    ids = ["short", "rating", "open"]
    resolved = resolve_calibration(None, CALIBRATION, ids)
    assert resolved["status"] == "provisional_target" and resolved["origin"] == "assumptions"
    assert resolved["target_reply_rate"] == 0.1778 and resolved["observed_checks"] == 17593

    override = resolve_calibration({"target_reply_rate": 0.2}, CALIBRATION, ids)
    assert override["target_reply_rate"] == 0.2 and override["baseline_variant_id"] == "rating"
    assert override["origin"] == "request+assumptions"
    assert override["source"] == "request_override" and override["observed_checks"] is None

    baseline_only = resolve_calibration({"baseline_variant_id": "open"}, CALIBRATION, ids)
    assert baseline_only["baseline_variant_id"] == "open"
    assert baseline_only["target_reply_rate"] == 0.1778


def test_resolve_provisional_defaults_and_validated_status():
    ids = ["rating"]
    assert resolve_calibration({"target_reply_rate": 0.3}, None, ids)["provisional"] is True
    validated = {**CALIBRATION, "provisional": False}
    assert resolve_calibration(None, validated, ids)["status"] == "calibrated"
    # A new target from the request is unverified even if the population's was validated.
    assert resolve_calibration({"target_reply_rate": 0.3}, validated, ids)["status"] == \
        "provisional_target"


def test_resolve_without_target_is_not_calibrated_with_warning():
    resolved = resolve_calibration(None, None, ["rating"])
    assert resolved["status"] == "not_calibrated" and resolved["warning"]


@pytest.mark.parametrize("request_block, code", [
    ({"target_reply_rate": 0}, "invalid"),
    ({"target_reply_rate": 1}, "invalid"),
    ({"target_reply_rate": "0.2"}, "invalid"),
    ({"target_reply_rate": True}, "invalid"),
    ({"target_reply_rate": 0.2, "provisional": "yes"}, "invalid"),
    ({"target_reply_rate": 0.2, "baseline_variant_id": "missing"}, "unknown_baseline"),
    ("0.2", "invalid"),
])
def test_resolve_rejects_bad_settings(request_block, code):
    with pytest.raises(CalibrationError) as caught:
        resolve_calibration(request_block, None, ["rating"])
    assert caught.value.code == code


def test_fit_uses_only_valid_baseline_rows():
    resolved = resolve_calibration(None, CALIBRATION, ["rating", "open"])
    rows = [
        {"variant_id": "rating", "guest_id": "a", "propensity_raw": 50},
        {"variant_id": "rating", "guest_id": "b", "propensity_raw": None},
        {"variant_id": "open", "guest_id": "a", "propensity_raw": 99},
    ]
    fit = fit_from_propensities(rows, resolved)
    assert fit["baseline_instances"] == 1 and fit["baseline_mean_p_raw"] == 0.5
    assert fit["baseline_mean_p_calibrated"] == pytest.approx(0.1778, abs=1e-4)
    assert fit["intercept"] == pytest.approx(math.log(0.1778 / 0.8222), abs=1e-4)
    assert fit_from_propensities(rows[2:], resolved)["status"] == "not_calibrated"


# Stage 1 and stage 2 parsing

def test_parse_propensity():
    assert parse_propensity({"reply_likelihood": 42}) == 42
    assert parse_propensity({"reply_likelihood": "55"}) == 55
    assert parse_propensity({"reply_likelihood": "70%"}) == 70
    assert parse_propensity({"reply_likelihood": 33.6}) == 34
    for bad in ({"reply_likelihood": True}, {"reply_likelihood": -1},
                {"reply_likelihood": 101}, {"reply_likelihood": "often"}, {}, ["x"]):
        with pytest.raises(ValueError):
            parse_propensity(bad)


def test_parse_reply():
    assert parse_reply({"reply_text": " ok\nthanks "}) == "ok thanks"
    for bad in ({"reply_text": ""}, {"reply_text": None}, {}, "text"):
        with pytest.raises(ValueError):
            parse_reply(bad)


# Two-stage runner

def test_gate_follows_shared_draw_and_only_responders_get_writing_calls():
    fake = FakeLLM()
    rows, stage1 = _gated_run(fake, n=6, replicates=2)
    assert len(rows) == len(stage1) == 6 * 3 * 2
    assert len(fake.propensity_calls) == 36
    responders = [r for r in rows if r["decision"] == "respond"]
    assert len(fake.writing_calls) == len(responders) > 0
    for row in rows:
        assert row["p_raw"] == p_from_propensity(row["propensity_raw"])
        assert row["random_draw"] == round(shared_draw(1, row["guest_id"], row["replicate"]), 6)
        assert (row["decision"] == "respond") == (row["random_draw"] < row["p_calibrated"])
        if row["decision"] == "ignore":
            assert row["reply_text"] is None


def test_draw_is_reused_across_variants_but_not_replicates():
    rows, _ = _gated_run(FakeLLM(), n=3, replicates=2)
    draws = {}
    for row in rows:
        draws.setdefault((row["guest_id"], row["replicate"]), set()).add(row["random_draw"])
    assert all(len(d) == 1 for d in draws.values())
    assert len({next(iter(d)) for d in draws.values()}) == 6


def test_higher_propensity_never_flips_to_ignore_under_shared_draw():
    rows, _ = _gated_run(FakeLLM(), n=6, replicates=3)
    by_key = {(r["guest_id"], r["replicate"], r["variant_id"]): r for r in rows}
    for (guest, replicate, variant), row in by_key.items():
        for other in ("short", "rating", "open"):
            peer = by_key[(guest, replicate, other)]
            if row["p_calibrated"] > peer["p_calibrated"] and peer["decision"] == "respond":
                assert row["decision"] == "respond"


def test_calibrator_runs_once_on_all_propensities_and_is_applied():
    seen = []

    def calibrator(stage1):
        seen.append(len(stage1))
        return -1.5

    rows, _ = _gated_run(FakeLLM(), n=3, calibrator=calibrator)
    assert seen == [9]
    for row in rows:
        assert row["calibration_intercept"] == -1.5
        expected = 1 / (1 + math.exp(-(math.log(row["p_raw"] / (1 - row["p_raw"])) - 1.5)))
        assert row["p_calibrated"] == pytest.approx(expected, abs=1e-6)


def test_propensity_failure_is_an_error_row_without_writing():
    fake = FakeLLM(propensity=lambda s, u: {"reply_likelihood": "often"})
    rows, stage1 = _gated_run(fake, n=1)
    assert all(r["decision"] == "error" and r["error_stage"] == "propensity" for r in rows)
    assert all(r["propensity_raw"] is None and r["attempts"] == 2 for r in rows)
    assert stage1[0]["error"] and not fake.writing_calls
    assert fake.max_tokens.count(1500) == fake.max_tokens.count(None) == 3


def test_writing_failure_keeps_the_propensity():
    fake = FakeLLM(propensity=lambda s, u: {"reply_likelihood": 100},
                   write=lambda s, u: {"reply_text": ""})
    rows, _ = _gated_run(fake, n=4, calibrator=lambda stage1: 20.0)
    assert rows and all(r["decision"] == "error" and r["error_stage"] == "writing" for r in rows)
    assert all(r["propensity_raw"] == 100 and r["attempts"] == 3 for r in rows)


def test_writing_retry_recovers():
    answers = iter([{"reply_text": ""}, {"reply_text": "Fine"}])
    fake = FakeLLM(propensity=lambda s, u: {"reply_likelihood": 100},
                   write=lambda s, u: next(answers))
    variant = Variant("open", "open", "Anything?")
    rows = []
    DirectMessageRunner(llm_client=fake).run(_spec(variants=[variant]), _snapshot(1), rows.append,
                                             calibrator=lambda stage1: 20.0)
    assert rows[0]["decision"] == "respond" and rows[0]["reply_text"] == "Fine"
    assert rows[0]["error"] is None and rows[0]["attempts"] == 3


def test_all_variants_reach_identical_guest_texts_in_both_stages():
    fake = FakeLLM(propensity=lambda s, u: {"reply_likelihood": 100})
    _gated_run(fake, n=3, replicates=2, calibrator=lambda stage1: 20.0)
    for calls in (fake.propensity_calls, fake.writing_calls):
        systems = {}
        for system, user in calls:
            systems.setdefault(user, set()).add(system)
        assert len(systems) == 3
        assert len({frozenset(s) for s in systems.values()}) == 1
    assert any("Hi from Brand A!" in user for _, user in fake.propensity_calls)


def test_resume_reuses_propensities_and_skips_completed_rows():
    first_fake = FakeLLM()
    rows, stage1 = _gated_run(first_fake, n=2)
    resumed_fake = FakeLLM()
    done = {row_key(rows[0])}
    more, new_stage1 = _gated_run(resumed_fake, n=2, existing_propensities=stage1,
                                  completed_keys=done)
    assert not resumed_fake.propensity_calls and not new_stage1
    assert len(more) == 5
    original = {row_key(r): r["decision"] for r in rows}
    assert all(original[row_key(r)] == r["decision"] for r in more)


# Scores

@pytest.mark.parametrize("text, scale, expected", [
    ("4", 5, 4),
    ("I'd say 4/5", 5, 4),
    ("4 out of 5, thanks", 5, 4),
    ("9/10", 10, 9),
    ("6", 5, None),
    ("3 or 4", 5, None),
    ("4.5", 5, None),
    ("Room 412 is fine", 5, None),
    ("8/10", 5, None),
    ("great stay", 5, None),
    ("4", None, None),
])
def test_parse_score(text, scale, expected):
    assert parse_score(text, scale) == expected


# Variants

@pytest.mark.parametrize("variants", [
    [],
    [{"variant_id": "a", "text": ""}],
    [{"variant_id": "a", "text": "x"}, {"variant_id": "a", "text": "y"}],
    [{"variant_id": "bad id", "text": "x"}],
    [{"variant_id": "a", "text": "x", "score_scale": 7}],
    [{"variant_id": str(i), "text": "x"} for i in range(6)],
])
def test_parse_variants_rejects(variants):
    with pytest.raises(SpecError):
        parse_variants(variants)


def test_brand_placeholder():
    variant = Variant("a", "a", "Welcome to {brand}")
    assert variant.message_for("Brand A") == "Welcome to Brand A"
    assert variant.message_for(None) == "Welcome to the hotel"


# Coding and evaluation

def test_coder_sees_only_reply_text_and_fixed_categories():
    fake = FakeLLM()
    snapshot = build_snapshot([_record(0, controlled=True), _record(1)], None)
    categories = issue_categories(snapshot)
    rows = [
        {"decision": "respond", "reply_text": "The AC still isn't working", "guest_id": "guest_00001",
         "variant_id": "open"},
        {"decision": "respond", "reply_text": "All fine", "guest_id": "guest_00002",
         "variant_id": "open"},
        {"decision": "ignore", "reply_text": None, "guest_id": "guest_00002", "variant_id": "open"},
    ]
    ReplyCoder(llm_client=fake).code_rows(rows, categories)

    assert len(fake.coder_prompts) == 2
    for prompt in fake.coder_prompts:
        assert "guest_0000" not in prompt
        assert "A/C Unit" not in prompt and "not been resolved" not in prompt
        assert "HVAC, other" in prompt
    assert rows[0]["coded"]["issue_categories"] == ["HVAC"]
    assert rows[0]["coded"]["actionable"] is True
    assert rows[0]["coded"]["coder_prompt_version"] == "reply-coder-v2"
    assert rows[2].get("coded") is None


def test_parse_coding_restricts_categories():
    coded = parse_coding({"feedback_type": "weird", "mentions_issue": True,
                          "issue_categories": ["Plumbing", "hvac"]}, ["HVAC", "other"])
    assert coded["feedback_type"] == "other" and coded["issue_categories"] == ["HVAC"]
    assert parse_coding({"mentions_issue": True, "issue_categories": []}, ["other"])[
        "issue_categories"] == ["other"]
    assert parse_coding({"mentions_issue": False, "issue_categories": ["HVAC"]}, ["HVAC"])[
        "issue_categories"] == []


def test_parse_coding_incidents_and_actionable():
    coded = parse_coding({"claims_incident": True, "incident_categories": ["Plumbing"],
                          "actionable": "yes"}, ["HVAC", "other"])
    assert coded["incident_categories"] == ["other"] and coded["actionable"] is False
    coded = parse_coding({"claims_incident": False, "incident_categories": ["HVAC"],
                          "actionable": True}, ["HVAC", "other"])
    assert coded["incident_categories"] == [] and coded["actionable"] is True


def test_evaluate_row():
    assigned = {"assigned_need": {"category": "HVAC"}}
    unassigned = {"assigned_need": None}
    hvac = {"decision": "respond", "coded": {"issue_categories": ["HVAC"], "claims_incident": True,
                                             "incident_categories": ["HVAC"]}}
    other = {"decision": "respond", "coded": {"issue_categories": ["other"]}}
    assert evaluate_row(hvac, assigned) == {"assigned_category": "HVAC", "need_detected": True,
                                            "unassigned_issue": False,
                                            "unsupported_incident": False}
    assert evaluate_row(other, assigned)["need_detected"] is False
    assert evaluate_row(other, assigned)["unassigned_issue"] is True
    assert evaluate_row(other, assigned)["unsupported_incident"] is False
    assert evaluate_row(hvac, unassigned)["unsupported_incident"] is True
    assert evaluate_row(hvac, unassigned)["need_detected"] is None
    assert evaluate_row({"decision": "ignore"}, assigned)["need_detected"] is False
    assert evaluate_row({"decision": "respond", "coded": None}, assigned) is None


# Statistics

def test_guest_level_averages_replicates_before_resampling():
    rows = [{"guest_id": "a", "decision": "respond"}, {"guest_id": "a", "decision": "ignore"},
            {"guest_id": "b", "decision": "respond"}, {"guest_id": "c", "decision": "error"}]
    result = guest_level(rows, lambda r: None if r["decision"] == "error"
                         else float(r["decision"] == "respond"), key="k")
    assert result["estimate"] == 0.75
    assert result["instances"] == 3 and result["unique_guests"] == 2
    low, high = result["ci95"]
    assert 0.5 <= low <= high <= 1.0
    assert result == guest_level(rows, lambda r: None if r["decision"] == "error"
                                 else float(r["decision"] == "respond"), key="k")


def test_paired_difference_is_deterministic():
    rows = []
    for g in range(20):
        rows.append({"variant_id": "a", "guest_id": str(g), "decision": "respond"})
        rows.append({"variant_id": "b", "guest_id": str(g),
                     "decision": "respond" if g < 5 else "ignore"})
    rows.append({"variant_id": "b", "guest_id": "x", "decision": "respond"})
    first = paired_difference(rows, "a", "b", seed=3)
    assert first == paired_difference(rows, "a", "b", seed=3)
    assert first["paired_guests"] == 20 and first["difference"] == 0.75
    assert first["instances_a"] == 20 and first["instances_b"] == 21
    low, high = first["ci95"]
    assert low <= 0.75 <= high


def test_reporting_cohort():
    assert reporting_cohort({"assigned_need": None}) == "unspecified"
    assert reporting_cohort({"assigned_need": {"need_kind": "request"}}) == "service_request"
    assert reporting_cohort({"assigned_need": {"need_kind": "problem"}}) == "repair_problem"
    assert reporting_cohort({"assigned_need": {"need_kind": None}}) == "unclassified_need"


def test_summary_splits_propensity_and_comparisons_by_cohort():
    variants = [Variant("a", "a", "A?"), Variant("b", "b", "B?")]
    kinds = {"g1": None, "g2": None, "g3": "request", "g4": "request", "g5": "problem"}
    snapshot = [{"guest_id": g, "assigned_need": {"category": "x", "need_kind": k} if k else None}
                for g, k in kinds.items()]
    # Requests and problems prefer b; unspecified guests prefer a.
    propensity = {("a", None): 60, ("b", None): 30, ("a", "request"): 20, ("b", "request"): 70,
                  ("a", "problem"): 40, ("b", "problem"): 90}
    rows = []
    for variant in ("a", "b"):
        for guest, kind in kinds.items():
            p = propensity[(variant, kind)] / 100
            rows.append({"variant_id": variant, "guest_id": guest, "cohort": "c",
                         "decision": "ignore", "reply_text": None, "propensity_raw": p * 100,
                         "p_raw": p, "p_calibrated": p})
    summary = summarize(_spec(variants=variants), rows, snapshot, "sha")

    assert summary["guests_by_cohort"] == {"unspecified": 2, "service_request": 2,
                                           "repair_problem": 1}
    by_cohort = summary["variants"][1]["by_cohort"]
    assert list(by_cohort) == ["unspecified", "service_request", "repair_problem"]
    assert by_cohort["service_request"]["mean_propensity_raw"]["estimate"] == 70
    assert by_cohort["service_request"]["mean_p_calibrated"]["unique_guests"] == 2
    assert by_cohort["repair_problem"]["mean_p_calibrated"]["estimate"] == 0.9
    assert by_cohort["unspecified"]["actionable_yield_per_message"]["estimate"] == 0

    within = summary["comparisons"][0]["within_cohort"]
    assert within["unspecified"]["p_calibrated"]["difference"] == 0.3
    assert within["service_request"]["p_calibrated"]["difference"] == -0.5
    assert within["repair_problem"]["p_calibrated"]["paired_guests"] == 1


def test_actionable_yield_counts_ignores_and_waits_for_coding():
    spec = _spec(variants=[Variant("open", "open", "Anything?")])
    snapshot = _snapshot(2, controlled=())
    rows = [
        {"variant_id": "open", "guest_id": "guest_00001", "cohort": "c", "decision": "respond",
         "reply_text": "No towels", "coded": {"actionable": True, "feedback_type": "complaint",
                                              "mentions_issue": True, "issue_categories": []}},
        {"variant_id": "open", "guest_id": "guest_00002", "cohort": "c", "decision": "ignore",
         "reply_text": None},
    ]
    variant = summarize(spec, rows, snapshot, "sha")["variants"][0]
    assert variant["actionable_yield_per_message"]["estimate"] == 0.5
    rows[0]["coded"] = None
    assert summarize(spec, rows, snapshot, "sha")["variants"][0][
        "actionable_yield_per_message"] is None


# Engines

def test_oasis_social_is_not_implemented():
    with pytest.raises(EngineNotImplemented):
        get_runner("oasis_social", llm_client=FakeLLM())
    with pytest.raises(ValueError):
        get_runner("telepathy", llm_client=FakeLLM())


# API

def test_experiment_end_to_end(env):
    client = env["client"]
    population_id, records = _population(client, n=4, controlled=(0,))
    data = _run(client, population_id, analyze=True)
    experiment_id = data["experiment_id"]
    assert data["expected_rows"] == 12
    assert data["calibration_status"] == "provisional_target" and data["warnings"] == []

    meta = client.get(f"/api/experiments/{experiment_id}").get_json()["data"]
    assert meta["status"] == "completed" and meta["result_rows_written"] == 12
    assert meta["snapshot_sha256"] == data["snapshot_sha256"]
    assert meta["analysis"]["status"] == "completed"
    assert meta["spec"]["seed"] == 42
    assert meta["spec"]["prompt_version"] == "direct-message-v2-gated"
    assert meta["spec"]["response_calibration"]["target_reply_rate"] == 0.1778
    assert meta["reasoning_effort"] == "low" and meta["coder_reasoning_effort"] == "low"
    fit = meta["calibration_fit"]
    assert fit["baseline_variant_id"] == "rating" and fit["baseline_instances"] == 4
    assert fit["baseline_mean_p_calibrated"] == pytest.approx(0.1778, abs=1e-4)
    assert set(env["fake"].efforts) == {"low"}

    experiment_dir = env["tmp"] / "uploads" / "experiments" / experiment_id
    stage1 = (experiment_dir / "propensities.jsonl").read_text().splitlines()
    assert len(stage1) == 12

    population = client.get(f"/api/population/{population_id}").get_json()["data"]
    assert population["frozen"] is True
    assert population["experiments"][0]["experiment_id"] == experiment_id
    assert client.post(f"/api/population/{population_id}/enrich", json={}).status_code == 409

    rows = client.get(f"/api/experiments/{experiment_id}/results?limit=1000").get_json()["data"]
    assert {r["calibration_intercept"] for r in rows["results"]} == {fit["intercept"]}
    assert len(env["fake"].writing_calls) == sum(
        r["decision"] == "respond" for r in rows["results"])

    summary = client.get(f"/api/experiments/{experiment_id}/summary").get_json()["data"]
    assert summary["complete"] is True
    by_id = {v["variant_id"]: v for v in summary["variants"]}
    rating = by_id["rating"]
    assert rating["mean_propensity_raw"]["estimate"] == pytest.approx((90 + 60 * 3) / 4)
    assert rating["mean_p_calibrated"]["estimate"] == pytest.approx(0.1778, abs=1e-3)
    assert rating["simulated_reply_rate"]["unique_guests"] == 4
    assert rating["simulated_reply_rate"]["instances"] == 4
    assert "scores" in rating and "scores" not in by_id["open"]
    assert by_id["short"]["actionable_yield_per_message"]["instances"] == 4
    assert by_id["short"]["unsupported_incidents"]["count"] == 0
    assert "need_discovery" in by_id["short"]["issue_discovery"]
    assert summary["guests_by_cohort"] == {"repair_problem": 1, "unspecified": 3}
    assert rating["by_cohort"]["repair_problem"]["mean_propensity_raw"]["estimate"] == 90
    assert set(summary["comparisons"][0]["within_cohort"]) == {"unspecified", "repair_problem"}
    comparison = summary["comparisons"][0]
    assert set(comparison) >= {"reply_rate", "p_calibrated", "actionable_yield", "log_odds_raw"}
    assert comparison["reply_rate"]["paired_guests"] == 4
    calibration = summary["calibration"]
    assert calibration["status"] == "provisional_target"
    assert calibration["intercept"] == fit["intercept"]
    assert calibration["score_calibration"] == "deferred_v2"
    assert "not a validated population response model" in calibration["note"]

    filtered = client.get(f"/api/experiments/{experiment_id}/results?variant=rating").get_json()["data"]
    assert filtered["total"] == 4
    flagged = client.get(f"/api/experiments/{experiment_id}/results?flag=unsupported_incident")
    assert flagged.status_code == 200 and flagged.get_json()["data"]["total"] == 0
    assert client.get(f"/api/experiments/{experiment_id}/results?flag=bogus").status_code == 400


def test_unsupported_incident_flag_filter(env):
    env["fake"].propensity = lambda s, u: {"reply_likelihood": 100}
    env["fake"].write = lambda s, u: {"reply_text": "There were no towels in my room."}
    client = env["client"]
    population_id, _ = _population(client, n=2, controlled=(0,), calibration=None)
    data = _run(client, population_id, analyze=True,
                variants=[{"variant_id": "open", "text": "Anything?"}])
    results = client.get(
        f"/api/experiments/{data['experiment_id']}/results?flag=unsupported_incident"
    ).get_json()["data"]
    responders = client.get(
        f"/api/experiments/{data['experiment_id']}/results?decision=respond").get_json()["data"]
    assert results["total"] == responders["total"] > 0


def test_request_override_and_uncalibrated_warning(env):
    client = env["client"]
    population_id, _ = _population(client, n=3)
    data = _run(client, population_id, response_calibration={"target_reply_rate": 0.4})
    meta = client.get(f"/api/experiments/{data['experiment_id']}").get_json()["data"]
    assert meta["response_calibration"]["target_reply_rate"] == 0.4
    assert meta["response_calibration"]["origin"] == "request+assumptions"
    assert meta["calibration_fit"]["baseline_mean_p_calibrated"] == pytest.approx(0.4, abs=1e-4)

    bare_id, _ = _population(client, n=3, calibration=None)
    bare = _run(client, bare_id)
    assert bare["calibration_status"] == "not_calibrated" and bare["warnings"]
    summary = client.get(f"/api/experiments/{bare['experiment_id']}/summary").get_json()["data"]
    assert summary["calibration"]["status"] == "not_calibrated"
    assert summary["calibration"]["warning"]
    rows = client.get(f"/api/experiments/{bare['experiment_id']}/results").get_json()["data"]
    assert all(r["p_calibrated"] == r["p_raw"] for r in rows["results"])


def test_snapshot_written_and_independent_of_population_changes(env):
    client = env["client"]
    population_id, records = _population(client, n=3)
    data = _run(client, population_id)
    experiment_dir = env["tmp"] / "uploads" / "experiments" / data["experiment_id"]
    snapshot = [json.loads(line) for line in (experiment_dir / "snapshot.jsonl").read_text().splitlines()]
    assert [g["guest_id"] for g in snapshot] == [r["guest_id"] for r in records]
    assert all(len(g["agent_text_sha256"]) == 64 for g in snapshot)

    guests_file = env["tmp"] / "uploads" / "populations" / population_id / "guests.jsonl"
    guests_file.write_text("")
    analyzed = client.post(f"/api/experiments/{data['experiment_id']}/analyze", json={})
    assert analyzed.status_code == 200
    assert _wait(client, analyzed.get_json()["data"]["task_id"])["status"] == "completed"
    summary = client.get(f"/api/experiments/{data['experiment_id']}/summary").get_json()["data"]
    assert summary["guests"] == 3 and summary["snapshot_sha256"] == data["snapshot_sha256"]


def test_two_experiments_share_snapshot_hash(env):
    client = env["client"]
    population_id, _ = _population(client, n=3)
    first = _run(client, population_id)
    second = _run(client, population_id, replicates=2)
    assert first["snapshot_sha256"] == second["snapshot_sha256"]
    assert second["expected_rows"] == 18


@pytest.mark.parametrize("body, status", [
    ({"population_id": ""}, 400),
    ({"population_id": "pp_000000000000"}, 404),
    ({"engine": "oasis_social"}, 400),
    ({"engine": "telepathy"}, 400),
    ({"variants": [{"variant_id": "a", "text": ""}]}, 400),
    ({"replicates": 11}, 400),
    ({"replicates": "two"}, 400),
    ({"analyze": "yes"}, 400),
    ({"persona_source": "llm"}, 400),
    ({"persona_source": "enriched"}, 400),
    ({"reasoning_effort": "max"}, 400),
    ({"response_calibration": "0.2"}, 400),
    ({"response_calibration": {"target_reply_rate": 1.5}}, 400),
    ({"response_calibration": {"baseline_variant_id": "missing"}}, 400),
    ({"variants": [{"variant_id": "open", "text": "Anything?"}]}, 400),
])
def test_create_rejects_bad_requests(env, body, status):
    population_id, _ = _population(env["client"], n=2)
    payload = {"population_id": population_id, "variants": VARIANTS, **body}
    response = env["client"].post("/api/experiments", json=payload)
    assert response.status_code == status, response.get_json()


def test_reasoning_effort_is_per_experiment(env):
    client = env["client"]
    population_id, _ = _population(client, n=2)
    data = _run(client, population_id, reasoning_effort="high")
    meta = client.get(f"/api/experiments/{data['experiment_id']}").get_json()["data"]
    assert meta["reasoning_effort"] == "high" and meta["spec"]["reasoning_effort"] == "high"
    assert set(env["fake"].efforts) == {"high"}


def test_unknown_experiment_is_404(env):
    assert env["client"].get("/api/experiments/ex_000000000000").status_code == 404
    assert env["client"].get("/api/experiments/../etc").status_code == 404


# Twitter-only runs

def test_twitter_platform_uses_parallel_script(monkeypatch, tmp_path):
    simulation_id = "sim-twitter-only"
    sim_dir = tmp_path / "runs" / simulation_id
    scripts_dir = tmp_path / "scripts"
    sim_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    (sim_dir / "simulation_config.json").write_text(json.dumps(
        {"time_config": {"total_simulation_hours": 1, "minutes_per_round": 60}}))
    (scripts_dir / "run_parallel_simulation.py").write_text("pass\n")

    commands = []

    class Process:
        pid = 1

        def poll(self):
            return None

    class Stop(Exception):
        pass

    def fake_popen(cmd, *args, **kwargs):
        commands.append(cmd)
        raise Stop()

    SimulationRunner = runner_module.SimulationRunner
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(SimulationRunner, "SCRIPTS_DIR", str(scripts_dir))
    monkeypatch.setattr(runner_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status",
                        classmethod(lambda _cls, *_a, **_k: None))
    with pytest.raises(Exception):
        SimulationRunner.start_simulation(simulation_id, platform="twitter",
                                          enable_graph_memory_update=False)
    assert commands and commands[0][1].endswith("run_parallel_simulation.py")
    assert "--twitter-only" in commands[0]
