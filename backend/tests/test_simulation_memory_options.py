import json
from types import SimpleNamespace

import pytest
from flask import Flask

from app.api import simulation as simulation_api
from app.config import Config
from app.utils.locale import get_locale, set_locale
from app.models.project import Project, ProjectStatus
from app.services.report_agent import summarize_simulation_activity
from app.services.simulation_manager import SimulationStatus


def _start_with_graph_memory(monkeypatch):
    started = []
    monkeypatch.setattr(
        simulation_api,
        "SimulationManager",
        lambda: SimpleNamespace(
            get_simulation=lambda _simulation_id: SimpleNamespace(
                simulation_id="sim-1",
                project_id="proj-1",
                graph_id="graph-1",
                status=SimulationStatus.READY,
            ),
        ),
    )
    monkeypatch.setattr(
        simulation_api.SimulationRunner,
        "start_simulation",
        classmethod(
            lambda _cls, **kwargs: started.append(kwargs)
            or SimpleNamespace(to_dict=lambda: {"runner_status": "running"})
        ),
    )
    app = Flask(__name__)
    with app.test_request_context(
        "/api/simulation/start",
        method="POST",
        json={"simulation_id": "sim-1", "enable_graph_memory_update": True},
    ):
        response = simulation_api.start_simulation()
    return response, started


def test_zep_simulation_memory_off_disables_graph_writes(monkeypatch):
    monkeypatch.setattr(Config, "ZEP_SIMULATION_MEMORY", False)

    response, started = _start_with_graph_memory(monkeypatch)

    assert response.get_json()["data"]["graph_memory_update_enabled"] is False
    assert started[0]["enable_graph_memory_update"] is False
    assert started[0]["graph_id"] is None


def _write_actions(sim_dir, platform, records):
    log_dir = sim_dir / platform
    log_dir.mkdir(parents=True)
    (log_dir / "actions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.fixture
def english_locale():
    previous = get_locale()
    set_locale("en")
    yield
    set_locale(previous)


def test_simulation_activity_filters_counts_and_caps(tmp_path, english_locale):
    _write_actions(tmp_path, "twitter", [
        {"event_type": "simulation_start"},
        {"round": 1, "agent_name": "Hotel A", "action_type": "CREATE_POST",
         "action_args": {"content": "Rate your stay from 1 to 10"}, "success": True},
        {"round": 1, "agent_name": "Guest B", "action_type": "LIKE_POST",
         "action_args": {"post_id": 1}, "success": True},
        {"round": 2, "agent_name": "Guest C", "action_type": "DO_NOTHING", "action_args": {}},
        {"round": 2, "agent_name": "Guest D", "action_type": "CREATE_POST",
         "action_args": {"content": "x" * 1000}, "success": False},
    ])
    _write_actions(tmp_path, "reddit", [
        {"round": 1, "agent_name": "Guest B", "action_type": "CREATE_COMMENT",
         "action_args": {"content": "My stay was great"}, "success": True},
    ])

    everything = summarize_simulation_activity(str(tmp_path))
    assert "Matching simulation actions: 3" in everything
    assert "twitter: CREATE_POST=1, LIKE_POST=1" in everything
    assert "DO_NOTHING" not in everything
    assert "Guest D" not in everything

    by_keyword = summarize_simulation_activity(str(tmp_path), keyword="STAY")
    assert "Matching simulation actions: 2" in by_keyword
    assert "Rate your stay" in by_keyword and "My stay was great" in by_keyword

    by_agent = summarize_simulation_activity(str(tmp_path), agent="guest b", platform="reddit")
    assert "Matching simulation actions: 1" in by_agent
    assert "[reddit round 1] Guest B CREATE_COMMENT" in by_agent

    capped = summarize_simulation_activity(str(tmp_path), limit=1)
    assert "First 1:" in capped
    assert capped.count("] ") == 1

    assert "No matching" in summarize_simulation_activity(str(tmp_path), keyword="nothing-like-this")


def test_new_projects_use_larger_default_chunks():
    project = Project(
        project_id="proj-1",
        name="p",
        status=ProjectStatus.CREATED,
        created_at="",
        updated_at="",
    )

    assert (project.chunk_size, project.chunk_overlap) == (2000, 200)
    assert Project.from_dict({"project_id": "proj-1"}).chunk_size == 2000
    assert Project.from_dict({"project_id": "proj-1", "chunk_size": 500}).chunk_size == 500
