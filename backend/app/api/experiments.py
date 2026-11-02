"""
Experiment API routes

Run engagement-message variants against a frozen population snapshot and
read back engine-neutral results and summaries.
"""

import json
import os
import subprocess
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from flask import jsonify, request

from . import experiments_bp
from ..experiments.calibration import (
    STATUS_NOT_CALIBRATED,
    CalibrationError,
    fit_from_propensities,
    resolve_calibration,
)
from ..experiments.coder import CODER_PROMPT_VERSION, ReplyCoder
from ..experiments.coder import REASONING_EFFORT as CODER_REASONING_EFFORT
from ..experiments.evaluate import evaluate_rows
from ..experiments.runners import EngineNotImplemented, get_runner
from ..experiments.runners.direct import REASONING_EFFORT as DEFAULT_REASONING_EFFORT
from ..experiments.runners.direct import REASONING_EFFORTS
from ..experiments.schema import (
    KNOWN_ENGINES,
    MAX_REPLICATES,
    ENGINE_DIRECT,
    ExperimentSpec,
    SpecError,
    new_experiment_id,
    parse_variants,
    row_key,
)
from ..experiments.snapshot import build_snapshot, issue_categories
from ..experiments.store import ExperimentNotFound, ExperimentStore
from ..experiments.summary import summarize
from ..models.task import TaskManager, TaskStatus
from ..population.mirofish_adapter import PERSONA_SOURCE_ENRICHED, PERSONA_SOURCE_TEMPLATE
from ..population.store import PopulationNotFound, PopulationStore
from ..utils.locale import get_locale, set_locale, t
from ..utils.logger import get_logger

logger = get_logger("mirofish.api.experiments")

MAX_PARALLEL = 10
RESULT_FLAGS = ("unsupported_incident", "unassigned_issue", "need_detected")
# Share of overall task progress at the start of each stage.
STAGE_PROGRESS = {"propensity": (0, 50), "writing": (50, 85), "coding": (85, 100)}

# Experiments with a live run or analysis thread in this process.
_running = set()
_running_guard = threading.Lock()

# Tests replace these to inject fake LLM clients.
runner_factory = get_runner
coder_factory = ReplyCoder


def _error(message: str, status: int):
    return jsonify({"success": False, "error": message}), status


def _not_found(experiment_id: str):
    return _error(t("experiments.notFound", id=experiment_id), 404)


def _bounded_int(data: Dict[str, Any], field: str, default: int, low: int, high: int) -> int:
    value = data.get(field, default)
    if value is None:
        value = default
    if isinstance(value, bool):
        raise ValueError(t("population.invalidNumber", field=field))
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(t("population.invalidNumber", field=field))
    if not low <= value <= high:
        raise ValueError(t("experiments.outOfRange", field=field, min=low, max=high))
    return value


def _bool_field(data: Dict[str, Any], field: str) -> bool:
    value = data.get(field, False)
    if not isinstance(value, bool):
        raise ValueError(t("population.invalidBoolean", field=field))
    return value


def _claim(experiment_id: str) -> bool:
    with _running_guard:
        if experiment_id in _running:
            return False
        _running.add(experiment_id)
        return True


def _release(experiment_id: str) -> None:
    with _running_guard:
        _running.discard(experiment_id)


def _is_running(experiment_id: str) -> bool:
    with _running_guard:
        return experiment_id in _running


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _assumed_calibration(population_store: PopulationStore, population_id: str) -> Optional[Dict]:
    path = population_store.assumptions_path(population_id)
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("response_calibration")
    except (OSError, ValueError, AttributeError):
        return None


def _calibration_error(exc: CalibrationError):
    if exc.code == "unknown_baseline":
        return _error(t("experiments.unknownBaseline", id=exc.detail), 400)
    return _error(t("experiments.invalidCalibration", error=exc.detail or exc.code), 400)


def _live_summary(store: ExperimentStore, experiment_id: str) -> Dict[str, Any]:
    meta = store.get(experiment_id)
    return summarize(
        store.spec(experiment_id),
        store.results(experiment_id),
        store.snapshot(experiment_id),
        meta["snapshot_sha256"],
        calibration_fit=meta.get("calibration_fit"),
    )


def _refresh_summary(store: ExperimentStore, experiment_id: str) -> Dict[str, Any]:
    summary = _live_summary(store, experiment_id)
    store.save_summary(experiment_id, summary)
    return summary


def _stage_progress(task_manager, task_id, stage, current, total, message_key):
    start, end = STAGE_PROGRESS[stage]
    task_manager.update_task(
        task_id,
        progress=start + int((end - start) * current / total) if total else end,
        message=t(message_key, current=current, total=total),
        progress_detail={"stage": stage, "current": current, "total": total},
    )


def _code_and_evaluate(store, experiment_id, coder, parallel, force, on_progress=None) -> Dict[str, Any]:
    snapshot = store.snapshot(experiment_id)
    rows = coder.code_rows(store.results(experiment_id), issue_categories(snapshot),
                           parallel=parallel, force=force, progress_callback=on_progress)
    rows = evaluate_rows(rows, snapshot)
    store.replace_results(experiment_id, rows)
    coded = [r for r in rows if r.get("coded")]
    return {
        "status": "completed",
        "model": coder.model_name,
        "coder_prompt_version": CODER_PROMPT_VERSION,
        "coder_reasoning_effort": CODER_REASONING_EFFORT,
        "coded_replies": sum(not r["coded"].get("error") for r in coded),
        "coding_errors": sum(bool(r["coded"].get("error")) for r in coded),
        "finished_at": datetime.now().isoformat(),
    }


@experiments_bp.route("", methods=["POST"])
@experiments_bp.route("/", methods=["POST"])
def create_experiment():
    """Body: {population_id, persona_source, engine, variants, replicates, seed, parallel,
    analyze, reasoning_effort, response_calibration}."""
    data = request.get_json(silent=True) or {}
    population_id = str(data.get("population_id") or "").strip()
    if not population_id:
        return _error(t("experiments.requirePopulation"), 400)
    persona_source = data.get("persona_source", PERSONA_SOURCE_TEMPLATE)
    if persona_source not in (PERSONA_SOURCE_ENRICHED, PERSONA_SOURCE_TEMPLATE):
        return _error(t("population.invalidPersonaSource"), 400)
    engine = data.get("engine", ENGINE_DIRECT)
    if engine not in KNOWN_ENGINES:
        return _error(t("experiments.invalidEngine", engine=engine), 400)
    try:
        variants = parse_variants(data.get("variants"))
    except SpecError as exc:
        return _error(t("experiments.invalidVariants", error=str(exc)), 400)
    try:
        replicates = _bounded_int(data, "replicates", 1, 1, MAX_REPLICATES)
        parallel = _bounded_int(data, "parallel", 5, 1, MAX_PARALLEL)
        seed = data.get("seed")
        if seed is not None:
            seed = _bounded_int(data, "seed", 0, 0, 2**31 - 1)
        analyze = _bool_field(data, "analyze")
    except ValueError as exc:
        return _error(str(exc), 400)
    reasoning_effort = data.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
    if reasoning_effort not in REASONING_EFFORTS:
        return _error(t("experiments.invalidEffort", values=", ".join(REASONING_EFFORTS)), 400)
    calibration_request = data.get("response_calibration")
    if calibration_request is not None and not isinstance(calibration_request, dict):
        return _error(t("experiments.invalidCalibration",
                        error="response_calibration must be an object"), 400)

    try:
        runner = runner_factory(engine, reasoning_effort=reasoning_effort)
    except EngineNotImplemented:
        return _error(t("experiments.engineNotImplemented", engine=engine), 400)
    except ValueError as exc:
        return _error(str(exc), 500)

    population_store = PopulationStore()
    store = ExperimentStore()
    experiment_id = new_experiment_id()
    try:
        with population_store.lock(population_id):
            population = population_store.get(population_id)
            if population.get("enrichment", {}).get("status") == "running":
                return _error(t("population.enrichmentRunning", id=population_id), 409)
            personas = None
            if persona_source == PERSONA_SOURCE_ENRICHED:
                if population.get("enrichment", {}).get("status") != "completed":
                    return _error(t("population.enrichmentRequired"), 400)
                personas = population_store.personas(population_id)
            try:
                calibration = resolve_calibration(
                    calibration_request, _assumed_calibration(population_store, population_id),
                    [v.variant_id for v in variants])
            except CalibrationError as exc:
                return _calibration_error(exc)
            seed = population["default_seed"] if seed is None else seed
            snapshot = build_snapshot(population_store.records(population_id), personas)
            spec = ExperimentSpec(
                experiment_id=experiment_id,
                population_id=population_id,
                persona_source=persona_source,
                engine=engine,
                variants=variants,
                replicates=replicates,
                seed=seed,
                prompt_version=runner.prompt_version,
                coder_prompt_version=CODER_PROMPT_VERSION if analyze else "",
                model=runner.model_name,
                reasoning_effort=runner.reasoning_effort,
                coder_reasoning_effort=CODER_REASONING_EFFORT if analyze else "",
                response_calibration=calibration,
            )
            meta = store.create(spec, snapshot, {
                "population_name": population.get("name"),
                "personas_sha256": (population.get("enrichment") or {}).get("personas_sha256")
                if persona_source == PERSONA_SOURCE_ENRICHED else None,
                "model": runner.model_name,
                "git_commit": _git_commit(),
                "response_calibration": calibration,
                "calibration_status": calibration["status"],
                "analyze_requested": analyze,
            })
            population_store.register_experiment(population_id, {
                "experiment_id": experiment_id,
                "engine": engine,
                "persona_source": persona_source,
                "snapshot_sha256": meta["snapshot_sha256"],
                "created_at": meta["created_at"],
            })
    except PopulationNotFound:
        return _error(t("population.notFound", id=population_id), 404)

    if calibration["status"] == STATUS_NOT_CALIBRATED:
        logger.warning(f"Experiment {experiment_id}: {t('experiments.notCalibrated')}")
    _claim(experiment_id)
    task_manager = TaskManager()
    task_id = task_manager.create_task(task_type="experiment_run",
                                       metadata={"experiment_id": experiment_id})
    store.update_meta(experiment_id, task_id=task_id)
    current_locale = get_locale()

    def run_experiment():
        set_locale(current_locale)
        try:
            store.update_meta(experiment_id, status="running", started_at=datetime.now().isoformat())
            task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0,
                                     message=t("experiments.runStarted"))

            def on_progress(stage, current, total):
                _stage_progress(task_manager, task_id, stage, current, total,
                                "experiments.propensityProgress" if stage == "propensity"
                                else "experiments.writingProgress")

            def calibrator(propensities):
                fit = fit_from_propensities(propensities, calibration)
                store.update_meta(experiment_id, calibration_fit=fit,
                                  calibration_status=fit["status"])
                return fit["intercept"]

            done = {row_key(r) for r in store.results(experiment_id)}
            runner.run(spec, store.snapshot(experiment_id),
                       on_result=lambda row: store.append_result(experiment_id, row),
                       progress_callback=on_progress, completed_keys=done, parallel=parallel,
                       existing_propensities=store.propensities(experiment_id),
                       on_propensity=lambda row: store.append_propensity(experiment_id, row),
                       calibrator=calibrator)

            if analyze:
                task_manager.update_task(task_id, message=t("experiments.analyzeStarted"))
                analysis = _code_and_evaluate(
                    store, experiment_id, coder_factory(), parallel, False,
                    lambda current, total: _stage_progress(
                        task_manager, task_id, "coding", current, total,
                        "experiments.analyzeProgress"))
                store.update_meta(experiment_id, analysis=analysis)
            else:
                store.replace_results(experiment_id,
                                      evaluate_rows(store.results(experiment_id),
                                                    store.snapshot(experiment_id)))
            summary = _refresh_summary(store, experiment_id)
            counts = {v["variant_id"]: v["decisions"] for v in summary["variants"]}
            store.update_meta(experiment_id, status="completed",
                              finished_at=datetime.now().isoformat(),
                              result_rows=summary["rows"], decision_counts=counts)
            task_manager.complete_task(task_id, result={
                "experiment_id": experiment_id, "rows": summary["rows"], "decisions": counts,
                "calibration_status": summary["calibration"]["status"]})
            logger.info(f"Experiment {experiment_id} completed: {counts}")
        except Exception as exc:
            logger.error(f"Experiment failed: {experiment_id}: {exc}")
            try:
                store.update_meta(experiment_id, status="failed", error=str(exc),
                                  finished_at=datetime.now().isoformat())
            finally:
                task_manager.fail_task(task_id, str(exc))
        finally:
            _release(experiment_id)

    threading.Thread(target=run_experiment, daemon=True).start()
    return jsonify({"success": True, "data": {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "population_id": population_id,
        "guest_count": meta["guest_count"],
        "expected_rows": meta["expected_rows"],
        "snapshot_sha256": meta["snapshot_sha256"],
        "response_calibration": calibration,
        "calibration_status": calibration["status"],
        "warnings": [t("experiments.notCalibrated")]
        if calibration["status"] == STATUS_NOT_CALIBRATED else [],
        "message": t("experiments.runStarted"),
    }})


@experiments_bp.route("/list", methods=["GET"])
def list_experiments():
    experiments = ExperimentStore().list()
    return jsonify({"success": True, "data": experiments, "count": len(experiments)})


@experiments_bp.route("/task/<task_id>", methods=["GET"])
def get_experiment_task(task_id: str):
    task = TaskManager().get_task(task_id)
    if not task:
        return _error(t("api.taskNotFound", id=task_id), 404)
    return jsonify({"success": True, "data": task.to_dict()})


@experiments_bp.route("/<experiment_id>", methods=["GET"])
def get_experiment(experiment_id: str):
    store = ExperimentStore()
    try:
        meta = store.get(experiment_id)
        meta["spec"] = store.spec(experiment_id).to_dict()
        meta["result_rows_written"] = len(store.results(experiment_id))
    except ExperimentNotFound:
        return _not_found(experiment_id)
    meta["running"] = _is_running(experiment_id)
    return jsonify({"success": True, "data": meta})


@experiments_bp.route("/<experiment_id>/results", methods=["GET"])
def get_experiment_results(experiment_id: str):
    args = request.args
    try:
        offset = _bounded_int(args, "offset", 0, 0, 10**9)
        limit = _bounded_int(args, "limit", 100, 1, 1000)
    except ValueError as exc:
        return _error(str(exc), 400)
    try:
        rows = ExperimentStore().results(experiment_id)
    except ExperimentNotFound:
        return _not_found(experiment_id)
    filters = {"variant_id": args.get("variant"), "cohort": args.get("cohort"),
               "decision": args.get("decision")}
    for field, value in filters.items():
        if value:
            rows = [r for r in rows if str(r.get(field)) == value]
    flag = args.get("flag")
    if flag:
        if flag not in RESULT_FLAGS:
            return _error(t("experiments.invalidFlag", values=", ".join(RESULT_FLAGS)), 400)
        rows = [r for r in rows if (r.get("evaluation") or {}).get(flag) is True]
    rows.sort(key=lambda r: (r["variant_id"], r["replicate"], r["guest_id"]))
    return jsonify({"success": True, "data": {
        "total": len(rows), "offset": offset, "limit": limit,
        "results": rows[offset:offset + limit],
    }})


@experiments_bp.route("/<experiment_id>/summary", methods=["GET"])
def get_experiment_summary(experiment_id: str):
    store = ExperimentStore()
    try:
        summary = store.summary(experiment_id)
        if summary is None or _is_running(experiment_id):
            summary = _live_summary(store, experiment_id)
    except ExperimentNotFound:
        return _not_found(experiment_id)
    return jsonify({"success": True, "data": summary})


@experiments_bp.route("/<experiment_id>/analyze", methods=["POST"])
def analyze_experiment(experiment_id: str):
    """Body: {"parallel": 5, "force": false}. Codes replies blind, then evaluates."""
    data = request.get_json(silent=True) or {}
    try:
        parallel = _bounded_int(data, "parallel", 5, 1, MAX_PARALLEL)
        force = _bool_field(data, "force")
    except ValueError as exc:
        return _error(str(exc), 400)
    store = ExperimentStore()
    try:
        meta = store.get(experiment_id)
    except ExperimentNotFound:
        return _not_found(experiment_id)
    if meta.get("status") != "completed":
        return _error(t("experiments.notCompleted", id=experiment_id), 409)
    if not _claim(experiment_id):
        return _error(t("experiments.running", id=experiment_id), 409)

    task_manager = TaskManager()
    task_id = task_manager.create_task(task_type="experiment_analyze",
                                       metadata={"experiment_id": experiment_id})
    store.update_meta(experiment_id, analysis={"status": "running", "task_id": task_id,
                                               "started_at": datetime.now().isoformat()})
    current_locale = get_locale()

    def run_analysis():
        set_locale(current_locale)
        try:
            task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0,
                                     message=t("experiments.analyzeStarted"))

            def on_progress(current, total):
                task_manager.update_task(
                    task_id,
                    progress=int(100 * current / total) if total else 100,
                    message=t("experiments.analyzeProgress", current=current, total=total),
                    progress_detail={"stage": "coding", "current": current, "total": total},
                )

            analysis = _code_and_evaluate(store, experiment_id, coder_factory(), parallel, force,
                                          on_progress)
            store.update_meta(experiment_id, analysis=analysis,
                              coder_prompt_version=CODER_PROMPT_VERSION,
                              coder_reasoning_effort=CODER_REASONING_EFFORT)
            _refresh_summary(store, experiment_id)
            task_manager.complete_task(task_id, result={"experiment_id": experiment_id, **analysis})
        except Exception as exc:
            logger.error(f"Experiment analysis failed: {experiment_id}: {exc}")
            try:
                store.update_meta(experiment_id, analysis={"status": "failed", "error": str(exc)})
            finally:
                task_manager.fail_task(task_id, str(exc))
        finally:
            _release(experiment_id)

    threading.Thread(target=run_analysis, daemon=True).start()
    return jsonify({"success": True, "data": {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "message": t("experiments.analyzeStarted"),
    }})
