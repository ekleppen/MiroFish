"""
Population API routes

Upload synthetic guest populations, optionally enrich their personas with the
LLM, and create reproducible OASIS simulations from them. Created simulations
run through the existing /api/simulation/start endpoint.
"""

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from flask import jsonify, request

from . import population_bp
from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..population.mirofish_adapter import (
    PERSONA_SOURCE_ENRICHED,
    PERSONA_SOURCE_TEMPLATE,
    GuestRecordError,
    TimeSettings,
    build_guest_simulation,
    new_simulation_id,
)
from ..population.persona_enricher import (
    MAX_PARALLEL,
    PROMPT_VERSION,
    REASONING_EFFORT,
    STATUS_ENRICHED,
    STATUS_FALLBACK,
    GuestPersonaEnricher,
)
from ..population.store import PopulationNotFound, PopulationStore
from ..population.traits import audit, validate_weights
from ..utils.locale import get_locale, set_locale, t
from ..utils.logger import get_logger

logger = get_logger("mirofish.api.population")

# Populations whose enrichment thread is alive in this process. A meta status of
# "running" without an entry here means the server restarted mid-enrichment.
_running_enrichments = set()
_running_guard = threading.Lock()

# Tests replace this to inject a fake LLM client.
enricher_factory = GuestPersonaEnricher


def _store() -> PopulationStore:
    return PopulationStore()


def _error(message: str, status: int):
    return jsonify({"success": False, "error": message}), status


def _not_found(population_id: str):
    return _error(t("population.notFound", id=population_id), 404)


def _int_field(data: Dict[str, Any], field: str, default: Optional[int]) -> Optional[int]:
    value = data.get(field, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(t("population.invalidNumber", field=field))
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(t("population.invalidNumber", field=field))


def _enrichment_running(population_id: str) -> bool:
    with _running_guard:
        return population_id in _running_enrichments


@population_bp.route("/upload", methods=["POST"])
def upload_population():
    """Multipart upload: guests (.jsonl, required), assumptions (.json), name."""
    guests_file = request.files.get("guests")
    if guests_file is None or not guests_file.filename:
        return _error(t("population.requireGuestsFile"), 400)
    if not guests_file.filename.lower().endswith(".jsonl"):
        return _error(t("population.invalidFileType"), 400)

    assumptions_file = request.files.get("assumptions")
    assumptions_bytes = assumptions_file.read() if assumptions_file and assumptions_file.filename else None

    try:
        meta = _store().create(
            guests_file.read(),
            guests_file.filename,
            name=request.form.get("name"),
            assumptions_bytes=assumptions_bytes,
        )
    except GuestRecordError as exc:
        return _error(str(exc), 400)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return _error(t("population.invalidAssumptions"), 400)

    logger.info(f"Population uploaded: {meta['population_id']} ({meta['guest_count']} guests)")
    return jsonify({"success": True, "data": {
        "population_id": meta["population_id"],
        "name": meta["name"],
        "guest_count": meta["guest_count"],
        "cohort_counts": meta["cohort_counts"],
        "default_seed": meta["default_seed"],
    }})


@population_bp.route("/list", methods=["GET"])
def list_populations():
    populations = _store().list()
    return jsonify({"success": True, "data": populations, "count": len(populations)})


@population_bp.route("/<population_id>", methods=["GET"])
def get_population(population_id: str):
    try:
        meta = _store().get(population_id)
    except PopulationNotFound:
        return _not_found(population_id)
    meta["enrichment_running"] = _enrichment_running(population_id)
    return jsonify({"success": True, "data": meta})


@population_bp.route("/<population_id>/guests", methods=["GET"])
def get_population_guests(population_id: str):
    try:
        offset = _int_field(request.args, "offset", 0)
        limit = _int_field(request.args, "limit", 50)
    except ValueError as exc:
        return _error(str(exc), 400)
    try:
        view = _store().guests_view(
            population_id,
            offset=offset,
            limit=limit,
            cohort=request.args.get("cohort") or None,
            query=request.args.get("q") or None,
        )
    except PopulationNotFound:
        return _not_found(population_id)
    return jsonify({"success": True, "data": view})


@population_bp.route("/<population_id>/audit", methods=["GET"])
def get_population_audit(population_id: str):
    store = _store()
    try:
        report = audit(store.records(population_id), store.personas(population_id))
        meta = store.get(population_id)
    except PopulationNotFound:
        return _not_found(population_id)
    report["enrichment"] = meta.get("enrichment", {})
    return jsonify({"success": True, "data": report})


@population_bp.route("/<population_id>/enrich", methods=["POST"])
def enrich_population(population_id: str):
    """Body: {"parallel": 5, "force": false, "trait_weights": {...}, "seed": null}."""
    data = request.get_json(silent=True) or {}
    store = _store()
    try:
        parallel = max(1, min(_int_field(data, "parallel", 5), MAX_PARALLEL))
        seed = _int_field(data, "seed", None)
        weights = validate_weights(data.get("trait_weights"))
    except ValueError as exc:
        return _error(str(exc), 400)
    force = data.get("force", False)
    if not isinstance(force, bool):
        return _error(t("population.invalidBoolean", field="force"), 400)

    try:
        with store.lock(population_id):
            meta = store.get(population_id)
            if meta["frozen"]:
                return _error(t("population.frozen", id=population_id), 409)
            with _running_guard:
                if population_id in _running_enrichments:
                    return _error(t("population.enrichmentRunning", id=population_id), 409)
                _running_enrichments.add(population_id)
            seed = meta["default_seed"] if seed is None else seed
            store.update_meta(population_id, enrichment={
                "status": "running",
                "prompt_version": PROMPT_VERSION,
                "reasoning_effort": REASONING_EFFORT,
                "seed": seed,
                "trait_weights": weights,
                "started_at": datetime.now().isoformat(),
            })
    except PopulationNotFound:
        return _not_found(population_id)

    task_manager = TaskManager()
    task_id = task_manager.create_task(
        task_type="population_enrich",
        metadata={"population_id": population_id},
    )
    current_locale = get_locale()

    def run_enrichment():
        set_locale(current_locale)
        try:
            enricher = enricher_factory()
            records = store.records(population_id)
            personas = store.personas(population_id)
            task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0,
                                     message=t("population.enrichStarted"))

            def on_result(guest_id, entry):
                personas[guest_id] = entry
                store.save_personas(population_id, personas)

            def on_progress(current, total, counts):
                task_manager.update_task(
                    task_id,
                    progress=int(100 * current / total) if total else 100,
                    message=t("population.enrichProgress", current=current, total=total),
                    progress_detail={"current": current, "total": total, **counts},
                )

            personas = enricher.enrich_all(
                records, seed=seed, weights=weights, existing=personas, force=force,
                parallel=parallel, progress_callback=on_progress, on_result=on_result,
            )
            store.save_personas(population_id, personas)
            counts = {
                STATUS_ENRICHED: sum(p.get("status") == STATUS_ENRICHED for p in personas.values()),
                STATUS_FALLBACK: sum(p.get("status") == STATUS_FALLBACK for p in personas.values()),
            }
            enrichment = store.get(population_id).get("enrichment", {})
            enrichment.update({
                "status": "completed",
                "model": enricher.model_name,
                "finished_at": datetime.now().isoformat(),
                "counts": counts,
                "personas_sha256": _sha256_file(os.path.join(
                    store.base_dir, population_id, "personas.json")),
            })
            store.update_meta(population_id, enrichment=enrichment)
            task_manager.complete_task(task_id, result={"population_id": population_id, **counts})
            logger.info(f"Population {population_id} enriched: {counts}")
        except Exception as exc:
            logger.error(f"Population enrichment failed: {population_id}: {exc}")
            try:
                enrichment = store.get(population_id).get("enrichment", {})
                enrichment.update({"status": "failed", "error": str(exc),
                                   "finished_at": datetime.now().isoformat()})
                store.update_meta(population_id, enrichment=enrichment)
            finally:
                task_manager.fail_task(task_id, str(exc))
        finally:
            with _running_guard:
                _running_enrichments.discard(population_id)

    threading.Thread(target=run_enrichment, daemon=True).start()
    return jsonify({"success": True, "data": {
        "population_id": population_id,
        "task_id": task_id,
        "message": t("population.enrichStarted"),
    }})


@population_bp.route("/task/<task_id>", methods=["GET"])
def get_population_task(task_id: str):
    task = TaskManager().get_task(task_id)
    if not task:
        return _error(t("api.taskNotFound", id=task_id), 404)
    return jsonify({"success": True, "data": task.to_dict()})


@population_bp.route("/<population_id>/simulations", methods=["POST"])
def create_population_simulation(population_id: str):
    """Create a ready-to-run simulation; the population's personas become frozen."""
    data = request.get_json(silent=True) or {}
    persona_source = data.get("persona_source", PERSONA_SOURCE_TEMPLATE)
    if persona_source not in (PERSONA_SOURCE_ENRICHED, PERSONA_SOURCE_TEMPLATE):
        return _error(t("population.invalidPersonaSource"), 400)
    try:
        settings = TimeSettings(
            total_hours=_int_field(data, "hours", 24),
            minutes_per_round=_int_field(data, "minutes_per_round", 60),
            agents_per_hour_min=_int_field(data, "agents_per_hour_min", 10),
            agents_per_hour_max=_int_field(data, "agents_per_hour_max", 25),
        )
        settings.validate()
        random_seed = _int_field(data, "random_seed", None)
    except ValueError as exc:
        return _error(str(exc), 400)

    store = _store()
    try:
        with store.lock(population_id):
            meta = store.get(population_id)
            if _enrichment_running(population_id):
                return _error(t("population.enrichmentRunning", id=population_id), 409)
            personas = None
            if persona_source == PERSONA_SOURCE_ENRICHED:
                if meta.get("enrichment", {}).get("status") != "completed":
                    return _error(t("population.enrichmentRequired"), 400)
                personas = store.personas(population_id)
            random_seed = meta["default_seed"] if random_seed is None else random_seed

            simulation_id = new_simulation_id()
            summary = build_guest_simulation(
                store.records(population_id),
                os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id),
                simulation_id,
                settings=settings,
                assumptions_path=store.assumptions_path(population_id),
                source_path=meta.get("source_filename"),
                personas=personas,
                population_id=population_id,
                random_seed=random_seed,
            )
            store.register_simulation(population_id, {
                "simulation_id": simulation_id,
                "persona_source": persona_source,
                "random_seed": random_seed,
                "created_at": datetime.now().isoformat(),
            })
    except PopulationNotFound:
        return _not_found(population_id)

    logger.info(f"Simulation {simulation_id} created from population {population_id} "
                f"({persona_source}, seed={random_seed})")
    return jsonify({"success": True, "data": {
        "simulation_id": simulation_id,
        "population_id": population_id,
        "agents": summary["agents"],
        "enriched_agents": summary["enriched_agents"],
        "distinct_prompts": summary["distinct_prompts"],
        "persona_source": persona_source,
        "random_seed": random_seed,
    }})


def _sha256_file(path: str) -> Optional[str]:
    import hashlib
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
