"""
Population store

A population is an uploaded set of synthetic guests saved once under
uploads/populations/<population_id>/. Its guest records never change after
upload, and its personas are frozen as soon as a simulation or experiment is
created from it, so every run on one population uses identical agents and
identical stay experiences.
"""

import json
import os
import random
import re
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config
from .mirofish_adapter import (
    ASSUMPTIONS_FILENAME,
    compose_agent_text,
    guest_cohort,
    guest_context,
    parse_guest_records,
)

POPULATION_ID_PATTERN = re.compile(r"^pp_[0-9a-f]{12}$")
META_FILENAME = "meta.json"
GUESTS_FILENAME = "guests.jsonl"
PERSONAS_FILENAME = "personas.json"


class PopulationNotFound(LookupError):
    pass


class PopulationFrozen(RuntimeError):
    """Raised when enrichment is attempted after a simulation was created."""


class PopulationStore:
    _locks: Dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(Config.UPLOAD_FOLDER, "populations")
        os.makedirs(self.base_dir, exist_ok=True)

    def lock(self, population_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(population_id, threading.RLock())

    def _dir(self, population_id: str) -> str:
        if not POPULATION_ID_PATTERN.match(population_id or ""):
            raise PopulationNotFound(population_id)
        path = os.path.join(self.base_dir, population_id)
        if not os.path.isdir(path):
            raise PopulationNotFound(population_id)
        return path

    def create(
        self,
        guests_bytes: bytes,
        filename: str,
        name: Optional[str] = None,
        assumptions_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Validate and save an uploaded synthetic_guests.jsonl."""
        text = guests_bytes.decode("utf-8-sig")
        records = parse_guest_records(text.splitlines(), source=filename)

        assumptions = None
        if assumptions_bytes:
            assumptions = json.loads(assumptions_bytes.decode("utf-8-sig"))
            if not isinstance(assumptions, dict):
                raise ValueError("Assumptions file must be a JSON object")

        seed = assumptions.get("random_seed") if assumptions else None
        default_seed = seed if isinstance(seed, int) else random.SystemRandom().randint(1, 2**31 - 1)

        population_id = f"pp_{uuid.uuid4().hex[:12]}"
        path = os.path.join(self.base_dir, population_id)
        os.makedirs(path)
        _write_text(os.path.join(path, GUESTS_FILENAME),
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        if assumptions is not None:
            _write_json(os.path.join(path, ASSUMPTIONS_FILENAME), assumptions)

        cohorts: Dict[str, int] = {}
        for record in records:
            cohort = guest_cohort(record)
            cohorts[cohort] = cohorts.get(cohort, 0) + 1

        meta = {
            "population_id": population_id,
            "name": (name or "").strip() or os.path.splitext(os.path.basename(filename))[0],
            "created_at": datetime.now().isoformat(),
            "source_filename": os.path.basename(filename),
            "guest_count": len(records),
            "cohort_counts": cohorts,
            "default_seed": default_seed,
            "has_assumptions": assumptions is not None,
            "enrichment": {"status": "none"},
            "simulations": [],
        }
        _write_json(os.path.join(path, META_FILENAME), meta)
        return meta

    def get(self, population_id: str) -> Dict[str, Any]:
        meta = _read_json(os.path.join(self._dir(population_id), META_FILENAME))
        meta["frozen"] = self.is_frozen(meta)
        return meta

    def list(self) -> List[Dict[str, Any]]:
        metas = []
        for entry in os.listdir(self.base_dir):
            if POPULATION_ID_PATTERN.match(entry):
                try:
                    metas.append(self.get(entry))
                except (PopulationNotFound, OSError, ValueError):
                    continue
        return sorted(metas, key=lambda m: m.get("created_at", ""), reverse=True)

    def records(self, population_id: str) -> List[Dict[str, Any]]:
        with open(os.path.join(self._dir(population_id), GUESTS_FILENAME), encoding="utf-8") as f:
            return parse_guest_records(f)

    def personas(self, population_id: str) -> Dict[str, Dict[str, Any]]:
        path = os.path.join(self._dir(population_id), PERSONAS_FILENAME)
        return _read_json(path) if os.path.exists(path) else {}

    def assumptions_path(self, population_id: str) -> Optional[str]:
        path = os.path.join(self._dir(population_id), ASSUMPTIONS_FILENAME)
        return path if os.path.exists(path) else None

    @staticmethod
    def is_frozen(meta: Dict[str, Any]) -> bool:
        return bool(meta.get("simulations") or meta.get("experiments"))

    def update_meta(self, population_id: str, **fields: Any) -> Dict[str, Any]:
        with self.lock(population_id):
            path = os.path.join(self._dir(population_id), META_FILENAME)
            meta = _read_json(path)
            meta.update(fields)
            _write_json(path, meta)
            return meta

    def save_personas(self, population_id: str, personas: Dict[str, Dict[str, Any]]) -> None:
        with self.lock(population_id):
            _write_json(os.path.join(self._dir(population_id), PERSONAS_FILENAME), personas)

    def register_simulation(self, population_id: str, entry: Dict[str, Any]) -> None:
        with self.lock(population_id):
            path = os.path.join(self._dir(population_id), META_FILENAME)
            meta = _read_json(path)
            meta.setdefault("simulations", []).append(entry)
            _write_json(path, meta)

    def register_experiment(self, population_id: str, entry: Dict[str, Any]) -> None:
        with self.lock(population_id):
            path = os.path.join(self._dir(population_id), META_FILENAME)
            meta = _read_json(path)
            meta.setdefault("experiments", []).append(entry)
            _write_json(path, meta)

    def guests_view(
        self,
        population_id: str,
        offset: int = 0,
        limit: int = 50,
        cohort: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Records joined with their personas, for browsing and comparison."""
        personas = self.personas(population_id)
        rows = []
        needle = (query or "").strip().lower()
        for record in self.records(population_id):
            if cohort and guest_cohort(record) != cohort:
                continue
            guest_id = str(record["guest_id"])
            persona = personas.get(guest_id)
            agent_text, source = compose_agent_text(record, persona)
            if needle and needle not in guest_id.lower() and needle not in agent_text.lower():
                continue
            rows.append({
                "guest_id": guest_id,
                "cohort": guest_cohort(record),
                "attributes": record.get("attributes") or {},
                "experience": guest_context(record),
                "template_persona": record.get("persona_text") or "",
                "traits": (persona or {}).get("traits"),
                "traits_paragraph": (persona or {}).get("traits_paragraph"),
                "persona_status": (persona or {}).get("status", "not_enriched"),
                "check_failures": (persona or {}).get("check_failures", []),
                "agent_text": agent_text,
                "agent_text_source": source,
            })
        offset = max(0, offset)
        limit = max(1, min(limit, 500))
        return {"total": len(rows), "offset": offset, "limit": limit,
                "guests": rows[offset:offset + limit]}


def _write_json(path: str, data: Any) -> None:
    _write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
