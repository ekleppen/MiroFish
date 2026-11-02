"""
Experiment store

Each experiment lives under uploads/experiments/<experiment_id>/:
spec.json, snapshot.jsonl, propensities.jsonl (stage 1), results.jsonl,
summary.json and meta.json.
"""

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config
from .schema import EXPERIMENT_ID_PATTERN, ExperimentSpec
from .snapshot import snapshot_lines

META_FILENAME = "meta.json"
SPEC_FILENAME = "spec.json"
SNAPSHOT_FILENAME = "snapshot.jsonl"
RESULTS_FILENAME = "results.jsonl"
PROPENSITIES_FILENAME = "propensities.jsonl"
SUMMARY_FILENAME = "summary.json"


class ExperimentNotFound(LookupError):
    pass


class ExperimentStore:
    _locks: Dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(Config.UPLOAD_FOLDER, "experiments")
        os.makedirs(self.base_dir, exist_ok=True)

    def lock(self, experiment_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(experiment_id, threading.RLock())

    def _dir(self, experiment_id: str) -> str:
        if not EXPERIMENT_ID_PATTERN.match(experiment_id or ""):
            raise ExperimentNotFound(experiment_id)
        path = os.path.join(self.base_dir, experiment_id)
        if not os.path.isdir(path):
            raise ExperimentNotFound(experiment_id)
        return path

    def create(
        self,
        spec: ExperimentSpec,
        snapshot: List[Dict[str, Any]],
        meta_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        path = os.path.join(self.base_dir, spec.experiment_id)
        os.makedirs(path)
        _write_json(os.path.join(path, SPEC_FILENAME), spec.to_dict())
        snapshot_text = snapshot_lines(snapshot)
        _write_text(os.path.join(path, SNAPSHOT_FILENAME), snapshot_text)
        _write_text(os.path.join(path, RESULTS_FILENAME), "")
        _write_text(os.path.join(path, PROPENSITIES_FILENAME), "")
        meta = {
            "experiment_id": spec.experiment_id,
            "population_id": spec.population_id,
            "engine": spec.engine,
            "persona_source": spec.persona_source,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "guest_count": len(snapshot),
            "variant_ids": [v.variant_id for v in spec.variants],
            "replicates": spec.replicates,
            "expected_rows": len(snapshot) * len(spec.variants) * spec.replicates,
            "snapshot_sha256": hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest(),
            "prompt_version": spec.prompt_version,
            "coder_prompt_version": spec.coder_prompt_version,
            "reasoning_effort": spec.reasoning_effort,
            "coder_reasoning_effort": spec.coder_reasoning_effort,
            "analysis": {"status": "none"},
            **meta_fields,
        }
        _write_json(os.path.join(path, META_FILENAME), meta)
        return meta

    def get(self, experiment_id: str) -> Dict[str, Any]:
        return _read_json(os.path.join(self._dir(experiment_id), META_FILENAME))

    def list(self) -> List[Dict[str, Any]]:
        metas = []
        for entry in os.listdir(self.base_dir):
            if EXPERIMENT_ID_PATTERN.match(entry):
                try:
                    metas.append(self.get(entry))
                except (ExperimentNotFound, OSError, ValueError):
                    continue
        return sorted(metas, key=lambda m: m.get("created_at", ""), reverse=True)

    def spec(self, experiment_id: str) -> ExperimentSpec:
        return ExperimentSpec.from_dict(
            _read_json(os.path.join(self._dir(experiment_id), SPEC_FILENAME)))

    def snapshot(self, experiment_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(os.path.join(self._dir(experiment_id), SNAPSHOT_FILENAME))

    def snapshot_sha256(self, experiment_id: str) -> str:
        with open(os.path.join(self._dir(experiment_id), SNAPSHOT_FILENAME), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def results(self, experiment_id: str) -> List[Dict[str, Any]]:
        return _read_jsonl(os.path.join(self._dir(experiment_id), RESULTS_FILENAME))

    def append_result(self, experiment_id: str, row: Dict[str, Any]) -> None:
        with self.lock(experiment_id):
            path = os.path.join(self._dir(experiment_id), RESULTS_FILENAME)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def propensities(self, experiment_id: str) -> List[Dict[str, Any]]:
        path = os.path.join(self._dir(experiment_id), PROPENSITIES_FILENAME)
        return _read_jsonl(path) if os.path.exists(path) else []

    def append_propensity(self, experiment_id: str, row: Dict[str, Any]) -> None:
        with self.lock(experiment_id):
            path = os.path.join(self._dir(experiment_id), PROPENSITIES_FILENAME)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def replace_results(self, experiment_id: str, rows: List[Dict[str, Any]]) -> None:
        with self.lock(experiment_id):
            _write_text(os.path.join(self._dir(experiment_id), RESULTS_FILENAME),
                        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    def summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self._dir(experiment_id), SUMMARY_FILENAME)
        return _read_json(path) if os.path.exists(path) else None

    def save_summary(self, experiment_id: str, summary: Dict[str, Any]) -> None:
        _write_json(os.path.join(self._dir(experiment_id), SUMMARY_FILENAME), summary)

    def update_meta(self, experiment_id: str, **fields: Any) -> Dict[str, Any]:
        with self.lock(experiment_id):
            path = os.path.join(self._dir(experiment_id), META_FILENAME)
            meta = _read_json(path)
            meta.update(fields)
            _write_json(path, meta)
            return meta


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


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
