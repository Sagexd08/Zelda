from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    """Persist experiment metadata and epoch metrics to disk."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.output_path = Path(output_dir) / f"{experiment_name}_{timestamp}.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._data: Dict[str, Any] = {
            "experiment": experiment_name,
            "created_at": datetime.utcnow().isoformat(),
            "config": config or {},
            "epochs": [],
            "summary": {},
        }

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        entry = {"epoch": epoch, **metrics, "logged_at": datetime.utcnow().isoformat()}
        self._data["epochs"].append(entry)
        self._write()

    def finalize(self, summary: Dict[str, Any]) -> None:
        self._data["summary"] = {**summary, "completed_at": datetime.utcnow().isoformat()}
        self._write()

    def _write(self) -> None:
        tmp_path = self.output_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._data, indent=2, default=_serialize), encoding="utf-8")
        tmp_path.replace(self.output_path)


def _serialize(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported type for serialization: {type(value)!r}")
