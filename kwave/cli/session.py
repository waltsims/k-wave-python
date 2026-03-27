"""Single file-backed session for the k-Wave CLI."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from kwave.cli.schema import SessionError

DEFAULT_SESSION_DIR = Path.home() / ".kwave"
SESSION_FILE = "session.json"


def _default_state() -> dict:
    return {
        "grid": None,
        "medium": None,
        "source": None,
        "sensor": None,
        "modality": None,
        "resolution_tier": None,
        "output_intent": None,
        "probe": None,
        "sim_options": {},
        "result_path": None,
    }


class Session:
    """Single file-backed simulation session.

    Stores parameters as JSON-serializable dicts. Array data is stored
    as .npy files in the session directory. Materializer methods construct
    kWave objects on demand.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_SESSION_DIR
        self.session_file = self.base_dir / SESSION_FILE
        self.data_dir = self.base_dir / "data"
        self._state: Optional[dict] = None
        self._id: Optional[str] = None
        self._created_at: Optional[str] = None

    @property
    def state(self) -> dict:
        if self._state is None:
            raise SessionError("No active session. Run 'kwave session init' first.")
        return self._state

    @property
    def id(self) -> str:
        if self._id is None:
            raise SessionError("No active session. Run 'kwave session init' first.")
        return self._id

    def init(self) -> dict:
        """Create a new session, overwriting any existing one."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._id = uuid.uuid4().hex[:12]
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._state = _default_state()
        self._save()
        return {"session_id": self._id, "created_at": self._created_at}

    def load(self) -> dict:
        """Load the current session from disk."""
        if not self.session_file.exists():
            raise SessionError("No active session. Run 'kwave session init' first.")
        raw = json.loads(self.session_file.read_text())
        self._id = raw["id"]
        self._created_at = raw["created_at"]
        self._state = raw["state"]
        return self.status()

    def reset(self) -> dict:
        """Clear session state, keep the session ID."""
        self.load()
        self._state = _default_state()
        # Clean up data files
        if self.data_dir.exists():
            for f in self.data_dir.iterdir():
                f.unlink()
        self._save()
        return {"session_id": self._id, "reset": True}

    def status(self) -> dict:
        """Return full current state."""
        return {
            "session_id": self.id,
            "created_at": self._created_at,
            "state": self.state,
            "completeness": self._completeness(),
        }

    def update(self, key: str, value) -> None:
        """Update a state field and persist."""
        self.state[key] = value
        self._save()

    def save_array(self, name: str, arr: np.ndarray) -> str:
        """Save an array to the session data dir, return the path."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / f"{name}.npy"
        np.save(path, arr)
        return str(path)

    def load_array(self, path: str) -> np.ndarray:
        """Load an array from a saved path."""
        return np.load(path)

    # --- Materializers: session state -> kWave objects ---

    def make_grid(self):
        """Construct a kWaveGrid from session state."""
        from kwave.kgrid import kWaveGrid

        g = self.state["grid"]
        if g is None:
            raise SessionError("Grid not defined. Run 'kwave phantom generate' first.")
        grid_size = tuple(g["N"])
        grid_spacing = tuple(g["spacing"])
        kgrid = kWaveGrid(grid_size, grid_spacing)
        if g.get("sound_speed_for_time") is not None:
            kgrid.makeTime(g["sound_speed_for_time"])
        return kgrid

    def make_medium(self):
        """Construct a kWaveMedium from session state."""
        from kwave.kmedium import kWaveMedium

        m = self.state["medium"]
        if m is None:
            raise SessionError("Medium not defined. Run 'kwave phantom generate' first.")
        kwargs = {}
        for field in ("sound_speed", "density", "alpha_coeff", "alpha_power", "BonA"):
            if field in m and m[field] is not None:
                kwargs[field] = m[field]
        return kWaveMedium(**kwargs)

    def make_source(self):
        """Construct a kSource from session state."""
        from kwave.ksource import kSource

        s = self.state["source"]
        if s is None:
            raise SessionError("Source not defined. It was auto-set by phantom generate.")
        source = kSource()
        if s.get("p0_path"):
            source.p0 = np.load(s["p0_path"])
        return source

    def make_sensor(self):
        """Construct a kSensor from session state."""
        from kwave.ksensor import kSensor

        sen = self.state["sensor"]
        if sen is None:
            raise SessionError("Sensor not defined. Run 'kwave sensor define' first.")
        record = sen.get("record", ["p", "p_final"])
        if sen.get("mask_type") == "full-grid":
            g = self.state["grid"]
            mask = np.ones(tuple(g["N"]), dtype=bool)
        elif sen.get("mask_path"):
            mask = np.load(sen["mask_path"])
        else:
            raise SessionError("Invalid sensor mask configuration.")
        sensor = kSensor(mask=mask, record=record)
        return sensor

    # --- Private ---

    def _completeness(self) -> dict:
        """Which steps have been completed."""
        s = self.state
        return {
            "grid": s["grid"] is not None,
            "medium": s["medium"] is not None,
            "source": s["source"] is not None,
            "sensor": s["sensor"] is not None,
        }

    def _save(self):
        raw = {
            "id": self._id,
            "created_at": self._created_at,
            "state": self._state,
        }
        self.session_file.write_text(json.dumps(raw, indent=2, default=str))
