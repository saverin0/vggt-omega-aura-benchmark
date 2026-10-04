"""Persistent state for an ephemeral runtime.

The Colab disk is wiped when a session ends, so anything that must survive
lives under a persist root. Two modes:

- "drive":   a folder on mounted Google Drive. Survives the session.
- "runtime": the project's results/ folder on the runtime disk. Does NOT
             survive the session by itself: download it before disconnecting.

The manifest records which (scene_id, stage) pairs are finished, so a dropped
session resumes instead of restarting. It is an append-only JSON Lines file,
one line per completed unit of work. Appending is safe to interrupt: a
half-written last line is ignored on read, so that unit simply runs again.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PERSIST_MODES = ("drive", "runtime")


def resolve_persist_root(mode, project_root, drive_root=None) -> Path:
    """Return the directory where small outputs and the manifest are kept."""
    if mode not in PERSIST_MODES:
        raise ValueError(f"mode must be one of {PERSIST_MODES}, got {mode!r}")
    if mode == "drive":
        if drive_root is None:
            raise ValueError("drive mode needs drive_root")
        root = Path(drive_root)
        if not root.parent.exists():
            raise FileNotFoundError(f"Drive does not look mounted: {root.parent} is missing")
    else:
        root = Path(project_root) / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


class Manifest:
    """Append-only record of finished (scene_id, stage) pairs."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _records(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A session died mid-write. The unit counts as not done.
                    continue
        return records

    def done(self, stage: str) -> set[str]:
        """Scene IDs finished for this stage."""
        return {r["scene_id"] for r in self._records() if r.get("stage") == stage}

    def is_done(self, scene_id: str, stage: str) -> bool:
        return scene_id in self.done(stage)

    def mark_done(self, scene_id: str, stage: str, **meta) -> None:
        """Record completion. Call only after the outputs are safely written."""
        record = {
            "scene_id": scene_id,
            "stage": stage,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **meta,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def pending(self, scene_ids: list[str], stage: str) -> list[str]:
        """Scene IDs still to do for this stage, original order kept."""
        finished = self.done(stage)
        return [s for s in scene_ids if s not in finished]
