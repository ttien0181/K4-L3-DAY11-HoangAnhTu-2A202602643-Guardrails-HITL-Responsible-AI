"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store an input until the matching output is recorded."""
        key = request_id or user_id
        self._open[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "start_time": time.monotonic(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Append a completed interaction with its elapsed time and decision."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "input": started["input"] if started else "",
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "started_at": started["started_at"] if started else None,
            "completed_at": utc_now_iso(),
            "latency_ms": round(
                (time.monotonic() - started["start_time"]) * 1000, 2
            ) if started else None,
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        # TODO: path = filepath or default_audit_log_path()
        #       ensure parent dirs exist, dump self.logs with indent=2
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
