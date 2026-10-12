from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

AWAITING = "awaiting_face_selection"


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                target_language TEXT NOT NULL,
                source_language TEXT NOT NULL,
                operation TEXT NOT NULL DEFAULT 'combine',
                masking_mode TEXT NOT NULL DEFAULT 'new',
                input_path TEXT NOT NULL,
                input_name TEXT NOT NULL,
                artifacts TEXT NOT NULL DEFAULT '{}',
                face_choice TEXT,
                awaiting_since TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "operation" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN operation TEXT NOT NULL DEFAULT 'combine'")
            if "masking_mode" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN masking_mode TEXT NOT NULL DEFAULT 'new'")
            if "face_choice" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN face_choice TEXT")
            if "awaiting_since" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN awaiting_since TEXT")
            # A worker process may have stopped while a job was active. Keep its
            # uploaded source and retry it from the beginning after restart.
            db.execute(
                "UPDATE jobs SET status='queued', stage='queued', updated_at=? WHERE status='running'",
                (self._now(),),
            )

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, job_id: str, target: str, source: str, input_path: Path, input_name: str,
               operation: str = "combine", masking_mode: str = "new") -> None:
        now = self._now()
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO jobs(job_id,status,stage,target_language,source_language,operation,masking_mode,input_path,input_name,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, "queued", "queued", target, source, operation, masking_mode, str(input_path), input_name, now, now),
            )
        self._write_manifest(job_id)

    def claim(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                db.commit()
                return None
            db.execute(
                "UPDATE jobs SET status='running',stage='starting',updated_at=? WHERE job_id=?",
                (self._now(), row["job_id"]),
            )
            db.commit()
            claimed = dict(row)
        self._write_manifest(claimed["job_id"])
        return claimed

    def stage(self, job_id: str, stage: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='running',stage=?,updated_at=? WHERE job_id=?",
                (stage, self._now(), job_id),
            )
        self._write_manifest(job_id)

    def finish(self, job_id: str, artifacts: dict[str, str]) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='completed',stage='completed',artifacts=?,error=NULL,updated_at=? WHERE job_id=?",
                (json.dumps(artifacts), self._now(), job_id),
            )
        self._write_manifest(job_id)

    def set_artifact(self, job_id: str, name: str, path: str) -> None:
        """Register (or replace) one artifact of an existing job without changing its state."""
        with self._lock, self._connect() as db:
            row = db.execute("SELECT artifacts FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return
            artifacts = json.loads(row["artifacts"])
            artifacts[name] = str(path)
            db.execute(
                "UPDATE jobs SET artifacts=?,updated_at=? WHERE job_id=?",
                (json.dumps(artifacts), self._now(), job_id),
            )
        self._write_manifest(job_id)

    def fail(self, job_id: str, stage: str, error: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed',stage=?,error=?,updated_at=? WHERE job_id=?",
                (stage, error, self._now(), job_id),
            )
        self._write_manifest(job_id)

    def await_face(self, job_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,stage=?,awaiting_since=?,updated_at=? WHERE job_id=?",
                (AWAITING, AWAITING, self._now(), self._now(), job_id),
            )
        self._write_manifest(job_id)

    def confirm_face(self, job_id: str, candidate_id: str) -> str:
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status, face_choice FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                db.commit()
                return "unknown_job"
            if row["status"] == AWAITING:
                db.execute(
                    "UPDATE jobs SET status='queued',stage='queued',face_choice=?,awaiting_since=NULL,updated_at=? WHERE job_id=?",
                    (candidate_id, self._now(), job_id),
                )
                db.commit()
                outcome = "queued"
            elif row["face_choice"] == candidate_id and row["status"] in ("queued", "running", "completed"):
                db.commit()
                return "already_confirmed"
            else:
                db.commit()
                return "not_awaiting"
        self._write_manifest(job_id)
        return outcome

    def expire_stale(self, max_age_seconds: float = 24 * 3600) -> list[str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT job_id FROM jobs WHERE status=? AND awaiting_since <= ?", (AWAITING, cutoff)
            ).fetchall()
            expired = [row["job_id"] for row in rows]
            for job_id in expired:
                db.execute(
                    "UPDATE jobs SET status='failed',stage='expired',error=?,updated_at=? WHERE job_id=?",
                    ("Face confirmation timed out after 24 hours; submit the video again.", self._now(), job_id),
                )
        for job_id in expired:
            self._write_manifest(job_id)
        return expired

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["artifacts"] = json.loads(result["artifacts"])
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return persisted jobs newest first for the queue/history dashboard."""
        limit = max(1, min(int(limit), 500))
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        jobs = []
        for row in rows:
            record = dict(row)
            record["artifacts"] = json.loads(record["artifacts"])
            jobs.append(record)
        return jobs

    def remove_queued(self, job_id: str) -> str:
        """Atomically remove a job only while it is still waiting in the queue."""
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                db.commit()
                return "unknown_job"
            if row["status"] != "queued":
                db.commit()
                return "not_queued"
            db.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
            db.commit()
        return "removed"

    def _write_manifest(self, job_id: str) -> None:
        record = self.get(job_id)
        if record is None:
            return
        path = self.path.parent / "jobs" / job_id / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": record["job_id"],
            "status": record["status"],
            "stage": record["stage"],
            "target_language": record["target_language"],
            "source_language": record["source_language"],
            "operation": record["operation"],
            "masking_mode": record["masking_mode"],
            "input": record["input_name"],
            "artifacts": {
                name: f"/api/jobs/{job_id}/artifacts/{name}"
                for name in record["artifacts"]
            },
            "log_url": f"/api/jobs/{job_id}/logs",
            "error": record["error"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
