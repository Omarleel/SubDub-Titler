from __future__ import annotations
"""Adaptador SQLite. Las conexiones son por operación (seguro entre threads)."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteJobs:
    def __init__(self, db: Path) -> None:
        self.db = db
        self.db.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                original_name TEXT NOT NULL,
                model_id TEXT NOT NULL,
                language TEXT NOT NULL,
                device TEXT NOT NULL,
                compute_type TEXT NOT NULL,
                status TEXT NOT NULL,
                output_path TEXT,
                transcript_json TEXT,
                profiling_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db, timeout=20)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, file_path: Path, original_name: str, model_id: str,
               language: str, device: str, compute_type: str) -> dict:
        identifier = uuid4().hex
        moment = now()
        with self._connect() as conn:
            conn.execute("""INSERT INTO jobs
              (id,file_path,original_name,model_id,language,device,compute_type,
               status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                         (identifier, str(file_path), original_name, model_id, language,
                          device, compute_type, "queued", moment, moment))
        return self.get(identifier)

    def get(self, identifier: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        return self._to_dict(row) if row else None

    def list(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._to_dict(row) for row in rows]

    def update(self, identifier: str, **fields) -> None:
        allowed = {"status", "output_path", "transcript_json", "profiling_json", "error"}
        if not fields or not set(fields).issubset(allowed):
            raise ValueError("Campos no permitidos")
        updates = {**fields, "updated_at": now()}
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id=?",
                         (*updates.values(), identifier))

    def recover(self) -> list[str]:
        # No marcar como completados procesos que murieron con el servidor.
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status='failed', error='Proceso interrumpido', "
                         "updated_at=? WHERE status='running'", (now(),))
            rows = conn.execute("SELECT id FROM jobs WHERE status='queued'").fetchall()
        return [str(row["id"]) for row in rows]

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict:
        data = dict(row)
        for key in ("transcript_json", "profiling_json"):
            data[key.removesuffix("_json")] = json.loads(data.pop(key)) if data[key] else None
        return data
