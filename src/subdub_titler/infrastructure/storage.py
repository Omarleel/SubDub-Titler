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
            self._migrate(conn)
            conn.execute("""CREATE TABLE IF NOT EXISTS ratings (
                job_id TEXT NOT NULL,
                cue_index INTEGER NOT NULL,
                score INTEGER NOT NULL CHECK (score BETWEEN 1 AND 5),
                note TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (job_id, cue_index)
            )""")

    # Columnas añadidas tras v0.1.0; bases existentes se amplían sin perder trabajos.
    _COLUMNS = {
        "kind": "TEXT NOT NULL DEFAULT 'transcribe'",
        "parent_id": "TEXT",
        "params_json": "TEXT",
        "result_json": "TEXT",
        "stage": "TEXT",
        "progress": "REAL",
    }

    def _migrate(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        for name, definition in self._COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db, timeout=20)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, file_path: Path, original_name: str, model_id: str,
               language: str, device: str, compute_type: str, kind: str = "transcribe",
               params: dict | None = None, parent_id: str | None = None) -> dict:
        identifier = uuid4().hex
        moment = now()
        with self._connect() as conn:
            conn.execute("""INSERT INTO jobs
              (id,file_path,original_name,model_id,language,device,compute_type,
               status,created_at,updated_at,kind,params_json,parent_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                         (identifier, str(file_path), original_name, model_id, language,
                          device, compute_type, "queued", moment, moment, kind,
                          json.dumps(params or {}, ensure_ascii=False), parent_id))
        return self.get(identifier)

    def get(self, identifier: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        return self._to_dict(row) if row else None

    def list(self, limit: int = 100, kind: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if kind is None:
                rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                                    (limit,))
            else:
                rows = conn.execute("SELECT * FROM jobs WHERE kind=? ORDER BY created_at DESC "
                                    "LIMIT ?", (kind, limit))
            return [self._to_dict(row) for row in rows]

    def update(self, identifier: str, **fields) -> None:
        allowed = {"status", "output_path", "transcript_json", "profiling_json", "error",
                   "result_json", "stage", "progress"}
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

    def rate(self, job_id: str, cue_index: int, score: int, note: str | None = None) -> None:
        """MOS humano por cue (1–5); una valoración por cue, la última prevalece."""
        if not 1 <= score <= 5:
            raise ValueError("La puntuación debe estar entre 1 y 5")
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?)",
                         (job_id, cue_index, score, note, now()))

    def ratings(self, job_id: str) -> dict[int, dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT cue_index, score, note FROM ratings WHERE job_id=?",
                                (job_id,)).fetchall()
        return {int(r["cue_index"]): {"score": r["score"], "note": r["note"]} for r in rows}

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict:
        data = dict(row)
        for key in ("transcript_json", "profiling_json", "params_json", "result_json"):
            data[key.removesuffix("_json")] = json.loads(data.pop(key)) if data[key] else None
        data["params"] = data["params"] or {}
        return data
