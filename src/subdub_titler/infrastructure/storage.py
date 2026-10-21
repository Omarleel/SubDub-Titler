"""Adaptador SQLite. Las conexiones son por operación (seguro entre threads)."""
from __future__ import annotations

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

            conn.execute("""CREATE TABLE IF NOT EXISTS revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                ratings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS revisions_job_idx ON revisions(job_id, id)")

    # Columnas añadidas tras v0.1.0; bases existentes se amplían sin perder trabajos.
    _COLUMNS = {
        "kind": "TEXT NOT NULL DEFAULT 'transcribe'",
        "parent_id": "TEXT",
        "params_json": "TEXT",
        "result_json": "TEXT",
        "stage": "TEXT",
        "progress": "REAL",
        "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
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

    def list(self, limit: int = 100, kind: str | None = None, *, offset: int = 0,
             visible: bool = False, summary: bool = False) -> list[dict]:
        where, args = [], []
        if kind is not None:
            where.append("kind=?")
            args.append(kind)
        if visible:
            where.append("kind IN ('transcribe', 'dub')")
        clause = " WHERE " + " AND ".join(where) if where else ""
        columns = "*"
        if summary:
            # No leer ni deserializar transcripciones y reportes en cada consulta de estado.
            columns = ("id, original_name, model_id, language, device, compute_type, status, "
                       "kind, parent_id, params_json, stage, progress, error, cancel_requested, "
                       "created_at, "
                       "updated_at, NULL AS transcript_json, NULL AS profiling_json, "
                       "NULL AS result_json")
        with self._connect() as conn:
            rows = conn.execute(f"SELECT {columns} FROM jobs{clause} "
                                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                                (*args, limit, offset))
            return [self._to_dict(row) for row in rows]

    def activate_revision(self, identifier: str, result: dict, *, cue_index: int) -> None:
        """Guarda el estado anterior y cambia todos los punteros en una sola transacción."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT result_json FROM jobs WHERE id=?",
                                   (identifier,)).fetchone()
            if current is None or current["result_json"] is None:
                raise ValueError("Doblaje inexistente")
            ratings = [dict(row) for row in conn.execute(
                "SELECT * FROM ratings WHERE job_id=?", (identifier,))]
            conn.execute("INSERT INTO revisions(job_id,result_json,ratings_json,created_at) "
                         "VALUES (?,?,?,?)", (identifier, current["result_json"],
                                              json.dumps(ratings), now()))
            outputs = result["outputs"]
            conn.execute("UPDATE jobs SET result_json=?, output_path=?, updated_at=? WHERE id=?",
                         (json.dumps(result, ensure_ascii=False),
                          outputs.get("video") or outputs["mix"], now(), identifier))
            # Una valoración del audio anterior no describe el audio recién generado.
            conn.execute("DELETE FROM ratings WHERE job_id=? AND cue_index=?",
                         (identifier, cue_index))

    def undo_revision(self, identifier: str) -> dict:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute("SELECT * FROM revisions WHERE job_id=? "
                                    "ORDER BY id DESC LIMIT 1", (identifier,)).fetchone()
            if previous is None:
                raise ValueError("No hay una revisión anterior")
            result = json.loads(previous["result_json"])
            outputs = result["outputs"]
            conn.execute("UPDATE jobs SET result_json=?, output_path=?, updated_at=? WHERE id=?",
                         (previous["result_json"], outputs.get("video") or outputs["mix"],
                          now(), identifier))
            conn.execute("DELETE FROM ratings WHERE job_id=?", (identifier,))
            for rating in json.loads(previous["ratings_json"]):
                conn.execute("INSERT INTO ratings VALUES (?,?,?,?,?)",
                             (identifier, rating["cue_index"], rating["score"],
                              rating["note"], rating["created_at"]))
            conn.execute("DELETE FROM revisions WHERE id=?", (previous["id"],))
        return result

    def revision_count(self, identifier: str) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM revisions WHERE job_id=?",
                                (identifier,)).fetchone()[0]

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

    def cancel(self, identifier: str) -> bool:
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE jobs SET cancel_requested=1, "
                "status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END, "
                "stage='cancelación solicitada', updated_at=? "
                "WHERE id=? AND status IN ('queued','running') AND cancel_requested=0",
                (now(), identifier))
            return changed.rowcount == 1

    def complete(self, identifier: str, fields: dict) -> None:
        allowed = {"output_path", "transcript_json", "profiling_json", "result_json"}
        if not set(fields).issubset(allowed):
            raise ValueError("Campos de resultado no permitidos")
        assignments = "".join(f", {key}=?" for key in fields)
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=CASE WHEN cancel_requested THEN 'cancelled' "
                "ELSE 'completed' END, stage=CASE WHEN cancel_requested THEN 'cancelado' "
                "ELSE 'listo' END, progress=1.0, updated_at=?" + assignments + " WHERE id=?",
                (now(), *fields.values(), identifier))

    def recover(self) -> list[str]:
        # No marcar como completados procesos que murieron con el servidor.
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status='failed', error='Proceso interrumpido', "
                         "updated_at=? WHERE status='running'", (now(),))
            rows = conn.execute("SELECT id FROM jobs WHERE status='queued' "
                                "ORDER BY created_at, id").fetchall()
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
