"""API local segura por defecto, separada de modelos e infraestructura."""
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from subdub_titler.application.jobs import LocalJobWorker
from subdub_titler.application.quality import evaluate
from subdub_titler.application.transcribe import TranscribeUseCase
from subdub_titler.infrastructure.doctor import diagnostics
from subdub_titler.infrastructure.registry import ModelRegistry, default_registry
from subdub_titler.infrastructure.storage import SqliteJobs

logger = logging.getLogger(__name__)
EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".mp4", ".mkv", ".mov", ".webm"}
MAX_UPLOAD_BYTES = 2 * 1024**3


class EvaluationRequest(BaseModel):
    reference: str = Field(min_length=1, max_length=6_000)
    hypothesis: str = Field(max_length=6_000)
    language: str = Field(pattern="^(en|es|ja|zh)$")


def create_app(home: Path | None = None, registry: ModelRegistry | None = None) -> FastAPI:
    home = home or Path(os.environ.get("SUBDUB_HOME", str(Path.home() / ".subdub-titler")))
    uploads = home / "uploads"
    output_dir = home / "outputs"
    uploads.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = SqliteJobs(home / "jobs.sqlite3")
    registry = registry or default_registry()
    worker = LocalJobWorker(jobs, TranscribeUseCase(registry), output_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start()
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(title="SubDub-Titler", version="0.1.0", lifespan=lifespan)
    app.state.jobs = jobs
    app.state.worker = worker

    @app.get("/api/v1/health")
    def health() -> dict:
        return {"status": "ok", "name": "SubDub-Titler"}

    @app.get("/api/v1/models")
    def models() -> list[dict]:
        return [asdict(model) for model in registry.list_models()]

    @app.get("/api/v1/doctor")
    def doctor() -> list[dict]:
        return [asdict(check) for check in diagnostics()]

    @app.post("/api/v1/evaluate")
    def validate(payload: EvaluationRequest) -> dict:
        try:
            return evaluate(payload.reference, payload.hypothesis, payload.language)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/jobs", status_code=202)
    async def upload_job(
        file: Annotated[UploadFile, File()],
        model_id: Annotated[str, Form()] = "small",
        language: Annotated[str, Form()] = "auto",
        device: Annotated[str, Form()] = "cuda",
        compute_type: Annotated[str, Form()] = "float16",
    ) -> dict:
        if model_id not in {m.id for m in registry.list_models()}:
            raise HTTPException(422, "Modelo no registrado")
        if language not in ("auto", "en", "es", "ja", "zh"):
            raise HTTPException(422, "Idioma inválido")
        if device not in ("cpu", "cuda"):
            raise HTTPException(422, "Dispositivo inválido")
        if compute_type not in ("float16", "int8_float16", "int8", "float32"):
            raise HTTPException(422, "Tipo de cómputo inválido")
        original_name = Path(file.filename or "unknown").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in EXTENSIONS:
            raise HTTPException(415, "Formato de archivo no admitido")
        destination = uploads / f"{uuid4().hex}{suffix}"
        length = 0
        persisted = False
        try:
            with destination.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    length += len(chunk)
                    if length > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Archivo excede 2 GiB")
                    target.write(chunk)
            if not length:
                raise HTTPException(422, "Archivo vacío")
            job = jobs.create(destination, original_name, model_id, language, device, compute_type)
            persisted = True
            worker.enqueue(job["id"])
            return job
        except Exception:
            # Si se persistió el job, conservar el archivo para recuperación.
            if not persisted:
                destination.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get("/api/v1/jobs")
    def list_jobs() -> list[dict]:
        return jobs.list()

    @app.get("/api/v1/jobs/{identifier}")
    def get_job(identifier: str) -> dict:
        job = jobs.get(identifier)
        if job is None:
            raise HTTPException(404, "Job no encontrado")
        return job

    @app.get("/api/v1/jobs/{identifier}/subtitles")
    def get_subtitles(identifier: str) -> FileResponse:
        job = jobs.get(identifier)
        if not job:
            raise HTTPException(404, "Job no encontrado")
        if job["status"] != "completed" or not job["output_path"]:
            raise HTTPException(409, "Transcripción aún no disponible")
        return FileResponse(job["output_path"], filename=f"{identifier}.srt",
                            media_type="application/x-subrip")

    # En distribución el frontend compilado se sirve en el mismo puerto (sin CORS).
    # El dev server Vite usa proxy hacia 127.0.0.1:8000.
    dist = Path(__file__).resolve().parents[3] / "apps" / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app

