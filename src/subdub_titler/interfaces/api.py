"""API local segura por defecto, separada de modelos e infraestructura."""
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from subdub_titler.application.jobs import LocalJobWorker
from subdub_titler.application.quality import evaluate
from subdub_titler.application.transcribe import TranscribeUseCase
from subdub_titler.domain.models import Segment, Transcript
from subdub_titler.infrastructure.doctor import diagnostics
from subdub_titler.infrastructure.registry import ModelRegistry, default_registry
from subdub_titler.infrastructure.storage import SqliteJobs
from subdub_titler.infrastructure.subtitles import VttRenderer

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 2 * 1024**3
# Explícitos: el registro de Windows no es fiable para tipos MIME.
MEDIA_TYPES = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".aac": "audio/aac",
    ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".mp4": "video/mp4", ".mkv": "video/x-matroska",
    ".mov": "video/quicktime", ".webm": "video/webm",
}
EXTENSIONS = set(MEDIA_TYPES)


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

    @app.get("/api/v1/jobs/{identifier}/media")
    def get_media(identifier: str) -> FileResponse:
        """Archivo original para el reproductor; FileResponse atiende Range (seek)."""
        job = jobs.get(identifier)
        if not job:
            raise HTTPException(404, "Job no encontrado")
        path = Path(job["file_path"])
        if not path.is_file():
            raise HTTPException(410, "El archivo original ya no está disponible")
        return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower()))

    @app.get("/api/v1/jobs/{identifier}/subtitles.vtt")
    def get_vtt(identifier: str) -> Response:
        job = jobs.get(identifier)
        if not job:
            raise HTTPException(404, "Job no encontrado")
        if job["status"] != "completed" or not job["transcript"]:
            raise HTTPException(409, "Transcripción aún no disponible")
        data = job["transcript"]
        transcript = Transcript(
            tuple(Segment(s["start"], s["end"], s["text"]) for s in data["segments"]),
            data["language"], data["model_id"], data["duration_seconds"],
        )
        return Response(VttRenderer().render(transcript), media_type="text/vtt")

    # En distribución el frontend compilado se sirve en el mismo puerto (sin CORS).
    # El dev server Vite usa proxy hacia 127.0.0.1:8000.
    dist = Path(__file__).resolve().parents[3] / "apps" / "web" / "dist"
    if dist.is_dir():
        # Windows puede mapear .js a text/plain en el registro; los módulos ES lo rechazan.
        mimetypes.add_type("text/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
        root = dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            # Rutas del SPA (/precision, /rendimiento…) devuelven index.html al recargar.
            if path == "api" or path.startswith("api/"):
                raise HTTPException(404, "Ruta no encontrada")
            target = (root / path).resolve()
            if path and target.is_file() and target.is_relative_to(root):
                return FileResponse(target)
            return FileResponse(root / "index.html")
    return app

