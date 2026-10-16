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

from subdub_titler.application.dubbing import DubComponentFactory, DubUseCase
from subdub_titler.application.jobs import (
    DubCueJobHandler,
    DubJobHandler,
    LocalJobWorker,
    TranscribeJobHandler,
)
from subdub_titler.application.quality import evaluate
from subdub_titler.application.transcribe import TranscribeUseCase
from subdub_titler.domain.dubbing import DUB_LANGUAGES
from subdub_titler.domain.models import Segment, Transcript
from subdub_titler.infrastructure.doctor import diagnostics
from subdub_titler.infrastructure.registry import ModelRegistry, default_registry
from subdub_titler.infrastructure.storage import SqliteJobs
from subdub_titler.infrastructure.subtitles import VttRenderer

logger = logging.getLogger(__name__)


def max_upload_bytes() -> int | None:
    raw_gb = os.environ.get("SUBDUB_MAX_UPLOAD_GB")
    if raw_gb is not None:
        try:
            val = float(raw_gb)
            return int(val * 1024**3) if val > 0 else None
        except ValueError:
            pass
    raw_bytes = os.environ.get("SUBDUB_MAX_UPLOAD_BYTES")
    if raw_bytes is not None:
        try:
            val_b = int(raw_bytes)
            return val_b if val_b > 0 else None
        except ValueError:
            pass
    return 10 * 1024**3  # Por defecto 10 GiB para admitir videos locales de alta resolución


MAX_UPLOAD_BYTES = max_upload_bytes() or 10 * 1024**3

# Explícitos: el registro de Windows no es fiable para tipos MIME.
MEDIA_TYPES = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".aac": "audio/aac",
    ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".mp4": "video/mp4", ".mkv": "video/x-matroska",
    ".mov": "video/quicktime", ".webm": "video/webm",
}
EXTENSIONS = set(MEDIA_TYPES)


class DubJobRequest(BaseModel):
    target_language: str = Field(pattern="^(en|es|ja|zh)$")
    tts_engine: str = "omnivoice"
    translator: str = "nllb-600m"
    separator: str | None = "htdemucs"
    num_step: int = Field(16, ge=1, le=64)
    tts_batch: int = Field(8, ge=1, le=64)
    n_best: int = Field(4, ge=1, le=8)
    evaluate: bool = True
    diarize: bool = False
    max_speakers: int = Field(4, ge=1, le=16)
    merge_fragments: bool = True
    verify_attempts: int = Field(1, ge=0, le=5)
    reference_candidates: int = Field(1, ge=1, le=8)
    text_guidance: float | None = Field(None, ge=0.0, le=20.0)
    reference_guidance: float | None = Field(None, ge=0.0, le=20.0)
    accent_guidance: float = Field(0.0, ge=0.0, le=50.0)
    accent_proxy_dir: str | None = None
    speed_penalty: float = Field(0.5, ge=0.0, le=10.0)
    concise_translation: bool = False
    per_phrase_reference: bool = False
    prosody_transfer: bool = False


class CueEdit(BaseModel):
    text: str = Field(min_length=1, max_length=1_000)


class CueRating(BaseModel):
    score: int = Field(ge=1, le=5)
    note: str | None = Field(None, max_length=500)


class EvaluationRequest(BaseModel):
    reference: str = Field(min_length=1, max_length=6_000)
    hypothesis: str = Field(max_length=6_000)
    language: str = Field(pattern="^(en|es|ja|zh)$")


def create_app(home: Path | None = None, registry: ModelRegistry | None = None,
               dub_factory: DubComponentFactory | None = None,
               transcripts=None) -> FastAPI:
    home = home or Path(os.environ.get("SUBDUB_HOME", str(Path.home() / ".subdub-titler")))
    uploads = home / "uploads"
    output_dir = home / "outputs"
    uploads.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = SqliteJobs(home / "jobs.sqlite3")
    registry = registry or default_registry()
    if dub_factory is None:
        # Composición por defecto; los motores pesados se importan al primer uso.
        from subdub_titler.infrastructure.dub_factory import AsrTranscripts, default_dub_factory
        dub_factory = default_dub_factory(registry)
        transcripts = transcripts or AsrTranscripts(registry)
    from subdub_titler.infrastructure.demucs_separator import LazyDemucsSeparator
    dub_use_case = DubUseCase(transcripts, dub_factory)
    dub_handler = DubJobHandler(dub_use_case, jobs, output_dir)
    transcribe_use_case = TranscribeUseCase(registry, separator=LazyDemucsSeparator())
    worker = LocalJobWorker(jobs, {
        "transcribe": TranscribeJobHandler(transcribe_use_case, output_dir),
        "dub": dub_handler,
        "dub-cue": DubCueJobHandler(dub_use_case, jobs, dub_handler),
    })

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
        batch_size: Annotated[int, Form(ge=1, le=64)] = 16,
        vad_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.35,
        vad_speech_pad_ms: Annotated[int, Form(ge=0, le=5000)] = 800,
        vad_min_silence_ms: Annotated[int, Form(ge=0, le=10000)] = 2000,
        no_speech_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.40,
        normalize_loudness: Annotated[bool, Form()] = False,
        isolate_dialogue: Annotated[bool, Form()] = False,
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
        limit = max_upload_bytes()
        try:
            with destination.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    length += len(chunk)
                    if limit is not None and length > limit:
                        limit_gib = limit / (1024**3)
                        raise HTTPException(
                            413, f"Archivo excede el límite permitido ({limit_gib:.1f} GiB)")
                    target.write(chunk)
            if not length:
                raise HTTPException(422, "Archivo vacío")
            job_params: dict[str, object] = {"batch_size": batch_size}
            if vad_threshold != 0.35:
                job_params["vad_threshold"] = vad_threshold
            if vad_speech_pad_ms != 800:
                job_params["vad_speech_pad_ms"] = vad_speech_pad_ms
            if vad_min_silence_ms != 2000:
                job_params["vad_min_silence_ms"] = vad_min_silence_ms
            if no_speech_threshold != 0.40:
                job_params["no_speech_threshold"] = no_speech_threshold
            if normalize_loudness:
                job_params["normalize_loudness"] = True
            if isolate_dialogue:
                job_params["isolate_dialogue"] = True
            job = jobs.create(destination, original_name, model_id, language, device,
                              compute_type, params=job_params)
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
    def list_jobs(kind: str | None = None) -> list[dict]:
        return jobs.list(kind=kind)

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

    # --- Doblaje ------------------------------------------------------------------------

    def _completed(identifier: str, kind: str) -> dict:
        job = jobs.get(identifier)
        if job is None or job["kind"] != kind:
            raise HTTPException(404, "Job no encontrado")
        if job["status"] != "completed":
            raise HTTPException(409, "El trabajo aún no está completado")
        return job

    @app.get("/api/v1/dub/components")
    def dub_components() -> dict:
        listing = getattr(dub_factory, "components", None)
        return {"languages": list(DUB_LANGUAGES),
                "components": listing() if callable(listing) else []}

    @app.post("/api/v1/jobs/{identifier}/dub", status_code=202)
    def create_dub(identifier: str, payload: DubJobRequest) -> dict:
        parent = _completed(identifier, "transcribe")
        if not Path(parent["file_path"]).is_file():
            raise HTTPException(410, "El archivo original ya no está disponible")
        job = jobs.create(Path(parent["file_path"]), parent["original_name"],
                          parent["model_id"], parent["transcript"]["language"],
                          parent["device"], parent["compute_type"], kind="dub",
                          params=payload.model_dump(), parent_id=identifier)
        worker.enqueue(job["id"])
        return job

    @app.get("/api/v1/jobs/{identifier}/dub/cues/{index}/audio")
    def dub_cue_audio(identifier: str, index: int) -> FileResponse:
        _completed(identifier, "dub")
        path = dub_handler.workdir(identifier) / "cues" / f"{index:04d}.wav"
        if not path.is_file():
            raise HTTPException(404, "Cue no encontrado")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/v1/jobs/{identifier}/dub/cues/{index}", status_code=202)
    def regenerate_cue(identifier: str, index: int, payload: CueEdit) -> dict:
        parent = _completed(identifier, "dub")
        if not any(c["index"] == index for c in (parent["result"] or {}).get("track", [])):
            raise HTTPException(404, "Cue no encontrado")
        job = jobs.create(Path(parent["file_path"]), parent["original_name"],
                          parent["model_id"], parent["language"], parent["device"],
                          parent["compute_type"], kind="dub-cue",
                          params={"index": index, "text": payload.text.strip()},
                          parent_id=identifier)
        worker.enqueue(job["id"])
        return job

    @app.post("/api/v1/jobs/{identifier}/dub/cues/{index}/rating")
    def rate_cue(identifier: str, index: int, payload: CueRating) -> dict:
        _completed(identifier, "dub")
        jobs.rate(identifier, index, payload.score, payload.note)
        return cue_ratings(identifier)

    @app.get("/api/v1/jobs/{identifier}/dub/ratings")
    def cue_ratings(identifier: str) -> dict:
        _completed(identifier, "dub")
        ratings = jobs.ratings(identifier)
        scores = [r["score"] for r in ratings.values()]
        return {"ratings": {str(k): v for k, v in ratings.items()}, "count": len(scores),
                "mean": round(sum(scores) / len(scores), 3) if scores else None}

    @app.get("/api/v1/jobs/{identifier}/dub/{artifact}")
    def dub_artifact(identifier: str, artifact: str) -> FileResponse:
        """Artefactos del doblaje: video (MP4 multipista), mix, voice o subtitles."""
        job = _completed(identifier, "dub")
        outputs = (job["result"] or {}).get("outputs", {})
        if artifact not in ("video", "mix", "voice", "subtitles") or artifact not in outputs:
            raise HTTPException(404, "Artefacto no disponible")
        path = Path(outputs[artifact])
        if not path.is_file():
            raise HTTPException(410, "El artefacto ya no está disponible")
        media_type = {".mp4": "video/mp4", ".flac": "audio/flac",
                      ".srt": "application/x-subrip"}.get(path.suffix.lower())
        return FileResponse(path, media_type=media_type)

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

