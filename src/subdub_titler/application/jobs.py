"""Worker único GPU local: recuperación de cola persistente y resultados durables.

Cada tipo de trabajo (`kind`) lo resuelve un manejador; añadir uno nuevo no cambia el worker
(OCP). Todos comparten el mismo hilo, así que nunca hay dos inferencias GPU a la vez.
"""
import json
import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from subdub_titler.application.dubbing import DubUseCase, report_to_dict
from subdub_titler.application.transcribe import TranscribeUseCase, write_subtitle
from subdub_titler.domain.dubbing import DubRequest
from subdub_titler.domain.models import Segment, Transcript, TranscriptionRequest, Word
from subdub_titler.infrastructure.profiling import SpanProfiler
from subdub_titler.infrastructure.storage import SqliteJobs

logger = logging.getLogger(__name__)
Progress = Callable[[str, float], None]


class JobHandler(Protocol):
    def run(self, job: dict, progress: Progress) -> dict:
        """Ejecuta el trabajo y devuelve los campos a persistir al completarse."""
        ...


class LocalJobWorker:
    def __init__(self, jobs: SqliteJobs, handlers: Mapping[str, JobHandler]):
        self.jobs = jobs
        self.handlers = dict(handlers)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="subdub-gpu-worker", daemon=True)

    def start(self) -> None:
        for identifier in self.jobs.recover():
            self._queue.put(identifier)
        self._thread.start()

    def enqueue(self, identifier: str) -> None:
        self._queue.put(identifier)

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            identifier = self._queue.get()
            try:
                if identifier is None:
                    return
                self._execute(identifier)
            finally:
                self._queue.task_done()

    def _execute(self, identifier: str) -> None:
        job = self.jobs.get(identifier)
        if not job or job["status"] != "queued":
            return
        self.jobs.update(identifier, status="running", stage="inicio", progress=0.0)

        def progress(stage: str, fraction: float) -> None:
            self.jobs.update(identifier, stage=stage, progress=round(fraction, 3))
        try:
            handler = self.handlers.get(job["kind"])
            if handler is None:
                raise ValueError(f"Tipo de trabajo sin manejador: {job['kind']}")
            fields = handler.run(job, progress)
            self.jobs.update(identifier, status="completed", stage="listo", progress=1.0,
                             **fields)
        except Exception as exc:
            logger.exception("Fallo de job %s", identifier)
            self.jobs.update(identifier, status="failed", error=f"{type(exc).__name__}: {exc}")


def transcript_from_dict(data: dict) -> Transcript:
    return Transcript(
        tuple(Segment(s["start"], s["end"], s["text"],
                      tuple(Word(**w) for w in s.get("words") or ()),
                      speaker=s.get("speaker"),
                      avg_logprob=s.get("avg_logprob"),
                      no_speech_prob=s.get("no_speech_prob"))
              for s in data["segments"]),
        data["language"], data["model_id"], data.get("duration_seconds"),
    )


class TranscribeJobHandler:
    def __init__(self, use_case: TranscribeUseCase, output_dir: Path) -> None:
        self.use_case = use_case
        self.output_dir = output_dir

    def run(self, job: dict, progress: Progress) -> dict:
        params = job.get("params") or {}
        request = TranscriptionRequest(
            source=Path(job["file_path"]), model_id=job["model_id"],
            language=job["language"], device=job["device"],
            compute_type=job["compute_type"], batch_size=params.get("batch_size", 16),
            vad_threshold=params.get("vad_threshold", 0.35),
            vad_speech_pad_ms=params.get("vad_speech_pad_ms", 800),
            vad_min_silence_ms=params.get("vad_min_silence_ms", 2000),
            no_speech_threshold=params.get("no_speech_threshold", 0.40),
            normalize_loudness=params.get("normalize_loudness", False),
            isolate_dialogue=params.get("isolate_dialogue", False),
            forced_alignment=params.get("forced_alignment", False),
            diarize=params.get("diarize", False),
            max_speakers=params.get("max_speakers", 4),
            conform_netflix=params.get("conform_netflix", False),
            suppress_hallucinations=params.get("suppress_hallucinations", True),
            adaptive_two_pass=params.get("adaptive_two_pass", False),
        )
        profiler = SpanProfiler()
        result = self.use_case.execute(request, profiler)
        outfile = self.output_dir / f"{job['id']}.srt"
        with profiler.measure("subtitles.write_srt"):
            write_subtitle(result, outfile, "srt")
        transcript = {
            "language": result.transcript.language,
            "model_id": result.transcript.model_id,
            "duration_seconds": result.transcript.duration_seconds,
            "text": result.transcript.text,
            "segments": [asdict(s) for s in result.transcript.segments],
            "elapsed_seconds": result.elapsed_seconds,
            "rtf": result.rtf,
        }
        return dict(output_path=str(outfile),
                    transcript_json=json.dumps(transcript, ensure_ascii=False),
                    profiling_json=json.dumps([asdict(e) for e in profiler.events]))


def dub_request(job: dict) -> DubRequest:
    params = job["params"]
    return DubRequest(
        source=Path(job["file_path"]), target_language=params["target_language"],
        source_language=job["language"], asr_model_id=job["model_id"], device=job["device"],
        tts_engine=params.get("tts_engine", "omnivoice"),
        translator=params.get("translator", "nllb-600m"),
        separator=params.get("separator", "htdemucs"), num_step=params.get("num_step", 16),
        tts_batch=params.get("tts_batch", 8), n_best=params.get("n_best", 4),
        evaluate=params.get("evaluate", True), diarize=params.get("diarize", False),
        max_speakers=params.get("max_speakers", 4),
        merge_fragments=params.get("merge_fragments", True),
        verify_attempts=params.get("verify_attempts", 1),
        text_guidance=params.get("text_guidance"),
        reference_guidance=params.get("reference_guidance"),
        accent_guidance=params.get("accent_guidance", 0.0),
        accent_proxy_dir=(Path(params["accent_proxy_dir"])
                          if params.get("accent_proxy_dir") else None),
        speed_penalty=params.get("speed_penalty", 0.5),
        concise_translation=params.get("concise_translation", False),
        per_phrase_reference=params.get("per_phrase_reference", False),
        prosody_transfer=params.get("prosody_transfer", False),
    )


class DubJobHandler:
    """Dobla a partir de la transcripción de un trabajo `transcribe` completado (padre)."""

    def __init__(self, use_case: DubUseCase, jobs: SqliteJobs, output_dir: Path) -> None:
        self.use_case = use_case
        self.jobs = jobs
        self.output_dir = output_dir

    def workdir(self, job_id: str) -> Path:
        return self.output_dir / "dub" / job_id

    def run(self, job: dict, progress: Progress) -> dict:
        parent = self.jobs.get(job["parent_id"] or "")
        if not parent or parent["status"] != "completed" or not parent["transcript"]:
            raise ValueError("El trabajo de transcripción de origen no está completado")
        profiler = SpanProfiler()
        with profiler.activate():
            report = self.use_case.execute(dub_request(job), self.workdir(job["id"]), profiler,
                                           transcript_from_dict(parent["transcript"]),
                                           progress)
        main = report.outputs.get("video") or report.outputs.get("mix")
        return dict(output_path=main,
                    result_json=json.dumps(report_to_dict(report), ensure_ascii=False),
                    profiling_json=json.dumps([asdict(e) for e in profiler.events]))


class DubCueJobHandler:
    """Re-sintetiza un cue de un doblaje (padre) con un texto editado."""

    def __init__(self, use_case: DubUseCase, jobs: SqliteJobs, dub: DubJobHandler) -> None:
        self.use_case = use_case
        self.jobs = jobs
        self.dub = dub

    def run(self, job: dict, progress: Progress) -> dict:
        parent = self.jobs.get(job["parent_id"] or "")
        if not parent or parent["status"] != "completed":
            raise ValueError("El doblaje de origen no está completado")
        params = job["params"]
        profiler = SpanProfiler()
        with profiler.activate():
            report = self.use_case.regenerate(dub_request(parent), self.dub.workdir(parent["id"]),
                                              int(params["index"]), params["text"], profiler)
        data = report_to_dict(report)
        self.jobs.update(parent["id"], result_json=json.dumps(data, ensure_ascii=False))
        cue = next(c for c in data["track"] if c["index"] == int(params["index"]))
        return dict(result_json=json.dumps({"cue": cue}, ensure_ascii=False),
                    profiling_json=json.dumps([asdict(e) for e in profiler.events]))
