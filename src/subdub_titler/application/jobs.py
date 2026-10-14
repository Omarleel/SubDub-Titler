"""Worker único GPU local: recuperación de cola persistente y resultados durables."""
import json
import logging
import queue
import threading
from dataclasses import asdict
from pathlib import Path

from subdub_titler.application.transcribe import TranscribeUseCase, write_subtitle
from subdub_titler.domain.models import TranscriptionRequest
from subdub_titler.infrastructure.profiling import SpanProfiler
from subdub_titler.infrastructure.storage import SqliteJobs

logger = logging.getLogger(__name__)


class LocalJobWorker:
    def __init__(self, jobs: SqliteJobs, use_case: TranscribeUseCase, output_dir: Path):
        self.jobs = jobs
        self.use_case = use_case
        self.output_dir = output_dir
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
        self.jobs.update(identifier, status="running")
        try:
            request = TranscriptionRequest(
                source=Path(job["file_path"]), model_id=job["model_id"],
                language=job["language"], device=job["device"],
                compute_type=job["compute_type"],
            )
            profiler = SpanProfiler()
            result = self.use_case.execute(request, profiler)
            outfile = self.output_dir / f"{identifier}.srt"
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
            self.jobs.update(identifier, status="completed", output_path=str(outfile),
                             transcript_json=json.dumps(transcript, ensure_ascii=False),
                             profiling_json=json.dumps([asdict(e) for e in profiler.events]))
        except Exception as exc:
            logger.exception("Fallo de job %s", identifier)
            self.jobs.update(identifier, status="failed", error=f"{type(exc).__name__}: {exc}")
