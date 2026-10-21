import hashlib
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from subdub_titler.domain.models import RunResult, Transcript, TranscriptionRequest
from subdub_titler.domain.ports import (
    EngineRegistry,
    ForcedAligner,
    Releasable,
    SourceSeparator,
    SpeakerDiarizer,
)
from subdub_titler.domain.segmentation import DEFAULT_POLICY, NETFLIX_POLICY, resegment
from subdub_titler.infrastructure.profiling import SpanProfiler
from subdub_titler.infrastructure.subtitles import SrtRenderer, VttRenderer


class TranscribeUseCase:
    def __init__(self, registry: EngineRegistry, separator: SourceSeparator | None = None,
                 aligner: ForcedAligner | None = None,
                 diarizer: SpeakerDiarizer | None = None) -> None:
        self._registry = registry
        self._separator = separator
        self._aligner = aligner
        self._diarizer = diarizer

    def execute(self, request: TranscriptionRequest,
                profiler: SpanProfiler | None = None,
                progress: Callable[[str, float], None] | None = None) -> RunResult:
        progress = progress or (lambda stage, fraction: None)
        profiler = profiler or SpanProfiler()
        start = time.perf_counter()
        actual_request = request
        policy = NETFLIX_POLICY if request.conform_netflix else DEFAULT_POLICY
        with profiler.activate():
            progress("preparar audio", 0.05)
            if request.isolate_dialogue:
                if self._separator is None:
                    raise ValueError(
                        "Aislamiento de diálogo solicitado (--isolate-dialogue) "
                        "pero no hay separador de voz configurado."
                    )
                with profiler.measure("asr.isolate_dialogue"):
                    identity = hashlib.sha256(
                        str(request.source.resolve()).encode("utf-8")).hexdigest()
                    workdir = Path(tempfile.gettempdir()) / "subdub_stems" / identity
                    try:
                        stems = self._separator.separate(request.source, workdir)
                    finally:
                        if isinstance(self._separator, Releasable):
                            self._separator.release()
                    actual_request = replace(request, source=stems.voice)

            progress("transcribir", 0.2)
            with profiler.measure("registry.engine_for"):
                engine = self._registry.engine_for(actual_request)
            with profiler.measure("asr.transcribe_complete"):
                transcript = engine.transcribe(actual_request)
            progress("segmentar", 0.65)
            with profiler.measure("subtitles.resegment"):
                transcript = resegment(transcript, policy=policy)
            progress("alinear", 0.75)
            if request.forced_alignment:
                if self._aligner is None:
                    raise ValueError(
                        "Alineación forzada solicitada (--align) "
                        "pero no hay alineador configurado."
                    )
                with profiler.measure("asr.forced_align"):
                    try:
                        transcript = self._aligner.align(actual_request.source, transcript)
                    finally:
                        if isinstance(self._aligner, Releasable):
                            self._aligner.release()
                with profiler.measure("subtitles.resegment_final"):
                    transcript = resegment(transcript, policy=policy)
            progress("hablantes", 0.85)
            if request.diarize:
                if self._diarizer is None:
                    raise ValueError(
                        "Diarización solicitada (--diarize) "
                        "pero no hay diarizador configurado."
                    )
                with profiler.measure("asr.diarize"):
                    try:
                        transcript = self._assign_speakers(
                            actual_request.source, transcript, request.max_speakers,
                        )
                    finally:
                        if isinstance(self._diarizer, Releasable):
                            self._diarizer.release()
        elapsed = time.perf_counter() - start
        duration = transcript.duration_seconds
        return RunResult(transcript, elapsed / duration if duration else None,
                         elapsed, profiler.events)

    def _assign_speakers(self, source: Path, transcript: Transcript,
                         max_speakers: int) -> Transcript:
        if not transcript.segments:
            return transcript
        spans = [(s.start, s.end) for s in transcript.segments]
        labels = self._diarizer.assign(source, spans, max_speakers=max_speakers)
        assigned = [
            replace(s, speaker=label)
            for s, label in zip(transcript.segments, labels, strict=True)
        ]
        return replace(transcript, segments=tuple(assigned))


def write_subtitle(result: RunResult, path: Path, format: str) -> None:
    renderer = {"srt": SrtRenderer(), "vtt": VttRenderer()}.get(format)
    if renderer is None:
        raise ValueError("Formato no soportado: usa srt o vtt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(renderer.render(result.transcript), encoding="utf-8")
