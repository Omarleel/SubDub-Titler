import tempfile
import time
from dataclasses import replace
from pathlib import Path

from subdub_titler.domain.models import RunResult, Segment, Transcript, TranscriptionRequest
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
                profiler: SpanProfiler | None = None) -> RunResult:
        profiler = profiler or SpanProfiler()
        start = time.perf_counter()
        actual_request = request
        policy = NETFLIX_POLICY if request.conform_netflix else DEFAULT_POLICY
        with profiler.activate():
            if request.isolate_dialogue:
                if self._separator is None:
                    raise ValueError(
                        "Aislamiento de diálogo solicitado (--isolate-dialogue) "
                        "pero no hay separador de voz configurado."
                    )
                with profiler.measure("asr.isolate_dialogue"):
                    workdir = Path(tempfile.gettempdir()) / "subdub_stems" / request.source.stem
                    stems = self._separator.separate(request.source, workdir)
                    if isinstance(self._separator, Releasable):
                        self._separator.release()
                    actual_request = replace(request, source=stems.voice)

            with profiler.measure("registry.engine_for"):
                engine = self._registry.engine_for(actual_request)
            with profiler.measure("asr.transcribe_complete"):
                transcript = engine.transcribe(actual_request)
            with profiler.measure("subtitles.resegment"):
                transcript = resegment(transcript, policy=policy)
            if request.forced_alignment:
                if self._aligner is None:
                    raise ValueError(
                        "Alineación forzada solicitada (--align) "
                        "pero no hay alineador configurado."
                    )
                with profiler.measure("asr.forced_align"):
                    transcript = self._aligner.align(actual_request.source, transcript)
                    if isinstance(self._aligner, Releasable):
                        self._aligner.release()
                with profiler.measure("subtitles.resegment_final"):
                    transcript = resegment(transcript, policy=policy)
            if request.diarize:
                if self._diarizer is None:
                    raise ValueError(
                        "Diarización solicitada (--diarize) "
                        "pero no hay diarizador configurado."
                    )
                with profiler.measure("asr.diarize"):
                    transcript = self._assign_speakers(
                        actual_request.source, transcript, request.max_speakers,
                    )
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
            Segment(s.start, s.end, s.text, s.words, speaker=label)
            for s, label in zip(transcript.segments, labels, strict=False)
        ]
        return replace(transcript, segments=tuple(assigned))


def write_subtitle(result: RunResult, path: Path, format: str) -> None:
    renderer = {"srt": SrtRenderer(), "vtt": VttRenderer()}.get(format)
    if renderer is None:
        raise ValueError("Formato no soportado: usa srt o vtt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(renderer.render(result.transcript), encoding="utf-8")
