import tempfile
import time
from dataclasses import replace
from pathlib import Path

from subdub_titler.domain.models import RunResult, TranscriptionRequest
from subdub_titler.domain.ports import EngineRegistry, Releasable, SourceSeparator
from subdub_titler.domain.segmentation import resegment
from subdub_titler.infrastructure.profiling import SpanProfiler
from subdub_titler.infrastructure.subtitles import SrtRenderer, VttRenderer


class TranscribeUseCase:
    def __init__(self, registry: EngineRegistry, separator: SourceSeparator | None = None) -> None:
        self._registry = registry
        self._separator = separator

    def execute(self, request: TranscriptionRequest,
                profiler: SpanProfiler | None = None) -> RunResult:
        profiler = profiler or SpanProfiler()
        start = time.perf_counter()
        actual_request = request
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
                transcript = resegment(transcript)
        elapsed = time.perf_counter() - start
        duration = transcript.duration_seconds
        return RunResult(transcript, elapsed / duration if duration else None,
                         elapsed, profiler.events)


def write_subtitle(result: RunResult, path: Path, format: str) -> None:
    renderer = {"srt": SrtRenderer(), "vtt": VttRenderer()}.get(format)
    if renderer is None:
        raise ValueError("Formato no soportado: usa srt o vtt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(renderer.render(result.transcript), encoding="utf-8")
