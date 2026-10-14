"""Único adaptador con dependencia explícita de faster_whisper."""
from contextlib import nullcontext

from subdub_titler.infrastructure.profiling import active_profiler
from subdub_titler.domain.models import Segment, Transcript, TranscriptionRequest


class FasterWhisperAdapter:
    def __init__(self, model_id: str, device: str, compute_type: str) -> None:
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                'ASR no instalado: ejecuta python -m pip install -e ".[asr]"'
            ) from exc
        profiler = active_profiler()
        with profiler.measure("asr.model_load") if profiler else nullcontext():
            if self._model is None:
                self._model = WhisperModel(
                    self.model_id, device=self.device, compute_type=self.compute_type,
                )
        with profiler.measure("asr.prepare_decode") if profiler else nullcontext():
            segments, info = self._model.transcribe(
                str(request.source),
                beam_size=request.beam_size,
                language=None if request.language == "auto" else request.language,
                vad_filter=request.vad_filter,
                word_timestamps=request.word_timestamps,
            )
        # Consumir el generador: de otro modo mediríamos solo la preparación.
        with profiler.measure("asr.consume_generator") if profiler else nullcontext():
            result_segments = tuple(
                Segment(float(s.start), float(s.end), s.text.strip())
                for s in segments if s.text.strip() and float(s.end) > float(s.start)
            )
        return Transcript(
            segments=result_segments,
            language=info.language,
            model_id=self.model_id,
            duration_seconds=float(info.duration) if info.duration else None,
        )
