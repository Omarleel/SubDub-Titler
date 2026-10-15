"""Único adaptador con dependencia explícita de faster_whisper."""
from collections.abc import Sequence
from contextlib import nullcontext

from subdub_titler.domain.dubbing import AudioClip
from subdub_titler.domain.models import Segment, Transcript, TranscriptionRequest, Word
from subdub_titler.infrastructure.audio import SAMPLING_RATE, decode, tune_vad_threads
from subdub_titler.infrastructure.gpu_runtime import ensure_cuda_libraries
from subdub_titler.infrastructure.profiling import active_profiler


class FasterWhisperAdapter:
    def __init__(self, model_id: str, device: str, compute_type: str) -> None:
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._batched = None

    def release(self) -> None:
        """Descarga los pesos de la GPU (CTranslate2 los recarga al volver a usarse)."""
        if self._model is not None and self._model.model.model_is_loaded:
            self._model.model.unload_model()

    def _load(self):
        ensure_cuda_libraries()
        if self._model is not None and not self._model.model.model_is_loaded:
            self._model.model.load_model()
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
        return self._model

    def transcribe_clips(self, clips: Sequence[AudioClip], language: str,
                         batch_size: int = 16, beam_size: int = 5) -> list[str]:
        """Clips cortos en memoria (voz sintetizada), sin VAD y con idioma fijo.

        Se concatenan y se pasan como `clip_timestamps`: el pipeline batched decodifica cada
        clip como un fragmento independiente (sin contexto entre clips), varios a la vez.
        Cada segmento se asigna a su clip por su instante de inicio.
        """
        import bisect

        import numpy as np
        from faster_whisper import BatchedInferencePipeline

        from subdub_titler.infrastructure.media import resample
        if not clips:
            return []
        self._load()
        gap = np.zeros(int(0.2 * SAMPLING_RATE), dtype=np.float32)
        parts, bounds, cursor = [], [], 0
        for clip in clips:
            audio = resample(clip.samples, clip.sample_rate, SAMPLING_RATE)[:30 * SAMPLING_RATE]
            parts += [audio, gap]
            bounds.append((cursor / SAMPLING_RATE, (cursor + len(audio)) / SAMPLING_RATE))
            cursor += len(audio) + len(gap)
        if self._batched is None:
            self._batched = BatchedInferencePipeline(self._model)
        segments, _ = self._batched.transcribe(
            np.concatenate(parts), language=language, beam_size=beam_size, batch_size=batch_size,
            clip_timestamps=[{"start": s, "end": max(e, s + 0.05)} for s, e in bounds],
        )
        starts = [s for s, _ in bounds]
        texts: list[list[str]] = [[] for _ in clips]
        for segment in segments:
            index = max(0, bisect.bisect_right(starts, float(segment.start) + 1e-3) - 1)
            texts[index].append(segment.text.strip())
        return [" ".join(t).strip() for t in texts]

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        from faster_whisper import BatchedInferencePipeline
        self._load()
        profiler = active_profiler()
        with profiler.measure("audio.decode") if profiler else nullcontext():
            audio = decode(request.source)
        if request.vad_filter:
            tune_vad_threads()
        options = dict(
            beam_size=request.beam_size,
            language=None if request.language == "auto" else request.language,
            vad_filter=request.vad_filter,
            word_timestamps=request.word_timestamps,
        )
        with profiler.measure("asr.prepare_decode") if profiler else nullcontext():
            if request.batch_size > 1:
                # Decodifica varios fragmentos VAD en paralelo: ~3.6x en RTX 5060 Ti.
                if self._batched is None:
                    self._batched = BatchedInferencePipeline(self._model)
                segments, info = self._batched.transcribe(
                    audio, batch_size=request.batch_size, **options,
                )
            else:
                segments, info = self._model.transcribe(audio, **options)
        # Consumir el generador: de otro modo mediríamos solo la preparación.
        with profiler.measure("asr.consume_generator") if profiler else nullcontext():
            result_segments = tuple(
                Segment(float(s.start), float(s.end), s.text.strip(), tuple(
                    Word(float(w.start), float(w.end), w.word) for w in (s.words or ())
                ))
                for s in segments if s.text.strip() and float(s.end) > float(s.start)
            )
        return Transcript(
            segments=result_segments,
            language=info.language,
            model_id=self.model_id,
            duration_seconds=float(info.duration) if info.duration else None,
        )
