"""Único adaptador con dependencia explícita de faster_whisper."""
from collections.abc import Sequence
from contextlib import nullcontext

from subdub_titler.domain.dubbing import AudioClip
from subdub_titler.domain.hallucinations import extract_dynamic_prompt, is_hallucination_segment
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

    def language_probability(self, clips: Sequence[AudioClip], language: str,
                             batch_size: int = 16) -> list[float]:
        """p(`language`) de la detección de idioma de Whisper para cada clip (primeros 30 s),
        en lotes: una pasada del codificador por lote."""
        import numpy as np
        from faster_whisper.audio import pad_or_trim

        from subdub_titler.infrastructure.media import resample
        if not clips:
            return []
        model = self._load()
        extractor = model.feature_extractor
        result: list[float] = []
        for first in range(0, len(clips), batch_size):
            features = np.stack([
                pad_or_trim(extractor(resample(c.samples, c.sample_rate, SAMPLING_RATE)
                                      [:extractor.n_samples]))
                for c in clips[first:first + batch_size]])
            for probs in model.model.detect_language(model.encode(features)):
                result.append(next((p for token, p in probs if token == f"<|{language}|>"), 0.0))
        return result

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        from faster_whisper import BatchedInferencePipeline
        self._load()
        profiler = active_profiler()
        with profiler.measure("audio.decode") if profiler else nullcontext():
            audio = decode(request.source, normalize_loudness=request.normalize_loudness)
        if request.vad_filter:
            tune_vad_threads()
        options = dict(
            beam_size=request.beam_size,
            language=None if request.language == "auto" else request.language,
            vad_filter=request.vad_filter,
            word_timestamps=request.word_timestamps,
            no_speech_threshold=request.no_speech_threshold,
            condition_on_previous_text=False,
        )
        if request.vad_filter:
            options["vad_parameters"] = dict(
                threshold=request.vad_threshold,
                min_silence_duration_ms=request.vad_min_silence_ms,
                speech_pad_ms=request.vad_speech_pad_ms,
            )
        with profiler.measure("asr.prepare_decode") if profiler else nullcontext():
            if request.batch_size > 1:
                # Decodifica varios fragmentos VAD en paralelo: ~3.6x en RTX 5060 Ti.
                if self._batched is None:
                    self._batched = BatchedInferencePipeline(self._model)
                if request.vad_filter:
                    from faster_whisper.vad import VadOptions, get_speech_timestamps
                    effective_threshold = min(request.vad_threshold, 0.20)
                    effective_pad = max(request.vad_speech_pad_ms, 1000)
                    vad_opts = VadOptions(
                        threshold=effective_threshold,
                        min_silence_duration_ms=request.vad_min_silence_ms,
                        speech_pad_ms=effective_pad,
                        max_speech_duration_s=30,
                    )
                    raw_ts = get_speech_timestamps(audio, vad_opts)
                    if not raw_ts:
                        return Transcript(
                            segments=(),
                            language="en" if request.language == "auto" else request.language,
                            model_id=self.model_id,
                            duration_seconds=float(len(audio) / SAMPLING_RATE),
                        )
                    clip_ts = [
                        {"start": float(t["start"] / SAMPLING_RATE),
                         "end": float(t["end"] / SAMPLING_RATE)}
                        for t in raw_ts
                    ]
                    # Fusionar micropausas breves (<= 2.5s) que quepan en la ventana de 30s
                    # para evitar cortes abruptos en susurros o colas de frase.
                    merged_clips: list[dict[str, float]] = []
                    for c in clip_ts:
                        if not merged_clips:
                            merged_clips.append(dict(c))
                        else:
                            last = merged_clips[-1]
                            gap = c["start"] - last["end"]
                            if gap <= 2.5 and (c["end"] - last["start"]) <= 30.0:
                                last["end"] = c["end"]
                            else:
                                merged_clips.append(dict(c))

                    batched_opts = dict(options)
                    batched_opts["vad_filter"] = False
                    batched_opts.pop("vad_parameters", None)
                    batched_opts["clip_timestamps"] = merged_clips
                    segments, info = self._batched.transcribe(
                        audio, batch_size=request.batch_size, **batched_opts,
                    )
                else:
                    segments, info = self._batched.transcribe(
                        audio, batch_size=request.batch_size, **options,
                    )
            else:
                segments, info = self._model.transcribe(audio, **options)
        # Consumir el generador: de otro modo mediríamos solo la preparación.
        with profiler.measure("asr.consume_generator") if profiler else nullcontext():
            result_segments = []
            for s in segments:
                text = s.text.strip()
                if not text:
                    continue
                avg_logprob = getattr(s, "avg_logprob", None)
                no_speech_prob = getattr(s, "no_speech_prob", None)
                compression_ratio = getattr(s, "compression_ratio", None)

                # Supresión activa de alucinaciones (bucles de repetición y frases fantasma)
                if request.suppress_hallucinations and is_hallucination_segment(
                    text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob,
                    compression_ratio=compression_ratio,
                ):
                    continue

                words = tuple(
                    Word(float(w.start), float(w.end), w.word) for w in (s.words or ())
                )
                start = float(s.start)
                end = float(s.end)
                if words:
                    end = max(end, words[-1].end + 0.12)
                if end > start:
                    nsp_val = float(no_speech_prob) if no_speech_prob is not None else None
                    result_segments.append(
                        Segment(
                            start, end, text, words,
                            avg_logprob=float(avg_logprob) if avg_logprob is not None else None,
                            no_speech_prob=nsp_val,
                        )
                    )

            # Inferencia adaptativa en dos pasadas con Dynamic Prompting
            if request.adaptive_two_pass and result_segments:
                with profiler.measure("asr.adaptive_second_pass") if profiler else nullcontext():
                    result_segments = self._adaptive_second_pass(
                        audio, result_segments, info.language, request,
                    )

            result_segments = tuple(result_segments)
        return Transcript(
            segments=result_segments,
            language=info.language,
            model_id=self.model_id,
            duration_seconds=float(info.duration) if info.duration else None,
        )

    def _adaptive_second_pass(
        self,
        audio,
        segments: list[Segment],
        language: str | None,
        request: TranscriptionRequest,
    ) -> list[Segment]:
        """Segunda pasada focalizada sobre segmentos de baja confianza con prompt dinámico."""
        dynamic_prompt = extract_dynamic_prompt(segments)
        if not dynamic_prompt:
            return segments

        prompt_str = f"Contexto y nombres propios clave: {dynamic_prompt}."
        second_pass_beam = max(request.beam_size + 2, 7)
        total_audio_len = len(audio) / SAMPLING_RATE
        improved: list[Segment] = []

        # Limitar a máx 15 segmentos para mantener tiempo de respuesta ultra-rápido
        dudosos_count = 0
        for s in segments:
            is_doubtful = (
                (s.avg_logprob is not None and s.avg_logprob < -0.65) or
                (s.no_speech_prob is not None and s.no_speech_prob > 0.35)
            )
            if not is_doubtful or dudosos_count >= 15:
                improved.append(s)
                continue

            dudosos_count += 1
            c_start = max(0.0, s.start - 0.20)
            c_end = min(total_audio_len, s.end + 0.20)
            chunk = audio[int(c_start * SAMPLING_RATE):int(c_end * SAMPLING_RATE)]
            if len(chunk) < int(0.15 * SAMPLING_RATE):
                improved.append(s)
                continue

            try:
                pass2_segs, _ = self._model.transcribe(
                    chunk,
                    language=language,
                    beam_size=second_pass_beam,
                    initial_prompt=prompt_str,
                    condition_on_previous_text=False,
                    word_timestamps=request.word_timestamps,
                    vad_filter=False,
                )
                pass2_list = list(pass2_segs)
                if not pass2_list:
                    # La segunda pasada confirmó que era silencio/ruido; suprimir
                    continue

                valid_pass2: list[Segment] = []
                for p in pass2_list:
                    p_text = p.text.strip()
                    if not p_text:
                        continue
                    p_logp = getattr(p, "avg_logprob", None)
                    p_nsp = getattr(p, "no_speech_prob", None)
                    p_ratio = getattr(p, "compression_ratio", None)
                    if is_hallucination_segment(p_text, p_logp, p_nsp, p_ratio):
                        continue
                    p_words = tuple(
                        Word(float(w.start) + c_start, float(w.end) + c_start, w.word)
                        for w in (p.words or ())
                    )
                    p_st = float(p.start) + c_start
                    p_en = float(p.end) + c_start
                    if p_words:
                        p_en = max(p_en, p_words[-1].end + 0.12)
                    if p_en > p_st:
                        valid_pass2.append(
                            Segment(
                                p_st, p_en, p_text, p_words,
                                speaker=s.speaker,
                                avg_logprob=float(p_logp) if p_logp is not None else None,
                                no_speech_prob=float(p_nsp) if p_nsp is not None else None,
                            )
                        )

                if not valid_pass2:
                    continue

                logps = [v.avg_logprob for v in valid_pass2 if v.avg_logprob is not None]
                p2_avg_logp = sum(logps) / len(logps) if logps else None
                if logps and (s.avg_logprob is None or p2_avg_logp > s.avg_logprob):
                    improved.extend(valid_pass2)
                else:
                    improved.append(s)
            except Exception:
                improved.append(s)

        return improved
