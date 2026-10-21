"""Caso de uso de doblaje: orquesta puertos y reglas de dominio; no conoce ningún motor.

Etapas medidas como spans `dub.*` para el laboratorio de rendimiento. Los componentes concretos
llegan de una `DubComponentFactory` (inversión de dependencias): un motor TTS, traductor o
separador nuevo se registra en infraestructura sin tocar este módulo.
"""
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from subdub_titler.application.dub_quality import DubEvaluator
from subdub_titler.application.quality import evaluate
from subdub_titler.domain.dubbing import (
    DEFAULT_TIMING,
    DEFAULT_VERIFY,
    Ambience,
    AudioClip,
    Candidate,
    CueQuality,
    DubbedCue,
    DubCue,
    DubReport,
    DubRequest,
    FitAction,
    Gate,
    QualitySummary,
    ReferenceSpan,
    Span,
    SyncReport,
    TimingPolicy,
    VerifyPolicy,
    ambience_spans,
    choose_candidate,
    cues_from_transcript,
    diarization_windows,
    fit_decision,
    merge_fragments,
    plan_slots,
    probe_indices,
    reference_candidates,
    speakers_by_presence,
    speech_spans,
    split_by_speaker,
    sync_metrics,
)
from subdub_titler.domain.models import Segment, Transcript
from subdub_titler.domain.ports import (
    AudioStore,
    ClipTranscriber,
    Mixer,
    Muxer,
    Profiler,
    ProsodyAnalyzer,
    Releasable,
    SourceSeparator,
    SpeakerDiarizer,
    SpeakerEncoder,
    Stems,
    SubtitleRenderer,
    Translator,
    VoiceCloner,
)

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm"}
Progress = Callable[[str, float], None]


@dataclass(frozen=True, slots=True)
class DubPipeline:
    """Puertos ya construidos para una petición concreta."""
    translator: Translator
    cloner: VoiceCloner
    audio: AudioStore
    mixer: Mixer
    subtitles: SubtitleRenderer
    muxer: Muxer | None = None
    separator: SourceSeparator | None = None
    evaluator: DubEvaluator | None = None
    diarizer: SpeakerDiarizer | None = None
    # Juez de inteligibilidad para reintentos y sondeo de referencias. Debe ser un ASR distinto
    # del de la evaluación final, para no optimizar contra la misma métrica que se reporta.
    verifier: ClipTranscriber | None = None
    speaker_encoder: SpeakerEncoder | None = None
    prosody: ProsodyAnalyzer | None = None


class DubComponentFactory(Protocol):
    def build(self, request: DubRequest) -> DubPipeline: ...

    def release_idle(self) -> None:
        """Descarga de la GPU los modelos que dejaron trabajos anteriores (se recargan solos)."""
        ...


class TranscriptProvider(Protocol):
    def __call__(self, request: DubRequest, profiler: Profiler) -> Transcript: ...


@dataclass(frozen=True, slots=True)
class Workspace:
    """Rutas de los artefactos de un doblaje."""
    root: Path

    @property
    def report(self) -> Path:
        return self.root / "report.json"

    @property
    def stems(self) -> Path:
        return self.root / "stems"

    def cue(self, index: int) -> Path:
        return self.root / "cues" / f"{index:04d}.wav"

    @property
    def subtitles(self) -> Path:
        return self.root / "dub.srt"

    @property
    def video(self) -> Path:
        return self.root / "dub.mp4"


@dataclass(frozen=True, slots=True)
class Voice:
    """Voz clonada de un hablante: fragmento de referencia, su audio y el handle del motor."""
    span: ReferenceSpan
    clip: AudioClip
    handle: object


class _NullProfiler:
    def measure(self, name: str):
        return nullcontext()


def _noop(stage: str, fraction: float) -> None:
    return None


def _release(*components: object) -> None:
    """Libera la GPU de los componentes que ya terminaron su etapa (si saben hacerlo)."""
    for component in components:
        if isinstance(component, Releasable):
            component.release()


class DubUseCase:
    def __init__(self, transcripts: TranscriptProvider, factory: DubComponentFactory,
                 policy: TimingPolicy = DEFAULT_TIMING,
                 verify: VerifyPolicy = DEFAULT_VERIFY) -> None:
        self._transcripts = transcripts
        self._factory = factory
        self._policy = policy
        self._verify = verify

    def execute(self, request: DubRequest, workdir: Path, profiler: Profiler | None = None,
                transcript: Transcript | None = None, progress: Progress = _noop,
                stems_dir: Path | None = None) -> DubReport:
        """`stems_dir` permite compartir la separación entre ejecuciones del mismo origen."""
        profiler = profiler or _NullProfiler()
        space = Workspace(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        with profiler.measure("dub.components"):
            # Un único proceso sirve todos los trabajos: un Whisper de una transcripción previa
            # más el del doblaje y OmniVoice desbordaban los 16 GB (verificación ~10x más lenta).
            self._factory.release_idle()
            pipe = self._factory.build(request)
        try:
            if transcript is None:
                progress("transcribir", 0.0)
                transcript = self._transcripts(request, profiler)
                self._factory.release_idle()  # Whisper de la transcripción: ~3.5 GB con large-v3
            cues = cues_from_transcript(transcript)
            if not cues:
                raise ValueError("La transcripción no tiene diálogo que doblar")
            source_language = transcript.language
            media_end = pipe.audio.duration(request.source)
            speech = speech_spans(cues)  # antes de fusionar: tiempos por palabra intactos

            stems = None
            if pipe.separator is not None:
                progress("separar", 0.1)
                with profiler.measure("dub.separate"):
                    stems = pipe.separator.separate(request.source, stems_dir or space.stems)
                _release(pipe.separator)
            voice_source = stems.voice if stems else request.source

            if request.diarize and pipe.diarizer is not None:
                progress("hablantes", 0.25)
                with profiler.measure("dub.diarize"):
                    # Ventanas alineadas a palabras: un cue puede contener un cambio de turno.
                    windows = diarization_windows(cues)
                    labels = pipe.diarizer.assign(voice_source, [(w.start, w.end) for w in windows],
                                                  request.max_speakers)
                    cues = split_by_speaker(cues, windows, labels)
                _release(pipe.diarizer)
            if request.merge_fragments:
                cues = merge_fragments(cues)

            progress("voz", 0.3)
            options = {speaker: reference_candidates(cues, speaker=speaker,
                                                     limit=request.reference_candidates)
                       for speaker in speakers_by_presence(cues)}
            with profiler.measure("dub.voice_prompt"):
                voices = {speaker: self._voice(pipe, voice_source, spans[0])
                          for speaker, spans in options.items()}

            progress("traducir", 0.35)
            with profiler.measure("dub.translate"):
                candidates = self._translate(pipe.translator, cues, source_language,
                                             request.target_language, request.n_best)
            _release(pipe.translator)

            with profiler.measure("dub.fit"):
                plan = self._plan(pipe.cloner, cues, candidates, media_end,
                                  request.target_language, voices, request)

            if request.reference_candidates > 1 and pipe.verifier and pipe.speaker_encoder:
                progress("voz", 0.4)
                with profiler.measure("dub.reference_probe"):
                    chosen = self._probe_references(pipe, request, plan, options, voices,
                                                    voice_source)
                if any(chosen[s].span != voices[s].span for s in voices):
                    voices = chosen
                    with profiler.measure("dub.fit"):
                        plan = self._plan(pipe.cloner, cues, candidates, media_end,
                                          request.target_language, voices, request)
                if not request.evaluate:
                    _release(pipe.speaker_encoder)

            progress("sintetizar", 0.45)
            with profiler.measure("dub.tts.generate"):
                clips = self._synthesize(pipe.cloner, plan, request, voices, progress,
                                         pipe=pipe, voice_source=voice_source)
            if request.verify_attempts and pipe.verifier is not None:
                progress("verificar", 0.8)
                with profiler.measure("dub.verify"):
                    plan = self._verify_and_retry(pipe, request, plan, clips, voices)
            _release(pipe.cloner, pipe.verifier)
            track = tuple(replace(item, generated_seconds=round(clips[item.index].duration, 3))
                          for item in plan)
            with profiler.measure("dub.save_cues"):
                for item in track:
                    pipe.audio.save(clips[item.index], space.cue(item.index))

            progress("mezclar", 0.85)
            outputs = self._render(pipe, request, space, track, clips, stems, media_end, profiler,
                                   speech)
            main = next(iter(voices.values()))
            report = DubReport(track, sync_metrics(track, self._policy), main.span, outputs,
                               references={s: v.span for s, v in voices.items()}, speech=speech)

            if request.evaluate and pipe.evaluator is not None:
                progress("evaluar", 0.92)
                with profiler.measure("dub.eval"):
                    report = self._evaluate(pipe, request, report, clips, voices, voice_source,
                                            stems, profiler)
                _release(pipe.evaluator)
            save_report(report, space.report)
            progress("listo", 1.0)
            return report
        except BaseException:
            _release(pipe.separator, pipe.diarizer, pipe.translator, pipe.cloner,
                     pipe.verifier, pipe.evaluator, pipe.speaker_encoder)
            raise


    def regenerate(self, request: DubRequest, workdir: Path, index: int, text: str,
                   profiler: Profiler | None = None, *, report: DubReport | None = None,
                   publish: bool = True) -> DubReport:
        """Construye una revisión aislada; nunca sobrescribe los artefactos activos.

        La CLI publica report.json atómicamente. La API usa publish=False y activa
        el resultado mediante una transacción SQLite después de completar la revisión.
        """
        profiler = profiler or _NullProfiler()
        report = report or load_report(Workspace(workdir).report)
        current = next((c for c in report.track if c.index == index), None)
        if current is None:
            raise ValueError(f"Cue inexistente: {index}")
        if not text.strip():
            raise ValueError("El texto no puede estar vacío")
        space = Workspace(workdir / "revisions" / uuid4().hex)
        source_cues = Path(report.outputs.get("cues") or workdir / "cues")
        space.cue(index).parent.mkdir(parents=True)
        for cue in report.track:
            if cue.index != index:
                shutil.copy2(source_cues / f"{cue.index:04d}.wav", space.cue(cue.index))
        self._factory.release_idle()
        pipe = self._factory.build(request)
        try:
            revised = self._regenerate(request, space, index, text.strip(), profiler,
                                       report, current, pipe)
        finally:
            _release(pipe.cloner, pipe.verifier, pipe.evaluator, pipe.speaker_encoder)
        if publish:
            save_report(revised, Workspace(workdir).report)
        return revised

    def _regenerate(self, request: DubRequest, space: Workspace, index: int, text: str,
                    profiler: Profiler, report: DubReport, current: DubbedCue,
                    pipe: DubPipeline) -> DubReport:
        voice_source = Path(report.outputs.get("voice_stem") or request.source)
        with profiler.measure("dub.voice_prompt"):
            voice = self._voice(pipe, voice_source, report.reference_for(current.speaker))
        with profiler.measure("dub.fit"):
            natural = pipe.cloner.natural_seconds([text], request.target_language,
                                                  voice.handle)[0]
            decision = fit_decision(natural, current.slot_seconds, self._policy)
        with profiler.measure("dub.tts.generate"):
            clip = pipe.cloner.synthesize([text], [decision.target_duration],
                                          request.target_language, voice.handle)[0]
        pipe.audio.save(clip, space.cue(index))
        updated = replace(current, target_text=text, natural_seconds=round(natural, 3),
                          target_seconds=round(decision.target_duration, 3),
                          generated_seconds=round(clip.duration, 3),
                          speed=round(decision.speed, 3), action=decision.action)
        track = tuple(updated if c.index == index else c for c in report.track)
        with profiler.measure("dub.load_cues"):
            clips = {c.index: (clip if c.index == index
                               else pipe.audio.load(space.cue(c.index), pipe.cloner.sample_rate))
                     for c in track}
        stems = None
        if report.outputs.get("background_stem"):
            stems = Stems(voice_source, Path(report.outputs["background_stem"]))
        media_end = pipe.audio.duration(request.source)
        outputs = self._render(pipe, request, space, track, clips, stems, media_end, profiler,
                               report.speech)
        cue_quality = report.cue_quality
        if pipe.evaluator is not None and cue_quality:
            with profiler.measure("dub.eval"):
                fresh = pipe.evaluator.evaluate_cues((updated,), {index: clip}, voice.clip,
                                                     request.target_language,
                                                     {index: pipe.audio.load(
                                                         voice_source, pipe.cloner.sample_rate,
                                                         current.start,
                                                         current.start + current.slot_seconds,
                                                     )})[0]
            cue_quality = tuple(fresh if q.index == index else q for q in cue_quality)
        sync = sync_metrics(track, self._policy)
        quality = report.quality
        gates = report.gates
        if pipe.evaluator is not None and quality is not None:
            quality = pipe.evaluator.summarize(cue_quality, track, quality,
                                               _float(outputs.get("lufs")),
                                               _float(outputs.get("true_peak")))
            gates = pipe.evaluator.gates(request.target_language, sync, quality)
        new_report = replace(report, track=track, sync=sync, outputs=outputs,
                             cue_quality=cue_quality, quality=quality, gates=gates)
        save_report(new_report, space.report)
        return new_report

    # --- etapas ------------------------------------------------------------------------

    @staticmethod
    def _translate(translator: Translator, cues: Sequence[DubCue], source: str, target: str,
                   n_best: int) -> list[list[Candidate]]:
        if source == target:
            return [[Candidate(c.source_text, 0.0)] for c in cues]
        return translator.translate([c.source_text for c in cues], source, target, n_best)

    @staticmethod
    def _voice(pipe: DubPipeline, source: Path, span: ReferenceSpan) -> Voice:
        clip = pipe.audio.load(source, pipe.cloner.sample_rate, span.start, span.end)
        return Voice(span, clip, pipe.cloner.prepare_voice(clip, span.text))

    def _cer(self, verifier: ClipTranscriber, texts: Sequence[str],
             clips: Sequence[AudioClip], language: str) -> list[float]:
        heard = verifier.transcribe_clips(clips, language)
        rates = []
        for text, hypothesis in zip(texts, heard, strict=True):
            rate = evaluate(text, hypothesis, language)["normalized"]["cer"]["rate"] \
                if text.strip() else 0.0
            rates.append(1.0 if rate is None else rate)
        return rates

    def _probe_references(self, pipe: DubPipeline, request: DubRequest,
                          plan: Sequence[DubbedCue], options: dict[str, list[ReferenceSpan]],
                          voices: dict[str, Voice], source: Path) -> dict[str, Voice]:
        """Elige por hablante la referencia que mejor clona la voz en unas frases de muestra:
        parecido con el original en la misma ranura menos CER, penalizando el ritmo que deja
        más cues sin caber. La referencia cambia el parecido de 0.17 a 0.37 y el CER de 2.9 % a
        6.7 % (la cercanía al centroide no lo predice); con 8 cues de muestra y sin penalizar el
        ritmo, el sondeo empeoraba el resultado (SIM por ranura 0.32 → 0.22)."""
        chosen = dict(voices)
        for speaker, spans in options.items():
            if len(spans) < 2:
                continue
            own = [c for c in plan if c.speaker == speaker]
            sample = [own[i] for i in probe_indices([c.target_seconds for c in own],
                                                     self._verify)]
            originals = [pipe.audio.load(source, pipe.cloner.sample_rate, c.start,
                                         c.start + c.slot_seconds) for c in sample]
            best: tuple[float, Voice] | None = None
            tight_default: float | None = None
            for span in spans:
                voice = voices[speaker] if span == voices[speaker].span \
                    else self._voice(pipe, source, span)
                clips = pipe.cloner.synthesize([c.target_text for c in sample],
                                               [c.target_seconds for c in sample],
                                               request.target_language, voice.handle)
                weights = [c.duration for c in clips]
                similarity = sum(w * pipe.speaker_encoder.similarity(o, [c])[0] for o, c, w
                                 in zip(originals, clips, weights, strict=True)) / sum(weights)
                cer = sum(self._cer(pipe.verifier, [c.target_text for c in sample], clips,
                                    request.target_language)) / len(clips)
                # El ritmo de la referencia cambia la duración natural estimada: una voz más
                # lenta deja más cues sin caber. Se penaliza el aumento frente a la primera.
                natural = pipe.cloner.natural_seconds([c.target_text for c in own],
                                                      request.target_language, voice.handle)
                tight = sum(n / c.slot_seconds > self._policy.max_speed
                            for n, c in zip(natural, own, strict=True)) / len(own)
                tight_default = tight if tight_default is None else tight_default
                quality = (pipe.audio.check_audio_quality(voice.clip)
                           if hasattr(pipe.audio, "check_audio_quality") else None)
                sat_penalty = 0.5 if (quality and quality.is_saturated) else 0.0
                quiet_penalty = 0.3 if (quality and quality.is_too_quiet) else 0.0
                score = (similarity - cer - max(0.0, tight - tight_default)
                         - sat_penalty - quiet_penalty)
                if best is None or score > best[0]:
                    best = (score, voice)
            chosen[speaker] = best[1]
        return chosen

    def _verify_and_retry(self, pipe: DubPipeline, request: DubRequest,
                          plan: Sequence[DubbedCue], clips: dict[int, AudioClip],
                          voices: dict[str, Voice]) -> list[DubbedCue]:
        """Regenera (variante 1, 2…) los cues que el verificador entiende mal y conserva la
        mejor versión. Video de prueba (175 cues): CER 4.75 % → 3.54 % (1 intento) en ~55 s:
        ~20 s de ASR aislado (fijo; empaquetar clips lo abarata 4× pero oculta errores y el CER
        solo baja a 4.2–4.7 %; beam 1, int8 o lotes mayores no aceleran) y ~35 s de reintentos."""
        language = request.target_language
        # TTS (PyTorch) y verificador (CTranslate2) nunca coinciden en la GPU: la caché de
        # activaciones de los lotes de TTS más Whisper large-v3 desbordaba los 16 GB (con lo
        # que ocupa el escritorio) y la etapa iba ~10x más lenta. Se alternan por fases.
        _release(pipe.cloner)
        rates = dict(zip((c.index for c in plan), self._cer(
            pipe.verifier, [c.target_text for c in plan], [clips[c.index] for c in plan],
            language), strict=True))
        attempts = {c.index: 0 for c in plan}
        for variant in range(1, request.verify_attempts + 1):
            failing = [c for c in plan if rates[c.index] > self._verify.max_cer]
            if not failing:
                break
            _release(pipe.verifier)
            retried: list[tuple[DubbedCue, AudioClip]] = []
            for speaker, voice in voices.items():
                group = sorted((c for c in failing if c.speaker == speaker),
                               key=lambda c: c.target_seconds)
                for first in range(0, len(group), request.tts_batch):
                    batch = group[first:first + request.tts_batch]
                    retried += zip(batch, pipe.cloner.synthesize(
                        [c.target_text for c in batch], [c.target_seconds for c in batch],
                        language, voice.handle, variant=variant), strict=True)
            _release(pipe.cloner)
            new_rates = self._cer(pipe.verifier, [c.target_text for c, _ in retried],
                                  [clip for _, clip in retried], language)
            for (cue, clip), rate in zip(retried, new_rates, strict=True):
                attempts[cue.index] = variant
                if rate < rates[cue.index]:
                    clips[cue.index], rates[cue.index] = clip, rate
        return [replace(c, attempts=attempts[c.index]) for c in plan]

    def _plan(self, cloner: VoiceCloner, cues: Sequence[DubCue],
              candidates: Sequence[Sequence[Candidate]], media_end: float, language: str,
              voices: dict[str, Voice], request: DubRequest | None = None) -> list[DubbedCue]:
        policy = (replace(self._policy, speed_penalty=request.speed_penalty)
                  if request else self._policy)
        # Duración natural con la voz de cada hablante (el ritmo depende de la referencia).
        natural: list[list[float]] = [[] for _ in cues]
        for speaker, voice in voices.items():
            positions = [i for i, c in enumerate(cues) if c.speaker == speaker]
            flat = [o.text for i in positions for o in candidates[i]]
            estimates = iter(cloner.natural_seconds(flat, language, voice.handle))
            for i in positions:
                natural[i] = [next(estimates) for _ in candidates[i]]
        # Huecos repartidos según la mejor traducción; después se elige candidato por ranura.
        slots = plan_slots(cues, policy, media_end, [n[0] for n in natural])
        plan = []
        for cue, options, durations, slot in zip(cues, candidates, natural, slots, strict=True):
            chosen = choose_candidate(options, durations, slot.duration, policy)
            decision = fit_decision(durations[chosen], slot.duration, policy)
            plan.append(DubbedCue(
                index=cue.index, start=slot.start, source_text=cue.source_text,
                target_text=options[chosen].text, slot_seconds=round(slot.duration, 3),
                natural_seconds=round(durations[chosen], 3),
                target_seconds=round(decision.target_duration, 3), generated_seconds=0.0,
                speed=round(decision.speed, 3), action=decision.action, speaker=cue.speaker,
                candidates=tuple(o.text for o in options),
            ))
        return plan

    @staticmethod
    def _synthesize(cloner: VoiceCloner, plan: Sequence[DubbedCue], request: DubRequest,
                    voices: dict[str, Voice], progress: Progress,
                    pipe: DubPipeline | None = None,
                    voice_source: Path | None = None) -> dict[int, AudioClip]:
        clips: dict[int, AudioClip] = {}
        for speaker, voice in voices.items():
            # Cubetas por duración: el coste de un lote lo marca su elemento más largo.
            ordered = sorted((c for c in plan if c.speaker == speaker),
                             key=lambda c: c.target_seconds)
            for first in range(0, len(ordered), request.tts_batch):
                batch = ordered[first:first + request.tts_batch]
                # Referencia por frase (Fase 5): si está activada y el cue dura >= 2.0 s
                if request.per_phrase_reference and pipe and voice_source:
                    prompt_list = []
                    for c in batch:
                        p_handle = voice.handle
                        if c.slot_seconds >= 2.0:
                            try:
                                phrase_clip = pipe.audio.load(voice_source, cloner.sample_rate,
                                                              c.start, c.start + c.slot_seconds)
                                q = (pipe.audio.check_audio_quality(phrase_clip)
                                     if hasattr(pipe.audio, "check_audio_quality") else None)
                                if not q or (not q.is_saturated and not q.is_too_quiet):
                                    p_handle = cloner.prepare_voice(phrase_clip, c.source_text)
                            except Exception:
                                p_handle = voice.handle
                        prompt_list.append(p_handle)
                    voice_arg = prompt_list if len(prompt_list) > 1 else prompt_list[0]
                else:
                    voice_arg = voice.handle

                generated = cloner.synthesize([c.target_text for c in batch],
                                              [c.target_seconds for c in batch],
                                              request.target_language, voice_arg)
                for c, clip in zip(batch, generated, strict=True):
                    # Transferencia de prosodia y energía (Fase 5)
                    if request.prosody_transfer and pipe and pipe.prosody and voice_source:
                        try:
                            orig_clip = pipe.audio.load(voice_source, cloner.sample_rate,
                                                        c.start, c.start + c.slot_seconds)
                            target_p = pipe.prosody.analyze(orig_clip)
                            clip = pipe.prosody.transfer_energy(clip, target_p)
                        except Exception:
                            pass
                    clips[c.index] = clip
                progress("sintetizar", 0.45 + 0.4 * len(clips) / len(plan))
        return clips

    @staticmethod
    def _render(pipe: DubPipeline, request: DubRequest, space: Workspace,
                track: Sequence[DubbedCue], clips: dict[int, AudioClip], stems,
                media_end: float, profiler: Profiler,
                speech: Sequence[Span] = ()) -> dict[str, str]:
        # Sin intervalos de habla (reportes antiguos) no se sabe qué parte del stem es diálogo.
        ambience = (Ambience(stems.voice, ambience_spans(speech, media_end))
                    if stems and speech else None)
        with profiler.measure("dub.mix"):
            mix = pipe.mixer.render([(c.start, clips[c.index]) for c in track],
                                    stems.background if stems else None, media_end, space.root,
                                    voice_reference=stems.voice if stems else None,
                                    ambience=ambience)
        transcript = Transcript(
            tuple(Segment(c.start, max(c.end, c.start + 0.05), c.target_text) for c in track),
            request.target_language, request.tts_engine, media_end,
        )
        space.subtitles.write_text(pipe.subtitles.render(transcript), encoding="utf-8")
        outputs = {"voice": str(mix.voice), "mix": str(mix.mix),
                   "subtitles": str(space.subtitles), "cues": str(space.root / "cues")}
        if stems:
            outputs |= {"voice_stem": str(stems.voice), "background_stem": str(stems.background)}
        if mix.bed is not None:
            outputs["bed"] = str(mix.bed)
        if mix.lufs is not None:
            outputs["lufs"] = str(mix.lufs)
        if mix.true_peak is not None:
            outputs["true_peak"] = str(mix.true_peak)
        if pipe.muxer is not None and request.source.suffix.lower() in VIDEO_SUFFIXES:
            with profiler.measure("dub.mux"):
                outputs["video"] = str(pipe.muxer.mux(request.source, mix.mix, space.subtitles,
                                                      request.target_language, space.video))
        return outputs

    @staticmethod
    def _evaluate(pipe: DubPipeline, request: DubRequest, report: DubReport,
                  clips: dict[int, AudioClip], voices: dict[str, Voice], voice_source: Path,
                  stems, profiler: Profiler) -> DubReport:
        evaluator = pipe.evaluator
        assert evaluator is not None
        slots = {c.index: pipe.audio.load(voice_source, pipe.cloner.sample_rate, c.start,
                                          c.start + c.slot_seconds) for c in report.track}
        by_index: dict[int, CueQuality] = {}
        with profiler.measure("dub.eval.cues"):
            for speaker, voice in voices.items():  # cada voz frente a su propia referencia
                own = [c for c in report.track if c.speaker == speaker]
                by_index.update((q.index, q) for q in evaluator.evaluate_cues(
                    own, clips, voice.clip, request.target_language, slots))
        cue_quality = [by_index[c.index] for c in report.track]
        with profiler.measure("dub.eval.original"):
            # Línea base del propio medio: la voz original en las mismas ranuras, salvo las que
            # forman la referencia (se parecerían a sí mismas).
            originals, similarities = [], []
            for speaker, voice in voices.items():
                ref = voice.span
                own = [slots[c.index] for c in report.track if c.speaker == speaker
                       and (c.start + c.slot_seconds <= ref.start or c.start >= ref.end)]
                originals += own
                similarity = evaluator.mean_similarity(voice.clip, own)
                if similarity is not None:
                    similarities.append((similarity, sum(c.duration for c in own)))
            weight = sum(w for _, w in similarities)
            baseline = QualitySummary(
                mos_original=evaluator.mean_mos(originals),
                similarity_original=round(sum(s * w for s, w in similarities) / weight, 4)
                if weight else None)
        if stems is not None:
            with profiler.measure("dub.eval.background"):
                # Se mide lo que realmente suena bajo la voz doblada (fondo + ambiente).
                bed = Path(report.outputs.get("bed") or stems.background)
                baseline = replace(baseline, background_wpm=evaluator.background_wpm(bed))
        quality = evaluator.summarize(cue_quality, report.track, baseline,
                                      _float(report.outputs.get("lufs")),
                                      _float(report.outputs.get("true_peak")))
        gates = evaluator.gates(request.target_language, report.sync, quality)
        return replace(report, quality=quality, cue_quality=tuple(cue_quality), gates=gates)


def _float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


# --- persistencia del informe ---------------------------------------------------------------


def report_to_dict(report: DubReport) -> dict:
    data = asdict(report)
    data["gates"] = [{**asdict(g), "passed": g.passed} for g in report.gates]
    data["schema_version"] = 1
    return data


def save_report(report: DubReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".report-", suffix=".json", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report_to_dict(report), stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def report_from_dict(data: dict) -> DubReport:
    track = tuple(DubbedCue(**{**c, "action": FitAction(c["action"]),
                               "candidates": tuple(c.get("candidates", ()))})
                  for c in data["track"])
    quality = QualitySummary(**data["quality"]) if data.get("quality") else None
    gates = tuple(Gate(g["name"], g["value"], g["threshold"], g["higher_is_better"])
                  for g in data.get("gates", ()))
    return DubReport(
        track=track, sync=SyncReport(**data["sync"]),
        reference=ReferenceSpan(**data["reference"]), outputs=dict(data.get("outputs", {})),
        quality=quality, cue_quality=tuple(CueQuality(**q) for q in data.get("cue_quality", ())),
        gates=gates, references={s: ReferenceSpan(**r)
                                 for s, r in (data.get("references") or {}).items()},
        speech=tuple((a, b) for a, b in data.get("speech", ())),
    )


def load_report(path: Path) -> DubReport:
    return report_from_dict(json.loads(path.read_text(encoding="utf-8")))

