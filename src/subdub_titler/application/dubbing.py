"""Caso de uso de doblaje: orquesta puertos y reglas de dominio; no conoce ningún motor.

Etapas medidas como spans `dub.*` para el laboratorio de rendimiento. Los componentes concretos
llegan de una `DubComponentFactory` (inversión de dependencias): un motor TTS, traductor o
separador nuevo se registra en infraestructura sin tocar este módulo.
"""
import json
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from subdub_titler.application.dub_quality import DubEvaluator
from subdub_titler.domain.dubbing import (
    DEFAULT_TIMING,
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
    SyncReport,
    TimingPolicy,
    choose_candidate,
    cues_from_transcript,
    diarization_windows,
    fit_decision,
    pick_reference,
    plan_slots,
    speakers_by_presence,
    split_by_speaker,
    sync_metrics,
)
from subdub_titler.domain.models import Segment, Transcript
from subdub_titler.domain.ports import (
    AudioStore,
    Mixer,
    Muxer,
    Profiler,
    Releasable,
    SourceSeparator,
    SpeakerDiarizer,
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


class DubComponentFactory(Protocol):
    def build(self, request: DubRequest) -> DubPipeline: ...


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
                 policy: TimingPolicy = DEFAULT_TIMING) -> None:
        self._transcripts = transcripts
        self._factory = factory
        self._policy = policy

    def execute(self, request: DubRequest, workdir: Path, profiler: Profiler | None = None,
                transcript: Transcript | None = None, progress: Progress = _noop,
                stems_dir: Path | None = None) -> DubReport:
        """`stems_dir` permite compartir la separación entre ejecuciones del mismo origen."""
        profiler = profiler or _NullProfiler()
        space = Workspace(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        with profiler.measure("dub.components"):
            pipe = self._factory.build(request)
        if transcript is None:
            progress("transcribir", 0.0)
            transcript = self._transcripts(request, profiler)
        cues = cues_from_transcript(transcript)
        if not cues:
            raise ValueError("La transcripción no tiene diálogo que doblar")
        source_language = transcript.language
        media_end = pipe.audio.duration(request.source)

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

        progress("voz", 0.3)
        with profiler.measure("dub.voice_prompt"):
            voices = {speaker: self._voice(pipe, voice_source, pick_reference(cues,
                                                                            speaker=speaker))
                      for speaker in speakers_by_presence(cues)}

        progress("traducir", 0.35)
        with profiler.measure("dub.translate"):
            candidates = self._translate(pipe.translator, cues, source_language,
                                         request.target_language, request.n_best)
        _release(pipe.translator)

        with profiler.measure("dub.fit"):
            plan = self._plan(pipe.cloner, cues, candidates, media_end,
                              request.target_language, voices)

        progress("sintetizar", 0.45)
        with profiler.measure("dub.tts.generate"):
            clips = self._synthesize(pipe.cloner, plan, request, voices, progress)
        _release(pipe.cloner)
        track = tuple(replace(item, generated_seconds=round(clips[item.index].duration, 3))
                      for item in plan)
        with profiler.measure("dub.save_cues"):
            for item in track:
                pipe.audio.save(clips[item.index], space.cue(item.index))

        progress("mezclar", 0.85)
        outputs = self._render(pipe, request, space, track, clips, stems, media_end, profiler)
        main = next(iter(voices.values()))
        report = DubReport(track, sync_metrics(track, self._policy), main.span, outputs,
                           references={s: v.span for s, v in voices.items()})

        if request.evaluate and pipe.evaluator is not None:
            progress("evaluar", 0.92)
            with profiler.measure("dub.eval"):
                report = self._evaluate(pipe, request, report, clips, voices, voice_source,
                                        stems, profiler)
            _release(pipe.evaluator)
        save_report(report, space.report)
        progress("listo", 1.0)
        return report

    def regenerate(self, request: DubRequest, workdir: Path, index: int, text: str,
                   profiler: Profiler | None = None) -> DubReport:
        """Re-sintetiza un cue con otro texto y vuelve a mezclar, sin repetir el resto."""
        profiler = profiler or _NullProfiler()
        space = Workspace(workdir)
        report = load_report(space.report)
        current = next((c for c in report.track if c.index == index), None)
        if current is None:
            raise ValueError(f"Cue inexistente: {index}")
        if not text.strip():
            raise ValueError("El texto no puede estar vacío")
        pipe = self._factory.build(request)
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
        outputs = self._render(pipe, request, space, track, clips, stems, media_end, profiler)
        cue_quality = report.cue_quality
        if pipe.evaluator is not None and cue_quality:
            with profiler.measure("dub.eval"):
                fresh = pipe.evaluator.evaluate_cues((updated,), {index: clip}, voice.clip,
                                                     request.target_language)[0]
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

    def _plan(self, cloner: VoiceCloner, cues: Sequence[DubCue],
              candidates: Sequence[Sequence[Candidate]], media_end: float, language: str,
              voices: dict[str, Voice]) -> list[DubbedCue]:
        # Duración natural con la voz de cada hablante (el ritmo depende de la referencia).
        natural: list[list[float]] = [[] for _ in cues]
        for speaker, voice in voices.items():
            positions = [i for i, c in enumerate(cues) if c.speaker == speaker]
            flat = [o.text for i in positions for o in candidates[i]]
            estimates = iter(cloner.natural_seconds(flat, language, voice.handle))
            for i in positions:
                natural[i] = [next(estimates) for _ in candidates[i]]
        # Huecos repartidos según la mejor traducción; después se elige candidato por ranura.
        slots = plan_slots(cues, self._policy, media_end, [n[0] for n in natural])
        plan = []
        for cue, options, durations, slot in zip(cues, candidates, natural, slots, strict=True):
            chosen = choose_candidate(options, durations, slot.duration, self._policy)
            decision = fit_decision(durations[chosen], slot.duration, self._policy)
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
                    voices: dict[str, Voice], progress: Progress) -> dict[int, AudioClip]:
        clips: dict[int, AudioClip] = {}
        for speaker, voice in voices.items():
            # Cubetas por duración: el coste de un lote lo marca su elemento más largo.
            ordered = sorted((c for c in plan if c.speaker == speaker),
                             key=lambda c: c.target_seconds)
            for first in range(0, len(ordered), request.tts_batch):
                batch = ordered[first:first + request.tts_batch]
                generated = cloner.synthesize([c.target_text for c in batch],
                                              [c.target_seconds for c in batch],
                                              request.target_language, voice.handle)
                clips.update({c.index: clip for c, clip in zip(batch, generated, strict=True)})
                progress("sintetizar", 0.45 + 0.4 * len(clips) / len(plan))
        return clips

    @staticmethod
    def _render(pipe: DubPipeline, request: DubRequest, space: Workspace,
                track: Sequence[DubbedCue], clips: dict[int, AudioClip], stems,
                media_end: float, profiler: Profiler) -> dict[str, str]:
        with profiler.measure("dub.mix"):
            mix = pipe.mixer.render([(c.start, clips[c.index]) for c in track],
                                    stems.background if stems else None, media_end, space.root,
                                    voice_reference=stems.voice if stems else None)
        transcript = Transcript(
            tuple(Segment(c.start, max(c.end, c.start + 0.05), c.target_text) for c in track),
            request.target_language, request.tts_engine, media_end,
        )
        space.subtitles.write_text(pipe.subtitles.render(transcript), encoding="utf-8")
        outputs = {"voice": str(mix.voice), "mix": str(mix.mix), "subtitles": str(space.subtitles)}
        if stems:
            outputs |= {"voice_stem": str(stems.voice), "background_stem": str(stems.background)}
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
        by_index: dict[int, CueQuality] = {}
        with profiler.measure("dub.eval.cues"):
            for speaker, voice in voices.items():  # cada voz frente a su propia referencia
                own = [c for c in report.track if c.speaker == speaker]
                by_index.update((q.index, q) for q in evaluator.evaluate_cues(
                    own, clips, voice.clip, request.target_language))
        cue_quality = [by_index[c.index] for c in report.track]
        with profiler.measure("dub.eval.original"):
            # Línea base del propio medio: la voz original en las mismas ranuras, salvo las que
            # forman la referencia (se parecerían a sí mismas).
            originals, similarities = [], []
            for speaker, voice in voices.items():
                ref = voice.span
                own = [pipe.audio.load(voice_source, pipe.cloner.sample_rate, c.start,
                                       c.start + c.slot_seconds)
                       for c in report.track if c.speaker == speaker
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
                baseline = replace(baseline,
                                   background_wpm=evaluator.background_wpm(stems.background))
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
    path.write_text(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
                    encoding="utf-8")


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
    )


def load_report(path: Path) -> DubReport:
    return report_from_dict(json.loads(path.read_text(encoding="utf-8")))

