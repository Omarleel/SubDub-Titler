"""Validación automática del doblaje: inteligibilidad (CER ida y vuelta), parecido de voz,
naturalidad relativa al original, acento (p del idioma destino), fuga de voz en el fondo y
umbrales.

Cada medidor es un puerto opcional: sin él, la métrica queda en `None` y su umbral no se evalúa.
"""
from collections.abc import Mapping, Sequence
from pathlib import Path

from subdub_titler.application.quality import evaluate
from subdub_titler.domain.dubbing import (
    DEFAULT_THRESHOLDS,
    AudioClip,
    CueQuality,
    DubbedCue,
    DubThresholds,
    Gate,
    QualitySummary,
    SyncReport,
    quality_gates,
)
from subdub_titler.domain.ports import (
    ClipTranscriber,
    LanguageIdentifier,
    NaturalnessPredictor,
    SpeakerEncoder,
    SpeechLeakMeter,
)


def corpus_cer(cues: Sequence[CueQuality]) -> float | None:
    """CER agregado (errores totales / caracteres totales), no media de tasas por cue."""
    units = sum(q.cer_units for q in cues)
    return round(sum(q.cer_errors for q in cues) / units, 4) if units else None


def _mean(values: Sequence[float | None], weights: Sequence[float] | None = None) -> float | None:
    """Media (ponderada si hay pesos) ignorando valores ausentes."""
    weights = weights or [1.0] * len(values)
    pairs = [(v, w) for v, w in zip(values, weights, strict=True) if v is not None and w > 0]
    total = sum(w for _, w in pairs)
    return round(sum(v * w for v, w in pairs) / total, 4) if total else None


class DubEvaluator:
    def __init__(self, transcriber: ClipTranscriber, speaker: SpeakerEncoder | None = None,
                 naturalness: NaturalnessPredictor | None = None,
                 leak_meter: SpeechLeakMeter | None = None,
                 language_id: LanguageIdentifier | None = None,
                 thresholds: DubThresholds = DEFAULT_THRESHOLDS) -> None:
        self._transcriber = transcriber
        self._speaker = speaker
        self._naturalness = naturalness
        self._leak_meter = leak_meter
        self._language_id = language_id
        self._thresholds = thresholds

    def evaluate_cues(self, track: Sequence[DubbedCue], clips: Mapping[int, AudioClip],
                      reference: AudioClip, language: str,
                      originals: Mapping[int, AudioClip] | None = None) -> list[CueQuality]:
        """`originals`: voz original en la ranura de cada cue → parecido por ranura, que no
        depende de la referencia elegida (permite comparar referencias de forma justa)."""
        ordered = [clips[c.index] for c in track]
        slot = [self._speaker.similarity(originals[c.index], [clips[c.index]])[0]
                if self._speaker and originals and c.index in originals else None
                for c in track]
        heard = self._transcriber.transcribe_clips(ordered, language)
        similarity = self._speaker.similarity(reference, ordered) if self._speaker else \
            [None] * len(track)
        mos = self._naturalness.mos(ordered) if self._naturalness else [None] * len(track)
        native = self._language_id.language_probability(ordered, language) \
            if self._language_id else [None] * len(track)
        result = []
        for cue, text, sim, score, same_slot, prob in zip(track, heard, similarity, mos, slot,
                                                          native, strict=True):
            counts = evaluate(cue.target_text, text, language)["normalized"]["cer"]
            errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
            result.append(CueQuality(
                index=cue.index, cer=None if counts["rate"] is None else round(counts["rate"], 4),
                cer_errors=errors, cer_units=counts["reference_units"], asr_text=text,
                similarity=None if sim is None else round(sim, 4),
                similarity_slot=None if same_slot is None else round(same_slot, 4),
                mos=None if score is None else round(score, 3),
                language_prob=None if prob is None else round(prob, 4),
            ))
        return result

    def mean_mos(self, clips: Sequence[AudioClip]) -> float | None:
        return _mean(self._naturalness.mos(clips)) if self._naturalness and clips else None

    def mean_similarity(self, reference: AudioClip, clips: Sequence[AudioClip]) -> float | None:
        """Parecido medio (ponderado por duración) de unos clips con la referencia."""
        if not self._speaker or not clips:
            return None
        return _mean(self._speaker.similarity(reference, clips), [c.duration for c in clips])

    def background_wpm(self, path: Path) -> float | None:
        """Última medición: los medidores de voz ya terminaron y se liberan antes de transcribir
        la pista entera (con todo cargado el pico llegaba a 12 GB)."""
        if not self._leak_meter:
            return None
        from subdub_titler.domain.ports import Releasable
        for part in (self._speaker, self._naturalness):
            if isinstance(part, Releasable):
                part.release()
        return round(self._leak_meter.words_per_minute(path), 2)

    def summarize(self, cue_quality: Sequence[CueQuality], track: Sequence[DubbedCue],
                  originals: QualitySummary, lufs: float | None,
                  true_peak: float | None) -> QualitySummary:
        """`originals` aporta las medidas de referencia (voz original, fondo)."""
        weights = {c.index: c.generated_seconds for c in track}
        return QualitySummary(
            cer=corpus_cer(cue_quality),
            # Embeddings de clips de 1 s son poco fiables: pesan menos que las frases largas.
            similarity=_mean([q.similarity for q in cue_quality],
                             [weights.get(q.index, 0.0) for q in cue_quality]),
            similarity_original=originals.similarity_original,
            similarity_slot=_mean([q.similarity_slot for q in cue_quality],
                                  [weights.get(q.index, 0.0) for q in cue_quality]),
            mos=_mean([q.mos for q in cue_quality]), mos_original=originals.mos_original,
            language_prob=_mean([q.language_prob for q in cue_quality]),
            background_wpm=originals.background_wpm, lufs=lufs, true_peak=true_peak,
        )

    def release(self) -> None:
        """Libera los medidores (el ASR de la fuga es el de transcripción: se conserva)."""
        from subdub_titler.domain.ports import Releasable
        for part in (self._transcriber, self._speaker, self._naturalness):
            if isinstance(part, Releasable):
                part.release()

    def gates(self, language: str, sync: SyncReport, quality: QualitySummary) -> tuple[Gate, ...]:
        return quality_gates(language, sync, quality, self._thresholds)
