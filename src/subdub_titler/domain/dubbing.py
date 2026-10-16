"""Doblaje: reglas puras de tiempos, elección de traducción, sincronía y umbrales de calidad.

Solo stdlib: el audio se representa con `AudioClip`, cuyo contenido (`samples`) es opaco para el
dominio (cualquier secuencia con `len`, p. ej. un array de numpy creado por un adaptador).
"""
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from .models import Transcript, Word

DUB_LANGUAGES = ("en", "es", "ja", "zh")


@dataclass(frozen=True, slots=True)
class AudioClip:
    samples: Any
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True, slots=True)
class DubCue:
    index: int
    start: float
    end: float
    source_text: str
    target_text: str = ""
    speaker: str = "S0"
    words: tuple[Word, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start


def cues_from_transcript(transcript: Transcript) -> tuple[DubCue, ...]:
    return tuple(DubCue(i, s.start, s.end, s.text.strip(), words=s.words)
                 for i, s in enumerate(s for s in transcript.segments if s.text.strip()))


Span = tuple[float, float]


def speech_spans(cues: Sequence[DubCue], padding: float = 0.15,
                 bridge: float = 0.3) -> tuple[Span, ...]:
    """Intervalos con habla original: cada palabra (o el cue entero si no hay tiempos por
    palabra) ampliada `padding` por lado; huecos menores que `bridge` se unen."""
    raw = sorted((w.start - padding, w.end + padding) for c in cues
                 for w in (c.words or (Word(c.start, c.end, c.source_text),)))
    merged: list[list[float]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] < bridge:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([max(0.0, start), end])
    return tuple((round(a, 3), round(b, 3)) for a, b in merged)


def ambience_spans(speech: Sequence[Span], duration: float,
                   min_seconds: float = 0.25) -> tuple[Span, ...]:
    """Complemento de `speech` en [0, duration]: donde el stem de voz no tiene diálogo
    (pájaros, pasos, respiraciones…) y debe conservarse en la mezcla."""
    spans, cursor = [], 0.0
    for start, end in sorted(speech):
        if start - cursor >= min_seconds:
            spans.append((round(cursor, 3), round(min(start, duration), 3)))
        cursor = max(cursor, end)
    if duration - cursor >= min_seconds:
        spans.append((round(cursor, 3), round(duration, 3)))
    return tuple(spans)


@dataclass(frozen=True, slots=True)
class Ambience:
    """Partes de una pista que se conservan en la mezcla (`spans` en segundos)."""
    source: Path
    spans: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class MergePolicy:
    """Une fragmentos de frase antes de traducir y sintetizar.

    La segmentación de subtítulos corta por duración/caracteres y deja fragmentos ("So",
    "embarrassing.") en ranuras de < 1 s donde ninguna traducción cabe y la voz sale sin prosodia.
    Un cue se une al siguiente si no termina frase (o es muy corto) y el hueco es breve.
    """
    min_seconds: float = 1.2  # un cue más corto se une aunque termine frase
    max_gap: float = 0.4
    max_seconds: float = 8.0
    max_chars: int = 160


DEFAULT_MERGE = MergePolicy()


def merge_fragments(cues: Sequence[DubCue],
                    policy: MergePolicy = DEFAULT_MERGE) -> tuple[DubCue, ...]:
    """Fusiona hacia delante cues incompletos o muy cortos con el siguiente del mismo hablante.
    Conserva palabras con tiempos y reasigna índices consecutivos."""
    def complete(cue: DubCue) -> bool:
        return cue.source_text.rstrip().endswith(SENTENCE_END)

    merged: list[DubCue] = []
    for cue in cues:
        if merged:
            last = merged[-1]
            joinable = (last.speaker == cue.speaker
                        and cue.start - last.end <= policy.max_gap
                        and cue.end - last.start <= policy.max_seconds
                        and len(last.source_text) + len(cue.source_text) < policy.max_chars)
            wants = not complete(last) or last.duration < policy.min_seconds \
                or cue.duration < policy.min_seconds
            if joinable and wants:
                merged[-1] = replace(last, end=cue.end, words=last.words + cue.words,
                                     source_text=f"{last.source_text} {cue.source_text}".strip())
                continue
        merged.append(cue)
    return tuple(replace(c, index=i) for i, c in enumerate(merged))


@dataclass(frozen=True, slots=True)
class VerifyPolicy:
    """Reintento de cues poco inteligibles: si un ASR (distinto del que mide la calidad final)
    entiende mal la voz generada, se regenera con otra variante y se conserva la mejor."""
    max_cer: float = 0.15
    probe_cues: int = 16  # cues de muestra para elegir la referencia de voz
    probe_min_seconds: float = 2.0  # el parecido de voz de clips cortos es poco fiable
    probe_max_seconds: float = 6.0


DEFAULT_VERIFY = VerifyPolicy()


def probe_indices(durations: Sequence[float], policy: VerifyPolicy = DEFAULT_VERIFY) -> list[int]:
    """Posiciones repartidas a lo largo del medio, con duración intermedia (las frases muy
    cortas o muy largas no representan bien la voz)."""
    eligible = [i for i, d in enumerate(durations)
                if policy.probe_min_seconds <= d <= policy.probe_max_seconds] \
        or list(range(len(durations)))
    if len(eligible) <= policy.probe_cues:
        return eligible
    step = len(eligible) / policy.probe_cues
    return [eligible[int(k * step)] for k in range(policy.probe_cues)]


@dataclass(frozen=True, slots=True)
class Window:
    """Tramo de un cue (palabras consecutivas) que se diariza como una unidad."""
    cue: int  # posición del cue en la secuencia
    first: int  # primera palabra (incluida)
    last: int  # última palabra (incluida)
    start: float
    end: float


def diarization_windows(cues: Sequence[DubCue], seconds: float = 1.2) -> list[Window]:
    """Ventanas de al menos `seconds` alineadas a palabras dentro de cada cue: permiten
    detectar un cambio de hablante en mitad de un cue (turnos con pausas cortas). Un cue sin
    palabras es una ventana; un resto corto se une a la ventana anterior del mismo cue."""
    windows: list[Window] = []
    for position, cue in enumerate(cues):
        if not cue.words:
            windows.append(Window(position, 0, -1, cue.start, cue.end))
            continue
        first = 0
        own: list[Window] = []
        for i, word in enumerate(cue.words):
            if word.end - cue.words[first].start >= seconds or i == len(cue.words) - 1:
                own.append(Window(position, first, i, cue.words[first].start, word.end))
                first = i + 1
        if len(own) > 1 and own[-1].end - own[-1].start < seconds / 2:
            tail = own.pop()
            own[-1] = Window(position, own[-1].first, tail.last, own[-1].start, tail.end)
        windows += own
    return windows


def split_by_speaker(cues: Sequence[DubCue], windows: Sequence[Window],
                     labels: Sequence[str]) -> tuple[DubCue, ...]:
    """Parte los cues donde cambia el hablante y reasigna índices consecutivos."""
    if len(windows) != len(labels):
        raise ValueError("Se necesita una etiqueta por ventana")
    result: list[DubCue] = []
    for position, cue in enumerate(cues):
        own = [(w, label) for w, label in zip(windows, labels, strict=True)
               if w.cue == position]
        if not own:
            continue
        runs: list[tuple[int, int, str]] = []  # (primera palabra, última, hablante)
        for window, label in own:
            if runs and runs[-1][2] == label:
                runs[-1] = (runs[-1][0], window.last, label)
            else:
                runs.append((window.first, window.last, label))
        if len(runs) == 1 or not cue.words:
            result.append(replace(cue, speaker=runs[0][2]))
            continue
        for first, last, label in runs:
            words = cue.words[first:last + 1]
            text = "".join(w.text for w in words).strip()
            if text:
                result.append(replace(cue, start=words[0].start,
                                      end=max(words[-1].end, words[0].start + 0.05),
                                      source_text=text, speaker=label, words=words))
    return tuple(replace(c, index=i) for i, c in enumerate(result))


@dataclass(frozen=True, slots=True)
class TimingPolicy:
    """Límites de ajuste de la voz sintetizada a la ranura del subtítulo."""
    min_speed: float = 0.85  # más lento suena arrastrado
    max_speed: float = 1.20  # más rápido pierde inteligibilidad: hay que acortar el texto
    # Silencio tomable tras el cue y adelanto del inicio. Con 0.5/0.25 s el 44 % de las frases
    # de un video real iba a más de max_speed (hasta ×2.2, voz atropellada y robótica); con
    # 1.5/0.4 s queda el 17 % (máx. ×1.8) a cambio de ~1 s de cola sobre el original.
    max_borrow: float = 1.50
    max_advance: float = 0.40
    min_gap: float = 0.05  # separación mínima entre voces consecutivas
    fill_slot: bool = False  # True: estirar hasta min_speed para llenar la ranura
    tolerance_s: float = 0.12
    tolerance_ratio: float = 0.05
    speed_penalty: float = 0.5  # penalización por exceso de velocidad al desempatar candidatos

    def __post_init__(self) -> None:
        if not 0 < self.min_speed <= 1 <= self.max_speed:
            raise ValueError("Se requiere 0 < min_speed <= 1 <= max_speed")
        if min(self.max_borrow, self.max_advance, self.min_gap) < 0:
            raise ValueError("max_borrow, max_advance y min_gap no pueden ser negativos")
        if self.speed_penalty < 0:
            raise ValueError("speed_penalty no puede ser negativo")


DEFAULT_TIMING = TimingPolicy()


@dataclass(frozen=True, slots=True)
class Slot:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_slots(cues: Sequence[DubCue], policy: TimingPolicy = DEFAULT_TIMING,
               media_end: float | None = None,
               natural: Sequence[float] | None = None) -> tuple[Slot, ...]:
    """Ranuras de voz: cada cue más el silencio que necesite de los huecos vecinos.

    Con `natural` (duración natural estimada por cue) se reparte en dos pasadas: primero cada
    cue se alarga sobre su hueco derecho (hasta `max_borrow`, con prioridad del anterior en un
    hueco compartido) y lo que aún falte se cubre adelantando su inicio (hasta `max_advance`).
    Sin estimación, cada cue solo toma hasta `max_borrow` al final. Nunca invade otro cue ni
    sale del medio, y una ranura nunca es más corta que su cue.
    """
    count = len(cues)
    if natural is not None and len(natural) != count:
        raise ValueError("Se necesita una duración natural por cue")
    need = [max(0.0, natural[i] - cues[i].duration) if natural is not None else policy.max_borrow
            for i in range(count)]
    starts = [c.start for c in cues]
    ends = [c.end for c in cues]
    for i in range(count):
        if i + 1 < count:
            limit = cues[i + 1].start - policy.min_gap
        else:
            limit = media_end if media_end is not None else cues[i].end + policy.max_borrow
        take = max(0.0, min(need[i], policy.max_borrow, limit - cues[i].end))
        ends[i] += take
        need[i] -= take
    if natural is not None:
        for i in range(count):
            floor = ends[i - 1] + policy.min_gap if i > 0 else 0.0
            advance = max(0.0, min(need[i], policy.max_advance, cues[i].start - floor))
            starts[i] -= advance
    return tuple(Slot(c.index, round(starts[i], 3), round(ends[i], 3))
                 for i, c in enumerate(cues))


class FitAction(StrEnum):
    OK = "ok"
    SHORTEN = "shorten"  # ni a velocidad máxima cabe: hace falta un texto más corto


@dataclass(frozen=True, slots=True)
class FitDecision:
    action: FitAction
    speed: float  # duración natural / duración objetivo
    target_duration: float


def fit_decision(natural_seconds: float, slot_seconds: float,
                 policy: TimingPolicy = DEFAULT_TIMING) -> FitDecision:
    """Duración a pedir al motor.

    Si la voz natural cabe, se respeta su ritmo (velocidad 1.0) y queda silencio al final;
    con `fill_slot` se estira hasta `min_speed`. Si no cabe, se comprime a la ranura; por
    encima de `max_speed` se marca para acortar el texto (y se fuerza como último recurso).
    """
    if natural_seconds <= 0 or slot_seconds <= 0:
        raise ValueError("Las duraciones deben ser positivas")
    stretched = natural_seconds / policy.min_speed if policy.fill_slot else natural_seconds
    target = min(slot_seconds, stretched)
    speed = natural_seconds / target
    action = FitAction.SHORTEN if speed > policy.max_speed + 1e-9 else FitAction.OK
    return FitDecision(action, speed, target)


@dataclass(frozen=True, slots=True)
class Candidate:
    text: str
    score: float  # log-probabilidad del traductor (mayor es mejor)


def choose_candidate(candidates: Sequence[Candidate], natural_seconds: Sequence[float],
                     slot_seconds: float, policy: TimingPolicy = DEFAULT_TIMING) -> int:
    """Índice de la traducción elegida: la mejor puntuada que cabe sin superar `max_speed`;
    si ninguna cabe, la que exige menor velocidad (normalmente la más corta).
    Con `speed_penalty > 0`, penaliza candidatos acelerados frente a los de ritmo natural.
    """
    if not candidates or len(candidates) != len(natural_seconds):
        raise ValueError("Candidatos y duraciones deben coincidir y no estar vacíos")
    fitting = [i for i in range(len(candidates))
               if natural_seconds[i] / slot_seconds <= policy.max_speed]
    if fitting:
        if policy.speed_penalty > 0.0:
            def score(i: int) -> float:
                speed = natural_seconds[i] / slot_seconds
                excess_speed = max(0.0, speed - 1.0)
                return candidates[i].score - policy.speed_penalty * excess_speed
            return max(fitting, key=score)
        return min(fitting, key=lambda i: -candidates[i].score)
    order = sorted(range(len(candidates)), key=lambda i: -candidates[i].score)
    return min(order, key=lambda i: natural_seconds[i])


@dataclass(frozen=True, slots=True)
class ReferenceSpan:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_rate(self) -> float:
        return _spoken_units(self.text) / self.duration if self.duration > 0 else 0.0


SENTENCE_END = (".", "?", "!", "。", "？", "！", "…")


def _spoken_units(text: str) -> float:
    """Unidades de habla aproximadas: letras/dígitos; un ideograma o kana pesa como ~3 letras."""
    units = 0.0
    for char in text:
        if "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff":
            units += 3.0
        elif char.isalnum():
            units += 1.0
    return units


def estimate_seconds_by_rate(text: str, reference_text: str, reference_seconds: float) -> float:
    """Duración natural con el ritmo del hablante de referencia (motores sin estimador)."""
    rate = _spoken_units(reference_text) / reference_seconds if reference_seconds > 0 else 0.0
    if rate <= 0:
        rate = 14.0  # ~14 letras/s, habla conversacional
    return max(0.3, _spoken_units(text) / rate)


def reference_quality_score(duration: float, text: str, optimal_duration: float = 6.0,
                            target_speech_rate: float = 14.0) -> float:
    """Puntuación de idoneidad como referencia de voz.
    Penaliza clips con ritmo anormalmente lento (< 8 unidades/s) o rápido (> 24)
    y duraciones muy alejadas de la zona óptima de OmniVoice (4.5–7.5 s).
    """
    rate = _spoken_units(text) / duration if duration > 0 else 0.0
    rate_penalty = max(0.0, 8.0 - rate) * 0.4 + max(0.0, rate - 24.0) * 0.2
    dur_penalty = abs(duration - optimal_duration) * 0.1
    return min(duration, optimal_duration) - dur_penalty - rate_penalty


def reference_candidates(cues: Sequence[DubCue], min_seconds: float = 3.0,
                         max_seconds: float = 10.0, max_pause: float = 1.0,
                         clean_gap: float = 0.25, margin: float = 0.2,
                         speaker: str | None = None, limit: int = 1) -> list[ReferenceSpan]:
    """Fragmentos de voz para clonar (OmniVoice recomienda 3–10 s), mejores primero.

    Preferencia: cues contiguos que empiezan y terminan en frontera de frase (texto acabado en
    puntuación y silencio de al menos `clean_gap` alrededor); entre ellos, los que tienen ritmo
    natural y duración adecuada. Una referencia cortada a media frase hace que el modelo la
    continúe en la voz generada. Con `speaker`, el fragmento solo contiene cues de ese hablante.
    Los candidatos no se solapan. El intervalo se amplía hasta `margin` en el silencio.
    """
    if not any(speaker is None or c.speaker == speaker for c in cues):
        raise ValueError("No hay voz para tomar como referencia")

    def gap_before(i: int) -> float:
        return cues[i].start - cues[i - 1].end if i > 0 else cues[i].start

    def gap_after(i: int) -> float:
        return cues[i + 1].start - cues[i].end if i + 1 < len(cues) else float("inf")

    def clean(first: int, last: int) -> bool:
        return (cues[last].source_text.rstrip().endswith(SENTENCE_END)
                and gap_after(last) >= clean_gap
                and (first == 0 or gap_before(first) >= clean_gap))

    def own(i: int) -> bool:
        return speaker is None or cues[i].speaker == speaker

    options: list[tuple[bool, float, int, int]] = []
    for first in range(len(cues)):
        if not own(first):
            continue
        for last in range(first, len(cues)):
            if not own(last) or (last > first and gap_before(last) > max_pause):
                break
            duration = cues[last].end - cues[first].start
            if duration > max_seconds:
                break
            if duration >= min_seconds:
                span_text = " ".join(cues[i].source_text for i in range(first, last + 1))
                score = reference_quality_score(duration, span_text)
                options.append((clean(first, last), score, first, last))
    options.sort(key=lambda o: (o[0], o[1]), reverse=True)
    chosen: list[tuple[int, int]] = []
    for _, _, first, last in options:
        if all(last < a or first > b for a, b in chosen):
            chosen.append((first, last))
        if len(chosen) == limit:
            break
    if not chosen:
        index = max((i for i in range(len(cues)) if own(i)), key=lambda i: cues[i].duration)
        chosen = [(index, index)]
    spans = []
    for first, last in chosen:
        start = cues[first].start - min(margin, max(0.0, gap_before(first)) / 2)
        end = cues[last].end + min(margin, gap_after(last) / 2)
        end = min(end, start + max_seconds + 2 * margin)
        spans.append(ReferenceSpan(round(max(0.0, start), 3), round(end, 3),
                                   " ".join(c.source_text for c in cues[first:last + 1])))
    return spans


def pick_reference(cues: Sequence[DubCue], min_seconds: float = 3.0, max_seconds: float = 10.0,
                   max_pause: float = 1.0, clean_gap: float = 0.25, margin: float = 0.2,
                   speaker: str | None = None) -> ReferenceSpan:
    """El mejor candidato de `reference_candidates`."""
    return reference_candidates(cues, min_seconds, max_seconds, max_pause, clean_gap, margin,
                                speaker, limit=1)[0]


def speakers_by_presence(cues: Sequence[DubCue]) -> list[str]:
    """Hablantes ordenados por tiempo de voz (el principal primero)."""
    totals: dict[str, float] = {}
    for cue in cues:
        totals[cue.speaker] = totals.get(cue.speaker, 0.0) + cue.duration
    return sorted(totals, key=lambda s: -totals[s])


@dataclass(frozen=True, slots=True)
class ProsodyProfile:
    """Perfil prosódico extraído de un segmento de habla original."""
    speech_rate: float  # unidades o caracteres hablados por segundo
    energy_rms: float  # nivel RMS promedio [0.0, 1.0]
    dynamic_range_db: float  # diferencia dB entre picos sonoros y base
    leading_pause: float  # silencio previo en segundos
    trailing_pause: float  # silencio posterior en segundos
    pitch_mean_hz: float | None = None  # frecuencia fundamental media estimada (Hz)
    pitch_std_hz: float | None = None  # desviación estándar tonal (expresividad)


@dataclass(frozen=True, slots=True)
class DubbedCue:
    index: int
    start: float
    source_text: str
    target_text: str
    slot_seconds: float
    natural_seconds: float
    target_seconds: float
    generated_seconds: float
    speed: float
    action: FitAction
    speaker: str = "S0"
    candidates: tuple[str, ...] = ()
    attempts: int = 0  # regeneraciones tras la verificación de inteligibilidad

    @property
    def error_seconds(self) -> float:
        return self.generated_seconds - self.target_seconds

    @property
    def end(self) -> float:
        return self.start + self.generated_seconds

    @property
    def is_rushed(self) -> bool:
        return self.speed > 1.15 or self.action is FitAction.SHORTEN


@dataclass(frozen=True, slots=True)
class SyncReport:
    cues: int
    within_tolerance: float  # fracción de cues con error dentro de tolerancia
    max_abs_error_ms: float
    mean_abs_error_ms: float
    overlaps: int
    speed_in_range: float  # fracción con velocidad en [min_speed, max_speed]
    speed_min: float
    speed_max: float
    shortened: int  # cues que no cupieron ni a velocidad máxima
    rushed: int = 0  # cues con velocidad excesiva (> 1.15) o que requirieron recorte


def sync_metrics(track: Sequence[DubbedCue],
                 policy: TimingPolicy = DEFAULT_TIMING) -> SyncReport:
    if not track:
        return SyncReport(0, 1.0, 0.0, 0.0, 0, 1.0, 1.0, 1.0, 0)
    errors = [abs(c.error_seconds) for c in track]
    within = sum(
        1 for c, e in zip(track, errors, strict=True)
        if e <= max(policy.tolerance_s, policy.tolerance_ratio * c.target_seconds)
    )
    ordered = sorted(track, key=lambda c: c.start)
    overlaps = sum(1 for a, b in zip(ordered, ordered[1:], strict=False)
                   if a.end > b.start + 1e-3)
    speeds = [c.speed for c in track]
    in_range = sum(1 for s in speeds if policy.min_speed - 1e-9 <= s <= policy.max_speed + 1e-9)
    rushed = sum(1 for c in track if c.speed > 1.15 or c.action is FitAction.SHORTEN)
    return SyncReport(
        cues=len(track), within_tolerance=within / len(track),
        max_abs_error_ms=round(max(errors) * 1000, 1),
        mean_abs_error_ms=round(sum(errors) / len(errors) * 1000, 1),
        overlaps=overlaps, speed_in_range=in_range / len(track),
        speed_min=round(min(speeds), 3), speed_max=round(max(speeds), 3),
        shortened=sum(1 for c in track if c.action is FitAction.SHORTEN),
        rushed=rushed,
    )


@dataclass(frozen=True, slots=True)
class DubRequest:
    source: Path
    target_language: str
    source_language: str = "auto"
    asr_model_id: str = "medium"
    device: str = "cuda"
    tts_engine: str = "omnivoice"
    translator: str = "nllb-600m"
    separator: str | None = "htdemucs"  # None = sin separación (mezcla sobre el original)
    num_step: int = 16
    tts_batch: int = 8
    n_best: int = 4
    evaluate: bool = False
    diarize: bool = False  # varios hablantes: una voz clonada por hablante
    max_speakers: int = 4
    merge_fragments: bool = True  # unir fragmentos de frase antes de traducir
    # Reintentos de cues poco inteligibles (0 = sin verificar). A/B: 1 intento CER 3.54 % en 55 s,
    # 2 intentos 3.44 % en 68 s (+0.1 puntos por +13 s): se queda en 1.
    verify_attempts: int = 1
    # Referencias sondeadas por hablante (1 = la mejor por regla). Medido en el video de prueba:
    # el sondeo eligió la misma que la regla (+42 s sin mejora); queda como opción.
    reference_candidates: int = 1
    # CFG dual (Cho & Lee, 2026): at (texto) y as (referencia).
    # None = usar guidance_scale estándar (equivalente a at=as=1+guidance_scale, típicamente 3.0).
    text_guidance: float | None = None
    reference_guidance: float | None = None
    # Accent Analogy Guidance (Cho & Lee, 2026): beta y carpeta de proxies
    accent_guidance: float = 0.0
    accent_proxy_dir: Path | None = None
    # Adaptación de duración y traducción (Fase 6)
    speed_penalty: float = 0.5  # peso de penalización por habla acelerada
    concise_translation: bool = False  # generar traducciones más breves
    # Expresividad y prosodia por frase (Fase 5)
    per_phrase_reference: bool = False  # usar fragmento de cada frase como prompt auxiliar
    prosody_transfer: bool = False  # transferir energía y dinámica original

    def __post_init__(self) -> None:
        if self.target_language not in DUB_LANGUAGES:
            raise ValueError("Idioma destino no soportado")
        if self.source_language not in ("auto", *DUB_LANGUAGES):
            raise ValueError("Idioma origen no soportado")
        if not 1 <= self.num_step <= 64:
            raise ValueError("num_step debe estar entre 1 y 64")
        if not 1 <= self.tts_batch <= 64:
            raise ValueError("tts_batch debe estar entre 1 y 64")
        if not 1 <= self.n_best <= 8:
            raise ValueError("n_best debe estar entre 1 y 8")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Dispositivo inválido")
        if not 1 <= self.max_speakers <= 16:
            raise ValueError("max_speakers debe estar entre 1 y 16")
        if self.text_guidance is not None and not 0.0 <= self.text_guidance <= 20.0:
            raise ValueError("text_guidance debe estar entre 0 y 20")
        if self.reference_guidance is not None and not 0.0 <= self.reference_guidance <= 20.0:
            raise ValueError("reference_guidance debe estar entre 0 y 20")
        if not 0.0 <= self.accent_guidance <= 50.0:
            raise ValueError("accent_guidance debe estar entre 0 y 50")
        if not 0 <= self.verify_attempts <= 5:
            raise ValueError("verify_attempts debe estar entre 0 y 5")
        if not 1 <= self.reference_candidates <= 8:
            raise ValueError("reference_candidates debe estar entre 1 y 8")
        if not 0.0 <= self.speed_penalty <= 10.0:
            raise ValueError("speed_penalty debe estar entre 0 y 10")
        if not isinstance(self.concise_translation, bool):
            raise ValueError("concise_translation debe ser booleano")
        if not isinstance(self.per_phrase_reference, bool):
            raise ValueError("per_phrase_reference debe ser booleano")
        if not isinstance(self.prosody_transfer, bool):
            raise ValueError("prosody_transfer debe ser booleano")


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    value: float | None
    threshold: float
    higher_is_better: bool = True

    @property
    def passed(self) -> bool | None:
        if self.value is None:
            return None
        return self.value >= self.threshold if self.higher_is_better \
            else self.value <= self.threshold


@dataclass(frozen=True, slots=True)
class DubThresholds:
    """Umbrales iniciales del plan; la naturalidad se mide relativa a la voz original
    porque el audio de origen (música, ruido) limita lo alcanzable."""
    max_cer_latin: float = 0.03
    max_cer_cjk: float = 0.08
    # SIM del doblaje / SIM de la propia voz original entre fragmentos del mismo medio: el
    # umbral absoluto del paper (mismo idioma, audio limpio) no aplica a doblaje entre idiomas.
    min_similarity_ratio: float = 0.85
    min_mos_delta: float = -0.1  # UTMOS(doblaje) - UTMOS(voz original); margen de ruido
    min_within_tolerance: float = 0.95
    max_overlaps: int = 0
    min_speed_in_range: float = 0.95
    max_background_wpm: float = 2.0
    target_lufs: float = -16.0
    lufs_tolerance: float = 1.0
    max_true_peak: float = -1.0


@dataclass(frozen=True, slots=True)
class QualitySummary:
    cer: float | None = None
    similarity: float | None = None
    similarity_original: float | None = None
    similarity_slot: float | None = None  # doblaje vs original en la misma ranura
    mos: float | None = None
    mos_original: float | None = None
    language_prob: float | None = None  # media de p(idioma destino): acento (sin umbral aún)
    background_wpm: float | None = None
    lufs: float | None = None
    true_peak: float | None = None


DEFAULT_THRESHOLDS = DubThresholds()


def quality_gates(language: str, sync: SyncReport, quality: QualitySummary,
                  thresholds: DubThresholds = DEFAULT_THRESHOLDS) -> tuple[Gate, ...]:
    max_cer = thresholds.max_cer_cjk if language in ("ja", "zh") else thresholds.max_cer_latin
    mos_delta = None if quality.mos is None or quality.mos_original is None \
        else round(quality.mos - quality.mos_original, 3)
    lufs_error = None if quality.lufs is None \
        else round(abs(quality.lufs - thresholds.target_lufs), 2)
    similarity_ratio = None if not quality.similarity or not quality.similarity_original \
        else round(quality.similarity / quality.similarity_original, 3)
    return (
        Gate("cer_ida_vuelta", quality.cer, max_cer, higher_is_better=False),
        Gate("similitud_voz_relativa", similarity_ratio, thresholds.min_similarity_ratio),
        Gate("utmos_vs_original", mos_delta, thresholds.min_mos_delta),
        Gate("sincronia", sync.within_tolerance, thresholds.min_within_tolerance),
        Gate("solapes", float(sync.overlaps), thresholds.max_overlaps, higher_is_better=False),
        Gate("velocidad_en_rango", sync.speed_in_range, thresholds.min_speed_in_range),
        Gate("fuga_voz_fondo_ppm", quality.background_wpm, thresholds.max_background_wpm,
             higher_is_better=False),
        Gate("error_lufs", lufs_error, thresholds.lufs_tolerance, higher_is_better=False),
        Gate("pico_real_dbtp", quality.true_peak, thresholds.max_true_peak,
             higher_is_better=False),
    )


@dataclass(frozen=True, slots=True)
class CueQuality:
    index: int
    cer: float | None = None
    cer_errors: int = 0
    cer_units: int = 0
    asr_text: str = ""
    similarity: float | None = None
    similarity_slot: float | None = None
    mos: float | None = None
    language_prob: float | None = None


@dataclass(frozen=True, slots=True)
class DubReport:
    track: tuple[DubbedCue, ...]
    sync: SyncReport
    reference: ReferenceSpan
    outputs: dict[str, str] = field(default_factory=dict)
    quality: QualitySummary | None = None
    cue_quality: tuple[CueQuality, ...] = ()
    gates: tuple[Gate, ...] = ()
    references: dict[str, ReferenceSpan] = field(default_factory=dict)  # por hablante
    speech: tuple[Span, ...] = ()  # habla original (el resto del stem de voz es ambiente)

    def reference_for(self, speaker: str) -> ReferenceSpan:
        return self.references.get(speaker, self.reference)
