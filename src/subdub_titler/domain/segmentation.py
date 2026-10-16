"""Política de cues de subtítulo: corta por pausa, duración y longitud de texto.

Whisper devuelve segmentos de decodificación (hasta decenas de segundos y cruzando
silencios); un subtítulo legible necesita cues cortos alineados con la voz.
"""
from dataclasses import dataclass, replace

from .models import Segment, Transcript, Word


@dataclass(frozen=True, slots=True)
class CuePolicy:
    max_duration: float = 6.0
    max_chars: int = 84  # dos líneas de 42 caracteres
    max_pause: float = 0.6
    min_duration: float = 0.05
    pad_end: float = 0.12  # margen de 120 ms tras la última palabra para evitar cortes
    merge_subsequent: bool = True
    merge_max_pause: float = 1.2


DEFAULT_POLICY = CuePolicy()

TERMINAL_PUNCTUATION = ('.', '?', '!', '…', ':', ';')
CONNECTORS = {
    # Español
    "y", "e", "o", "u", "pero", "sino", "porque", "que", "cual", "quien",
    "donde", "cuando", "si", "entonces", "como", "para", "por", "con", "de",
    "en", "hacia", "hasta", "ni", "aunque", "mas",
    # Inglés
    "and", "or", "but", "so", "because", "that", "which", "who", "whom",
    "where", "when", "if", "then", "as", "than", "to", "for", "with",
    "at", "by", "from", "in", "on", "into", "onto", "nor", "yet",
}


def _is_continuation(word_text: str) -> bool:
    clean = word_text.strip()
    if not clean:
        return False
    if clean[0].islower():
        return True
    return clean.lower().rstrip(".,?!;:…") in CONNECTORS


def _cue(words: list[Word], policy: CuePolicy, next_start: float | None = None) -> Segment:
    text = "".join(w.text for w in words).strip()
    start = words[0].start
    end = words[-1].end + policy.pad_end
    if next_start is not None and end > next_start:
        end = next_start
    # Se conservan las palabras: el doblaje parte cues por cambio de hablante con ellas.
    return Segment(start, max(end, start + policy.min_duration), text, tuple(words))


def resegment(transcript: Transcript, policy: CuePolicy = DEFAULT_POLICY) -> Transcript:
    """Sin palabras con tiempos (motor sin soporte) devuelve el transcript intacto."""
    words = [w for s in transcript.segments for w in s.words if w.text.strip()]
    if not words:
        return transcript

    raw_cues: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            text = "".join(w.text for w in current + [word]).strip()
            if (word.start - current[-1].end > policy.max_pause
                    or word.end - current[0].start > policy.max_duration
                    or len(text) > policy.max_chars):
                raw_cues.append(current)
                current = []
        current.append(word)
    if current:
        raw_cues.append(current)

    # Fusión sintáctica: une fragmentos partidos por pausas menores a merge_max_pause
    # si la cláusula gramatical no concluyó y cabe en las restricciones de lectura.
    merged: list[list[Word]] = []
    for cue_words in raw_cues:
        if not merged or not policy.merge_subsequent:
            merged.append(cue_words)
            continue
        prev = merged[-1]
        prev_text = "".join(w.text for w in prev).strip()
        curr_text = "".join(w.text for w in cue_words).strip()
        combined_text = f"{prev_text} {curr_text}".strip()
        combined_duration = cue_words[-1].end - prev[0].start
        pause = cue_words[0].start - prev[-1].end

        if (pause <= policy.merge_max_pause
                and not prev_text.endswith(TERMINAL_PUNCTUATION)
                and _is_continuation(cue_words[0].text)
                and len(combined_text) <= policy.max_chars
                and combined_duration <= policy.max_duration):
            merged[-1] = prev + cue_words
        else:
            merged.append(cue_words)

    cues: list[Segment] = []
    for i, cue_words in enumerate(merged):
        next_start = merged[i + 1][0].start if i + 1 < len(merged) else None
        cues.append(_cue(cue_words, policy, next_start=next_start))
    return replace(transcript, segments=tuple(cues))
