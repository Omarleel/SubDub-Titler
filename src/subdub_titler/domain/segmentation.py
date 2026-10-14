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


DEFAULT_POLICY = CuePolicy()


def _cue(words: list[Word], policy: CuePolicy) -> Segment:
    text = "".join(w.text for w in words).strip()
    start = words[0].start
    return Segment(start, max(words[-1].end, start + policy.min_duration), text)


def resegment(transcript: Transcript, policy: CuePolicy = DEFAULT_POLICY) -> Transcript:
    """Sin palabras con tiempos (motor sin soporte) devuelve el transcript intacto."""
    words = [w for s in transcript.segments for w in s.words if w.text.strip()]
    if not words:
        return transcript
    cues: list[Segment] = []
    current: list[Word] = []
    for word in words:
        if current:
            text = "".join(w.text for w in current + [word]).strip()
            if (word.start - current[-1].end > policy.max_pause
                    or word.end - current[0].start > policy.max_duration
                    or len(text) > policy.max_chars):
                cues.append(_cue(current, policy))
                current = []
        current.append(word)
    cues.append(_cue(current, policy))
    return replace(transcript, segments=tuple(cues))
