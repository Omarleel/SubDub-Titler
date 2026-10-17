"""Detección y supresión de alucinaciones y utilidades de dynamic prompting (stdlib pura)."""
import re
import unicodedata
from collections.abc import Sequence

from .models import Segment

SINK_PHRASES_NORM = {
    # Español
    "gracias por ver el video", "gracias por ver", "gracias por mirar",
    "muchas gracias por ver", "muchas gracias por ver el video",
    "suscribete al canal", "suscribete", "suscribanse al canal",
    "dale like y suscribete", "no olvides suscribirte", "no olviden suscribirse",
    "subtitulos por", "subtitulos realizados por", "subtitulos de la comunidad",
    "nos vemos en el proximo video", "hasta el proximo video", "hasta la proxima",
    # Inglés
    "thank you for watching", "thanks for watching", "thank you for watching this video",
    "please subscribe", "subscribe to my channel", "like and subscribe",
    "don t forget to subscribe", "dont forget to subscribe",
    "subtitles by", "closed captions by", "captioned by", "amara org",
    "see you next time", "see you in the next video", "bye bye",
}

STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "este", "esta", "ese", "esa", "aquel",
    "mi", "tu", "su", "yo", "ella", "nosotros", "ellos",
    "pero", "aunque", "porque", "pues", "que", "como", "cuando", "donde",
    "y", "o", "si", "no", "bien", "muy", "mas", "tan", "ya", "ahora",
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her",
    "its", "our", "their", "i", "you", "he", "she", "it", "we", "they",
    "and", "or", "but", "so", "if", "when", "where", "how", "why", "what",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
}


def normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return re.sub(r"[^\w\s]", " ", ascii_text).strip()


def is_repetition_loop(text: str) -> bool:
    """Detecta si el texto está atrapado en un bucle autorregresivo de repetición."""
    words = [w.lower().strip(".,?!;:-_\"'()[]") for w in text.split() if w.strip()]
    if not words:
        return False

    # 1. Monopalabra repetida 3+ veces consecutivas (ej. "de de de", "la casa de de de campo")
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True

    # 2. Bigrama repetido 3+ veces consecutivas (6+ tokens, ej. "en la en la en la")
    if len(words) >= 6:
        for i in range(len(words) - 5):
            if (words[i] == words[i + 2] == words[i + 4] and
                words[i + 1] == words[i + 3] == words[i + 5]):
                return True

    # 3. Palabra dominante (60%+ de tokens iguales en 5+ palabras, ej. "sí claro sí sí sí")
    if len(words) >= 5:
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        top_count = max(counts.values())
        if top_count / len(words) >= 0.60:
            return True

    return False


def is_hallucination_sink_phrase(text: str) -> bool:
    """Detecta frases fantasma comunes generadas en silencios o música de fondo."""
    norm = normalize_text(text)
    norm = " ".join(norm.split())
    if not norm:
        return True
    if norm in SINK_PHRASES_NORM:
        return True
    sink_starts = (
        "subtitulos por", "subtitulos realizados por", "subtitulos de la comunidad",
        "subtitles by", "closed captions by", "captioned by",
    )
    if any(norm.startswith(s) for s in sink_starts):
        return True
    if "amara org" in norm:
        return True
    if ("gracias por ver" in norm or "thank you for watching" in norm) and len(norm) <= 60:
        return True
    if ("suscribete" in norm or "subscribe" in norm) and len(norm) <= 50:
        return True
    return False


def is_hallucination_segment(
    text: str,
    avg_logprob: float | None = None,
    no_speech_prob: float | None = None,
    compression_ratio: float | None = None,
) -> bool:
    """Clasifica si un segmento ASR es una alucinación sobre silencio o ruido."""
    clean = text.strip()
    # 1. Sin texto alfanumérico útil
    alphanumeric = re.sub(r"[^\w]", "", clean)
    if not alphanumeric:
        return True

    # 2. Salvaguarda de alta certeza acústica:
    # Si Whisper tiene certeza acústica alta, baja probabilidad de no-habla y compresión normal,
    # es habla legítima del usuario y nunca debe suprimirse.
    is_high_acoustic_confidence = (
        avg_logprob is not None and avg_logprob >= -0.50 and
        (no_speech_prob is None or no_speech_prob <= 0.35) and
        (compression_ratio is None or compression_ratio <= 2.2)
    )
    if is_high_acoustic_confidence:
        return False

    # 3. Bucle patológico por ratio de compresión zlib de Whisper (> 2.4)
    if compression_ratio is not None and compression_ratio > 2.4:
        return True

    # 4. Bucle n-gram detectado sintácticamente
    if is_repetition_loop(clean):
        # Si tiene certeza acústica moderada/alta, o el texto es rico en vocabulario variado
        # (ej. habla coloquial con 'yeah, yeah, yeah' dentro de una frase amplia),
        # NO se descarta como alucinación.
        words = [w.lower().strip(".,?!;:-_\"'()[]") for w in clean.split() if w.strip()]
        if avg_logprob is not None and avg_logprob >= -0.60 and (no_speech_prob is None or no_speech_prob <= 0.40):
            return False
        if len(words) >= 8 and (len(set(words)) / len(words)) >= 0.50:
            return False
        return True

    # 4. Sink phrases de entrenamiento
    if is_hallucination_sink_phrase(clean):
        if no_speech_prob is not None and no_speech_prob > 0.25:
            return True
        if avg_logprob is not None and avg_logprob < -0.60:
            return True
        # Si no hay métricas, la coincidencia estricta se descarta
        if normalize_text(clean) in SINK_PHRASES_NORM:
            return True

    # 5. Silencio / ruido puro con muy baja probabilidad de habla
    if no_speech_prob is not None and no_speech_prob > 0.70:
        if avg_logprob is None or avg_logprob < -0.75:
            return True

    return False


def extract_dynamic_prompt(segments: Sequence[Segment], max_terms: int = 25) -> str:
    """Extrae entidades, nombres propios y vocabulario clave de segmentos de alta confianza."""
    terms_count: dict[str, int] = {}

    for s in segments:
        if s.avg_logprob is not None and s.avg_logprob < -0.65:
            continue
        if s.no_speech_prob is not None and s.no_speech_prob > 0.45:
            continue

        words = s.text.split()
        for idx, raw_w in enumerate(words):
            clean = raw_w.strip(".,?!;:-_\"'()[]{}«»¿¡")
            if not clean:
                continue
            lower = clean.lower()
            if lower in STOPWORDS or len(clean) < 3:
                continue
            is_proper = clean[0].isupper() and (idx > 0 or len(clean) >= 4)
            is_long_term = len(clean) >= 6
            if is_proper or is_long_term:
                key = clean.capitalize()
                terms_count[key] = terms_count.get(key, 0) + 1

    sorted_terms = sorted(terms_count.items(), key=lambda x: x[1], reverse=True)
    selected = [term for term, _ in sorted_terms[:max_terms]]
    if not selected:
        return ""
    return ", ".join(selected)

