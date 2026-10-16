"""Pruebas unitarias para detección de alucinaciones y dynamic prompting."""
from subdub_titler.domain.hallucinations import (
    extract_dynamic_prompt,
    is_hallucination_segment,
    is_hallucination_sink_phrase,
    is_repetition_loop,
    normalize_text,
)
from subdub_titler.domain.models import Segment


def test_normalize_text():
    assert normalize_text("¡Hola, Mundo! ¿Cómo estás?") == "hola  mundo   como estas"
    assert normalize_text("   Gracias por ver   ") == "gracias por ver"


def test_is_repetition_loop_single_word():
    assert is_repetition_loop("de de de") is True
    assert is_repetition_loop("la casa de de de campo") is True
    assert is_repetition_loop("hola, hola, hola!") is True
    assert is_repetition_loop("una casa bonita y grande") is False


def test_is_repetition_loop_bigram():
    assert is_repetition_loop("en la en la en la") is True
    assert is_repetition_loop("y luego y luego y luego") is True
    assert is_repetition_loop("por favor por favor por favor") is True
    assert is_repetition_loop("el rápido zorro marrón salta") is False


def test_is_repetition_loop_dominant_word():
    # 4 de 5 palabras son "si" -> 80% > 60%
    assert is_repetition_loop("sí claro sí sí sí") is True
    # Frase equilibrada
    assert is_repetition_loop("sí claro que iremos al parque") is False


def test_is_hallucination_sink_phrase():
    assert is_hallucination_sink_phrase("Gracias por ver el video.") is True
    assert is_hallucination_sink_phrase("¡Muchas gracias por ver!") is True
    assert is_hallucination_sink_phrase("Thank you for watching!") is True
    amara = "Subtítulos realizados por la comunidad de Amara.org"
    assert is_hallucination_sink_phrase(amara) is True
    assert is_hallucination_sink_phrase("Suscríbete al canal para más contenido.") is True
    assert is_hallucination_sink_phrase("Hoy vamos a hablar sobre física cuántica.") is False


def test_is_hallucination_segment_conditions():
    # Solo puntuación / vacío
    assert is_hallucination_segment("   ...   ") is True
    assert is_hallucination_segment("¿?") is True

    # Ratio de compresión zlib anómalo (> 2.4)
    assert is_hallucination_segment("texto aparentemente normal", compression_ratio=2.6) is True

    # Repetición en bucle
    assert is_hallucination_segment("bla bla bla bla") is True

    # Sink phrase con probabilidad de no-habla moderada
    assert is_hallucination_segment("Gracias por ver", no_speech_prob=0.35) is True

    # Ruido / silencio puro con probabilidad de habla casi nula
    assert is_hallucination_segment(
        "algo inaudible", avg_logprob=-0.85, no_speech_prob=0.82
    ) is True

    # Segmento legítimo de alta confianza
    assert is_hallucination_segment(
        "Bienvenidos a la conferencia de inteligencia artificial.",
        avg_logprob=-0.22,
        no_speech_prob=0.03,
        compression_ratio=1.3,
    ) is False


def test_extract_dynamic_prompt():
    segments = [
        Segment(
            start=0.0,
            end=4.0,
            text="El profesor Oppenheimer y Albert Einstein discutieron sobre la relatividad.",
            avg_logprob=-0.25,
            no_speech_prob=0.05,
        ),
        Segment(
            start=4.5,
            end=8.0,
            text="En Los Alamos se realizaron los experimentos de fisión nuclear.",
            avg_logprob=-0.30,
            no_speech_prob=0.02,
        ),
        # Segmento de baja certeza que debe ignorarse para evitar contaminar el prompt
        Segment(
            start=8.5,
            end=10.0,
            text="Murmullo dudoso e incierto con alucinaciones raras.",
            avg_logprob=-0.80,
            no_speech_prob=0.60,
        ),
    ]

    prompt = extract_dynamic_prompt(segments, max_terms=10)
    assert "Oppenheimer" in prompt
    assert "Einstein" in prompt
    assert "Alamos" in prompt
    assert "Relatividad" in prompt or "relatividad" in prompt.lower()
    # No debe contener palabras vacías
    terms = [t.strip().lower() for t in prompt.split(",")]
    assert "el" not in terms
    assert "en" not in terms
    assert "sobre" not in terms

