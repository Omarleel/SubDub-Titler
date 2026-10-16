from pathlib import Path

import pytest

from subdub_titler.domain.models import Segment, Transcript, TranscriptionRequest, Word
from subdub_titler.domain.segmentation import CuePolicy, resegment


def _words(*items):
    return tuple(Word(start, end, text) for start, end, text in items)


def _transcript(*segments):
    return Transcript(tuple(segments), "en", "fake", 100.0)


def test_without_words_keeps_engine_segments():
    original = _transcript(Segment(0, 40, "Segmento largo sin palabras"))
    assert resegment(original) is original


def test_splits_on_pause_and_drops_silence():
    words = _words((0.0, 0.4, " Hello"), (0.5, 0.9, " there."), (5.0, 5.4, " Again"))
    cues = resegment(_transcript(Segment(0, 5.4, "Hello there. Again", words))).segments
    assert [(c.start, round(c.end, 2), c.text) for c in cues] == [
        (0.0, 1.02, "Hello there."), (5.0, 5.52, "Again")]


def test_limits_duration_and_characters():
    words = _words(*[(i * 0.5, i * 0.5 + 0.45, " word") for i in range(40)])
    cues = resegment(_transcript(Segment(0, 20, "x", words)),
                     CuePolicy(max_duration=6.0, max_chars=30)).segments
    assert all(c.end - c.start <= 6.0 for c in cues)
    assert all(len(c.text) <= 30 for c in cues)
    assert " ".join(c.text for c in cues) == " ".join(["word"] * 40)


def test_merges_words_across_engine_segments():
    first = Segment(0, 1, "Hola", _words((0.0, 0.5, " Hola")))
    second = Segment(1, 2, "mundo", _words((0.6, 1.0, " mundo")))
    cues = resegment(_transcript(first, second)).segments
    assert [c.text for c in cues] == ["Hola mundo"]


def test_syntactic_merge_on_pause_without_terminal_punctuation():
    # Pausa de 0.8s (> max_pause 0.6s pero <= merge_max_pause 1.2s)
    # Sin puntuación terminal en la primera parte + continuación en minúscula:
    first_words = [(0.0, 0.5, " estamos"), (0.6, 1.0, " listos")]
    second_words = [(1.8, 2.2, " para"), (2.3, 2.7, " comenzar.")]
    words = _words(*(first_words + second_words))
    cues = resegment(_transcript(Segment(0, 2.7, "estamos listos para comenzar.", words))).segments
    assert len(cues) == 1
    assert cues[0].text == "estamos listos para comenzar."


def test_no_syntactic_merge_when_terminal_punctuation_present():
    # Con puntuación terminal: no debe unirse aunque la pausa sea <= 1.2s
    first_words = [(0.0, 0.5, " Adiós.")]
    second_words = [(1.2, 1.8, " hasta"), (1.9, 2.3, " luego.")]
    words = _words(*(first_words + second_words))
    cues = resegment(_transcript(Segment(0, 2.3, "Adiós. hasta luego.", words))).segments
    assert len(cues) == 2
    assert [c.text for c in cues] == ["Adiós.", "hasta luego."]


def test_no_syntactic_merge_when_pause_exceeds_threshold():
    # Pausa de 1.5s (> merge_max_pause 1.2s): debe mantenerse separado
    first_words = [(0.0, 0.5, " primero")]
    second_words = [(2.0, 2.5, " segundo")]
    words = _words(*(first_words + second_words))
    cues = resegment(_transcript(Segment(0, 2.5, "primero segundo", words))).segments
    assert len(cues) == 2


def test_pad_end_does_not_overlap_with_next_cue():
    # Pausa de 0.65s (> max_pause 0.6s) genera 2 cues.
    # pad_end (0.20s) llevaría cue 0 de 0.5 a 0.70. Si cue 1 inicia a 0.65s,
    # con max_pause=0.1s para forzar separación:
    policy = CuePolicy(max_pause=0.1, pad_end=0.20)
    words = _words((0.0, 0.5, " Uno."), (0.65, 1.0, " Dos."))
    cues = resegment(_transcript(Segment(0, 1.0, "Uno. Dos.", words)), policy=policy).segments
    assert len(cues) == 2
    # 0.5 + 0.20 = 0.70 > 0.65, por tanto c[0].end debe limitarse exactamente a c[1].start (0.65)
    assert cues[0].end == pytest.approx(0.65)
    assert cues[0].end <= cues[1].start


def test_zero_length_word_yields_valid_cue():
    cues = resegment(_transcript(Segment(0, 1, "Ok", _words((2.0, 2.0, " Ok"))))).segments
    assert cues[0].end > cues[0].start


def test_batch_size_bounds(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    assert TranscriptionRequest(audio).batch_size == 16
    with pytest.raises(ValueError):
        TranscriptionRequest(audio, batch_size=0)
    with pytest.raises(ValueError):
        TranscriptionRequest(audio, vad_threshold=-0.1)
    with pytest.raises(ValueError):
        TranscriptionRequest(audio, vad_speech_pad_ms=-1)
    with pytest.raises(ValueError):
        TranscriptionRequest(audio, no_speech_threshold=1.5)


def test_smart_line_break_preserves_short_text():
    from subdub_titler.domain.segmentation import balance_line_breaks
    short_text = "Esta es una frase corta."
    assert balance_line_breaks(short_text, max_cpl=42) == short_text


def test_smart_line_break_splits_long_text_within_bounds():
    from subdub_titler.domain.segmentation import balance_line_breaks
    long_text = "Okay, I know this is gonna sound a little crazy, but I thought maybe we could have"
    broken = balance_line_breaks(long_text, max_cpl=42)
    lines = broken.split("\n")
    assert len(lines) == 2
    assert all(len(line_text) <= 42 for line_text in lines)
    assert " ".join(lines) == long_text


def test_conform_timing_stretches_high_cps_if_gap_allows():
    from subdub_titler.domain.segmentation import NETFLIX_POLICY, conform_timing_netflix
    # Segmento con 33 caracteres que dura 0.8s -> CPS = 41.25 (> target 17.0)
    # Siguiente segmento empieza a los 4.0s (hay silencio de 3.2s disponible)
    seg1 = Segment(0.0, 0.8, "Texto de treinta y cuatro letras.")
    seg2 = Segment(4.0, 5.0, "Siguiente frase.")
    adjusted = conform_timing_netflix((seg1, seg2), NETFLIX_POLICY)
    # Duración necesaria para 33 caracteres a 17 CPS = 33 / 17 = 1.941 s.
    # nuevo end = 1.941 s, y respeta min_gap con seg2 (4.0 - 0.08 = 3.92)
    assert adjusted[0].end == pytest.approx(1.941, abs=0.01)
    assert adjusted[0].end <= 4.0 - NETFLIX_POLICY.min_gap


def test_evaluate_audiovisual_compliance_metrics():
    from subdub_titler.domain.segmentation import evaluate_audiovisual_compliance
    # Subtítulo de 3 segundos con 42 caracteres -> CPS = 14 (<= 17)
    seg = Segment(0.0, 3.0, "Línea uno muy clara\ny línea dos balanceada.")
    t = Transcript((seg,), "es", "small", 3.0)
    report = evaluate_audiovisual_compliance(t)
    assert report["total_cues"] == 1
    assert report["cps_compliance_rate"] == 100.0
    assert report["cpl_compliance_rate"] == 100.0
    assert report["lines_compliance_rate"] == 100.0
    assert report["is_compliant"] is True

