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
    assert [(c.start, c.end, c.text) for c in cues] == [
        (0.0, 0.9, "Hello there."), (5.0, 5.4, "Again")]


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


def test_zero_length_word_yields_valid_cue():
    cues = resegment(_transcript(Segment(0, 1, "Ok", _words((2.0, 2.0, " Ok"))))).segments
    assert cues[0].end > cues[0].start


def test_batch_size_bounds(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    assert TranscriptionRequest(audio).batch_size == 16
    with pytest.raises(ValueError):
        TranscriptionRequest(audio, batch_size=0)
