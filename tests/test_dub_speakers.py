import shutil
from pathlib import Path

import pytest
from dub_fakes import FakeFactory, no_transcripts

from subdub_titler.application.dubbing import DubUseCase, load_report
from subdub_titler.domain.dubbing import (
    DubCue,
    DubRequest,
    diarization_windows,
    estimate_seconds_by_rate,
    pick_reference,
    speakers_by_presence,
    split_by_speaker,
)
from subdub_titler.domain.models import Segment, Transcript, Word


def words(*items):
    return tuple(Word(s, e, t) for s, e, t in items)


TURN = DubCue(0, 0.0, 4.0, "Hola qué tal. Muy bien gracias.", words=words(
    (0.0, 0.6, " Hola"), (0.6, 1.3, " qué tal."), (1.5, 2.2, " Muy"), (2.2, 2.9, " bien"),
    (2.9, 3.3, " gracias.")))


def test_windows_align_to_words_and_absorb_short_tail():
    windows = diarization_windows([TURN, DubCue(1, 5.0, 6.0, "Sin palabras")], seconds=1.2)
    assert [(w.first, w.last) for w in windows if w.cue == 0] == [(0, 1), (2, 4)]
    assert windows[-1].cue == 1 and windows[-1].last == -1  # cue sin palabras: una ventana


def test_split_by_speaker_at_turn_change_and_reindex():
    second = DubCue(1, 5.0, 6.0, "Adiós.", words=words((5.0, 6.0, " Adiós.")))
    windows = diarization_windows([TURN, second], seconds=1.2)
    cues = split_by_speaker([TURN, second], windows, ["S0", "S1", "S1"])
    assert [(c.index, c.source_text, c.speaker) for c in cues] == [
        (0, "Hola qué tal.", "S0"), (1, "Muy bien gracias.", "S1"), (2, "Adiós.", "S1")]
    assert cues[1].start == 1.5 and cues[1].end == 3.3
    same = split_by_speaker([TURN], diarization_windows([TURN], 1.2), ["S0", "S0"])
    assert len(same) == 1 and same[0].source_text == TURN.source_text
    with pytest.raises(ValueError):
        split_by_speaker([TURN], windows, ["S0"])


def test_reference_never_crosses_another_speaker():
    cues = [DubCue(0, 0.0, 2.0, "Uno.", speaker="A"), DubCue(1, 2.1, 2.6, "Dos.", speaker="B"),
            DubCue(2, 2.7, 6.0, "Tres frases largas.", speaker="A")]
    reference = pick_reference(cues, min_seconds=1.0, speaker="A")
    assert reference.text == "Tres frases largas."
    assert speakers_by_presence(cues) == ["A", "B"]
    with pytest.raises(ValueError):
        pick_reference(cues, speaker="Z")


def test_rate_estimate_uses_reference_pace():
    assert estimate_seconds_by_rate("abcdefghij", "abcde", 1.0) == pytest.approx(2.0)
    assert estimate_seconds_by_rate("今日", "abcdef", 1.0) == pytest.approx(1.0)
    assert estimate_seconds_by_rate("hola", "", 0.0) == pytest.approx(0.3)


class AlternatingDiarizer:
    """Ventanas pares = S0, impares = S1."""

    def assign(self, source, spans, max_speakers):
        return [f"S{i % 2}" for i in range(len(spans))]


class SpeakerFactory(FakeFactory):
    def build(self, request):
        from dataclasses import replace
        return replace(super().build(request), diarizer=AlternatingDiarizer())


def test_multi_speaker_dub_uses_one_voice_per_speaker(tmp_path: Path):
    source = tmp_path / "v.mp4"
    source.write_bytes(b"x")
    transcript = Transcript((
        Segment(1.0, 4.0, "This is speaker zero.", words((1.0, 4.0, " This is speaker zero."))),
        Segment(5.0, 8.0, "And this is speaker one.",
                words((5.0, 8.0, " And this is speaker one."))),
        Segment(9.0, 12.0, "Zero again here.", words((9.0, 12.0, " Zero again here."))),
    ), "en", "fake", 20.0)
    factory = SpeakerFactory()
    voices = []
    original = factory.cloner.prepare_voice
    factory.cloner.prepare_voice = lambda clip, text: voices.append(text) or original(clip, text)
    request = DubRequest(source, "es", diarize=True, evaluate=True)
    report = DubUseCase(no_transcripts, factory).execute(request, tmp_path / "w",
                                                          transcript=transcript)
    assert [c.speaker for c in report.track] == ["S0", "S1", "S0"]
    assert set(report.references) == {"S0", "S1"} and len(voices) == 2
    assert report.reference == report.references["S0"]  # el principal
    assert factory.cloner.batches == [2, 1]  # síntesis agrupada por hablante
    assert load_report(tmp_path / "w" / "report.json").references == report.references
    updated = DubUseCase(no_transcripts, factory).regenerate(request, tmp_path / "w", 1,
                                                             "Hola")
    assert updated.track[1].speaker == "S1" and voices[-1] == report.references["S1"].text


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no disponible")
def test_time_stretch_exact_length_keeps_pitch():
    np = pytest.importorskip("numpy")
    from subdub_titler.infrastructure.media import time_stretch
    rate = 16_000
    tone = np.sin(2 * np.pi * 200 * np.arange(rate) / rate).astype(np.float32)
    for seconds in (0.6, 1.0, 1.7, 2.5):
        stretched = time_stretch(tone, rate, seconds)
        assert len(stretched) == round(seconds * rate)
        spectrum = np.abs(np.fft.rfft(stretched[: rate // 2]))
        assert abs(spectrum.argmax() * 2 - 200) <= 4


def test_diarization_windows_cover_cue_boundaries_with_short_words():
    # Cue donde la única palabra termina mucho antes que el cue (ej. interjección 'Uh')
    cue = DubCue(75, 615.13, 615.963, "Uh", words=words((615.13, 615.171, " Uh")))
    windows = diarization_windows([cue], seconds=1.2)
    assert len(windows) == 1
    # La ventana no debe quedar truncada a 41 ms sino abarcar el cue
    assert windows[0].start == 615.13
    assert windows[0].end == pytest.approx(615.963)
    assert windows[0].end - windows[0].start > 0.8


def test_waveform_pads_short_and_empty_clips():
    pytest.importorskip("torch")
    from subdub_titler.domain.dubbing import AudioClip
    from subdub_titler.infrastructure.dub_eval import MIN_EVAL_SAMPLES, _waveform

    # Audio vacío
    empty_clip = AudioClip([], 16_000)
    w_empty = _waveform(empty_clip, "cpu")
    assert w_empty.shape[-1] >= MIN_EVAL_SAMPLES

    # Audio ultracorto (400 muestras = 25 ms, tamaño que causaba error espacial L=1)
    short_clip = AudioClip([0.1] * 400, 16_000)
    w_short = _waveform(short_clip, "cpu")
    assert w_short.shape[-1] >= MIN_EVAL_SAMPLES

