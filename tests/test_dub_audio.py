"""Ambiente conservado (stem de voz sin diálogo) y control de ráfagas de la voz sintetizada."""
from pathlib import Path

import numpy as np
import pytest
from dub_fakes import FakeCloner, FakeFactory, no_transcripts, transcript

from subdub_titler.application.dubbing import DubUseCase, load_report
from subdub_titler.domain.dubbing import (
    AudioClip,
    DubCue,
    DubRequest,
    ambience_spans,
    speech_spans,
)
from subdub_titler.domain.models import Word
from subdub_titler.infrastructure.media import FfmpegMixer
from subdub_titler.infrastructure.voice_level import BurstGuard, tame_bursts


def test_speech_spans_follow_words_and_bridge_short_gaps():
    words = (Word(1.0, 1.4, " Hola"), Word(1.5, 1.9, " qué"), Word(3.0, 3.5, " tal"))
    cues = [DubCue(0, 1.0, 3.5, "Hola qué tal", words=words), DubCue(1, 6.0, 7.0, "Adiós.")]
    # Sin tiempos por palabra se usa el cue entero; el hueco 1.9–3.0 no es habla.
    assert speech_spans(cues, padding=0.1, bridge=0.3) == (
        (0.9, 2.0), (2.9, 3.6), (5.9, 7.1))
    assert speech_spans(cues, padding=0.1, bridge=1.0) == ((0.9, 3.6), (5.9, 7.1))


def test_ambience_is_the_complement_without_tiny_gaps():
    speech = ((0.9, 2.0), (2.1, 3.6), (5.9, 7.1))
    assert ambience_spans(speech, 10.0, min_seconds=0.25) == (
        (0.0, 0.9), (3.6, 5.9), (7.1, 10.0))
    assert ambience_spans(((0.0, 10.0),), 10.0) == ()


def test_mixer_envelope_ramps_inside_ambience_spans():
    mixer = FfmpegMixer(sample_rate=1000, ambience_fade=0.1)
    gain = mixer._envelope(((1.0, 2.0),), 3000)
    assert gain[:1000].max() == 0 and gain[2000:].max() == 0  # el habla nunca se cuela
    assert gain[1100:1900].min() == 1.0
    assert 0 < gain[1050] < 1 and 0 < gain[1950] < 1


def test_dub_keeps_ambience_outside_speech_and_persists_it(tmp_path: Path):
    factory = FakeFactory()
    use_case = DubUseCase(no_transcripts, factory)
    req = DubRequest(source=tmp_path / "in.mp4", target_language="es")
    report = use_case.execute(req, tmp_path / "w", transcript=transcript())
    ambience = factory.mixer.ambience
    assert ambience is not None and ambience.source.name == "voice.flac"
    assert report.speech == ((0.85, 6.65), (7.85, 9.15))
    assert ambience.spans == ((0.0, 0.85), (6.65, 7.85), (9.15, 20.0))
    assert load_report(tmp_path / "w" / "report.json").speech == report.speech
    factory.mixer.ambience = None
    use_case.regenerate(req, tmp_path / "w", 1, "Otra frase")
    assert factory.mixer.ambience.spans == ambience.spans


def test_tame_bursts_only_touches_bursts():
    rate = 1000
    t = np.arange(4 * rate) / rate
    speech = (0.1 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    assert np.array_equal(tame_bursts(speech, rate), speech)
    burst = speech.copy()
    burst[2000:2200] *= 8  # +18 dB durante 0.2 s
    out = tame_bursts(burst, rate)
    assert np.abs(out[2050:2150]).max() < 0.1 * 10 ** (9 / 20)
    assert np.allclose(out[:1500], burst[:1500]) and np.allclose(out[2800:], burst[2800:])


def test_burst_guard_delegates_to_the_wrapped_cloner():
    class Releasing(FakeCloner):
        released = False

        def release(self):
            self.released = True

    inner = Releasing()
    guard = BurstGuard(inner)
    voice = guard.prepare_voice(AudioClip(np.zeros(300, np.float32), inner.sample_rate), "ref")
    clips = guard.synthesize(["hola"], [1.5], "es", voice, variant=1)
    assert guard.sample_rate == inner.sample_rate and inner.variants == [1]
    assert clips[0].duration == pytest.approx(1.5)
    guard.release()
    assert inner.released
