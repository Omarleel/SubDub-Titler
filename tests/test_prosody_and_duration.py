"""Pruebas unitarias para las Fases 4, 5 y 6:
- Fase 4: Selección de referencia (ritmo, duración óptima, calidad acústica).
- Fase 5: Expresividad y prosodia (F0, pausas, rango dinámico, modulación).
- Fase 6: Adaptación de traducción y duración (penalización velocidad, rushed).
"""
from pathlib import Path

import numpy as np
import pytest

from subdub_titler.domain.dubbing import (
    AudioClip,
    Candidate,
    DubbedCue,
    DubCue,
    DubRequest,
    FitAction,
    ProsodyProfile,
    TimingPolicy,
    choose_candidate,
    reference_candidates,
    reference_quality_score,
    sync_metrics,
)
from subdub_titler.domain.models import Word
from subdub_titler.infrastructure.prosody import (
    NumpyProsodyAnalyzer,
    extract_prosody,
    transfer_prosodic_energy,
)
from subdub_titler.infrastructure.voice_level import check_reference_audio

# ==============================================================================
# Fase 4: Selección robusta de la referencia
# ==============================================================================


def test_reference_quality_score_penalizes_slow_and_fast_speech():
    # Frase conversacional típica (~14 letras/s, 5 s)
    ideal_text = "Esta es una frase con ritmo completamente normal y fluido."
    score_ideal = reference_quality_score(5.0, ideal_text)

    # Frase arrastrada / anormalmente lenta (< 8 letras/s)
    slow_text = "Uhm... sí."  # 7 caracteres en 5 s = 1.4 letras/s
    score_slow = reference_quality_score(5.0, slow_text)

    # Frase excesivamente rápida / atropellada (> 24 letras/s)
    fast_text = "Estafraseesmuyrapidainteligibleconmuchasletrasporsegundoimposibledeimitar" * 2
    score_fast = reference_quality_score(3.0, fast_text)

    assert score_ideal > score_slow
    assert score_ideal > score_fast


def test_reference_candidates_prefers_conversational_cadence_over_dragging_cadence():
    # Cue 0: 6 s con habla arrastrada/lenta
    # Cue 1: 5 s con habla fluida y natural
    cues = [
        DubCue(0, 0.0, 6.0, "Uhm... sí."),
        DubCue(1, 8.0, 13.0, "Esta es una frase con ritmo perfectamente natural y fluido."),
    ]
    spans = reference_candidates(cues, min_seconds=3.0, max_seconds=10.0, limit=2)
    assert len(spans) == 2
    # El candidato 1 (ritmo fluido) debe quedar en primera posición
    assert spans[0].text == "Esta es una frase con ritmo perfectamente natural y fluido."
    assert spans[0].speech_rate > 5.0


def test_check_reference_audio_detects_clipping_and_quietness():
    sr = 16000
    # Audio limpio normal
    clean_samples = 0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False))
    q_clean = check_reference_audio(clean_samples, sr)
    assert not q_clean.is_saturated
    assert not q_clean.is_too_quiet
    assert q_clean.clipping_ratio == 0.0

    # Audio saturado con clipping
    clipped_samples = clean_samples.copy()
    clipped_samples[:500] = 1.0  # > 0.1% de muestras saturadas
    q_clipped = check_reference_audio(clipped_samples, sr)
    assert q_clipped.is_saturated
    assert q_clipped.clipping_ratio > 0.001

    # Audio casi inaudible (demasiado bajo)
    quiet_samples = 0.001 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False))
    q_quiet = check_reference_audio(quiet_samples, sr)
    assert q_quiet.is_too_quiet
    assert q_quiet.rms < 0.01


# ==============================================================================
# Fase 5: Expresividad y prosodia por frase
# ==============================================================================


def test_extract_prosody_computes_f0_pauses_and_dynamic_range():
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Crear señal armónica con tono fundamental F0 ≈ 150 Hz
    # Añadir 0.2s de silencio inicial y 0.2s de silencio final
    signal = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.15 * np.sin(2 * np.pi * 300 * t)
    signal[:int(0.2 * sr)] = 0.0
    signal[-int(0.2 * sr):] = 0.0

    words = (
        Word(0.2, 0.8, "Hola"),
        Word(0.9, 1.8, "mundo"),
    )
    clip = AudioClip(signal.astype(np.float32), sr)
    profile = extract_prosody(clip, words)

    assert isinstance(profile, ProsodyProfile)
    assert profile.energy_rms > 0.05
    assert profile.leading_pause == pytest.approx(0.2, abs=0.05)
    assert profile.trailing_pause == pytest.approx(0.2, abs=0.05)
    assert profile.dynamic_range_db > 5.0
    # F0 debe estimarse alrededor de 150 Hz
    assert profile.pitch_mean_hz is not None
    assert 130.0 <= profile.pitch_mean_hz <= 170.0


def test_transfer_prosodic_energy_modulates_without_clipping():
    sr = 16000
    synth_samples = 0.1 * np.sin(2 * np.pi * 200 * np.linspace(0, 1, sr, endpoint=False))
    synth_clip = AudioClip(synth_samples.astype(np.float32), sr)

    # Target con mayor energía (énfasis)
    target_loud = ProsodyProfile(
        speech_rate=14.0,
        energy_rms=0.25,
        dynamic_range_db=15.0,
        leading_pause=0.0,
        trailing_pause=0.0,
    )
    loud_clip = transfer_prosodic_energy(synth_clip, target_loud, max_boost_db=6.0)
    rms_orig = np.sqrt(np.mean(synth_clip.samples ** 2))
    rms_loud = np.sqrt(np.mean(loud_clip.samples ** 2))
    assert rms_loud > rms_orig
    assert np.max(np.abs(loud_clip.samples)) <= 0.95

    # Analyzer protocol
    analyzer = NumpyProsodyAnalyzer()
    p = analyzer.analyze(synth_clip)
    assert p.energy_rms > 0.0


# ==============================================================================
# Fase 6: Adaptación de traducción y duración
# ==============================================================================


def test_choose_candidate_balances_score_and_speed_penalty():
    # Candidato 0: score -0.10, pero requiere speed = 1.18 (atropellado)
    # Candidato 1: score -0.15, pero requiere speed = 1.00 (ritmo natural cómodo)
    options = [
        Candidate("Traducción larga que apenas cabe", -0.10),
        Candidate("Traducción natural", -0.15),
    ]
    slot = 2.0
    durations = [2.36, 2.00]  # speed = 1.18 y 1.00

    # Sin penalización de velocidad (speed_penalty = 0.0): elige el de mayor score (-0.10)
    idx_no_pen = choose_candidate(options, durations, slot, TimingPolicy(speed_penalty=0.0))
    assert idx_no_pen == 0

    # Con penalización de velocidad (speed_penalty = 0.5):
    # Candidate 0: -0.10 - 0.5 * (1.18 - 1.0) = -0.19
    # Candidate 1: -0.15 - 0.5 * (1.00 - 1.0) = -0.15
    # Elige la traducción natural cómoda (Candidate 1)
    idx_pen = choose_candidate(options, durations, slot, TimingPolicy(speed_penalty=0.5))
    assert idx_pen == 1


def test_sync_metrics_tracks_rushed_cues():
    track = [
        DubbedCue(0, 0.0, "Orig 1", "Trad 1", 2.0, 2.0, 2.0, 2.0, 1.00, FitAction.OK),
        DubbedCue(1, 3.0, "Orig 2", "Trad 2", 2.0, 2.36, 2.0, 2.0, 1.18, FitAction.OK),
        DubbedCue(2, 6.0, "Orig 3", "Trad 3", 1.0, 1.5, 1.0, 1.0, 1.50, FitAction.SHORTEN),
    ]
    report = sync_metrics(track)
    assert report.cues == 3
    # cues con speed > 1.15 o SHORTEN son el índice 1 y 2
    assert report.rushed == 2
    assert track[0].is_rushed is False
    assert track[1].is_rushed is True
    assert track[2].is_rushed is True


def test_dub_request_validates_new_phase_parameters():
    req = DubRequest(
        source=Path("test.mp4"),
        target_language="es",
        speed_penalty=0.8,
        concise_translation=True,
        per_phrase_reference=True,
        prosody_transfer=True,
    )
    assert req.speed_penalty == 0.8
    assert req.concise_translation is True
    assert req.per_phrase_reference is True
    assert req.prosody_transfer is True

    # Validaciones de rango y tipos
    with pytest.raises(ValueError, match="speed_penalty"):
        DubRequest(source=Path("test.mp4"), target_language="es", speed_penalty=15.0)

    with pytest.raises(ValueError, match="concise_translation"):
        DubRequest(source=Path("test.mp4"), target_language="es", concise_translation="si")  # type: ignore

    with pytest.raises(ValueError, match="per_phrase_reference"):
        DubRequest(source=Path("test.mp4"), target_language="es", per_phrase_reference=1)  # type: ignore

    with pytest.raises(ValueError, match="prosody_transfer"):
        DubRequest(source=Path("test.mp4"), target_language="es", prosody_transfer="yes")  # type: ignore
