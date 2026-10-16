from pathlib import Path

import pytest

from subdub_titler.domain.models import Segment, Transcript, Word
from subdub_titler.infrastructure.ctc_aligner import (
    LazyMmsForcedAligner,
    clean_word_for_ctc,
)


def test_clean_word_for_ctc_normalization():
    assert clean_word_for_ctc("¡Hola!") == "hola"
    assert clean_word_for_ctc("canción") == "cancion"
    assert clean_word_for_ctc("niño") == "nino"
    assert clean_word_for_ctc("vergüenza") == "verguenza"
    assert clean_word_for_ctc("l'oiseau") == "l'oiseau"
    assert clean_word_for_ctc("ça") == "ca"
    assert clean_word_for_ctc("¿Qué tal?") == "quetal"
    assert clean_word_for_ctc("12345") == ""
    assert clean_word_for_ctc("...") == ""


def test_mms_forced_aligner_end_to_end_cpu(tmp_path: Path):
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    from subdub_titler.infrastructure.ctc_aligner import MmsForcedAligner

    # Generar un archivo WAV sintético mono a 16 kHz
    sr = 16000
    duration_sec = 2.0
    num_samples = int(duration_sec * sr)
    # Forma de onda sintética con dos ráfagas de energía
    t = torch.linspace(0, duration_sec, num_samples)
    waveform = 0.3 * torch.sin(2 * 3.14159 * 440 * t)
    wav_path = tmp_path / "test_synth.wav"
    torchaudio.save(str(wav_path), waveform.unsqueeze(0), sr)

    orig_words = (
        Word(0.1, 0.7, " Hola"),
        Word(0.8, 1.6, " mundo"),
    )
    transcript = Transcript(
        segments=(
            Segment(0.1, 1.6, "Hola mundo", orig_words),
        ),
        language="es",
        model_id="test",
        duration_seconds=duration_sec,
    )

    aligner = MmsForcedAligner(device="cpu")
    aligned = aligner.align(wav_path, transcript)

    # Verificar estructura
    assert len(aligned.segments) == 1
    seg = aligned.segments[0]
    assert len(seg.words) == 2
    w1, w2 = seg.words

    # Textos originales preservados
    assert w1.text == " Hola"
    assert w2.text == " mundo"

    # Invariantes de tiempo
    assert w1.start < w1.end
    assert w2.start < w2.end
    assert w1.end <= w2.start
    assert seg.start == w1.start
    assert seg.end == w2.end

    # Test release
    aligner.release()
    assert aligner._model is None


def test_mms_forced_aligner_handles_empty_or_special_segments(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from subdub_titler.infrastructure.ctc_aligner import MmsForcedAligner

    audio = tmp_path / "dummy.wav"
    audio.write_bytes(b"dummy")

    # Segmento sin palabras o con palabras vacías (ej. sólo puntuación)
    transcript = Transcript(
        segments=(
            Segment(0.0, 1.0, "..."),
            Segment(1.2, 2.5, "???", (Word(1.2, 2.5, "???"),)),
        ),
        language="es",
        model_id="test",
        duration_seconds=3.0,
    )

    aligner = MmsForcedAligner(device="cpu")
    # Si no hay tokens o el archivo es ilegible, no debe fallar catastróficamente
    result = aligner.align(audio, transcript)
    assert len(result.segments) == 2


def test_lazy_mms_forced_aligner_proxy():
    lazy = LazyMmsForcedAligner(device="cpu")
    assert lazy._aligner is None
    lazy.release()
    assert lazy._aligner is None
