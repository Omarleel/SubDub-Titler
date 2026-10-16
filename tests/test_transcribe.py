from pathlib import Path

import pytest

from subdub_titler.application.transcribe import TranscribeUseCase
from subdub_titler.domain.models import (
    ModelDescriptor,
    Segment,
    Transcript,
    TranscriptionRequest,
    Word,
)
from subdub_titler.domain.ports import Stems
from subdub_titler.infrastructure.profiling import SpanProfiler
from subdub_titler.infrastructure.registry import ModelRegistry


class FakeEngine:
    def __init__(self) -> None:
        self.last_request = None

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        self.last_request = request
        words = (Word(0.0, 0.4, " Hola"), Word(0.5, 0.9, " mundo"))
        return Transcript(
            segments=(
                Segment(0.0, 1.0, "Hola mundo", words),
            ),
            language=request.language if request.language != "auto" else "es",
            model_id=request.model_id,
            duration_seconds=1.5,
        )


class FakeSeparator:
    def __init__(self, stems: Stems) -> None:
        self.stems = stems
        self.released = False
        self.calls = []

    def separate(self, source: Path, workdir: Path) -> Stems:
        self.calls.append((source, workdir))
        return self.stems

    def release(self) -> None:
        self.released = True


def test_transcription_request_defaults_and_validation(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy")

    req = TranscriptionRequest(audio)
    assert req.vad_threshold == 0.35
    assert req.vad_speech_pad_ms == 800
    assert req.vad_min_silence_ms == 2000
    assert req.no_speech_threshold == 0.40
    assert req.normalize_loudness is False
    assert req.isolate_dialogue is False

    with pytest.raises(ValueError, match="vad_threshold"):
        TranscriptionRequest(audio, vad_threshold=1.5)
    with pytest.raises(ValueError, match="vad_threshold"):
        TranscriptionRequest(audio, vad_threshold=-0.1)
    with pytest.raises(ValueError, match="vad_speech_pad_ms"):
        TranscriptionRequest(audio, vad_speech_pad_ms=-10)
    with pytest.raises(ValueError, match="vad_min_silence_ms"):
        TranscriptionRequest(audio, vad_min_silence_ms=-1)
    with pytest.raises(ValueError, match="no_speech_threshold"):
        TranscriptionRequest(audio, no_speech_threshold=2.0)


def test_transcribe_use_case_with_dialogue_isolation(tmp_path: Path):
    audio = tmp_path / "mixed.mp4"
    audio.write_bytes(b"video_bytes")
    voice = tmp_path / "voice.flac"
    voice.write_bytes(b"voice_bytes")
    bg = tmp_path / "background.flac"
    bg.write_bytes(b"bg_bytes")

    stems = Stems(voice, bg)
    separator = FakeSeparator(stems)

    engine = FakeEngine()
    registry = ModelRegistry()
    registry.register(ModelDescriptor("small", "test"), lambda r: engine)

    use_case = TranscribeUseCase(registry, separator=separator)
    req = TranscriptionRequest(
        audio, "small", device="cpu", compute_type="int8", isolate_dialogue=True,
    )
    profiler = SpanProfiler()
    result = use_case.execute(req, profiler)

    # El separador debió aislar el audio y liberar memoria
    assert len(separator.calls) == 1
    assert separator.calls[0][0] == audio
    assert separator.released is True

    # El motor ASR debió recibir el stem vocal aislado
    assert engine.last_request is not None
    assert engine.last_request.source == voice

    # El resultado debe incluir el span de aislamiento
    event_names = [e.name for e in result.events]
    assert "asr.isolate_dialogue" in event_names
    assert "asr.transcribe_complete" in event_names


def test_transcribe_use_case_fails_if_dialogue_isolation_without_separator(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    engine = FakeEngine()
    registry = ModelRegistry()
    registry.register(ModelDescriptor("small", "test"), lambda r: engine)

    use_case = TranscribeUseCase(registry, separator=None)
    req = TranscriptionRequest(
        audio, "small", device="cpu", compute_type="int8", isolate_dialogue=True,
    )
    with pytest.raises(ValueError, match="--isolate-dialogue"):
        use_case.execute(req)
