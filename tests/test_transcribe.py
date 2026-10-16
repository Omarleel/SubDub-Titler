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
    assert req.forced_alignment is False

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


class FakeAligner:
    def __init__(self) -> None:
        self.calls = []
        self.released = False

    def align(self, audio_source: Path, transcript: Transcript) -> Transcript:
        from dataclasses import replace
        self.calls.append((audio_source, transcript))
        aligned_segments = []
        for s in transcript.segments:
            new_words = tuple(Word(w.start + 0.05, w.end + 0.05, w.text) for w in s.words)
            aligned_segments.append(
                Segment(new_words[0].start, new_words[-1].end, s.text, new_words)
            )
        return replace(transcript, segments=tuple(aligned_segments))

    def release(self) -> None:
        self.released = True


def test_transcribe_use_case_with_forced_alignment(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio_bytes")

    aligner = FakeAligner()
    engine = FakeEngine()
    registry = ModelRegistry()
    registry.register(ModelDescriptor("small", "test"), lambda r: engine)

    use_case = TranscribeUseCase(registry, aligner=aligner)
    req = TranscriptionRequest(
        audio, "small", device="cpu", compute_type="int8", forced_alignment=True,
    )
    profiler = SpanProfiler()
    result = use_case.execute(req, profiler)

    # El alineador debió procesar el audio y liberar memoria
    assert len(aligner.calls) == 1
    assert aligner.calls[0][0] == audio
    assert aligner.released is True

    # El span asr.forced_align debe estar registrado
    event_names = [e.name for e in result.events]
    assert "asr.forced_align" in event_names

    # Los tiempos deben reflejar el ajuste del alineador (+0.05s)
    first_seg = result.transcript.segments[0]
    assert first_seg.words[0].start == pytest.approx(0.05)


def test_transcribe_use_case_fails_if_forced_alignment_without_aligner(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    engine = FakeEngine()
    registry = ModelRegistry()
    registry.register(ModelDescriptor("small", "test"), lambda r: engine)

    use_case = TranscribeUseCase(registry, aligner=None)
    req = TranscriptionRequest(
        audio, "small", device="cpu", compute_type="int8", forced_alignment=True,
    )
    with pytest.raises(ValueError, match="--align"):
        use_case.execute(req)


class FakeDiarizer:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.calls = []
        self.released = False

    def assign(self, source: Path, spans, max_speakers: int):
        self.calls.append((source, spans, max_speakers))
        return self.labels[: len(spans)]

    def release(self) -> None:
        self.released = True


def test_transcribe_use_case_with_diarization(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio_bytes")

    diarizer = FakeDiarizer(["S0", "S1"])
    engine = FakeEngine()
    registry = ModelRegistry()
    registry.register(ModelDescriptor("small", "test"), lambda r: engine)

    use_case = TranscribeUseCase(registry, diarizer=diarizer)
    req = TranscriptionRequest(
        audio, "small", device="cpu", compute_type="int8", diarize=True, max_speakers=2,
    )
    profiler = SpanProfiler()
    result = use_case.execute(req, profiler)

    # El diarizador debió ejecutarse y liberar memoria
    assert len(diarizer.calls) == 1
    assert diarizer.calls[0][0] == audio
    assert diarizer.calls[0][2] == 2
    assert diarizer.released is True

    # Span asr.diarize registrado
    event_names = [e.name for e in result.events]
    assert "asr.diarize" in event_names

    # Segmento con speaker asignado
    assert result.transcript.segments[0].speaker == "S0"


def test_transcribe_use_case_fails_if_diarize_without_diarizer(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    engine = FakeEngine()
    registry = ModelRegistry()
    registry.register(ModelDescriptor("small", "test"), lambda r: engine)

    use_case = TranscribeUseCase(registry, diarizer=None)
    req = TranscriptionRequest(
        audio, "small", device="cpu", compute_type="int8", diarize=True,
    )
    with pytest.raises(ValueError, match="--diarize"):
        use_case.execute(req)


def test_subtitles_render_with_speaker():
    from subdub_titler.infrastructure.subtitles import SrtRenderer, VttRenderer

    transcript = Transcript(
        segments=(
            Segment(0.0, 1.5, "Hola mundo", speaker="S0"),
            Segment(2.0, 3.5, "Respuesta", speaker="S1"),
            Segment(4.0, 5.0, "Sin speaker", speaker=None),
        ),
        language="es",
        model_id="test",
    )

    srt = SrtRenderer().render(transcript)
    assert "[Hablante 1] Hola mundo" in srt
    assert "[Hablante 2] Respuesta" in srt
    assert "Sin speaker" in srt
    assert "[None]" not in srt

    vtt = VttRenderer().render(transcript)
    assert "[Hablante 1] Hola mundo" in vtt
    assert "[Hablante 2] Respuesta" in vtt
