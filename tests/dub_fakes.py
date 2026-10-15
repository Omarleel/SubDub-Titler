"""Puertos falsos del doblaje: sin GPU, sin pesos y sin ffmpeg (ver FakeEngine en test_api)."""
from pathlib import Path

from subdub_titler.application.dub_quality import DubEvaluator
from subdub_titler.application.dubbing import DubPipeline
from subdub_titler.domain.dubbing import AudioClip, Candidate
from subdub_titler.domain.models import Segment, Transcript
from subdub_titler.domain.ports import MixResult, Stems
from subdub_titler.infrastructure.subtitles import SrtRenderer

RATE = 100  # muestras por segundo: suficiente para medir duraciones


class FakeTranslator:
    def __init__(self) -> None:
        self.calls = 0

    def translate(self, texts, source, target, n_best=1):
        self.calls += 1
        # Primer candidato largo y bien puntuado; segundo corto: prueba la elección por ranura.
        return [[Candidate(f"{t} traducido con mucho detalle", -0.1),
                 Candidate(f"{t[:4]}", -0.9)][:n_best] for t in texts]


class FakeCloner:
    sample_rate = RATE

    def __init__(self) -> None:
        self.batches: list[int] = []
        self.variants: list[int] = []

    def prepare_voice(self, reference, text):
        return {"text": text, "seconds": reference.duration}

    def natural_seconds(self, texts, language, voice):
        return [0.08 * len(t) for t in texts]

    def synthesize(self, texts, durations, language, voice, variant=0):
        self.batches.append(len(texts))
        self.variants.append(variant)
        return [AudioClip([0.1] * int(round(d * RATE)), RATE) for d in durations]


class FakeAudioStore:
    def load(self, path, sample_rate, start=None, end=None):
        if start is None:
            return AudioClip([0.1] * sample_rate, sample_rate)
        return AudioClip([0.1] * int(round(((end or start + 1) - start) * sample_rate)),
                         sample_rate)

    def save(self, clip, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF" + bytes(len(clip.samples)))
        return path

    def duration(self, path):
        return 20.0


class FakeMixer:
    def __init__(self) -> None:
        self.placements = []
        self.ambience = None

    def render(self, placements, background, duration, out_dir, voice_reference=None,
               ambience=None):
        self.placements = list(placements)
        self.ambience = ambience
        out_dir.mkdir(parents=True, exist_ok=True)
        voice, mix = out_dir / "dub_voice.flac", out_dir / "dub_mix.flac"
        voice.write_bytes(b"fLaC")
        mix.write_bytes(b"fLaC")
        return MixResult(voice, mix, -16.2, -1.5)


class FakeMuxer:
    def mux(self, video, dub_audio, subtitles, language, out):
        out.write_bytes(b"mp4")
        return out


class FakeSeparator:
    def separate(self, source, workdir):
        workdir.mkdir(parents=True, exist_ok=True)
        stems = Stems(workdir / "voice.flac", workdir / "background.flac")
        stems.voice.write_bytes(b"v")
        stems.background.write_bytes(b"b")
        return stems


class EchoTranscriber:
    """ASR perfecto: devuelve el texto que se pidió sintetizar (vía registro externo)."""

    def __init__(self) -> None:
        self.expected: list[str] = []

    def transcribe_clips(self, clips, language):
        return ["texto"] * len(clips)


class FixedSpeaker:
    def similarity(self, reference, clips):
        return [0.6] * len(clips)


class FixedMos:
    def mos(self, clips):
        return [3.0] * len(clips)


class FixedLeak:
    def words_per_minute(self, path):
        return 0.5


class FakeFactory:
    def __init__(self) -> None:
        self.cloner = FakeCloner()
        self.translator = FakeTranslator()
        self.mixer = FakeMixer()

    def release_idle(self):
        self.idle_released = getattr(self, "idle_released", 0) + 1

    def components(self):
        return [{"id": "fake", "kind": "motor TTS", "description": "", "license": ""}]

    def build(self, request):
        evaluator = DubEvaluator(EchoTranscriber(), FixedSpeaker(), FixedMos(), FixedLeak()) \
            if request.evaluate else None
        return DubPipeline(translator=self.translator, cloner=self.cloner,
                           audio=FakeAudioStore(), mixer=self.mixer, subtitles=SrtRenderer(),
                           muxer=FakeMuxer(),
                           separator=FakeSeparator() if request.separator else None,
                           evaluator=evaluator)


def transcript() -> Transcript:
    return Transcript((
        Segment(1.0, 2.0, "Hello there."),
        Segment(2.5, 6.5, "This is a long and complete sentence."),
        Segment(8.0, 9.0, "Bye."),
    ), "en", "fake", 20.0)


def no_transcripts(request, profiler):
    raise AssertionError("No debería transcribir: se pasa la transcripción")
