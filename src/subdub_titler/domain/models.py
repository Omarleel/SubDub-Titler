from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Language = Literal["auto", "en", "es", "ja", "zh"]
Device = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class Word:
    """Palabra con tiempos del motor; `text` conserva el espacio inicial del tokenizador."""
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Intervalo de segmento inválido")


@dataclass(frozen=True, slots=True)
class Transcript:
    segments: tuple[Segment, ...]
    language: str
    model_id: str
    duration_seconds: float | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    source: Path
    model_id: str = "small"
    language: Language = "auto"
    device: Device = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    batch_size: int = 16  # 1 = decodificación secuencial
    vad_threshold: float = 0.35
    vad_speech_pad_ms: int = 800
    vad_min_silence_ms: int = 2000
    no_speech_threshold: float = 0.40
    normalize_loudness: bool = False
    isolate_dialogue: bool = False

    def __post_init__(self) -> None:
        if not self.source.is_file():
            raise FileNotFoundError(f"Archivo no encontrado: {self.source}")
        if self.beam_size < 1 or self.beam_size > 20:
            raise ValueError("beam_size debe estar entre 1 y 20")
        if self.batch_size < 1 or self.batch_size > 64:
            raise ValueError("batch_size debe estar entre 1 y 64")
        if self.language not in ("auto", "en", "es", "ja", "zh"):
            raise ValueError("Idioma no soportado")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Dispositivo inválido")
        if not (0.0 <= self.vad_threshold <= 1.0):
            raise ValueError("vad_threshold debe estar entre 0.0 y 1.0")
        if self.vad_speech_pad_ms < 0:
            raise ValueError("vad_speech_pad_ms debe ser >= 0")
        if self.vad_min_silence_ms < 0:
            raise ValueError("vad_min_silence_ms debe ser >= 0")
        if not (0.0 <= self.no_speech_threshold <= 1.0):
            raise ValueError("no_speech_threshold debe estar entre 0.0 y 1.0")


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    id: str
    backend: str
    languages: tuple[str, ...] = ("en", "es", "ja", "zh")
    supports_word_timestamps: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class ProfileEvent:
    name: str
    milliseconds: float
    parent: str | None = None
    depth: int = 0
    vram_peak_mb: float | None = None  # pico de memoria CUDA de PyTorch dentro del span
    vram_device_mb: float | None = None  # memoria ocupada en la GPU al cerrar el span (todo)


@dataclass(frozen=True, slots=True)
class RunResult:
    transcript: Transcript
    rtf: float | None
    elapsed_seconds: float
    events: tuple[ProfileEvent, ...] = field(default_factory=tuple)
