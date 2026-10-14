from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Language = Literal["auto", "en", "es", "ja", "zh"]
Device = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str

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
    word_timestamps: bool = False

    def __post_init__(self) -> None:
        if not self.source.is_file():
            raise FileNotFoundError(f"Archivo no encontrado: {self.source}")
        if self.beam_size < 1 or self.beam_size > 20:
            raise ValueError("beam_size debe estar entre 1 y 20")
        if self.language not in ("auto", "en", "es", "ja", "zh"):
            raise ValueError("Idioma no soportado")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Dispositivo inválido")


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


@dataclass(frozen=True, slots=True)
class RunResult:
    transcript: Transcript
    rtf: float | None
    elapsed_seconds: float
    events: tuple[ProfileEvent, ...] = field(default_factory=tuple)
