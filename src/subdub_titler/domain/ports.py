from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .dubbing import Ambience, AudioClip, Candidate, ProsodyProfile
from .models import ModelDescriptor, Transcript, TranscriptionRequest, Word


class ASREngine(Protocol):
    """Contrato estable: un adaptador devuelve siempre un Transcript del dominio."""

    def transcribe(self, request: TranscriptionRequest) -> Transcript: ...


class EngineRegistry(Protocol):
    def list_models(self) -> tuple[ModelDescriptor, ...]: ...

    def engine_for(self, request: TranscriptionRequest) -> ASREngine: ...


class SubtitleRenderer(Protocol):
    def render(self, transcript: Transcript) -> str: ...


ProfilerFactory = Callable[[], object]


@runtime_checkable
class Releasable(Protocol):
    """Componente que puede liberar su memoria de GPU al terminar su etapa y recargarse solo
    al volver a usarse. Opcional: con todos los modelos del doblaje cargados a la vez un único
    proceso supera los 16 GB y Windows desborda a memoria compartida (mucho más lento)."""

    def release(self) -> None: ...


class Profiler(Protocol):
    """Spans con nombre; la implementación decide qué mide (tiempo, VRAM, NVTX)."""

    def measure(self, name: str) -> AbstractContextManager[None]: ...


# --- Doblaje -------------------------------------------------------------------------------


class Translator(Protocol):
    def translate(self, texts: Sequence[str], source: str, target: str,
                  n_best: int = 1) -> list[list[Candidate]]:
        """Una lista de candidatos (mejor primero) por texto de entrada."""
        ...


class VoiceCloner(Protocol):
    """Síntesis con clonación de voz y duración controlada."""

    sample_rate: int

    def prepare_voice(self, reference: AudioClip, text: str) -> object:
        """Handle opaco de la voz (p. ej. prompt codificado), reutilizable entre llamadas."""
        ...

    def natural_seconds(self, texts: Sequence[str], language: str, voice: object) -> list[float]:
        """Duración que el motor elegiría sin restricción para cada texto."""
        ...

    def synthesize(self, texts: Sequence[str], durations: Sequence[float], language: str,
                   voice: object, variant: int = 0) -> list[AudioClip]:
        """`variant` > 0: generación alternativa para reintentos (otra semilla). Misma
        variante, mismo texto y voz = misma salida."""
        ...


@dataclass(frozen=True, slots=True)
class Stems:
    voice: Path
    background: Path


class SourceSeparator(Protocol):
    def separate(self, source: Path, workdir: Path) -> Stems: ...


@dataclass(frozen=True, slots=True)
class AudioQuality:
    clipping_ratio: float
    is_saturated: bool
    rms: float
    is_too_quiet: bool


class AudioStore(Protocol):
    def load(self, path: Path, sample_rate: int, start: float | None = None,
             end: float | None = None) -> AudioClip: ...

    def save(self, clip: AudioClip, path: Path) -> Path: ...

    def duration(self, path: Path) -> float: ...

    def check_audio_quality(self, clip: AudioClip) -> AudioQuality: ...


class ProsodyAnalyzer(Protocol):
    """Extracción y transferencia de prosodia y expresividad de la voz original."""

    def analyze(self, clip: AudioClip, words: Sequence[Word] = ()) -> ProsodyProfile: ...

    def transfer_energy(self, clip: AudioClip, target: ProsodyProfile,
                        max_boost_db: float = 4.0) -> AudioClip: ...


@dataclass(frozen=True, slots=True)
class MixResult:
    voice: Path
    mix: Path
    lufs: float | None
    true_peak: float | None
    bed: Path | None = None  # fondo + ambiente conservado (lo que no es voz doblada)


class Mixer(Protocol):
    def render(self, placements: Sequence[tuple[float, AudioClip]], background: Path | None,
               duration: float, out_dir: Path,
               voice_reference: Path | None = None,
               ambience: Ambience | None = None) -> MixResult:
        """`voice_reference`: voz original separada; la doblada se iguala a su sonoridad.
        `ambience`: tramos de esa voz sin diálogo que se suman al fondo."""
        ...


class Muxer(Protocol):
    def mux(self, video: Path, dub_audio: Path, subtitles: Path | None, language: str,
            out: Path) -> Path: ...


class ClipTranscriber(Protocol):
    def transcribe_clips(self, clips: Sequence[AudioClip], language: str) -> list[str]: ...


class LanguageIdentifier(Protocol):
    """Probabilidad de que cada clip suene al idioma dado: mide el acento extranjero que la
    clonación arrastra de la referencia (más alta = más nativo)."""

    def language_probability(self, clips: Sequence[AudioClip], language: str) -> list[float]: ...


class SpeakerEncoder(Protocol):
    def similarity(self, reference: AudioClip, clips: Sequence[AudioClip]) -> list[float]: ...


class NaturalnessPredictor(Protocol):
    def mos(self, clips: Sequence[AudioClip]) -> list[float]: ...


class SpeakerDiarizer(Protocol):
    def assign(self, source: Path, spans: Sequence[tuple[float, float]],
               max_speakers: int) -> list[str]:
        """Etiqueta de hablante (S0, S1… por orden de aparición) para cada intervalo."""
        ...


class SpeechLeakMeter(Protocol):
    """Palabras por minuto que un ASR reconoce en una pista que no debería tener voz."""

    def words_per_minute(self, path: Path) -> float: ...
