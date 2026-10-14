from collections.abc import Callable
from typing import Protocol

from .models import ModelDescriptor, Transcript, TranscriptionRequest


class ASREngine(Protocol):
    """Contrato estable: un adaptador devuelve siempre un Transcript del dominio."""

    def transcribe(self, request: TranscriptionRequest) -> Transcript: ...


class EngineRegistry(Protocol):
    def list_models(self) -> tuple[ModelDescriptor, ...]: ...

    def engine_for(self, request: TranscriptionRequest) -> ASREngine: ...


class SubtitleRenderer(Protocol):
    def render(self, transcript: Transcript) -> str: ...


ProfilerFactory = Callable[[], object]
