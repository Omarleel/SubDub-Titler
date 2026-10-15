"""OCP: registros de fábricas, importación diferida de dependencias GPU."""
from collections.abc import Callable

from subdub_titler.domain.models import ModelDescriptor, TranscriptionRequest
from subdub_titler.domain.ports import ASREngine, Releasable


# Presets verificados como nombres del catálogo de faster-whisper.
PRESETS = (
    ModelDescriptor("tiny", "faster-whisper", description="Pruebas rápidas"),
    ModelDescriptor("base", "faster-whisper", description="GPU/CPU de desarrollo"),
    ModelDescriptor("small", "faster-whisper", description="Predeterminado"),
    ModelDescriptor("medium", "faster-whisper", description="Más precisión"),
    ModelDescriptor("large-v3", "faster-whisper", description="Alta calidad multilingüe"),
    ModelDescriptor("turbo", "faster-whisper", description="Menor latencia"),
)


class ModelRegistry:
    def __init__(self) -> None:
        self._descriptors: dict[str, ModelDescriptor] = {}
        self._factories: dict[str, Callable[[TranscriptionRequest], ASREngine]] = {}
        self._engine_cache: dict[tuple[str, str, str], ASREngine] = {}

    def register(self, descriptor: ModelDescriptor,
                 factory: Callable[[TranscriptionRequest], ASREngine]) -> None:
        if descriptor.id in self._descriptors:
            raise ValueError(f"Modelo duplicado: {descriptor.id}")
        self._descriptors[descriptor.id] = descriptor
        self._factories[descriptor.id] = factory

    def list_models(self) -> tuple[ModelDescriptor, ...]:
        return tuple(self._descriptors.values())

    def engine_for(self, request: TranscriptionRequest) -> ASREngine:
        if request.model_id not in self._factories:
            raise ValueError(f"Modelo no registrado: {request.model_id}")
        key = (request.model_id, request.device, request.compute_type)
        if key not in self._engine_cache:
            self._engine_cache[key] = self._factories[request.model_id](request)
        return self._engine_cache[key]

    def cached_engine(self, model_id: str, device: str, compute_type: str) -> ASREngine | None:
        """Motor ya construido (comparte pesos cargados con otros usos, p. ej. evaluación)."""
        return self._engine_cache.get((model_id, device, compute_type))

    def remember(self, model_id: str, device: str, compute_type: str, engine: ASREngine) -> None:
        self._engine_cache[(model_id, device, compute_type)] = engine

    def release_cached(self) -> None:
        """Descarga de la GPU los motores en caché; siguen registrados y recargan al usarse."""
        for engine in self._engine_cache.values():
            if isinstance(engine, Releasable):
                engine.release()


def default_registry() -> ModelRegistry:
    from subdub_titler.infrastructure.whisper import FasterWhisperAdapter
    registry = ModelRegistry()
    for descriptor in PRESETS:
        registry.register(descriptor, lambda request: FasterWhisperAdapter(
            request.model_id, request.device, request.compute_type,
        ))
    return registry
