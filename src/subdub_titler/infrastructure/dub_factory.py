"""Composición del doblaje. OCP: motores TTS, traductores y separadores se registran con una
fábrica; un motor nuevo (CosyVoice, IndexTTS, la API de VoiceStudio…) es un adaptador más un
`register`, sin tocar los casos de uso. Las dependencias pesadas se importan de forma diferida.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from subdub_titler.application.dub_quality import DubEvaluator
from subdub_titler.application.dubbing import DubPipeline
from subdub_titler.domain.dubbing import DubRequest
from subdub_titler.domain.models import TranscriptionRequest
from subdub_titler.domain.ports import SourceSeparator, Translator, VoiceCloner
from subdub_titler.infrastructure.registry import ModelRegistry
from subdub_titler.infrastructure.voice_level import BurstGuard

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    id: str
    kind: str
    description: str
    license: str = ""


class ComponentRegistry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[[DubRequest], T]] = {}
        self._info: dict[str, ComponentInfo] = {}

    def register(self, identifier: str, factory: Callable[[DubRequest], T],
                 description: str = "", license: str = "") -> None:
        if identifier in self._factories:
            raise ValueError(f"{self.kind} duplicado: {identifier}")
        self._factories[identifier] = factory
        self._info[identifier] = ComponentInfo(identifier, self.kind, description, license)

    def create(self, identifier: str, request: DubRequest) -> T:
        if identifier not in self._factories:
            raise ValueError(f"{self.kind} no registrado: {identifier}")
        return self._factories[identifier](request)

    def list(self) -> tuple[ComponentInfo, ...]:
        return tuple(self._info.values())


class Memo:
    """Caché de instancias pesadas por clave (pesos cargados una sola vez por proceso)."""

    def __init__(self) -> None:
        self._items: dict[tuple, object] = {}

    def get(self, key: tuple, build: Callable[[], T]) -> T:
        if key not in self._items:
            self._items[key] = build()
        return self._items[key]  # type: ignore[return-value]


class DubFactory:
    def __init__(self, asr_registry: ModelRegistry, eval_asr_model: str = "large-v3",
                 compute_type: str = "float16") -> None:
        self.asr_registry = asr_registry
        self.eval_asr_model = eval_asr_model
        self.compute_type = compute_type
        self.translators: ComponentRegistry[Translator] = ComponentRegistry("traductor")
        self.cloners: ComponentRegistry[VoiceCloner] = ComponentRegistry("motor TTS")
        self.separators: ComponentRegistry[SourceSeparator] = ComponentRegistry("separador")
        self.memo = Memo()

    def components(self) -> list[dict]:
        from dataclasses import asdict
        return [asdict(info) for registry in (self.cloners, self.translators, self.separators)
                for info in registry.list()]

    def release_idle(self) -> None:
        """Whisper de trabajos anteriores fuera de la GPU (OmniVoice se conserva: 2 GB que se
        reutilizan al regenerar cues; sus activaciones ya se liberan por etapa)."""
        self.asr_registry.release_cached()

    def build(self, request: DubRequest) -> DubPipeline:
        from subdub_titler.infrastructure.media import FfmpegAudioStore, FfmpegMixer, FfmpegMuxer
        from subdub_titler.infrastructure.prosody import NumpyProsodyAnalyzer
        from subdub_titler.infrastructure.subtitles import SrtRenderer
        return DubPipeline(
            translator=self.translators.create(request.translator, request),
            cloner=self.cloner(request),
            audio=FfmpegAudioStore(), mixer=self.memo.get(("mixer",), FfmpegMixer),
            subtitles=SrtRenderer(),
            muxer=FfmpegMuxer(),
            separator=self.separators.create(request.separator, request)
            if request.separator else None,
            evaluator=self.evaluator(request) if request.evaluate else None,
            diarizer=self.diarizer(request) if request.diarize else None,
            # El verificador es el ASR de la transcripción (ya cargado); la evaluación final
            # usa otro (large-v3), así los reintentos no optimizan la métrica reportada.
            verifier=self.whisper(request.asr_model_id, request.device)
            if request.verify_attempts or request.reference_candidates > 1 else None,
            speaker_encoder=self.speaker_encoder(request.device)
            if request.reference_candidates > 1 else None,
            prosody=NumpyProsodyAnalyzer(),
        )

    def cloner(self, request: DubRequest) -> VoiceCloner:
        """Motor TTS pedido con el control de ráfagas (A/B en un video real: frases con
        ráfagas 44 -> 0, CER/parecido/UTMOS sin cambio significativo)."""
        return BurstGuard(self.cloners.create(request.tts_engine, request))

    def speaker_encoder(self, device: str):
        from subdub_titler.infrastructure.dub_eval import WavLmSpeakerEncoder
        return self.memo.get(("sim", device), lambda: WavLmSpeakerEncoder(device))

    def diarizer(self, request: DubRequest):
        from subdub_titler.infrastructure.diarization import EmbeddingDiarizer
        from subdub_titler.infrastructure.media import FfmpegAudioStore
        return EmbeddingDiarizer(self.speaker_encoder(request.device), FfmpegAudioStore())

    def whisper(self, model_id: str, device: str):
        """Adaptador Whisper compartido con el registro ASR (no carga dos veces los pesos)."""
        from subdub_titler.infrastructure.whisper import FasterWhisperAdapter
        compute = self.compute_type if device == "cuda" else "int8"
        engine = self.asr_registry.cached_engine(model_id, device, compute)
        if engine is None:
            engine = FasterWhisperAdapter(model_id, device, compute)
            self.asr_registry.remember(model_id, device, compute, engine)
        return engine

    def evaluator(self, request: DubRequest) -> DubEvaluator:
        from subdub_titler.infrastructure.dub_eval import AsrLeakMeter, Utmos22Predictor
        device = request.device
        whisper = self.whisper(self.eval_asr_model, device)
        compute = self.compute_type if device == "cuda" else "int8"
        return DubEvaluator(
            transcriber=whisper,
            speaker=self.speaker_encoder(device),
            naturalness=self.memo.get(("mos", device), lambda: Utmos22Predictor(device)),
            leak_meter=AsrLeakMeter(self.whisper(request.asr_model_id, device),
                                    request.asr_model_id, device, compute),
            language_id=whisper,
        )


def default_dub_factory(asr_registry: ModelRegistry) -> DubFactory:
    factory = DubFactory(asr_registry)

    def omnivoice(split_cfg: bool) -> Callable[[DubRequest], VoiceCloner]:
        def build(request: DubRequest) -> VoiceCloner:
            from subdub_titler.infrastructure.omnivoice_tts import (
                OmniVoiceCloner,
                OmniVoiceModel,
                OmniVoiceSettings,
            )
            model = factory.memo.get(("omnivoice", request.device),
                                     lambda: OmniVoiceModel(request.device))
            return OmniVoiceCloner(model, OmniVoiceSettings(
                num_step=request.num_step,
                split_cfg=split_cfg,
                text_guidance=request.text_guidance,
                reference_guidance=request.reference_guidance,
                accent_guidance_beta=request.accent_guidance,
                accent_proxy_dir=request.accent_proxy_dir,
            ))
        return build

    factory.cloners.register("omnivoice", omnivoice(True),
                             "OmniVoice 0.6B (k2-fsa), clonación zero-shot y duración exacta; "
                             "CFG en pasadas separadas", "pesos CC-BY-NC")
    factory.cloners.register("omnivoice-base", omnivoice(False),
                             "OmniVoice con la decodificación original (referencia A/B)",
                             "pesos CC-BY-NC")

    def voxcpm(request: DubRequest) -> VoiceCloner:
        from subdub_titler.infrastructure.voxcpm_tts import VoxCpmCloner, VoxCpmModel
        model = factory.memo.get(("voxcpm", request.device), lambda: VoxCpmModel(request.device))
        return VoxCpmCloner(model)

    factory.cloners.register("voxcpm2", voxcpm,
                             "VoxCPM2 2B (OpenBMB): sin control de duración, se ajusta con "
                             "time-stretch", "pesos Apache-2.0 (uso comercial)")

    def nllb(size: str) -> Callable[[DubRequest], Translator]:
        def build(request: DubRequest) -> Translator:
            from subdub_titler.infrastructure.nllb import NLLB_MODELS, NllbTranslator
            length_penalty = 0.8 if request.concise_translation else 1.0
            return factory.memo.get((size, request.device, length_penalty), lambda: NllbTranslator(
                NLLB_MODELS[size], request.device, length_penalty=length_penalty))
        return build

    factory.translators.register("nllb-600m", nllb("nllb-600m"),
                                 "NLLB-200 distilled 600M sobre CTranslate2", "pesos CC-BY-NC")
    factory.translators.register("nllb-1.3b", nllb("nllb-1.3b"),
                                 "NLLB-200 distilled 1.3B sobre CTranslate2", "pesos CC-BY-NC")

    def demucs(name: str) -> Callable[[DubRequest], SourceSeparator]:
        def build(request: DubRequest) -> SourceSeparator:
            from subdub_titler.infrastructure.demucs_separator import DemucsSeparator
            return factory.memo.get((name, request.device),
                                    lambda: DemucsSeparator(name, request.device))
        return build

    factory.separators.register("htdemucs", demucs("htdemucs"), "Demucs v4 híbrido", "MIT")
    factory.separators.register("htdemucs_ft", demucs("htdemucs_ft"),
                                "Demucs v4 afinado (4 modelos, ~4x más lento)", "MIT")
    return factory


class AsrTranscripts:
    """Proveedor de transcripción para el doblaje basado en el caso de uso de ASR."""

    def __init__(self, registry: ModelRegistry, compute_type: str = "float16") -> None:
        from subdub_titler.application.transcribe import TranscribeUseCase
        self._use_case = TranscribeUseCase(registry)
        self._compute_type = compute_type

    def __call__(self, request: DubRequest, profiler):
        compute = self._compute_type if request.device == "cuda" else "int8"
        result = self._use_case.execute(TranscriptionRequest(
            request.source, request.asr_model_id, request.source_language, request.device,
            compute), profiler)
        return result.transcript
