"""Único adaptador con dependencia explícita de OmniVoice (k2-fsa, motor de VoiceStudio).

Pesos `k2-fsa/OmniVoice`: licencia CC-BY-NC (uso no comercial). Clonar solo voces con permiso.
"""
import threading
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

from subdub_titler.domain.dubbing import AudioClip
from subdub_titler.infrastructure.profiling import active_profiler

MODEL_ID = "k2-fsa/OmniVoice"


def _require_cuda(device: str) -> None:
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        # Nunca degradar a CPU en silencio: el usuario pidió GPU.
        raise RuntimeError("CUDA no disponible para PyTorch; usa --device cpu explícitamente")


class OmniVoiceModel:
    """Carga diferida y compartida del modelo (un único proceso, un único worker GPU)."""

    def __init__(self, device: str = "cuda", dtype: str = "float16",
                 model_id: str = MODEL_ID) -> None:
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self._model = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._model is None:
                profiler = active_profiler()
                with profiler.measure("dub.tts.load") if profiler else nullcontext():
                    self._model = self._load()
            return self._model

    def _load(self):
        try:
            import torch
            from omnivoice import OmniVoice
        except ImportError as exc:
            raise RuntimeError(
                'Doblaje no instalado: python -m pip install -e ".[dub]" '
                "--extra-index-url https://download.pytorch.org/whl/cu128"
            ) from exc
        _require_cuda(self.device)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[self.dtype]
        device_map = "cuda:0" if self.device == "cuda" else "cpu"
        return OmniVoice.from_pretrained(self.model_id, device_map=device_map, dtype=dtype)


@dataclass(frozen=True, slots=True)
class OmniVoiceSettings:
    num_step: int = 16
    guidance_scale: float = 2.0
    position_temperature: float = 5.0
    seed: int | None = 0  # misma semilla + mismas formas = misma salida (reproducible)
    split_cfg: bool = True  # ver omnivoice_fast
    retry_num_step: int | None = None  # pasos de los reintentos (None = 2 × num_step)
    fade_seconds: float = 0.05
    # RMS al que se lleva la referencia. OmniVoice solo sube las referencias suaves (< 0.1) y
    # genera al nivel de la referencia: con una voz original muy alta (-9 dBFS) la salida
    # pasaba de 1.0 y se recortaba. None = referencia sin tocar.
    reference_rms: float | None = 0.1


class OmniVoiceCloner:
    sample_rate = 24_000

    def __init__(self, model: OmniVoiceModel, settings: OmniVoiceSettings | None = None):
        self._holder = model
        self.settings = settings or OmniVoiceSettings()

    def _model(self):
        model = self._holder.get()
        if self.settings.split_cfg and not getattr(model, "_subdub_split_cfg", False):
            from subdub_titler.infrastructure import omnivoice_fast
            model._subdub_split_cfg = omnivoice_fast.install(model)
        if not self.settings.split_cfg and getattr(model, "_subdub_split_cfg", False):
            del model._generate_iterative  # vuelve al método de la clase
            model._subdub_split_cfg = False
        return model

    def release(self) -> None:
        """Conserva el modelo (2 GB, regenerar cues es habitual) y devuelve la caché del
        asignador: las activaciones de los lotes dejan varios GB reservados."""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_voice(self, reference: AudioClip, text: str) -> object:
        import torch
        model = self._model()
        waveform = torch.as_tensor(reference.samples, dtype=torch.float32)
        if self.settings.reference_rms is not None:
            rms = float(waveform.square().mean().sqrt())
            if rms > self.settings.reference_rms:
                waveform = waveform * (self.settings.reference_rms / rms)
        with torch.inference_mode():
            return model.create_voice_clone_prompt((waveform, reference.sample_rate),
                                                   ref_text=text or None)

    def natural_seconds(self, texts: Sequence[str], language: str, voice: object) -> list[float]:
        """Mismo estimador por reglas que usa OmniVoice para fijar la longitud objetivo."""
        model = self._model()
        frame_rate = model.audio_tokenizer.config.frame_rate
        ref_frames = voice.ref_audio_tokens.size(-1)
        return [max(1, int(model.duration_estimator.estimate_duration(
            text, voice.ref_text, ref_frames))) / frame_rate for text in texts]

    def synthesize(self, texts: Sequence[str], durations: Sequence[float], language: str,
                   voice: object, variant: int = 0) -> list[AudioClip]:
        import torch
        if not texts:
            return []
        model = self._model()
        if self.settings.seed is not None:
            torch.manual_seed(self.settings.seed + variant)
        # Reintentos: otra semilla y el doble de pasos. A/B en la misma ejecución (video de
        # 23 min): CER 4.75 % sin reintentos, 4.07 % reintentando a 16 pasos, 3.44 % a 32.
        steps = self.settings.num_step if variant == 0 else (
            self.settings.retry_num_step or min(64, self.settings.num_step * 2))
        with torch.inference_mode():
            audios = model.generate(
                text=list(texts), language=language, voice_clone_prompt=[voice] * len(texts),
                duration=[float(d) for d in durations], num_step=steps,
                guidance_scale=self.settings.guidance_scale,
                position_temperature=self.settings.position_temperature,
                class_temperature=0.0,
                # Duración exacta: sin recorte de silencios ni relleno (añadía 2 x 0.1 s).
                postprocess_output=False, pad_duration=0.0,
                fade_duration=self.settings.fade_seconds,
            )
        if torch.cuda.is_available():
            # Cada lote tiene otra longitud: sin vaciar, la caché del asignador se fragmenta
            # (decodificación original: 10.3 GB reservados para 3.2 GB usados; vaciando 3.7 GB,
            # +1.8 % de tiempo) y desborda los 16 GB junto al verificador.
            torch.cuda.empty_cache()
        return [AudioClip(audio, model.sampling_rate) for audio in audios]
