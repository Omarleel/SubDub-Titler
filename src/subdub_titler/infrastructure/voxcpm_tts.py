"""Adaptador VoxCPM2 (OpenBMB): motor alternativo con pesos Apache-2.0 (uso comercial).

Clona en modo «solo referencia». No controla la duración: se genera a ritmo natural y se
ajusta a la duración pedida con time-stretch (ffmpeg `atempo`, conserva el tono). La duración
natural se estima con el ritmo del hablante en la referencia (`estimate_seconds_by_rate`).
Medido con el corpus dorado (40 frases, es): CER 8.7 %, parecido 0.27, RTF 1.0; OmniVoice:
4.9 %, 0.59, 0.13. Útil cuando la licencia no comercial de OmniVoice no es aceptable.
"""
import tempfile
import threading
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from subdub_titler.domain.dubbing import AudioClip, estimate_seconds_by_rate
from subdub_titler.infrastructure.profiling import active_profiler

MODEL_ID = "openbmb/VoxCPM2"


@dataclass(frozen=True, slots=True)
class VoxVoice:
    cache: object
    text: str
    seconds: float


class VoxCpmModel:
    def __init__(self, device: str = "cuda", model_id: str = MODEL_ID) -> None:
        self.device = device
        self.model_id = model_id
        self._model = None
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            if self._model is None:
                import torch
                from voxcpm import VoxCPM
                if self.device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA no disponible para VoxCPM; usa --device cpu")
                profiler = active_profiler()
                with profiler.measure("dub.tts.load") if profiler else nullcontext():
                    # optimize=False: torch.compile necesita Triton, ausente en Windows.
                    self._model = VoxCPM.from_pretrained(self.model_id, load_denoiser=False,
                                                         optimize=False, device=self.device)
            return self._model


class VoxCpmCloner:
    sample_rate = 48_000

    def __init__(self, model: VoxCpmModel, timesteps: int = 10, cfg: float = 2.0,
                 seed: int | None = 0) -> None:
        self._holder = model
        self.timesteps = timesteps
        self.cfg = cfg
        self.seed = seed

    def release(self) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_voice(self, reference: AudioClip, text: str) -> object:
        import soundfile
        model = self._holder.get()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.wav"
            soundfile.write(str(path), reference.samples, reference.sample_rate)
            # Solo referencia (aislada): en modo continuación (prompt_text + audio) la voz en
            # otro idioma se degrada (CER ida y vuelta 23.7 % frente a 3.2 %, corpus es).
            cache = model.tts_model.build_prompt_cache(reference_wav_path=str(path))
        return VoxVoice(cache, text, reference.duration)

    def natural_seconds(self, texts: Sequence[str], language: str, voice: object) -> list[float]:
        return [estimate_seconds_by_rate(t, voice.text, voice.seconds) for t in texts]

    def synthesize(self, texts: Sequence[str], durations: Sequence[float], language: str,
                   voice: object) -> list[AudioClip]:
        import torch

        from subdub_titler.infrastructure.media import time_stretch
        model = self._holder.get()
        clips = []
        for text, duration in zip(texts, durations, strict=True):
            if self.seed is not None:
                torch.manual_seed(self.seed)
            with torch.inference_mode():
                result = model.tts_model._generate_with_prompt_cache(
                    target_text=text, prompt_cache=voice.cache, min_len=2, max_len=4096,
                    inference_timesteps=self.timesteps, cfg_value=self.cfg,
                    retry_badcase=True, retry_badcase_max_times=3,
                    retry_badcase_ratio_threshold=6.0, streaming=False)
                wav, _, _ = next(result)
                result.close()
            samples = wav.squeeze(0).float().cpu().numpy()
            clips.append(AudioClip(time_stretch(samples, self.sample_rate, duration),
                                   self.sample_rate))
        return clips
