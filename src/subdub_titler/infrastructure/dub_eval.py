"""Medidores de calidad de voz con el mismo protocolo que el paper de OmniVoice.

- Parecido de voz (SIM-o): WavLM-large + ECAPA-TDNN afinado (`k2-fsa/TTS_eval_models`).
- Naturalidad: UTMOS22 strong (mismo repositorio).
- Fuga de voz: palabras por minuto que reconoce el ASR en el stem de fondo.
Solo se descargan los pesos necesarios (~3 GB), no el repositorio completo (14.5 GB).
"""
import os
import threading
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

from subdub_titler.domain.dubbing import AudioClip
from subdub_titler.domain.models import TranscriptionRequest
from subdub_titler.domain.ports import ASREngine
from subdub_titler.infrastructure.profiling import active_profiler

EVAL_REPO = "k2-fsa/TTS_eval_models"
EVAL_RATE = 16_000
MIN_EVAL_SAMPLES = 1600  # 100 ms a 16 kHz: margen seguro para WavLM (L >= 4) y UTMOS


def eval_models_dir() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(EVAL_REPO, allow_patterns=[
        "mos/*", "speaker_similarity/wavlm_large/*",
        "speaker_similarity/wavlm_large_finetune.pth",
    ]))


def _waveform(clip: AudioClip, device: str, min_samples: int = MIN_EVAL_SAMPLES):
    import torch
    import torchaudio
    wave = torch.as_tensor(clip.samples, dtype=torch.float32)
    if clip.sample_rate != EVAL_RATE:
        wave = torchaudio.functional.resample(wave, clip.sample_rate, EVAL_RATE)
    if wave.ndim == 1:
        if wave.numel() == 0:
            wave = torch.zeros(min_samples, dtype=torch.float32)
        elif wave.numel() < min_samples:
            pad_needed = min_samples - wave.numel()
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            wave = torch.nn.functional.pad(wave, (pad_left, pad_right), mode="constant", value=0.0)
    elif wave.ndim == 2:
        if wave.shape[-1] == 0:
            wave = torch.zeros((wave.shape[0], min_samples), dtype=torch.float32)
        elif wave.shape[-1] < min_samples:
            pad_needed = min_samples - wave.shape[-1]
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            wave = torch.nn.functional.pad(wave, (pad_left, pad_right), mode="constant", value=0.0)
    return wave.to(device)


class _LazyTorchModel:
    def __init__(self, device: str, span: str) -> None:
        self.device = device
        self._span = span
        self._model = None
        self._lock = threading.Lock()

    def _build(self):
        raise NotImplementedError

    def release(self) -> None:
        with self._lock:
            if self._model is not None:
                import torch
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def model(self):
        with self._lock:
            if self._model is None:
                import torch
                if self.device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA no disponible para los evaluadores")
                profiler = active_profiler()
                with profiler.measure(self._span) if profiler else nullcontext():
                    self._model = self._build().to(self.device).eval()
            return self._model


class WavLmSpeakerEncoder(_LazyTorchModel):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device, "dub.eval.load_sim")

    def _build(self):
        import torch
        from omnivoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
        root = eval_models_dir() / "speaker_similarity"
        model = ECAPA_TDNN_WAVLM(feat_dim=1024, channels=512, emb_dim=256, sr=EVAL_RATE,
                                 ssl_model_path=os.path.join(root, "wavlm_large", ""))
        state = torch.load(root / "wavlm_large_finetune.pth", map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        return model

    def embeddings(self, clips: Sequence[AudioClip]) -> list[list[float]]:
        if not clips:
            return []
        import torch
        model = self.model()
        with torch.inference_mode():
            return [model([_waveform(clip, self.device)])[0].float().cpu().tolist()
                    for clip in clips]

    def similarity(self, reference: AudioClip, clips: Sequence[AudioClip]) -> list[float]:
        if not clips:
            return []
        import torch
        model = self.model()
        with torch.inference_mode():
            anchor = model([_waveform(reference, self.device)])
            return [float(torch.nn.functional.cosine_similarity(
                anchor, model([_waveform(clip, self.device)])).item()) for clip in clips]


class Utmos22Predictor(_LazyTorchModel):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device, "dub.eval.load_mos")

    def _build(self):
        import torch
        from omnivoice.eval.models.utmos import UTMOS22Strong
        model = UTMOS22Strong()
        model.load_state_dict(torch.load(
            eval_models_dir() / "mos" / "utmos22_strong_step7459_v1.pt", map_location="cpu"))
        return model

    def mos(self, clips: Sequence[AudioClip]) -> list[float]:
        if not clips:
            return []
        import torch
        model = self.model()
        with torch.inference_mode():
            return [float(model(_waveform(clip, self.device).unsqueeze(0), EVAL_RATE).item())
                    for clip in clips]


class AsrLeakMeter:
    """Palabras por minuto reconocidas en una pista sin voz (con VAD, como en producción)."""

    def __init__(self, engine: ASREngine, model_id: str, device: str, compute_type: str) -> None:
        self._engine = engine
        self._request = dict(model_id=model_id, device=device, compute_type=compute_type)

    def words_per_minute(self, path: Path) -> float:
        transcript = self._engine.transcribe(TranscriptionRequest(path, **self._request))
        words = sum(len(s.text.split()) for s in transcript.segments)
        minutes = (transcript.duration_seconds or 0) / 60
        return words / minutes if minutes else 0.0
