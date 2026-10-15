"""Separación voz/fondo con Demucs (MIT). Único adaptador con dependencia de demucs.

La voz separada sirve de referencia limpia para clonar; el fondo, de base para la mezcla.
Los stems se reutilizan si ya existen y son más recientes que el origen (benchmarks).
"""
import shutil
import subprocess
import threading
from contextlib import nullcontext
from pathlib import Path

from subdub_titler.domain.ports import Stems
from subdub_titler.infrastructure.profiling import active_profiler


class DemucsSeparator:
    def __init__(self, model_name: str = "htdemucs", device: str = "cuda", shifts: int = 0,
                 overlap: float = 0.25) -> None:
        self.model_name = model_name
        self.device = device
        self.shifts = shifts
        self.overlap = overlap
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                import torch
                from demucs.pretrained import get_model
                if self.device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA no disponible para Demucs; usa --device cpu")
                profiler = active_profiler()
                with profiler.measure("dub.separate.load") if profiler else nullcontext():
                    model = get_model(self.model_name)
                    model.eval()
                    model.to(self.device)
                    self._model = model
            return self._model

    def release(self) -> None:
        with self._lock:
            if self._model is not None:
                import torch
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def separate(self, source: Path, workdir: Path) -> Stems:
        stems = Stems(workdir / "voice.flac", workdir / "background.flac")
        if all(p.is_file() and p.stat().st_mtime >= source.stat().st_mtime
               for p in (stems.voice, stems.background)):
            return stems
        import numpy as np
        import soundfile
        import torch
        from demucs.apply import apply_model

        model = self._load()
        rate, channels = model.samplerate, model.audio_channels
        raw = subprocess.run(
            [shutil.which("ffmpeg") or "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
             "-vn", "-ac", str(channels), "-ar", str(rate), "-f", "f32le", "-"],
            capture_output=True, check=True).stdout
        mix = torch.from_numpy(np.frombuffer(raw, np.float32).reshape(-1, channels).T.copy())
        # Misma normalización que demucs.api: la red se entrenó con mezclas estandarizadas.
        reference = mix.mean(0)
        mean, std = reference.mean(), reference.std() + 1e-8
        profiler = active_profiler()
        with profiler.measure("dub.separate.apply") if profiler else nullcontext():
            with torch.inference_mode():
                sources = apply_model(model, ((mix - mean) / std)[None], shifts=self.shifts,
                                      split=True, overlap=self.overlap, device=self.device,
                                      progress=False)[0]
        sources = sources * std + mean
        vocals = model.sources.index("vocals")
        voice = sources[vocals]
        background = sources.sum(0) - voice
        workdir.mkdir(parents=True, exist_ok=True)
        for path, data in ((stems.voice, voice), (stems.background, background)):
            soundfile.write(str(path), data.T.numpy(), rate)
        del sources, voice, background
        if self.device == "cuda":
            # Las activaciones de Demucs dejan GB reservados en el asignador; en un único proceso
            # con todos los modelos cargados (16 GB) Windows desbordaría a memoria compartida.
            torch.cuda.empty_cache()
        return stems
