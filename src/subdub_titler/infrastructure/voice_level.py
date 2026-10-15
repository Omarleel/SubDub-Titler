"""Control de ráfagas de la voz sintetizada (decorador de cualquier `VoiceCloner`).

Los motores generativos a veces producen tramos de ~0.1–0.3 s muy por encima del nivel de la
frase (un «grito» que se estabiliza enseguida). Se atenúa lo que supera en `headroom_db` la
mediana de los tramos sonoros de la propia frase; la dinámica normal del habla no se toca.
"""
from collections.abc import Sequence

from subdub_titler.domain.dubbing import AudioClip
from subdub_titler.domain.ports import Releasable, VoiceCloner


def tame_bursts(samples, sample_rate: int, headroom_db: float = 8.0, frame: float = 0.02,
                smooth: int = 5):
    """Ganancia por tramos de `frame` s: 0 dB salvo donde el nivel supera la mediana sonora
    en más de `headroom_db`; suavizada (`smooth` tramos) e interpolada por muestra."""
    import numpy as np
    samples = np.asarray(samples, dtype=np.float32)
    hop = max(1, int(frame * sample_rate))
    count = len(samples) // hop
    if count < 8:
        return samples
    level = 10 * np.log10((samples[:count * hop].reshape(count, hop) ** 2).mean(1) + 1e-12)
    voiced = level > level.max() - 35
    if voiced.sum() < 5:
        return samples
    ceiling = np.median(level[voiced]) + headroom_db
    reduction = np.minimum(0.0, ceiling - level)
    if not reduction.any():
        return samples
    # Se extiende un tramo a cada lado y se suaviza: sin escalones audibles.
    padded = np.pad(reduction, 1, mode="edge")
    reduction = np.minimum(np.minimum(padded[:-2], padded[1:-1]), padded[2:])
    padded = np.pad(reduction, smooth // 2, mode="edge")
    reduction = np.convolve(padded, np.ones(smooth) / smooth, mode="valid")[:count]
    centers = (np.arange(count) + 0.5) * hop
    gain = 10 ** (np.interp(np.arange(len(samples)), centers, reduction) / 20)
    return (samples * gain).astype(np.float32)


class BurstGuard:
    """`VoiceCloner` que delega en otro y atenúa las ráfagas de cada frase generada."""

    def __init__(self, inner: VoiceCloner, headroom_db: float = 8.0) -> None:
        self.inner = inner
        self.headroom_db = headroom_db

    @property
    def sample_rate(self) -> int:
        return self.inner.sample_rate

    def prepare_voice(self, reference: AudioClip, text: str) -> object:
        return self.inner.prepare_voice(reference, text)

    def natural_seconds(self, texts: Sequence[str], language: str, voice: object) -> list[float]:
        return self.inner.natural_seconds(texts, language, voice)

    def synthesize(self, texts: Sequence[str], durations: Sequence[float], language: str,
                   voice: object, variant: int = 0) -> list[AudioClip]:
        clips = self.inner.synthesize(texts, durations, language, voice, variant)
        return [AudioClip(tame_bursts(c.samples, c.sample_rate, self.headroom_db), c.sample_rate)
                for c in clips]

    def release(self) -> None:
        if isinstance(self.inner, Releasable):
            self.inner.release()
