"""Extracción y transferencia de prosodia y expresividad de la voz original.

Módulo de infraestructura: utiliza NumPy de forma diferida.
Calcula pausas, ritmo, rango dinámico, contorno de energía y curva tonal fundamental (F0).
"""
from collections.abc import Sequence

from subdub_titler.domain.dubbing import AudioClip, ProsodyProfile
from subdub_titler.domain.models import Word


def _estimate_f0_contour(samples, sample_rate: int, frame_sec: float = 0.04,
                         hop_sec: float = 0.02) -> tuple[float | None, float | None]:
    """Estima la frecuencia fundamental (F0) media y su desviación estándar (entonación)
    mediante autocorrelación normalizada sobre tramos sonoros.
    Rango de búsqueda vocal: 70 Hz a 400 Hz.
    """
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32)
    frame_len = int(round(frame_sec * sample_rate))
    hop_len = int(round(hop_sec * sample_rate))
    if len(samples) < frame_len or frame_len <= 0:
        return None, None

    min_lag = max(1, int(round(sample_rate / 400.0)))
    max_lag = min(frame_len - 1, int(round(sample_rate / 70.0)))
    if min_lag >= max_lag:
        return None, None

    num_frames = (len(samples) - frame_len) // hop_len + 1
    if num_frames <= 0:
        return None, None

    # Tramos sonoros por nivel RMS
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_len)[::hop_len]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    max_rms = np.max(rms)
    if max_rms < 1e-4:
        return None, None

    voiced_mask = rms > max(0.01, max_rms * 0.15)
    voiced_frames = frames[voiced_mask]
    if len(voiced_frames) < 3:
        return None, None

    f0_values: list[float] = []
    for frame in voiced_frames:
        frame_zero = frame - np.mean(frame)
        energy = np.dot(frame_zero, frame_zero)
        if energy < 1e-8:
            continue
        corr = np.correlate(frame_zero, frame_zero, mode="full")
        corr = corr[len(frame_zero) - 1:]
        search = corr[min_lag:max_lag + 1]
        best_rel = int(np.argmax(search))
        best_lag = min_lag + best_rel
        norm_val = corr[best_lag] / energy
        if norm_val > 0.35:  # umbral de periodicidad armónica
            f0_values.append(sample_rate / float(best_lag))

    if not f0_values:
        return None, None

    arr = np.array(f0_values, dtype=np.float32)
    return round(float(np.mean(arr)), 1), round(float(np.std(arr)), 1)


def extract_prosody(clip: AudioClip, words: Sequence[Word] = ()) -> ProsodyProfile:
    """Extrae el perfil prosódico completo de un clip de voz."""
    import numpy as np

    samples = np.asarray(clip.samples, dtype=np.float32)
    duration = clip.duration
    if len(samples) == 0 or duration <= 0:
        return ProsodyProfile(0.0, 0.0, 0.0, 0.0, 0.0, None, None)

    energy_rms = float(np.sqrt(np.mean(samples ** 2)))

    # Pausas de inicio y fin
    if words:
        leading_pause = max(0.0, words[0].start)
        trailing_pause = max(0.0, duration - words[-1].end)
        units = sum(len(w.text.strip()) for w in words)
        speech_rate = units / duration if duration > 0 else 0.0
    else:
        # Detección por umbral de silencio
        hop = max(1, int(0.02 * clip.sample_rate))
        num_hops = len(samples) // hop
        if num_hops > 0:
            frame_rms = np.sqrt(
                np.mean((samples[:num_hops * hop].reshape(num_hops, hop)) ** 2, axis=1) + 1e-12
            )
            threshold = max(0.005, np.max(frame_rms) * 0.1)
            active = np.where(frame_rms > threshold)[0]
            if len(active) > 0:
                leading_pause = round(float(active[0] * 0.02), 3)
                trailing_pause = round(float((num_hops - active[-1] - 1) * 0.02), 3)
            else:
                leading_pause, trailing_pause = 0.0, 0.0
        else:
            leading_pause, trailing_pause = 0.0, 0.0
        speech_rate = 14.0  # valor neutro conversacional

    # Rango dinámico en dB
    hop = max(1, int(0.02 * clip.sample_rate))
    num_hops = len(samples) // hop
    if num_hops >= 4:
        frame_rms = np.sqrt(
            np.mean((samples[:num_hops * hop].reshape(num_hops, hop)) ** 2, axis=1) + 1e-12
        )
        p95 = np.percentile(frame_rms, 95)
        p10 = np.percentile(frame_rms, 10)
        dyn_range = 20 * np.log10((p95 + 1e-6) / (p10 + 1e-6))
        dynamic_range_db = round(float(max(0.0, dyn_range)), 2)
    else:
        dynamic_range_db = 0.0

    pitch_mean, pitch_std = _estimate_f0_contour(samples, clip.sample_rate)
    return ProsodyProfile(
        speech_rate=round(speech_rate, 2),
        energy_rms=round(energy_rms, 4),
        dynamic_range_db=dynamic_range_db,
        leading_pause=leading_pause,
        trailing_pause=trailing_pause,
        pitch_mean_hz=pitch_mean,
        pitch_std_hz=pitch_std,
    )


def transfer_prosodic_energy(clip: AudioClip, target: ProsodyProfile,
                             max_boost_db: float = 4.0) -> AudioClip:
    """Modula la sonoridad del clip generado para reflejar el énfasis y dinámica original,
    evitando saturación o distorsión.
    """
    import numpy as np

    samples = np.asarray(clip.samples, dtype=np.float32)
    if len(samples) == 0 or target.energy_rms <= 0.005:
        return clip

    curr_rms = float(np.sqrt(np.mean(samples ** 2)))
    if curr_rms < 1e-5:
        return clip

    ratio = target.energy_rms / curr_rms
    # Limitar el factor de ganancia entre [-max_boost_db, +max_boost_db]
    max_gain = 10.0 ** (max_boost_db / 20.0)
    min_gain = 10.0 ** (-max_boost_db / 20.0)
    clamped_gain = float(np.clip(ratio, min_gain, max_gain))

    scaled = samples * clamped_gain
    # Prevenir clipping
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.95:
        scaled = scaled * (0.95 / peak)

    return AudioClip(scaled.astype(np.float32), clip.sample_rate)


class NumpyProsodyAnalyzer:
    """Implementación del puerto ProsodyAnalyzer basada en NumPy."""

    def analyze(self, clip: AudioClip, words: Sequence[Word] = ()) -> ProsodyProfile:
        return extract_prosody(clip, words)

    def transfer_energy(self, clip: AudioClip, target: ProsodyProfile,
                        max_boost_db: float = 4.0) -> AudioClip:
        return transfer_prosodic_energy(clip, target, max_boost_db)

