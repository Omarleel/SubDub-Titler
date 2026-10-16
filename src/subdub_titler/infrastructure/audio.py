"""Preparación de audio en CPU para faster-whisper: decodificación y VAD.

numpy/onnxruntime llegan con el extra `asr`; se importan de forma diferida.
"""
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
SAMPLING_RATE = 16_000
_vad_tuned = False


def decode(source: Path, normalize_loudness: bool = False):
    """PCM mono 16 kHz float32. ffmpeg multihilo es ~2x más rápido que PyAV y produce
    las mismas muestras (ambos s16 → /32768); PyAV queda como respaldo."""
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            cmd = [
                ffmpeg, "-nostdin", "-v", "error", "-threads", "0", "-i", str(source),
                "-vn", "-ac", "1", "-ar", str(SAMPLING_RATE),
            ]
            if normalize_loudness:
                cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
            cmd.extend(["-f", "s16le", "-"])
            pcm = subprocess.run(
                cmd, capture_output=True, check=True,
            ).stdout
            return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("ffmpeg falló (%s); se usa PyAV", exc)
    from faster_whisper.audio import decode_audio
    return decode_audio(str(source), sampling_rate=SAMPLING_RATE)


def tune_vad_threads(threads: int | None = None) -> None:
    """faster-whisper fija Silero VAD a 1 hilo; en CPUs híbridas ese hilo puede caer en un
    núcleo de eficiencia (0.4–2.3 s por 23 min). Varios hilos estabilizan y aceleran.
    Acoplado a faster-whisper==1.2.1 (SileroVADModel.session): si cambia, no hace nada."""
    global _vad_tuned
    if _vad_tuned:
        return
    _vad_tuned = True
    try:
        import onnxruntime
        from faster_whisper.vad import get_vad_model

        model = get_vad_model()
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = threads or min(8, os.cpu_count() or 1)
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 4
        model.session = onnxruntime.InferenceSession(
            model.session._model_path, providers=["CPUExecutionProvider"],
            sess_options=options,
        )
    except Exception as exc:  # optimización opcional: nunca bloquear la inferencia
        logger.warning("No se pudo ajustar hilos del VAD: %s", exc)
