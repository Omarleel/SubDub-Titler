"""Audio/video con ffmpeg: recortes, mezcla con loudness EBU R128 y multiplexado MP4.

La sonoridad se normaliza con `loudnorm` en dos pasadas (modo lineal) y después se **mide**
con `ebur128`, un filtro distinto: la cifra que se valida no es la que se pidió.
"""
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from subdub_titler.domain.dubbing import Ambience, AudioClip
from subdub_titler.domain.ports import MixResult

ISO639_2 = {"en": "eng", "es": "spa", "ja": "jpn", "zh": "zho"}


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg no encontrado: instálalo en el entorno (conda install ffmpeg)")
    return path


def _run(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([_ffmpeg(), "-nostdin", "-hide_banner", *args],
                              capture_output=capture, check=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode("utf-8", "replace")[-600:]
        raise RuntimeError(f"ffmpeg falló: {tail}") from exc


def _span(name: str):
    from contextlib import nullcontext

    from subdub_titler.infrastructure.profiling import active_profiler
    profiler = active_profiler()
    return profiler.measure(name) if profiler else nullcontext()


def resample(samples, source_rate: int, target_rate: int):
    import numpy as np
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    import soxr
    return soxr.resample(np.asarray(samples, dtype=np.float32), source_rate, target_rate,
                         quality="HQ").astype(np.float32)


def time_stretch(samples, sample_rate: int, seconds: float, tolerance: float = 0.01):
    """Ajusta la duración conservando el tono (ffmpeg `atempo`, encadenado fuera de 0.5–2.0).
    Termina con recorte/relleno a la longitud exacta pedida."""
    import numpy as np
    target = max(1, int(round(seconds * sample_rate)))
    samples = np.asarray(samples, dtype=np.float32)
    factor = len(samples) / target
    if abs(factor - 1) > tolerance:
        steps = []
        while factor > 2.0:
            steps.append(2.0)
            factor /= 2.0
        while factor < 0.5:
            steps.append(0.5)
            factor /= 0.5
        steps.append(factor)
        chain = ",".join(f"atempo={s:.6f}" for s in steps)
        raw = subprocess.run(
            [_ffmpeg(), "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(sample_rate),
             "-ac", "1", "-i", "-", "-af", chain, "-f", "f32le", "-"],
            input=samples.tobytes(), capture_output=True, check=True).stdout
        samples = np.frombuffer(raw, np.float32)
    if len(samples) >= target:
        return samples[:target].copy()
    return np.concatenate([samples, np.zeros(target - len(samples), np.float32)])


class FfmpegAudioStore:
    """Recortes servidos desde una decodificación completa en memoria del último archivo:
    cientos de recortes de un mismo stem cuestan una decodificación (0.4 s para 23 min) en
    lugar de un proceso ffmpeg con seek por recorte (~66 ms cada uno)."""

    def __init__(self) -> None:
        self._cached: tuple[tuple[str, float, int], object] | None = None

    def _decode(self, path: Path, sample_rate: int):
        import numpy as np
        key = (str(path.resolve()), path.stat().st_mtime, sample_rate)
        if self._cached is None or self._cached[0] != key:
            raw = _run(["-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar",
                        str(sample_rate), "-f", "f32le", "-"]).stdout
            self._cached = (key, np.frombuffer(raw, np.float32))
        return self._cached[1]

    def load(self, path: Path, sample_rate: int, start: float | None = None,
             end: float | None = None) -> AudioClip:
        import soundfile
        try:
            info = soundfile.info(str(path))
        except RuntimeError:
            info = None
        if info is not None and info.samplerate == sample_rate and info.channels == 1:
            # Clips propios (WAV mono a la tasa pedida): lectura directa, sin proceso ffmpeg.
            first = 0 if start is None else max(0, int(round(start * sample_rate)))
            last = -1 if end is None else int(round(end * sample_rate))
            samples, _ = soundfile.read(str(path), dtype="float32", start=first, stop=last
                                        if last >= 0 else None)
            return AudioClip(samples, sample_rate)
        samples = self._decode(path, sample_rate)
        first = 0 if start is None else max(0, int(round(start * sample_rate)))
        last = len(samples) if end is None else int(round(end * sample_rate))
        return AudioClip(samples[first:last].copy(), sample_rate)

    def save(self, clip: AudioClip, path: Path) -> Path:
        import soundfile
        path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(path), clip.samples, clip.sample_rate, subtype="PCM_16")
        return path

    def duration(self, path: Path) -> float:
        probe = shutil.which("ffprobe")
        if probe:
            out = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=nw=1:nk=1", str(path)],
                                 capture_output=True, text=True, check=False).stdout.strip()
            try:
                return float(out)
            except ValueError:
                pass
        import soundfile
        return soundfile.info(str(path)).duration


def _ebur128(args: list[str], peak: bool, stdin: bytes | None = None) -> str:
    filters = "ebur128=peak=true" if peak else "ebur128"
    command = [_ffmpeg(), "-nostdin" if stdin is None else "-y", "-hide_banner", "-v", "info",
               *args, "-filter_complex", filters, "-f", "null", "-"]
    try:
        return subprocess.run(command, input=stdin, capture_output=True,
                              check=True).stderr.decode("utf-8", "replace")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg falló: {exc.stderr[-600:]!r}") from exc


def measure_loudness(path: Path) -> tuple[float | None, float | None]:
    """(LUFS integrados, pico real dBTP) medidos con el filtro ebur128."""
    err = _ebur128(["-i", str(path)], peak=True)
    summary = err[err.rfind("Summary:"):]
    lufs = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary)

    def number(match):
        if not match or match.group(1) == "-inf":
            return None
        return float(match.group(1))
    return number(lufs), number(peak)


class FfmpegMixer:
    """Voz doblada colocada en su tiempo + fondo separado (+ ambiente: tramos del stem de voz
    sin diálogo), normalizada a `target_lufs`.

    Ganancia lineal calculada con `ebur128` y limitador de picos (`alimiter`) con 0.5 dB de
    margen para el pico real. `loudnorm` daba lo mismo, pero sobremuestrea a 192 kHz: 2 x 22 s
    para 23 min frente a ~2 s.
    """

    def __init__(self, sample_rate: int = 48_000, target_lufs: float = -16.0,
                 true_peak: float = -1.5, ambience_fade: float = 0.06) -> None:
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.true_peak = true_peak
        self.ambience_fade = ambience_fade
        self._reference_lufs: dict[tuple[str, float], float | None] = {}
        self._tracks: dict[str, tuple[tuple[str, float], object]] = {}

    def render(self, placements: Sequence[tuple[float, AudioClip]], background: Path | None,
               duration: float, out_dir: Path,
               voice_reference: Path | None = None,
               ambience: Ambience | None = None) -> MixResult:
        import numpy as np
        import soundfile
        span = _span
        out_dir.mkdir(parents=True, exist_ok=True)
        total = int(round(duration * self.sample_rate))
        with span("dub.mix.place"):
            voice = np.zeros(total, dtype=np.float32)
            for start, clip in placements:
                samples = resample(clip.samples, clip.sample_rate, self.sample_rate)
                first = int(round(start * self.sample_rate))
                if first >= total:
                    continue
                samples = samples[: total - first]
                voice[first:first + len(samples)] += samples
        if voice_reference is not None:
            with span("dub.mix.level"):
                # OmniVoice iguala el RMS de la referencia (unos segundos); la sonoridad
                # integrada de toda la voz original respeta mejor el balance con el fondo.
                voice *= self._match_gain(voice, voice_reference)
        voice_path = out_dir / "dub_voice.flac"
        with span("dub.mix.write_voice"):
            soundfile.write(str(voice_path), voice, self.sample_rate)
        bed_path = None
        with span("dub.mix.background"):
            if background is not None:
                bed = self._stereo("background", background)
                mix = np.zeros((max(total, len(bed)), 2), dtype=np.float32)
                mix[: len(bed)] += bed
            else:
                mix = np.zeros((total, 2), dtype=np.float32)
            if ambience is not None and ambience.spans:
                kept = self._stereo("ambience", ambience.source)
                length = min(len(kept), len(mix))
                mix[:length] += kept[:length] * self._envelope(ambience.spans, length)[:, None]
                bed_path = out_dir / "dub_bed.flac"
                soundfile.write(str(bed_path), mix, self.sample_rate)
            mix[:total] += voice[:, None]
        mix_path = out_dir / "dub_mix.flac"
        with span("dub.mix.normalize"):
            self._normalize(mix, mix_path)
        with span("dub.mix.measure"):
            lufs, peak = measure_loudness(mix_path)
        return MixResult(voice_path, mix_path, lufs, peak, bed_path)

    def _stereo(self, role: str, path: Path):
        """Pista decodificada a la tasa de mezcla; se conserva para regenerar cues sin volver
        a decodificar y remuestrear (3 s para 23 min)."""
        import numpy as np
        key = (str(path.resolve()), path.stat().st_mtime)
        cached = self._tracks.get(role)
        if cached is None or cached[0] != key:
            raw = _run(["-v", "error", "-i", str(path), "-ac", "2", "-ar", str(self.sample_rate),
                        "-f", "f32le", "-"]).stdout
            self._tracks[role] = (key, np.frombuffer(raw, np.float32).reshape(-1, 2))
        return self._tracks[role][1]

    def _envelope(self, spans, length: int):
        """1 dentro de los tramos, 0 fuera; rampas de coseno hacia dentro de cada tramo (el
        habla original de los bordes nunca se cuela)."""
        import numpy as np
        gain = np.zeros(length, dtype=np.float32)
        for start, end in spans:
            first = max(0, int(round(start * self.sample_rate)))
            last = min(length, int(round(end * self.sample_rate)))
            if last <= first:
                continue
            gain[first:last] = 1.0
            ramp = min(int(self.ambience_fade * self.sample_rate), (last - first) // 2)
            if ramp:
                curve = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp, dtype=np.float32))
                gain[first:first + ramp] = curve
                gain[last - ramp:last] = curve[::-1]
        return gain

    def _loudness_pcm(self, samples, channels: int) -> float | None:
        """LUFS integrados de PCM en memoria. Los ceros exactos se sustituyen por 1e-7
        (-140 dBFS, muy por debajo de la compuerta de -70 LUFS): no cambia la medida, pero
        ebur128 de ffmpeg es ~8x más lento con voz alternada con ceros exactos."""
        import numpy as np
        floor = np.where(samples == 0, np.float32(1e-7), samples).astype(np.float32)
        return self._loudness(["-f", "f32le", "-ar", str(self.sample_rate), "-ac",
                               str(channels), "-i", "-"], floor.tobytes())

    def _loudness(self, args: list[str], stdin: bytes | None = None) -> float | None:
        err = _ebur128(args, peak=False, stdin=stdin)
        match = re.search(r"I:\s+(-?[\d.]+) LUFS", err[err.rfind("Summary:"):])
        value = float(match.group(1)) if match else None
        return None if value is None or value <= -69.0 else value  # -70 = silencio

    def _match_gain(self, voice, reference: Path, limit_db: float = 20.0) -> float:
        dub = self._loudness_pcm(voice, 1)
        key = (str(reference.resolve()), reference.stat().st_mtime)
        if key not in self._reference_lufs:  # el stem no cambia entre regeneraciones
            self._reference_lufs[key] = self._loudness(["-i", str(reference)])
        original = self._reference_lufs[key]
        if dub is None or original is None:
            return 1.0
        return 10 ** (max(-limit_db, min(limit_db, original - dub)) / 20)

    def _normalize(self, mix, target: Path) -> None:
        pcm = ["-f", "f32le", "-ar", str(self.sample_rate), "-ac", "2"]
        measured = self._loudness_pcm(mix, 2)
        gain = 1.0 if measured is None else 10 ** ((self.target_lufs - measured) / 20)
        limit = 10 ** ((self.true_peak - 0.5) / 20)  # pico de muestra < pico real objetivo
        command = [_ffmpeg(), "-y", "-hide_banner", "-v", "error", *pcm, "-i", "-", "-af",
                   f"alimiter=limit={limit:.4f}:level=false:attack=2:release=50",
                   "-sample_fmt", "s16", str(target)]
        try:
            subprocess.run(command, input=(mix * gain).astype("float32").tobytes(),
                           capture_output=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg falló: {exc.stderr[-600:]!r}") from exc


class FfmpegMuxer:
    """MP4 con vídeo copiado, pista doblada (por defecto), pista original y subtítulos."""

    def mux(self, video: Path, dub_audio: Path, subtitles: Path | None, language: str,
            out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        code = ISO639_2.get(language, "und")
        args = ["-v", "error", "-y", "-i", str(video), "-i", str(dub_audio)]
        if subtitles is not None:
            args += ["-i", str(subtitles)]
        args += ["-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0?"]
        if subtitles is not None:
            args += ["-map", "2:0", "-c:s", "mov_text", "-metadata:s:s:0", f"language={code}"]
        tail = ["-metadata:s:a:0", f"language={code}",
                "-metadata:s:a:0", f"title=Doblaje sintético ({language})",
                "-metadata:s:a:1", "title=Original",
                "-disposition:a:0", "default", "-disposition:a:1", "0",
                "-metadata", "comment=Voz generada con clonación (OmniVoice)",
                "-movflags", "+faststart", str(out)]
        # Solo se codifica el doblaje, con el AAC rápido (2x frente al predeterminado a igual
        # bitrate); la pista original se copia. Si su códec no cabe en MP4, se recodifica.
        dub = ["-c:v", "copy", "-c:a:0", "aac", "-aac_coder", "fast", "-b:a:0", "192k"]
        try:
            _run(args + dub + ["-c:a:1", "copy"] + tail)
        except RuntimeError:
            _run(args + dub + ["-c:a:1", "aac", "-b:a:1", "192k"] + tail)
        return out
