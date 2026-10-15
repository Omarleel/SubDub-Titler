"""Matriz de optimización del doblaje: cada variante se mide en velocidad y en calidad.

Una variante solo es *adoptable* si no empeora ninguna métrica de calidad respecto a la
línea base más allá de su tolerancia. La transcripción se calcula una vez y se reutiliza;
los stems quedan en caché en la carpeta de trabajo (mismo origen).
"""
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from subdub_titler.application.dubbing import DubUseCase
from subdub_titler.domain.dubbing import DubReport, DubRequest
from subdub_titler.domain.models import ProfileEvent, Transcript

# Tolerancias de no-regresión frente a la línea base (positivo = permitido empeorar).
TOLERANCES = {"cer": 0.01, "similarity": 0.02, "mos": 0.10, "within_tolerance": 0.02}


@dataclass(frozen=True, slots=True)
class BenchRow:
    variant: dict
    seconds: float
    tts_seconds: float
    audio_seconds: float
    tts_rtf: float
    vram_peak_mb: float | None
    quality: dict
    gates_passed: int
    gates_total: int
    events: tuple[ProfileEvent, ...]
    regressions: tuple[str, ...] = ()

    @property
    def adoptable(self) -> bool:
        return not self.regressions


def _span(events: Sequence[ProfileEvent], name: str) -> float:
    return sum(e.milliseconds for e in events if e.name == name) / 1000


def _row(variant: dict, report: DubReport, events: tuple[ProfileEvent, ...]) -> BenchRow:
    audio = sum(c.generated_seconds for c in report.track)
    tts = _span(events, "dub.tts.generate")
    peaks = [e.vram_peak_mb for e in events if e.vram_peak_mb is not None]
    quality = asdict(report.quality) if report.quality else {}
    quality["within_tolerance"] = report.sync.within_tolerance
    judged = [g for g in report.gates if g.passed is not None]
    pipeline = sum(e.milliseconds for e in events
                   if e.depth == 0 and not e.name.startswith("dub.eval")) / 1000
    return BenchRow(variant, round(pipeline, 3), round(tts, 3), round(audio, 3),
                    round(tts / audio, 4) if audio else 0.0, max(peaks) if peaks else None,
                    quality, sum(1 for g in judged if g.passed), len(judged), events)


def regressions(row: BenchRow, base: BenchRow) -> tuple[str, ...]:
    found = []
    for key, tolerance in TOLERANCES.items():
        new, old = row.quality.get(key), base.quality.get(key)
        if new is None or old is None:
            continue
        lower_is_better = key == "cer"
        worse = new - old if lower_is_better else old - new
        if worse > tolerance:
            found.append(f"{key}: {old:.4g} → {new:.4g}")
    return tuple(found)


class DubBenchmark:
    def __init__(self, use_case: DubUseCase, profiler_factory: Callable[[], object]) -> None:
        self._use_case = use_case
        self._profiler_factory = profiler_factory

    def run(self, base: DubRequest, variants: Sequence[dict], workdir: Path,
            transcript: Transcript, warmup: bool = True) -> list[BenchRow]:
        """`variants`: cambios sobre `base` (p. ej. {"num_step": 8, "tts_batch": 16})."""
        if warmup:  # cargas en frío (pesos, conversiones, kernels) fuera de la medida
            self._use_case.execute(replace(base, **variants[0]), workdir / "warmup",
                                   transcript=transcript, stems_dir=workdir / "stems")
        rows = []
        for variant in variants:
            request = replace(base, **variant)
            profiler = self._profiler_factory()
            with profiler.activate():
                report = self._use_case.execute(request, workdir / _name(variant), profiler,
                                                transcript=transcript,
                                                stems_dir=workdir / "stems")
            rows.append(_row(variant, report, profiler.events))
        return [replace(row, regressions=regressions(row, rows[0])) for row in rows]


def _name(variant: dict) -> str:
    return "-".join(f"{k}={v}" for k, v in variant.items()) or "base"


def rows_to_dict(rows: Sequence[BenchRow]) -> list[dict]:
    return [{**{k: v for k, v in asdict(row).items() if k != "events"},
             "adoptable": row.adoptable, "events": [asdict(e) for e in row.events]}
            for row in rows]

