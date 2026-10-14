"""Profiling con spans de baja sobrecarga y cProfile opcional."""
import cProfile
import os
from contextlib import nullcontext
import json
import pstats
import time
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from subdub_titler.domain.models import ProfileEvent


_active: ContextVar["SpanProfiler | None"] = ContextVar("subdub_profile", default=None)


def active_profiler() -> "SpanProfiler | None":
    return _active.get()


class SpanProfiler:
    def __init__(self) -> None:
        self._stack: list[str] = []
        self._events: list[ProfileEvent] = []

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        parent = self._stack[-1] if self._stack else None
        depth = len(self._stack)
        self._stack.append(name)
        start = time.perf_counter_ns()
        # Opcional: anotar rangos CUDA en Nsight Systems; no dep principal.
        annotation = nullcontext()
        if os.environ.get("SUBDUB_ENABLE_NVTX") == "1":
            try:
                import nvtx
                annotation = nvtx.annotate(name, domain="subdub-titler")
            except ImportError:
                # La falta del extra de profiling no debe bloquear inferencia.
                pass
        try:
            with annotation:
                yield
        finally:
            elapsed = (time.perf_counter_ns() - start) / 1e6
            self._stack.pop()
            self._events.append(ProfileEvent(name, round(elapsed, 3), parent, depth))

    @property
    def events(self) -> tuple[ProfileEvent, ...]:
        return tuple(self._events)

    def save(self, target: Path, metadata: dict | None = None) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"schema_version": 1, "metadata": metadata or {},
                 "events": [asdict(e) for e in self._events]},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )


@contextmanager
def profile_python(stats_path: Path | None) -> Iterator[None]:
    """Si hay ruta, produce .pstats para snakeviz/pstats (no inspecciona kernels GPU)."""
    if stats_path is None:
        yield
        return
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        profiler.dump_stats(str(stats_path))
