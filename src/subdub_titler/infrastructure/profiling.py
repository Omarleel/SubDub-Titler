"""Profiling con spans de baja sobrecarga y cProfile opcional."""
import cProfile
import json
import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path

from subdub_titler.domain.models import ProfileEvent

_active: ContextVar["SpanProfiler | None"] = ContextVar("subdub_profile", default=None)


def active_profiler() -> "SpanProfiler | None":
    return _active.get()


class TorchCudaMemory:
    """Pico de memoria CUDA de PyTorch por span, con spans anidados.

    No importa torch: si el proceso aún no lo cargó o CUDA no se inicializó, no mide (el ASR con
    CTranslate2 no pasa por el asignador de PyTorch y no aparece aquí).
    """

    def __init__(self) -> None:
        self._running: list[int] = []

    @staticmethod
    def _cuda():
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        return torch.cuda

    def enter(self) -> None:
        cuda = self._cuda()
        if cuda is not None:
            if self._running:
                self._running[-1] = max(self._running[-1], cuda.max_memory_allocated())
            cuda.reset_peak_memory_stats()
        self._running.append(0)

    def exit(self) -> float | None:
        running = self._running.pop()
        cuda = self._cuda()
        if cuda is None:
            return round(running / 2**20, 1) if running else None
        peak = max(running, cuda.max_memory_allocated())
        if self._running:
            self._running[-1] = max(self._running[-1], peak)
        cuda.reset_peak_memory_stats()
        return round(peak / 2**20, 1)

    def device_used(self) -> float | None:
        """Ocupación total de la GPU (todos los procesos y runtimes, incluido CTranslate2)."""
        cuda = self._cuda()
        if cuda is None:
            return None
        free, total = cuda.mem_get_info()
        return round((total - free) / 2**20, 1)


class SpanProfiler:
    def __init__(self, memory: TorchCudaMemory | None = None) -> None:
        self._stack: list[str] = []
        self._events: list[ProfileEvent] = []
        self._memory = memory or TorchCudaMemory()

    @contextmanager
    def activate(self) -> Generator[None, None, None]:
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        parent = self._stack[-1] if self._stack else None
        depth = len(self._stack)
        self._stack.append(name)
        self._memory.enter()
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
            vram = self._memory.exit()
            self._events.append(ProfileEvent(name, round(elapsed, 3), parent, depth, vram,
                                             self._memory.device_used()))

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
def profile_python(stats_path: Path | None) -> Generator[None, None, None]:
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
