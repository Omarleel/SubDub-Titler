"""Registra las DLL CUDA de los paquetes pip nvidia-* antes de importar CTranslate2.

En Windows, CTranslate2 carga cuBLAS/cuDNN bajo demanda vía PATH. Hacerlo aquí evita
depender de ejecutar scripts/activate_gpu_windows.ps1 antes de cada arranque.
En Linux el cargador lee LD_LIBRARY_PATH al iniciar el proceso: usar el script .sh.
"""
import importlib.util
import os
import sys
from pathlib import Path

_PACKAGES = ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc")
_registered = False


def nvidia_dll_dirs() -> list[Path]:
    directories = []
    for module in _PACKAGES:
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            continue
        if spec and spec.submodule_search_locations:
            directory = Path(next(iter(spec.submodule_search_locations))) / "bin"
            if directory.is_dir():
                directories.append(directory)
    return directories


def ensure_cuda_libraries() -> None:
    """Idempotente; sin efecto fuera de Windows o sin el runtime pip instalado."""
    global _registered
    if _registered or sys.platform != "win32":
        return
    _registered = True
    current = os.environ.get("PATH", "").split(os.pathsep)
    for directory in nvidia_dll_dirs():
        os.add_dll_directory(str(directory))
        if str(directory) not in current:
            # CTranslate2 usa LoadLibrary con la búsqueda estándar, que consulta PATH.
            os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
