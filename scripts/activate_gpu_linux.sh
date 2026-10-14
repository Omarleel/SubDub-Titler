#!/usr/bin/env bash
# Invocar con 'source scripts/activate_gpu_linux.sh' después de conda activate.
# Las librerías deben estar en LD_LIBRARY_PATH ANTES de iniciar Python.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo 'Activa primero el entorno conda subdub-titler.' >&2
  return 1 2>/dev/null || exit 1
fi
GPU_PATH=$(python - <<'PY'
from pathlib import Path
import importlib.util
paths = []
for module, suffix in [('nvidia.cublas', 'lib'), ('nvidia.cudnn', 'lib')]:
    spec = importlib.util.find_spec(module)
    if spec and spec.submodule_search_locations:
        base = Path(next(iter(spec.submodule_search_locations)))
        directory = base / suffix
        if directory.is_dir():
            paths.append(str(directory))
print(':'.join(paths))
PY
)
if [[ -z "$GPU_PATH" ]]; then
  echo 'No se encontraron bibliotecas NVIDIA pip: revisa requirements/gpu-runtime.txt.' >&2
  return 1 2>/dev/null || exit 1
fi
export LD_LIBRARY_PATH="$GPU_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "CUDA paths activados: $GPU_PATH"
