import importlib.metadata
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass

from subdub_titler.infrastructure.gpu_runtime import ensure_cuda_libraries


@dataclass(frozen=True)
class Check:
    component: str
    ok: bool
    detail: str


def _version(package: str) -> Check:
    try:
        return Check(package, True, importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        return Check(package, False, "no instalado")


def diagnostics() -> list[Check]:
    result = [_version("faster-whisper"), _version("ctranslate2")]
    result.append(Check("ffmpeg", shutil.which("ffmpeg") is not None,
                        shutil.which("ffmpeg") or "ausente (PyAV puede decodificar sin FFmpeg CLI)"))
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            proc = subprocess.run(
                [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=8, check=True,
            )
            result.append(Check("NVIDIA GPU", True, proc.stdout.strip()))
        except (OSError, subprocess.SubprocessError) as exc:
            result.append(Check("NVIDIA GPU", False, str(exc)))
    else:
        result.append(Check("NVIDIA GPU", False, "nvidia-smi no encontrado"))
    try:
        ensure_cuda_libraries()
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        result.append(Check("CTranslate2 CUDA", True, ", ".join(sorted(supported))))
    except Exception as exc:
        result.append(Check("CTranslate2 CUDA", False, f"{type(exc).__name__}: {exc}"))
    result.extend(_dubbing_checks())
    # Verificar bibliotecas es orientativo: solo una inferencia prueba la cadena entera.
    return result


def _dubbing_checks() -> list[Check]:
    """Extra `dub` opcional: PyTorch con kernels para esta GPU (Blackwell exige cu128)."""
    try:
        importlib.metadata.version("omnivoice")
    except importlib.metadata.PackageNotFoundError:
        return [Check("Doblaje (extra dub)", True, "no instalado; opcional")]
    try:
        import torch
        if not torch.cuda.is_available():
            return [Check("PyTorch CUDA", False, f"torch {torch.__version__} sin CUDA")]
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        supported = arch in torch.cuda.get_arch_list()
        checks = [Check("PyTorch CUDA", supported,
                        f"torch {torch.__version__}, CUDA {torch.version.cuda}, {arch} "
                        f"{'compilado' if supported else 'SIN kernels: usa ruedas cu128'}")]
    except Exception as exc:
        checks = [Check("PyTorch CUDA", False, f"{type(exc).__name__}: {exc}")]
    checks.append(Check("Licencias de pesos", True,
                        "OmniVoice y NLLB: CC-BY-NC (uso no comercial); clonar voces con permiso"))
    return checks


def diagnostic_json() -> str:
    return json.dumps([asdict(c) for c in diagnostics()], ensure_ascii=False, indent=2)
