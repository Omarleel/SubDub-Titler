"""Preflight de dependencias: ejecutar DENTRO del conda env instalado."""
import importlib.metadata
import subprocess
import sys

wanted = {
    "fastapi": "0.142.4",
    "typer": "0.27.3",
    "uvicorn": "0.54.0",
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.3",
}
# Extra opcional de doblaje: se verifica solo si está instalado.
optional = {
    "torch": "2.11.0+cu128",
    "torchaudio": "2.11.0+cu128",
    "omnivoice": "0.2.1",
    "demucs": "4.1.0",
}
for package, expected in optional.items():
    try:
        importlib.metadata.version(package)
        wanted[package] = expected
    except importlib.metadata.PackageNotFoundError:
        print(f"--   {package}: extra 'dub' no instalado")
failed = False
for package, expected in wanted.items():
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        actual = "MISSING"
    status = "OK" if actual == expected else "FAIL"
    print(f"{status:4} {package}: esperado {expected}, encontrado {actual}")
    if status == "FAIL":
        failed = True
result = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
if failed or result.returncode:
    sys.exit(1)
