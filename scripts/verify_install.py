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
