"""Capturar snapshot REAL por OS (no se incluyen locks ficticios)."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path("env-locks")
root.mkdir(exist_ok=True)
ident = f"{platform.system().lower()}-{platform.machine().lower()}-py{sys.version_info.major}{sys.version_info.minor}"
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
freezed = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(root / f"{ident}.pip.txt").write_text(freezed, encoding="utf-8")
try:
    conda = subprocess.check_output(["conda", "list", "--explicit"], text=True)
    (root / f"{ident}.conda.txt").write_text(conda, encoding="utf-8")
except (OSError, subprocess.CalledProcessError):
    print("Conda no disponible: solo se guardó snapshot pip")
(root / f"{ident}.json").write_text(json.dumps({
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
}, indent=2), encoding="utf-8")
print(f"Snapshot de dependencias: {root} ({ident})")
