# SubDub-Titler

**Subtitulado offline multilingüe y laboratorio local de precisión/rendimiento.**
Backend Python con Clean Architecture + SOLID y motor `faster-whisper`/CTranslate2;
CLI Typer, API FastAPI, React + TypeScript. Diseñado para NVIDIA RTX 5060 Ti.

> **Estado: v0.1.0 / MVP de fases 1–4.** El núcleo, la API, UI y pruebas unitarias
> están implementados. GPU real, Qwen3-ASR, alineación forzada externa, editor
> profesional y optimización CUDA profunda quedan pendientes de validación/desarrollo.

## Instalación — Conda

Prerequisitos: Miniconda/Miniforge; driver NVIDIA compatible con RTX 5060 Ti;
Node.js 24 para desarrollo frontend. Se requiere acceso a internet solo al instalar
paquetes/descargar modelos (después puede funcionar offline).

```bash
conda env create -f environment.yml
conda activate subdub-titler
python -m pip install -e '.[asr,dev]' -r requirements/gpu-runtime.txt
python -m pip check
python scripts/verify_install.py
```

**Linux**: antes de arrancar SubDub en esa terminal:

```bash
source scripts/activate_gpu_linux.sh
subdub doctor --strict-gpu
```

**Windows PowerShell**: después de `conda activate`:

```powershell
.\scripts\activate_gpu_windows.ps1
subdub doctor --strict-gpu
```

**CPU/CI sin GPU** (sin dependencias ASR, tests/API):

```bash
conda env create -f environment.yml
conda activate subdub-titler
python -m pip install -e '.[dev]'
python -m pytest -q
```

En CPU con ASR real instala `-e '.[asr,dev]'` sin runtime NVIDIA y usa
`--device cpu --compute-type int8`. La UI ofrece el mismo perfil.

### Smoke real de GPU

```bash
subdub doctor --strict-gpu
subdub transcribe examples/tu_audio.mp3 --model tiny --language es --device cuda --compute-type float16 --profile reports/smoke.json
```

**La primera inferencia descargará pesos** automáticamente de Hugging Face; puedes
sustituir el archivo por un audio tuyo. La CLI no cambia a CPU silenciosamente.

## CLI

```bash
subdub --help
subdub models list
subdub transcribe video.mkv --model large-v3 --language ja --device cuda --format srt
subdub transcribe audio.wav --model small --device cpu --compute-type int8 --format vtt
subdub evaluate-text --reference examples/reference.txt --hypothesis examples/hypothesis.txt --language en
subdub benchmark audio.wav --model small --device cuda --repeats 3 --warmups 1 --output reports/benchmark.json
subdub ui
```

`transcribe` puede generar un `.pstats` mediante `--cprofile reports/run.pstats` y
spans mediante `--profile reports/run.json`. `benchmark` ejecuta la inferencia
**completa** (consume el generador de faster-whisper), calienta el modelo y calcula
mediana/dispersiones. `RTF` usa la duración del audio cuando el backend la informa.

## Frontend

```bash
cd apps/web
npm install
npm run build
cd ../..
subdub ui
```

Abre **http://127.0.0.1:8000**. API Swagger:
**http://127.0.0.1:8000/docs**.

Durante desarrollo ejecuta en terminales separadas:

```bash
subdub ui
# otra terminal:
cd apps/web && npm run dev
```

Visita **http://127.0.0.1:5173**. Vite usa proxy `/api` a la API local.

## Funcionalidad disponible

- Audio/video a SRT y VTT; español, inglés, japonés y chino (Whisper multilingüe).
- Presets tiny/base/small/medium/large-v3/turbo; descarga diferida de modelos.
- Cola local SQLite persistente; **un único worker** de inferencia desde UI.
- Calidad WER/CER cruda y normalizada; CER primaria para JA/ZH (sin WER engañoso).
- Trazas wall-clock por etapa y cProfile opcional; benchmark reproducible en CLI.
- Adaptadores ASR aislados, registro abierto a nuevos motores, tests de contratos.
- API escucha localhost; carga restringida a extensiones y archivos de 2 GiB.

## Limitaciones explícitas

- Sin pruebas de CUDA en la RTX 5060 Ti del usuario en este entorno de entrega.
- La app no incorpora todavía Qwen/Kotoba ni votación multi-modelo.
- Timestamps son los que proporciona faster-whisper; no hay aligner externo.
- Profiling nativo GPU individual (Nsight), NVTX, VRAM por operación: fase siguiente.
- La UI es un dashboard operativo, aún no un editor visual detallado de subtítulos.
- La validación lingüística completa de una película debe segmentarse en clips;
  la API limita a 6000 caracteres por referencia para evitar cálculos excesivos.
- El trabajo `running` interrumpido se marca `failed`; no hay reanudación por chunk.
- Exports integrados: SRT/VTT en CLI; UI ofrece descarga SRT.

## Arquitectura / desarrollo

```text
src/subdub_titler/
  domain/          # entidades, contratos y reglas independientes
  application/     # casos de uso ASR, evaluación y trabajos
  infrastructure/  # faster-whisper, SQLite, profiling, subtítulos, NVIDIA doctor
  interfaces/      # CLI Typer y API FastAPI
apps/web/          # React 19 + TypeScript
requirements/      # extras y runtime GPU opcional
scripts/           # CUDA paths, verificación y snapshots
```

No usar `pip install` sobre el entorno base. Para cambiar versiones,
**reconstruir el entorno Conda** y verificar con `pip check`, `verify_install.py`,
`pytest` y smoke GPU. Consulta `docs/DEPENDENCIAS.md` y `docs/FASES.md`.

## Fuentes técnicas

- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- CTranslate2: https://opennmt.net/CTranslate2/
- Conda + pip: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- NVIDIA CUDA GPU: https://developer.nvidia.com/cuda/gpus

MIT — código fuente inicial del proyecto.

## Profundidad del profiling

```bash
subdub profile audio.mp3 --level spans --output reports/span
subdub profile audio.mp3 --level functions --output reports/python
subdub profile audio.mp3 --level gpu --output reports/nsys
subdub profile audio.mp3 --level kernel --output reports/kernel
```

Para rangos NVTX: incluir el extra `profiling` en la instalación conda/pip inicial,
por ejemplo `python -m pip install -e '.[asr,dev,profiling]' -r requirements/gpu-runtime.txt`.
`gpu` requiere NVIDIA Nsight Systems (`nsys`); `kernel`, Nsight Compute (`ncu`).
No se incluyen en Conda para no forzar conflictos entre toolkits CUDA.
