# SubDub-Titler

**Subtitulado y doblaje de voz offline multilingüe, con laboratorio local de precisión/rendimiento.**
Backend Python con Clean Architecture + SOLID y motor `faster-whisper`/CTranslate2; doblaje con
traducción NLLB, clonación de voz OmniVoice y separación Demucs; CLI Typer, API FastAPI, React + TypeScript.
Diseñado y validado en NVIDIA RTX 5060 Ti 16 GB (Windows 11).

> **Estado:** subtitulado (fases 0–7) y doblaje de voz (fases 8–13) implementados y medidos en GPU real.
> Pendiente: segundo motor ASR (Qwen3-ASR) y editor profesional con forma de onda.
> Alineación forzada MMS disponible; requiere sus dependencias y pesos.

## Fiabilidad y revisión de resultados

- Los hablantes identificados en una transcripción se conservan al doblar. La opción
  **Recalcular hablantes** fuerza una nueva diarización.
- **Referencias por voz** permite comparar 1, 2, 4 u 8 referencias; la selección llega al motor.
- Cada edición de doblaje genera una revisión aislada. **Deshacer última edición** restaura
  el audio, informe y valoraciones anteriores. Una edición fallida no reemplaza el resultado activo.
- El informe vigente se descarga desde **Informe de la revisión activa**; en la API SQLite es
  la autoridad de la revisión activa. El report.json raíz corresponde al doblaje inicial;
  cada revisión conserva su propio report.json. La CLI publica su report.json atómicamente.
- El historial carga páginas de 30 trabajos y excluye las regeneraciones internas antes de paginar.
  Los informes completos se consultan sólo para los trabajos seleccionados.
- **Sólo frases que necesitan revisión** filtra el banco de doblaje por inteligibilidad y velocidad.
- **Cancelar trabajo** actúa entre etapas; no interrumpe un kernel o subproceso activo.
  **Reintentar** crea otro trabajo con la misma configuración desde el inicio; no reanuda por chunks.
- CPU/CUDA se aplica también a aislamiento, alineación y diarización. CUDA ausente produce un error
  explícito. La caché de Demucs comprueba SHA-256 y parámetros, no sólo nombre y fecha.

Las revisiones se conservan en disco para permitir deshacer. La limpieza automática, la reanudación
por etapa y el editor de subtítulos con forma de onda siguen pendientes.

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

## Doblaje de voz

Del video original a un MP4 con una **pista doblada con la voz clonada del hablante**, sincronizada con los
subtítulos y mezclada con el fondo original (la pista original se conserva como segunda pista).

```text
transcripción ─► traducción NLLB-200 (n-best, frase a frase) ─► ajuste a la duración de cada subtítulo
     │                                                                │
     └─► separación voz/fondo (Demucs) ─► voz de referencia por hablante ─► OmniVoice (duración exacta)
                                                                            │
                     mezcla −16 LUFS, nivel de la voz original ◄────────────┘ ─► MP4 multipista + SRT
```

Instalación (mismo entorno; PyTorch cu128 es obligatorio para la arquitectura Blackwell de la RTX 50):

```bash
python -m pip install -r requirements/dub.txt
python -m pip check
subdub doctor --strict-gpu          # incluye «PyTorch CUDA … sm_120 compilado»
```

```bash
subdub dub video.mp4 --to es --evaluate                 # doblaje + validación de calidad
subdub dub video.mp4 --to es --diarize --evaluate       # varios hablantes: una voz clonada por hablante
subdub dub-bench video.mp4 --to es --num-step 8,16,32 --tts-batch 1,8,16   # matriz A/B velocidad + calidad
subdub tts-eval --from-report video.dub-es/report.json --limit 40 --save-manifest corpus/tts-es.jsonl
subdub tts-eval --manifest corpus/tts-es.jsonl --baseline reports/tts-eval-base.json   # regresión
subdub translate-eval --manifest corpus/mt.jsonl --translator nllb-600m             # chrF
subdub listen-panel video.dub-es otra-variante --limit 20   # escucha a ciegas: acento y parecido
```

En la UI, la sección **Doblaje** lanza trabajos sobre una transcripción, muestra el progreso por etapa, los
umbrales de calidad y un banco de trabajo para escuchar original/doblaje, editar la traducción de un subtítulo,
regenerarlo (Ctrl+Enter) y puntuarlo de 1 a 5.

**Cómo se valida** (cada doblaje con `--evaluate`, y cada optimización con `dub-bench`/`tts-eval`):

| Dimensión | Métrica | Umbral |
|---|---|---|
| Inteligibilidad | CER «ida y vuelta»: Whisper large-v3 transcribe la voz doblada y se compara con la traducción | ≤ 3 % (es/en), ≤ 8 % (ja/zh) |
| Parecido de voz | WavLM-ECAPA (protocolo del paper de OmniVoice), relativo al parecido de la voz original consigo misma | ≥ 85 % |
| Naturalidad | UTMOS22 del doblaje frente a la voz original separada | ≥ −0.1 |
| Sincronía | error de duración por subtítulo, solapes, velocidad de habla | ≥ 95 % en ±120 ms/5 %, 0 solapes |
| Separación | palabras/min que el ASR reconoce en el fondo | ≤ 2 |
| Mezcla | sonoridad integrada y pico real (`ebur128`) | −16 ± 1 LUFS, ≤ −1 dBTP |
| Traducción | chrF frente a referencias humanas (idéntico a sacreBLEU) | sin regresión |
| Acento | probabilidad media del idioma destino en la detección de idioma de Whisper large-v3 («suena nativo») | informativa |

Resultados en RTX 5060 Ti (`subdub dub-bench`, video real de 23 min, inglés → español):

| Variante (video 23 min, 187 cues, en→es) | Síntesis | RTF | Pipeline* | CER | SIM | Adoptable |
|---|---|---|---|---|---|---|
| OmniVoice original, lote 8 | 98.7 s | 0.225 | 117 s | 6.2 % | 0.562 | línea base |
| OmniVoice original, lote 16 | 107.3 s | 0.244 | 126 s | 6.8 % | 0.565 | sí |
| **CFG separado, lote 8 (predeterminado)** | **55.1 s** | **0.125** | **74 s** | 6.2 % | 0.568 | sí |
| CFG separado, lote 16 | 57.9 s | 0.132 | 95 s | 7.1 % | 0.560 | sí |

\*Sin ASR ni separación (reutilizados entre variantes). Doblaje completo en frío: 139 s + 70 s de evaluación;
pico de GPU 11.2 GB. Umbrales cumplidos 7/9: el CER ida y vuelta (6.2 %) lo elevan los subtítulos de ≤1.5 s y la
velocidad en rango queda en 94.7 % (10 de 187 cues no caben ni a 1.2×: candidatos para editar en la UI).

Mejoras de calidad (mismo video, validadas con la matriz A/B; parecido medido con el original en la misma ranura):

| Variante | Subtítulos | CER | Parecido por ranura | Pipeline* |
|---|---|---|---|---|
| Sin mejoras | 187 | 6.21 % | 0.319 | 75 s |
| + unir fragmentos de frase | 175 | 4.75 % | 0.322 | 72 s |
| + verificar y reintentar ×2 | 175 | 3.44 % | 0.323 | 140 s |
| **+ verificar y reintentar ×1 (predeterminado)** | 175 | **3.54 %** | 0.321 | 126 s |
| + sondear 4 referencias de voz (opcional) | 175 | 3.44 % | 0.323 | 182 s |

La verificación transcribe cada frase con Whisper medium y regenera con el doble de pasos las que no se
entienden; la calidad final se mide con otro ASR (large-v3) para no optimizar contra la propia métrica.

Motores registrados (extensibles sin tocar los casos de uso): `omnivoice` (predeterminado), `omnivoice-base`
(decodificación original, para A/B), `voxcpm2` (opcional: `pip install voxcpm==2.0.3`), `nllb-600m`, `nllb-1.3b`,
`htdemucs`, `htdemucs_ft`.

Coqui XTTS-v2 se evaluó con el mismo corpus y se descartó: CER 4.55 % (OmniVoice 4.9 %) y mejor UTMOS, pero el
parecido de voz cae de 0.59 a 0.21, es 2.6× más lento, tiene la misma licencia no comercial y no puede convivir
con OmniVoice (exige transformers < 5).

Correcciones de audio (video real de 13 min con voz original muy alta y saturada, inglés → español):

| Problema | Causa medida | Corrección | Antes → después |
|---|---|---|---|
| Voz áspera/robótica | referencia a −9 dBFS: OmniVoice genera a ese nivel y la salida pasaba de 1.0 | referencia a RMS 0.1 (nivel de entrenamiento) | CER 9.3 → 3.1 %, parecido 0.35 → 0.47, saturación 0.9 → 0 % |
| Voz atropellada | solo 0.5 s de silencio tomable tras cada frase | 1.5 s tras la frase, 0.4 s de adelanto | velocidad en rango 56 → 83 %, máx. ×2.2 → ×1.8 |
| Ráfagas («gritos») | tramos de 0.1–0.3 s +10 dB sobre la frase | limitador relativo al nivel de cada frase | frases con ráfagas 44 → 0, sin cambio de CER/parecido |
| Se pierde el ambiente | Demucs mete pájaros, pasos y respiraciones en «voz» | se conserva el stem de voz fuera de las palabras | 498 s de ambiente recuperado, fuga 0.08 palabras/min |

Doblaje completo: umbrales 6/9 → 8/9 (CER 6.2 → 1.7 %). El acento se mide con la probabilidad de español de la
detección de idioma de Whisper: 0.56 → 0.72; una voz nativa sin clonar llega a 0.92 pero no se parece al hablante
(0.08). Convertir esa voz nativa al timbre del hablante con kNN-VC se descartó: 0.85 de español, pero parecido
0.21 y CER 9.6 %.

La detección de idioma es poco fiable como medida de acento cuando el destino es español, así que los cambios de
voz se validan además con `subdub listen-panel`: genera una página local con las mismas frases de varias
variantes, en orden aleatorio y con nombres de archivo que no las delatan; se puntúa el acento y el parecido
(1–5) y al final muestra la media por variante y descarga las respuestas. En un video de 13 min la medida
distingue claramente la voz clonada (0.32, 146 de 216 frases por debajo de 0.5) de la nativa sin clonar (0.92).

**Licencias y uso responsable:** los pesos de OmniVoice y NLLB son **CC-BY-NC** (uso no comercial); VoxCPM2 es
Apache-2.0. Clona solo voces con permiso de su titular; el MP4 marca la pista como «Doblaje sintético».

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
- Doblaje de voz con clonación, varios hablantes, edición y regeneración por subtítulo y validación automática.

## Limitaciones explícitas

- La app no incorpora todavía Qwen/Kotoba ni votación multi-modelo.
- Doblaje: la diarización no separa voces solapadas. Pico de VRAM ~11 GB en el video de prueba (cada etapa
  libera sus modelos al terminar); con menos de 12 GB conviene `--no-evaluate`.
- Timestamps son los que proporciona faster-whisper; no hay aligner externo.
- VRAM por etapa medida solo para PyTorch (el ASR usa CTranslate2, fuera de su asignador).
- La UI es un dashboard operativo, aún no un editor visual detallado de subtítulos.
- La validación lingüística completa de una película debe segmentarse en clips;
  la API limita a 6000 caracteres por referencia para evitar cálculos excesivos.
- El trabajo `running` interrumpido se marca `failed`; no hay reanudación por chunk.
- Exports integrados: SRT/VTT en CLI; UI ofrece descarga SRT.

## Arquitectura / desarrollo

```text
src/subdub_titler/
  domain/          # entidades, puertos y reglas puras (tiempos de doblaje, umbrales), solo stdlib
  application/     # casos de uso: ASR, doblaje, evaluación de calidad, benchmark, corpus, trabajos
  infrastructure/  # faster-whisper, OmniVoice, NLLB/CT2, Demucs, ffmpeg, evaluadores, SQLite, profiling
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
- OmniVoice: https://github.com/k2-fsa/OmniVoice (arXiv 2604.00688); VoiceStudio: https://github.com/debpalash/VoiceStudio
- NLLB-200: https://huggingface.co/facebook/nllb-200-distilled-600M · Demucs: https://github.com/adefossez/demucs
- VoxCPM: https://github.com/OpenBMB/VoxCPM · NTREX-128: https://github.com/MicrosoftTranslator/NTREX
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
