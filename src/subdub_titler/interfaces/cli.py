"""CLI: mismo caso de uso que API. Nunca forzar fallback CPU silencioso."""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from subdub_titler.application.dubbing import DubUseCase, report_to_dict
from subdub_titler.application.quality import evaluate
from subdub_titler.application.transcribe import TranscribeUseCase, write_subtitle
from subdub_titler.domain.dubbing import DubReport, DubRequest
from subdub_titler.domain.models import TranscriptionRequest
from subdub_titler.infrastructure.doctor import diagnostics
from subdub_titler.infrastructure.profiling import SpanProfiler, profile_python
from subdub_titler.infrastructure.registry import default_registry

app = typer.Typer(name="subdub", no_args_is_help=True, help="SubDub-Titler | subtítulos locales")
models_app = typer.Typer(help="Catálogo de motores ASR registrados")
app.add_typer(models_app, name="models")
console = Console()
error_console = Console(stderr=True)


def _request(path: Path, model: str, device: str, compute_type: str,
             language: str, beam_size: int, batch_size: int,
             vad_threshold: float = 0.35, vad_speech_pad_ms: int = 800,
             vad_min_silence_ms: int = 2000, no_speech_threshold: float = 0.40,
             normalize_loudness: bool = False,
             isolate_dialogue: bool = False) -> TranscriptionRequest:
    return TranscriptionRequest(
        source=path.resolve(), model_id=model, device=device,
        compute_type=compute_type, language=language, beam_size=beam_size,
        batch_size=batch_size, vad_threshold=vad_threshold,
        vad_speech_pad_ms=vad_speech_pad_ms, vad_min_silence_ms=vad_min_silence_ms,
        no_speech_threshold=no_speech_threshold, normalize_loudness=normalize_loudness,
        isolate_dialogue=isolate_dialogue,
    )


@app.command()
def doctor(strict_gpu: bool = typer.Option(False, help="Error si CUDA no está listo")) -> None:
    """Inspección de GPU y bibliotecas; inferencia real debe probarse por separado."""
    table = Table("Componente", "OK", "Detalle")
    checks = diagnostics()
    for check in checks:
        table.add_row(check.component, "✓" if check.ok else "✗", check.detail)
    console.print(table)
    if strict_gpu and not next(c.ok for c in checks if c.component == "CTranslate2 CUDA"):
        raise typer.Exit(1)


@models_app.command("list")
def list_models() -> None:
    """Modelos registrados, no necesariamente descargados."""
    table = Table("Modelo", "Motor", "Idiomas", "Uso")
    for desc in default_registry().list_models():
        table.add_row(desc.id, desc.backend, ", ".join(desc.languages), desc.description)
    console.print(table)


@app.command()
def transcribe(
    input: Annotated[Path, typer.Argument(help="Video/audio de entrada")],
    model: Annotated[str, typer.Option()] = "small",
    device: Annotated[str, typer.Option()] = "cuda",
    compute_type: Annotated[str, typer.Option()] = "float16",
    language: Annotated[str, typer.Option()] = "auto",
    beam_size: Annotated[int, typer.Option()] = 5,
    batch_size: Annotated[int, typer.Option(help="Fragmentos en paralelo; 1 = secuencial")] = 16,
    format: Annotated[str, typer.Option(help="srt|vtt")] = "srt",
    output: Annotated[Path | None, typer.Option()] = None,
    vad_threshold: Annotated[float, typer.Option(help="Umbral de voz Silero VAD")] = 0.35,
    speech_pad_ms: Annotated[int, typer.Option(help="Margen temporal de voz en ms")] = 800,
    vad_min_silence_ms: Annotated[int, typer.Option(help="Silencio mínimo en ms")] = 2000,
    no_speech_threshold: Annotated[float, typer.Option(help="Umbral de no-habla")] = 0.40,
    normalize_loudness: Annotated[bool, typer.Option(help="Normalizar sonoridad EBU R128")] = False,
    isolate_dialogue: Annotated[bool, typer.Option(help="Aislar voz con Demucs")] = False,
    profile: Annotated[Path | None, typer.Option(help="Reporte JSON de spans")] = None,
    cprofile: Annotated[Path | None, typer.Option(help="Archivo .pstats")] = None,
) -> None:
    """Generar subtítulos mediante faster-whisper (descarga pesos al primer uso)."""
    try:
        request = _request(
            input, model, device, compute_type, language, beam_size, batch_size,
            vad_threshold=vad_threshold, vad_speech_pad_ms=speech_pad_ms,
            vad_min_silence_ms=vad_min_silence_ms, no_speech_threshold=no_speech_threshold,
            normalize_loudness=normalize_loudness, isolate_dialogue=isolate_dialogue,
        )
        target = output or input.with_suffix(f".{format}")
        if format not in ("srt", "vtt"):
            raise ValueError("Formato admitido: srt, vtt")
        registry = default_registry()
        separator = None
        if isolate_dialogue:
            from subdub_titler.infrastructure.demucs_separator import LazyDemucsSeparator
            separator = LazyDemucsSeparator(device=device)
        use_case = TranscribeUseCase(registry, separator=separator)
        profiler = SpanProfiler()
        with profile_python(cprofile):
            result = use_case.execute(request, profiler)
            with profiler.measure("subtitles.write"):
                write_subtitle(result, target, format)
        if profile:
            profiler.save(profile, {"source": str(input), "model": model,
                                    "device": device, "compute_type": compute_type,
                                    "elapsed_seconds": result.elapsed_seconds,
                                    "rtf": result.rtf})
        console.print(f"[green]Guardado:[/green] {target}")
        console.print(f"Idioma: {result.transcript.language}; "
                      f"tiempo ASR: {result.elapsed_seconds:.2f}s; "
                      f"RTF: {result.rtf if result.rtf is not None else 'N/D'}")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        error_console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def evaluate_text(
    reference: Annotated[Path, typer.Option(help="TXT humano UTF-8")],
    hypothesis: Annotated[Path, typer.Option(help="TXT predicción UTF-8")],
    language: Annotated[str, typer.Option()] = "es",
    output: Annotated[Path | None, typer.Option(help="Reporte JSON")]=None,
) -> None:
    """WER/CER crudos y normalizados. Para CJK, WER queda N/D."""
    result = evaluate(reference.read_text(encoding="utf-8"),
                      hypothesis.read_text(encoding="utf-8"), language)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    console.print(rendered)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


@app.command()
def benchmark(
    input: Annotated[Path, typer.Argument()],
    model: Annotated[str, typer.Option()] = "small",
    device: Annotated[str, typer.Option()] = "cuda",
    compute_type: Annotated[str, typer.Option()] = "float16",
    language: Annotated[str, typer.Option()] = "auto",
    batch_size: Annotated[int, typer.Option(min=1, max=64)] = 16,
    repeats: Annotated[int, typer.Option(min=1, max=30)] = 3,
    warmups: Annotated[int, typer.Option(min=0, max=10)] = 1,
    output: Annotated[Path, typer.Option()] = Path("reports/benchmark.json"),
) -> None:
    """Repeticiones con modelo en caché; tiempos completos, incluye decodificación."""
    request = _request(input, model, device, compute_type, language, beam_size=5,
                       batch_size=batch_size)
    use_case = TranscribeUseCase(default_registry())
    for _ in range(warmups):
        use_case.execute(request)
    runs = [use_case.execute(request) for _ in range(repeats)]
    seconds = [r.elapsed_seconds for r in runs]
    result = {
        "schema_version": 1, "model": model, "device": device,
        "compute_type": compute_type, "language": language, "batch_size": batch_size,
        "repeats": repeats, "warmups": warmups,
        "seconds": seconds, "mean_seconds": statistics.mean(seconds),
        "median_seconds": statistics.median(seconds),
        "stdev_seconds": statistics.stdev(seconds) if repeats > 1 else None,
        "rtfs": [r.rtf for r in runs],
        "spans": [[asdict(e) for e in r.events] for r in runs],
        "environment": {"python": sys.version.split()[0]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Benchmark:[/green] {output} "
                  f"(mediana {result['median_seconds']:.3f} s)")


@app.command()
def profile(
    input: Annotated[Path, typer.Argument()],
    level: Annotated[str, typer.Option(help="spans|functions|gpu|kernel")] = "spans",
    model: Annotated[str, typer.Option()] = "small",
    language: Annotated[str, typer.Option()] = "auto",
    device: Annotated[str, typer.Option()] = "cuda",
    compute_type: Annotated[str, typer.Option()] = "float16",
    batch_size: Annotated[int, typer.Option(min=1, max=64)] = 16,
    output: Annotated[Path, typer.Option(help="Prefijo de resultados")] = Path("reports/profile"),
) -> None:
    """Spans, funciones Python, CUDA timeline (nsys) o kernels (ncu)."""
    if level not in ("spans", "functions", "gpu", "kernel"):
        raise typer.BadParameter("Nivel permitido: spans, functions, gpu, kernel")
    if not input.is_file():
        raise typer.BadParameter(f"Archivo no encontrado: {input}")
    output.parent.mkdir(parents=True, exist_ok=True)
    args = ["transcribe", str(input.resolve()), "--model", model,
            "--language", language, "--device", device, "--compute-type", compute_type,
            "--batch-size", str(batch_size),
            "--profile", str(output.with_suffix(".json")),
            "--output", str(output.with_suffix(".srt"))]
    if level in ("spans", "functions"):
        if level == "functions":
            args.extend(["--cprofile", str(output.with_suffix(".pstats"))])
        transcribe(input=input, model=model, language=language, device=device,
                   compute_type=compute_type, beam_size=5, batch_size=batch_size, format="srt",
                   output=output.with_suffix(".srt"),
                   profile=output.with_suffix(".json"),
                   cprofile=output.with_suffix(".pstats") if level == "functions" else None)
        return
    executable = "nsys" if level == "gpu" else "ncu"
    if not shutil.which(executable):
        error_console.print(f"Falta {executable}; instala NVIDIA Nsight (no incluido en conda).")
        raise typer.Exit(1)
    subcommand = (["nsys", "profile", "--force-overwrite", "true",
                   "--trace=cuda,nvtx,osrt", "-o", str(output)]
                  if level == "gpu" else ["ncu", "--set", "full",
                                           "-o", str(output)])
    env = os.environ.copy()
    env["SUBDUB_ENABLE_NVTX"] = "1"
    result = subprocess.run(subcommand + [sys.executable, "-m", "subdub_titler", *args],
                            check=False, env=env)
    if result.returncode:
        raise typer.Exit(result.returncode)
    console.print(f"Reporte NVIDIA guardado con prefijo: {output}")


def _dub_use_case() -> tuple[DubUseCase, object]:
    from subdub_titler.infrastructure.dub_factory import AsrTranscripts, default_dub_factory
    registry = default_registry()
    factory = default_dub_factory(registry)
    return DubUseCase(AsrTranscripts(registry), factory), factory


def _print_dub(report: DubReport, events, elapsed: float) -> None:
    sync = report.sync
    if len(report.references) > 1:
        console.print("Hablantes: " + ", ".join(
            f"{s} ({sum(1 for c in report.track if c.speaker == s)} cues)"
            for s in report.references))
    console.print(f"Cues: {sync.cues} · dentro de tolerancia {sync.within_tolerance:.1%} · "
                  f"error medio {sync.mean_abs_error_ms} ms · solapes {sync.overlaps} · "
                  f"velocidad {sync.speed_min}–{sync.speed_max} · acortados {sync.shortened}")
    table = Table("Etapa", "ms", "VRAM pico MiB")
    for event in events:
        if event.depth <= 1:
            table.add_row("  " * event.depth + event.name, f"{event.milliseconds:,.0f}",
                          "" if event.vram_peak_mb is None else f"{event.vram_peak_mb:,.0f}")
    console.print(table)
    if report.gates:
        gates = Table("Umbral", "Valor", "Límite", "OK")
        for gate in report.gates:
            mark = {True: "[green]✓[/green]", False: "[red]✗[/red]", None: "–"}[gate.passed]
            gates.add_row(gate.name, "N/D" if gate.value is None else f"{gate.value:.4g}",
                          ("≥ " if gate.higher_is_better else "≤ ") + f"{gate.threshold:g}", mark)
        console.print(gates)
    if report.quality and report.quality.language_prob is not None:
        console.print(f"Suena nativo (p del idioma destino, sin umbral): "
                      f"{report.quality.language_prob:.1%}")
    for key, value in report.outputs.items():
        console.print(f"[green]{key}:[/green] {value}")
    console.print(f"Tiempo total: {elapsed:.1f} s")


@app.command()
def dub(
    input: Annotated[Path, typer.Argument(help="Video/audio de entrada")],
    to: Annotated[str, typer.Option("--to", help="Idioma destino: es|en|ja|zh")],
    source_language: Annotated[str, typer.Option("--from")] = "auto",
    model: Annotated[str, typer.Option(help="Modelo ASR")] = "medium",
    device: Annotated[str, typer.Option()] = "cuda",
    tts: Annotated[str, typer.Option(help="Motor TTS registrado")] = "omnivoice",
    translator: Annotated[str, typer.Option(help="Traductor registrado")] = "nllb-600m",
    separator: Annotated[str, typer.Option(help="Separador registrado o 'none'")] = "htdemucs",
    num_step: Annotated[int, typer.Option(min=1, max=64)] = 16,
    tts_batch: Annotated[int, typer.Option(min=1, max=64)] = 8,
    n_best: Annotated[int, typer.Option(min=1, max=8)] = 4,
    evaluate_quality: Annotated[bool, typer.Option("--evaluate/--no-evaluate")] = False,
    diarize: Annotated[bool, typer.Option(help="Varios hablantes: una voz por hablante")] = False,
    max_speakers: Annotated[int, typer.Option(min=1, max=16)] = 4,
    merge: Annotated[bool, typer.Option("--merge/--no-merge",
                                        help="Unir fragmentos de frase")] = True,
    verify_attempts: Annotated[int, typer.Option(min=0, max=5,
                                                 help="Reintentos por inteligibilidad")] = 1,
    reference_candidates: Annotated[int, typer.Option(min=1, max=8,
                                                      help="Referencias sondeadas")] = 1,
    text_guidance: Annotated[float | None, typer.Option(
        help="CFG dual: peso de texto (at)")] = None,
    reference_guidance: Annotated[float | None, typer.Option(
        help="CFG dual: peso de referencia (as)")] = None,
    accent_guidance: Annotated[float, typer.Option(
        help="AAG: peso de analogía de acento (beta; 0 = desactivado)")] = 0.0,
    accent_proxy_dir: Annotated[Path | None, typer.Option(
        help="Carpeta con conjunto de proxies AAG guardados")] = None,
    speed_penalty: Annotated[float, typer.Option(
        help="Penalización por habla acelerada al elegir traducción")] = 0.5,
    concise_translation: Annotated[bool, typer.Option(
        help="Favorecer traducciones más compactas que eviten aceleración")] = False,
    per_phrase_reference: Annotated[bool, typer.Option(
        help="Usar fragmento de cada frase como referencia auxiliar de expresividad")] = False,
    prosody_transfer: Annotated[bool, typer.Option(
        help="Transferir contorno de energía y dinámica de la voz original")] = False,
    workdir: Annotated[Path | None, typer.Option(help="Carpeta de artefactos")] = None,
    profile: Annotated[Path | None, typer.Option(help="Reporte JSON de spans")] = None,
) -> None:
    """Doblar un video: ASR → traducción → clonación de voz con duración exacta → mezcla."""
    try:
        request = DubRequest(
            source=input.resolve(), target_language=to, source_language=source_language,
            asr_model_id=model, device=device, tts_engine=tts, translator=translator,
            separator=None if separator == "none" else separator, num_step=num_step,
            tts_batch=tts_batch, n_best=n_best, evaluate=evaluate_quality, diarize=diarize,
            max_speakers=max_speakers, merge_fragments=merge, verify_attempts=verify_attempts,
            reference_candidates=reference_candidates,
            text_guidance=text_guidance, reference_guidance=reference_guidance,
            accent_guidance=accent_guidance, accent_proxy_dir=accent_proxy_dir,
            speed_penalty=speed_penalty, concise_translation=concise_translation,
            per_phrase_reference=per_phrase_reference, prosody_transfer=prosody_transfer,
        )
        if not request.source.is_file():
            raise FileNotFoundError(f"Archivo no encontrado: {input}")
        target = workdir or input.with_name(f"{input.stem}.dub-{to}")
        use_case, _ = _dub_use_case()
        profiler = SpanProfiler()
        start = time.perf_counter()
        with profiler.activate():
            report = use_case.execute(request, target, profiler)
        elapsed = time.perf_counter() - start
        if profile:
            profiler.save(profile, {"source": str(input), "request": {
                k: str(v) for k, v in asdict(request).items()}, "elapsed_seconds": elapsed,
                "report": report_to_dict(report)})
        _print_dub(report, profiler.events, elapsed)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        error_console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc


def _values(text: str, cast=str) -> list:
    return [cast(v.strip()) for v in text.split(",") if v.strip()]


@app.command("dub-bench")
def dub_bench(
    input: Annotated[Path, typer.Argument(help="Video/audio de entrada")],
    to: Annotated[str, typer.Option("--to")],
    source_language: Annotated[str, typer.Option("--from")] = "auto",
    model: Annotated[str, typer.Option(help="Modelo ASR")] = "medium",
    num_step: Annotated[str, typer.Option(help="Lista, p. ej. 32,16,8")] = "16",
    tts_batch: Annotated[str, typer.Option(help="Lista, p. ej. 1,8,16")] = "8",
    tts: Annotated[str, typer.Option(help="Lista de motores TTS")] = "omnivoice",
    translator: Annotated[str, typer.Option(help="Lista de traductores")] = "nllb-600m",
    merge: Annotated[str, typer.Option(help="Lista: true,false")] = "true",
    verify_attempts: Annotated[str, typer.Option(help="Lista, p. ej. 0,1,2")] = "1",
    reference_candidates: Annotated[str, typer.Option(help="Lista, p. ej. 1,4")] = "1",
    text_guidance: Annotated[str | None, typer.Option(
        help="Lista de pesos de texto (at), p. ej. 3.0,5.0")] = None,
    reference_guidance: Annotated[str | None, typer.Option(
        help="Lista de pesos de referencia (as), p. ej. 2.0,3.0")] = None,
    accent_guidance: Annotated[str | None, typer.Option(
        help="Lista de pesos AAG (beta), p. ej. 0.0,2.0,5.0")] = None,
    accent_proxy_dir: Annotated[Path | None, typer.Option(
        help="Carpeta con proxies AAG guardados")] = None,
    speed_penalty: Annotated[str, typer.Option(help="Lista: 0.0,0.5")] = "0.5",
    concise_translation: Annotated[str, typer.Option(help="Lista: false,true")] = "false",
    per_phrase_reference: Annotated[str, typer.Option(help="Lista: false,true")] = "false",
    prosody_transfer: Annotated[str, typer.Option(help="Lista: false,true")] = "false",
    workdir: Annotated[Path, typer.Option()] = Path("reports/dub-bench"),
    output: Annotated[Path, typer.Option()] = Path("reports/dub-bench.json"),
) -> None:
    """Matriz A/B de doblaje: velocidad y calidad por variante; la primera es la línea base."""
    from itertools import product

    from subdub_titler.application.dub_bench import DubBenchmark, rows_to_dict
    base = DubRequest(source=input.resolve(), target_language=to,
                      source_language=source_language, asr_model_id=model, evaluate=True)
    tg_values = _values(text_guidance, float) if text_guidance else [None]
    rg_values = _values(reference_guidance, float) if reference_guidance else [None]
    ag_values = _values(accent_guidance, float) if accent_guidance else [0.0]
    sp_values = _values(speed_penalty, float)
    ct_values = _values(concise_translation, lambda x: x.lower() == "true")
    ppr_values = _values(per_phrase_reference, lambda x: x.lower() == "true")
    pt_values = _values(prosody_transfer, lambda x: x.lower() == "true")
    variants = [dict(tts_engine=e, translator=t, num_step=n, tts_batch=b, merge_fragments=m,
                     verify_attempts=v, reference_candidates=r,
                     text_guidance=tg, reference_guidance=rg,
                     accent_guidance=ag, accent_proxy_dir=accent_proxy_dir,
                     speed_penalty=sp, concise_translation=ct,
                     per_phrase_reference=ppr, prosody_transfer=pt)
                for e, t, n, b, m, v, r, tg, rg, ag, sp, ct, ppr, pt in product(
                    _values(tts), _values(translator), _values(num_step, int),
                    _values(tts_batch, int), _values(merge, lambda x: x.lower() == "true"),
                    _values(verify_attempts, int), _values(reference_candidates, int),
                    tg_values, rg_values, ag_values, sp_values, ct_values, ppr_values, pt_values)]
    from subdub_titler.infrastructure.dub_factory import AsrTranscripts, default_dub_factory
    registry = default_registry()
    transcript = AsrTranscripts(registry)(base, SpanProfiler())
    bench = DubBenchmark(DubUseCase(AsrTranscripts(registry), default_dub_factory(registry)),
                         SpanProfiler)
    rows = bench.run(base, variants, workdir, transcript)
    table = Table("Variante", "Pipeline s", "TTS s", "RTF TTS", "VRAM MiB", "CER",
                  "SIM ranura", "UTMOS", "Umbrales", "Adoptable")
    for row in rows:
        q = row.quality
        table.add_row(
            " ".join(f"{k}={v}" for k, v in row.variant.items()), f"{row.seconds:.1f}",
            f"{row.tts_seconds:.2f}", f"{row.tts_rtf:.3f}", f"{row.vram_peak_mb or 0:,.0f}",
            f"{q.get('cer') or 0:.3f}", f"{q.get('similarity_slot') or 0:.3f}",
            f"{q.get('mos') or 0:.2f}", f"{row.gates_passed}/{row.gates_total}",
            "[green]sí[/green]" if row.adoptable else "[red]no[/red]: " + "; ".join(
                row.regressions))
    console.print(table)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": 1, "source": str(input), "to": to,
                                  "rows": rows_to_dict(rows)}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    console.print(f"[green]Informe:[/green] {output}")


def _print_summary(title: str, summary: dict, found: list[str] | None) -> None:
    table = Table(title, "Valor")
    for key, value in summary.items():
        table.add_row(key, json.dumps(value, ensure_ascii=False) if isinstance(value, dict)
                      else str(value))
    console.print(table)
    if found is not None:
        console.print("[green]Sin regresiones frente a la línea base[/green]" if not found
                      else "[red]Regresiones:[/red] " + "; ".join(found))


@app.command("tts-eval")
def tts_eval(
    manifest: Annotated[Path | None, typer.Option(help="Corpus JSONL de síntesis")] = None,
    from_report: Annotated[Path | None, typer.Option(help="report.json de un doblaje")] = None,
    limit: Annotated[int | None, typer.Option(help="Máximo de elementos del informe")] = None,
    save_manifest: Annotated[Path | None, typer.Option(help="Guardar el corpus generado")] = None,
    language: Annotated[str, typer.Option(help="Idioma de los textos (con --from-report)")] = "es",
    tts: Annotated[str, typer.Option()] = "omnivoice",
    num_step: Annotated[int, typer.Option(min=1, max=64)] = 16,
    tts_batch: Annotated[int, typer.Option(min=1, max=64)] = 8,
    device: Annotated[str, typer.Option()] = "cuda",
    text_guidance: Annotated[float | None, typer.Option(
        help="CFG dual: peso de texto (at)")] = None,
    reference_guidance: Annotated[float | None, typer.Option(
        help="CFG dual: peso de referencia (as)")] = None,
    accent_guidance: Annotated[float, typer.Option(
        help="AAG: peso de analogía de acento (beta)")] = 0.0,
    accent_proxy_dir: Annotated[Path | None, typer.Option(
        help="Carpeta con proxies AAG guardados")] = None,
    baseline: Annotated[Path | None, typer.Option(help="Informe previo para comparar")] = None,
    output: Annotated[Path, typer.Option()] = Path("reports/tts-eval.json"),
) -> None:
    """Corpus dorado de síntesis: CER ida y vuelta, parecido de voz, UTMOS, duración y RTF."""
    from subdub_titler.application.dub_corpus import (
        TtsCorpusEvaluation,
        TtsItem,
        items_from_report,
        read_jsonl,
        regressions,
        write_jsonl,
    )
    from subdub_titler.application.dubbing import load_report
    from subdub_titler.infrastructure.dub_factory import default_dub_factory
    from subdub_titler.infrastructure.media import FfmpegAudioStore
    if (manifest is None) == (from_report is None):
        raise typer.BadParameter("Indica --manifest o --from-report (uno de los dos)")
    if from_report is not None:
        report = load_report(from_report)
        stem = report.outputs.get("voice_stem")
        items = items_from_report(report, Path(stem) if stem else from_report, language, limit)
        if save_manifest:
            write_jsonl(items, save_manifest)
    else:
        items = [TtsItem(**row) for row in read_jsonl(manifest)]
    request = DubRequest(source=(manifest or from_report).resolve(),
                         target_language=items[0].language, tts_engine=tts, num_step=num_step,
                         tts_batch=tts_batch, device=device, evaluate=True,
                         text_guidance=text_guidance, reference_guidance=reference_guidance,
                         accent_guidance=accent_guidance, accent_proxy_dir=accent_proxy_dir)
    factory = default_dub_factory(default_registry())
    evaluation = TtsCorpusEvaluation(factory.cloner(request), FfmpegAudioStore(),
                                     factory.evaluator(request), tts_batch)
    result = evaluation.run(items)
    found = regressions(result["summary"], json.loads(baseline.read_text(
        encoding="utf-8"))["summary"]) if baseline else None
    result |= {"schema_version": 1, "settings": {"tts": tts, "num_step": num_step,
                                                 "tts_batch": tts_batch,
                                                 "text_guidance": text_guidance,
                                                 "reference_guidance": reference_guidance,
                                                 "accent_guidance": accent_guidance},
               "regressions": found}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary("Síntesis", result["summary"], found)
    console.print(f"[green]Informe:[/green] {output}")
    if found:
        raise typer.Exit(1)


@app.command("listen-panel")
def listen_panel(
    workdirs: Annotated[list[Path], typer.Argument(help="Carpetas de doblaje del mismo video")],
    names: Annotated[str | None, typer.Option(help="Nombres de las variantes, separados por "
                                                   "comas (por defecto, el de la carpeta)")] = None,
    limit: Annotated[int, typer.Option(min=1, max=200)] = 20,
    min_seconds: Annotated[float, typer.Option(help="Duración mínima de las frases")] = 1.5,
    seed: Annotated[int, typer.Option()] = 0,
    output: Annotated[Path, typer.Option()] = Path("reports/listen-panel"),
) -> None:
    """Panel de escucha a ciegas: acento y parecido de voz de varias variantes de un doblaje."""
    from subdub_titler.application.listening_panel import ListeningPanel
    from subdub_titler.infrastructure.media import FfmpegAudioStore
    from subdub_titler.interfaces.listening_page import render_listening_page
    labels = _values(names) if names else [w.resolve().name for w in workdirs]
    if len(labels) != len(workdirs) or len(set(labels)) != len(labels):
        raise typer.BadParameter("Hace falta un nombre distinto por carpeta")
    if output.exists():
        shutil.rmtree(output / "audio", ignore_errors=True)
    try:
        items = ListeningPanel(FfmpegAudioStore()).build(dict(zip(labels, workdirs, strict=True)),
                                                         output, limit, min_seconds, seed)
    except FileNotFoundError as exc:
        error_console.print(f"[red]Error: archivo o carpeta no encontrado: {exc}[/red]")
        error_console.print("Pasa rutas a carpetas de doblaje con 'report.json'.")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        error_console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc
    title = "Panel de escucha · " + " / ".join(labels)
    page = output / "index.html"
    page.write_text(render_listening_page(items, title, f"{'-'.join(labels)}-{seed}"),
                    encoding="utf-8")
    console.print(f"{len(items)} frases × {len(labels)} variantes. [green]Abre:[/green] {page}")


@app.command("translate-eval")
def translate_eval(
    manifest: Annotated[Path, typer.Option(help="JSONL: source, reference, source_language, "
                                                "target_language")],
    translator: Annotated[str, typer.Option()] = "nllb-600m",
    device: Annotated[str, typer.Option()] = "cuda",
    baseline: Annotated[Path | None, typer.Option(help="Informe previo para comparar")] = None,
    output: Annotated[Path, typer.Option()] = Path("reports/translate-eval.json"),
) -> None:
    """chrF de la traducción frente a referencias humanas (misma métrica que sacreBLEU)."""
    from subdub_titler.application.dub_corpus import (
        TranslationItem,
        evaluate_translations,
        read_jsonl,
        regressions,
    )
    from subdub_titler.infrastructure.dub_factory import default_dub_factory
    items = [TranslationItem(**row) for row in read_jsonl(manifest)]
    request = DubRequest(source=manifest.resolve(), target_language=items[0].target_language,
                         translator=translator, device=device)
    factory = default_dub_factory(default_registry())
    result = evaluate_translations(factory.translators.create(translator, request), items)
    found = regressions(result["summary"], json.loads(baseline.read_text(
        encoding="utf-8"))["summary"]) if baseline else None
    result |= {"schema_version": 1, "translator": translator, "regressions": found}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary("Traducción", result["summary"], found)
    console.print(f"[green]Informe:[/green] {output}")
    if found:
        raise typer.Exit(1)


@app.command()
def ui(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    max_upload_gb: Annotated[float, typer.Option(
        help="Límite máximo de subida en GiB para videos o audios (0 para ilimitado).")] = 10.0,
) -> None:
    """API + UI compilada. Durante desarrollo ejecutar npm run dev en apps/web."""
    if host not in ("127.0.0.1", "localhost"):
        console.print("[red]Por seguridad sólo se permite loopback en esta versión.[/red]")
        raise typer.Exit(1)
    os.environ["SUBDUB_MAX_UPLOAD_GB"] = str(max_upload_gb)
    import uvicorn
    console.print(f"API: http://127.0.0.1:{port}/docs")
    console.print("UI: compilar primero con cd apps/web && npm run build")
    if max_upload_gb > 0:
        console.print(f"Límite de carga: {max_upload_gb:.1f} GiB (--max-upload-gb)")
    else:
        console.print("Límite de carga: sin límite")
    # Sin límite, Ctrl+C se queda en «Shutting down» mientras el reproductor tenga abierta una
    # descarga del video (FileResponse con Range): uvicorn espera a que el navegador la cierre.
    uvicorn.run("subdub_titler.interfaces.api:create_app", host="127.0.0.1", port=port,
                workers=1, factory=True, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    app()
