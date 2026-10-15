from pathlib import Path

import pytest
from dub_fakes import FakeFactory, no_transcripts, transcript

from subdub_titler.application.dub_bench import BenchRow, regressions
from subdub_titler.application.dubbing import DubUseCase, load_report
from subdub_titler.domain.dubbing import DubRequest
from subdub_titler.infrastructure.profiling import SpanProfiler


def request(tmp_path: Path, **changes) -> DubRequest:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"x")
    return DubRequest(source=source, target_language="es", **changes)


def test_dub_pipeline_end_to_end_with_fakes(tmp_path: Path):
    factory = FakeFactory()
    use_case = DubUseCase(no_transcripts, factory)
    profiler = SpanProfiler()
    stages = []
    with profiler.activate():
        report = use_case.execute(request(tmp_path, tts_batch=2, evaluate=True),
                                  tmp_path / "work", profiler, transcript(),
                                  lambda stage, fraction: stages.append(stage))
    assert len(report.track) == 3
    # Cue corto (1 s): la traducción larga no cabe y se elige la corta.
    assert report.track[0].target_text == "Hell"
    # Cue largo (4 s): cabe la mejor puntuada.
    assert report.track[1].target_text.endswith("traducido con mucho detalle")
    assert all(c.generated_seconds == pytest.approx(c.target_seconds) for c in report.track)
    assert report.sync.overlaps == 0 and report.sync.within_tolerance == 1.0
    assert factory.cloner.batches == [2, 1]  # lotes de tts_batch ordenados por duración
    assert [p[0] for p in factory.mixer.placements] == [c.start for c in report.track]
    assert {"video", "mix", "voice", "subtitles", "voice_stem"} <= set(report.outputs)
    assert report.reference.text == "Hello there. This is a long and complete sentence."
    names = [e.name for e in profiler.events]
    for span in ("dub.separate", "dub.voice_prompt", "dub.translate", "dub.fit",
                 "dub.tts.generate", "dub.mix", "dub.mux", "dub.eval"):
        assert span in names
    assert stages[0] == "separar" and stages[-1] == "listo"
    assert report.quality is not None and report.quality.similarity_original == 0.6
    assert report.quality.background_wpm == 0.5
    gates = {g.name: g.passed for g in report.gates}
    assert gates["sincronia"] is True and gates["cer_ida_vuelta"] is False
    assert load_report(tmp_path / "work" / "report.json") == report
    assert (tmp_path / "work" / "dub.srt").read_text(encoding="utf-8").startswith("1\n")


def test_same_language_skips_translation_and_no_separator(tmp_path: Path):
    factory = FakeFactory()
    data = transcript()
    from dataclasses import replace
    report = DubUseCase(no_transcripts, factory).execute(
        request(tmp_path, separator=None), tmp_path / "w", transcript=replace(data, language="es"))
    assert factory.translator.calls == 0
    assert report.track[1].target_text == "This is a long and complete sentence."
    assert "voice_stem" not in report.outputs and report.quality is None


def test_regenerate_single_cue_keeps_others(tmp_path: Path):
    factory = FakeFactory()
    use_case = DubUseCase(no_transcripts, factory)
    req = request(tmp_path, evaluate=True)
    first = use_case.execute(req, tmp_path / "w", transcript=transcript())
    factory.cloner.batches.clear()
    updated = use_case.regenerate(req, tmp_path / "w", 2, "Adiós a todos")
    assert factory.cloner.batches == [1]
    assert updated.track[2].target_text == "Adiós a todos"
    assert updated.track[:2] == first.track[:2]
    assert load_report(tmp_path / "w" / "report.json").track[2].target_text == "Adiós a todos"
    with pytest.raises(ValueError):
        use_case.regenerate(req, tmp_path / "w", 99, "x")


def test_empty_transcript_is_rejected(tmp_path: Path):
    from subdub_titler.domain.models import Transcript
    with pytest.raises(ValueError):
        DubUseCase(no_transcripts, FakeFactory()).execute(
            request(tmp_path), tmp_path / "w", transcript=Transcript((), "en", "x"))


def row(**quality):
    return BenchRow({}, 1, 1, 1, 1, None, quality, 0, 0, ())


def test_bench_flags_quality_regressions_with_tolerance():
    base = row(cer=0.02, similarity_slot=0.6, mos=3.0, within_tolerance=1.0)
    assert regressions(row(cer=0.025, similarity_slot=0.59, mos=2.95, within_tolerance=0.99),
                       base) == ()
    found = regressions(row(cer=0.05, similarity_slot=0.5, mos=3.0, within_tolerance=1.0), base)
    assert [f.split(":")[0] for f in found] == ["cer", "similarity_slot"]


def test_components_release_gpu_after_their_stage(tmp_path: Path):
    """Con todos los modelos cargados a la vez un proceso desborda 16 GB: cada etapa libera."""
    factory = FakeFactory()
    released = []
    factory.translator.release = lambda: released.append("traductor")
    factory.cloner.release = lambda: released.append("tts")
    DubUseCase(no_transcripts, factory).execute(request(tmp_path), tmp_path / "w",
                                                transcript=transcript())
    assert released == ["traductor", "tts"]
