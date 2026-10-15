"""Fase 0 del acento: p(idioma destino) en la evaluación y panel de escucha a ciegas."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
from dub_fakes import (
    EchoTranscriber,
    FakeAudioStore,
    FakeCloner,
    FakeFactory,
    FixedLanguage,
    no_transcripts,
    transcript,
)

from subdub_titler.application.dub_corpus import TtsCorpusEvaluation, TtsItem
from subdub_titler.application.dub_quality import DubEvaluator
from subdub_titler.application.dubbing import DubUseCase, load_report
from subdub_titler.application.listening_panel import ListeningPanel, shared_cues
from subdub_titler.domain.dubbing import DubRequest
from subdub_titler.interfaces.listening_page import render_listening_page


def dub(tmp_path: Path, name: str, evaluate: bool = False) -> Path:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"x")
    workdir = tmp_path / name
    DubUseCase(no_transcripts, FakeFactory()).execute(
        DubRequest(source=source, target_language="es", evaluate=evaluate), workdir,
        transcript=transcript())
    return workdir


def test_report_measures_target_language_probability(tmp_path: Path):
    report = load_report(dub(tmp_path, "w", evaluate=True) / "report.json")
    assert {q.language_prob for q in report.cue_quality} == {0.4}
    assert report.quality.language_prob == 0.4
    # Medida informativa: aún no es un umbral (sin calibrar frente al panel de escucha).
    assert "acento" not in {g.name for g in report.gates}


def test_language_probability_is_optional_and_reaches_the_tts_corpus():
    items = [TtsItem("i", "ref.wav", "Ref.", "texto", "es", 1.0, 0.0, 3.0)]
    without = TtsCorpusEvaluation(FakeCloner(), FakeAudioStore(),
                                  DubEvaluator(EchoTranscriber()), 8).run(items)
    assert without["summary"]["language_prob"] is None
    with_lid = TtsCorpusEvaluation(FakeCloner(), FakeAudioStore(), DubEvaluator(
        EchoTranscriber(), language_id=FixedLanguage()), 8).run(items)
    assert with_lid["summary"]["language_prob"] == 0.4
    assert with_lid["items"][0]["language_prob"] == 0.4


def test_shared_cues_require_same_text_and_minimum_length(tmp_path: Path):
    report = load_report(dub(tmp_path, "w") / "report.json")
    assert [c.index for c in shared_cues([report, report], 10, 0.0, 0)] == \
        [c.index for c in report.track]
    changed = replace(report, track=(replace(report.track[0], target_text="otro"),
                                     *report.track[1:]))
    assert report.track[0].index not in {c.index for c in shared_cues([report, changed], 10,
                                                                      0.0, 0)}
    longest = max(c.generated_seconds for c in report.track)
    assert shared_cues([report], 10, longest + 1, 0) == []


def test_listening_panel_is_blind_and_reproducible(tmp_path: Path):
    variants = {"clonada": dub(tmp_path, "a"), "sin clonar": dub(tmp_path, "b")}
    out = tmp_path / "panel"
    items = ListeningPanel(FakeAudioStore()).build(variants, out, limit=10, min_seconds=0.0)
    assert items and all(len(i.clips) == 2 for i in items)
    files = [i.original for i in items] + [c.file for i in items for c in i.clips]
    assert len(set(files)) == len(files) and all((out / f).exists() for f in files)
    assert not any("clon" in f for f in files)  # el nombre no delata la variante
    again = ListeningPanel(FakeAudioStore()).build(variants, tmp_path / "otra", 10, 0.0)
    assert again == items
    page = render_listening_page(items, "Panel </script>", "p-0")
    assert "</script>\"" not in page and "Panel &lt;/script&gt;" in page
    data = page.split("const PANEL = ", 1)[1].split(";\nconst KEY", 1)[0]
    assert json.loads(data)["items"][0]["text"] == items[0].text


def test_listening_panel_rejects_variants_without_shared_phrases(tmp_path: Path):
    with pytest.raises(ValueError):
        ListeningPanel(FakeAudioStore()).build({}, tmp_path / "p")
