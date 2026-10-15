import random

import pytest
from dub_fakes import (
    EchoTranscriber,
    FakeAudioStore,
    FakeCloner,
    FakeTranslator,
    FixedMos,
    FixedSpeaker,
)

from subdub_titler.application.dub_corpus import (
    TranslationItem,
    TtsCorpusEvaluation,
    TtsItem,
    evaluate_translations,
    read_jsonl,
    regressions,
    write_jsonl,
)
from subdub_titler.application.dub_quality import DubEvaluator
from subdub_titler.application.quality import chrf
from subdub_titler.infrastructure.nllb import cjk_punctuation


def test_chrf_matches_sacrebleu():
    sacrebleu = pytest.importorskip("sacrebleu.metrics")
    generator = random.Random(7)
    words = "la venganza es el camino hola que tal 今日は 元気 です a".split()
    for _ in range(300):
        size = generator.randint(1, 4)
        hyps = [" ".join(generator.choices(words, k=generator.randint(0, 7))) for _ in range(size)]
        refs = [" ".join(generator.choices(words, k=generator.randint(0, 7))) for _ in range(size)]
        expected = sacrebleu.CHRF().corpus_score(hyps, [refs]).score
        assert chrf(hyps, refs) == pytest.approx(expected, abs=1e-9)


def test_chrf_bounds():
    assert chrf(["hola mundo"], ["hola mundo"]) == pytest.approx(100)
    assert chrf(["xyz"], ["abc"]) == 0
    with pytest.raises(ValueError):
        chrf(["a"], [])


def test_cjk_punctuation_only_after_cjk():
    assert cjk_punctuation("价格是3.5元,很好!", "zh") == "价格是3.5元，很好！"
    assert cjk_punctuation("計画が起因した. 次に", "ja") == "計画が起因した。次に"
    assert cjk_punctuation('担心"看起来像蛋",由于', "zh") == '担心"看起来像蛋"，由于'
    assert cjk_punctuation("Hola, mundo.", "es") == "Hola, mundo."


def test_tts_corpus_is_grouped_by_voice_and_measures_duration(tmp_path):
    cloner = FakeCloner()
    evaluator = DubEvaluator(EchoTranscriber(), FixedSpeaker(), FixedMos())
    items = [TtsItem(f"i{n}", "ref.wav", "Referencia.", "texto", "es", duration=d,
                     ref_start=0.0, ref_end=3.0) for n, d in enumerate((1.0, 2.5, 0.5))]
    items.append(TtsItem("libre", "otra.wav", "Otra.", "texto", "es"))
    result = TtsCorpusEvaluation(cloner, FakeAudioStore(), evaluator, batch=2).run(items)
    summary = result["summary"]
    assert summary["items"] == 4 and summary["cer"] == 0.0
    assert summary["mean_abs_duration_error_ms"] == 0.0
    assert summary["similarity"] == 0.6 and summary["mos"] == 3.0
    assert cloner.batches == [2, 1, 1]  # 3 de una voz (lotes de 2) + 1 de otra
    durations = {row["id"]: row["requested_seconds"] for row in result["items"]}
    assert durations["libre"] == pytest.approx(0.08 * len("texto"))
    write_jsonl(items, tmp_path / "c.jsonl")
    assert [TtsItem(**row) for row in read_jsonl(tmp_path / "c.jsonl")] == items


def test_translation_corpus_by_language_pair():
    items = [TranslationItem("Hello there.", "Hello there. traducido con mucho detalle", "en",
                             "es"),
             TranslationItem("Bye.", "algo distinto", "en", "ja")]
    result = evaluate_translations(FakeTranslator(), items)
    assert result["summary"]["chrf_by_pair"]["en->es"] == pytest.approx(100)
    assert result["summary"]["chrf_by_pair"]["en->ja"] < 30
    assert result["items"][0]["hypothesis"].endswith("traducido con mucho detalle")


def test_corpus_regressions_respect_direction_and_tolerance():
    base = {"cer": 0.05, "similarity": 0.6, "mean_abs_duration_error_ms": 5.0, "chrf": 50.0}
    assert regressions({"cer": 0.055, "similarity": 0.59, "mean_abs_duration_error_ms": 20.0,
                        "chrf": 49.5}, base) == []
    found = regressions({"cer": 0.08, "similarity": 0.6, "mean_abs_duration_error_ms": 40.0,
                         "chrf": 45.0}, base)
    assert [f.split(":")[0] for f in found] == ["cer", "mean_abs_duration_error_ms", "chrf"]
