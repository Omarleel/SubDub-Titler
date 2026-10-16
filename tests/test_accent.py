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


def test_listening_page_includes_naturalness_scale(tmp_path: Path):
    variants = {"clonada": dub(tmp_path, "a")}
    out = tmp_path / "panel"
    items = ListeningPanel(FakeAudioStore()).build(variants, out, limit=1, min_seconds=0.0)
    page = render_listening_page(items, "Test Panel", "p-test")
    assert "Naturalidad" in page
    assert 'scale(clip.file, "naturalness", "Naturalidad (1–5)")' in page
    assert "<th>Naturalidad</th>" in page


def test_dual_cfg_log_probs_equivalence_and_normalization():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from subdub_titler.infrastructure.omnivoice_fast import dual_cfg_log_probs

    torch.manual_seed(42)
    # Forma (batch=1, channels=1, seq=4, vocab=32)
    u_logits = torch.randn(1, 1, 4, 32)
    c_logits = torch.randn(1, 1, 4, 32)
    t_logits = torch.randn(1, 1, 4, 32)

    # 1. guidance_scale = 2.0 estándar en OmniVoice aplica: c + 2*(c - u) = 3*c - 2*u
    # En log_softmax:
    u_log = F.log_softmax(u_logits, dim=-1)
    c_log = F.log_softmax(c_logits, dim=-1)
    expected_standard = F.log_softmax(c_log + 2.0 * (c_log - u_log), dim=-1)

    # Con CFG dual at=3.0, as=3.0 (sin rama t o con t cancelado)
    dual_result = dual_cfg_log_probs(u_logits, c_logits, t_logits, at=3.0, as_=3.0)
    assert torch.allclose(expected_standard, dual_result, atol=1e-5)

    # 2. Pesos asimétricos at=5.0, as=2.0
    asym_result = dual_cfg_log_probs(u_logits, c_logits, t_logits, at=5.0, as_=2.0)
    t_log = F.log_softmax(t_logits, dim=-1)
    manual_formula = u_log + 5.0 * (t_log - u_log) + 2.0 * (c_log - t_log)
    expected_asym = F.log_softmax(manual_formula, dim=-1)
    assert torch.allclose(expected_asym, asym_result, atol=1e-5)

    # 3. Suma de probabilidades exponenciadas es 1.0 en el vocabulario
    probs = torch.exp(asym_result)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5)


def test_dub_request_guidance_validation(tmp_path: Path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"x")
    # Válidos
    req = DubRequest(source=source, target_language="es", text_guidance=5.0,
                     reference_guidance=2.0)
    assert req.text_guidance == 5.0
    assert req.reference_guidance == 2.0

    # Rango negativo
    with pytest.raises(ValueError, match="text_guidance"):
        DubRequest(source=source, target_language="es", text_guidance=-0.1)
    with pytest.raises(ValueError, match="reference_guidance"):
        DubRequest(source=source, target_language="es", reference_guidance=-0.5)

    # Rango superior > 20
    with pytest.raises(ValueError, match="text_guidance"):
        DubRequest(source=source, target_language="es", text_guidance=25.0)
    with pytest.raises(ValueError, match="reference_guidance"):
        DubRequest(source=source, target_language="es", reference_guidance=20.1)


def test_omnivoice_fast_install_routes_batch_1_with_dual_cfg(monkeypatch):
    from types import SimpleNamespace

    import subdub_titler.infrastructure.omnivoice_fast as fast_mod
    from subdub_titler.infrastructure.omnivoice_fast import install

    monkeypatch.setattr(fast_mod, "supported", lambda: True)

    calls = []

    def mock_original(task, gen_config):
        calls.append("original")
        return ["original_result"]

    def mock_split(model, task, gen_config):
        calls.append("split")
        return ["split_result"]

    monkeypatch.setattr(fast_mod, "_generate_split", mock_split)

    dummy_model = SimpleNamespace(_generate_iterative=mock_original)
    installed = install(dummy_model)
    assert installed is True

    # 1. Batch size = 1, sin dual CFG -> llama original
    task_b1 = SimpleNamespace(batch_size=1)
    cfg_standard = SimpleNamespace(guidance_scale=2.0)
    res1 = dummy_model._generate_iterative(task_b1, cfg_standard)
    assert calls[-1] == "original"
    assert res1 == ["original_result"]

    # 2. Batch size = 1, con dual CFG asimétrico (at=5.0, as=2.0) -> llama split
    cfg_dual = SimpleNamespace(guidance_scale=2.0, text_guidance=5.0, reference_guidance=2.0)
    res2 = dummy_model._generate_iterative(task_b1, cfg_dual)
    assert calls[-1] == "split"
    assert res2 == ["split_result"]

    # 3. Batch size = 4 -> siempre llama split
    task_b4 = SimpleNamespace(batch_size=4)
    res3 = dummy_model._generate_iterative(task_b4, cfg_standard)
    assert calls[-1] == "split"
    assert res3 == ["split_result"]

    # 4. Batch size = 1, con AAG (beta=3.0) -> llama split
    cfg_aag = SimpleNamespace(guidance_scale=2.0, accent_guidance_beta=3.0)
    res4 = dummy_model._generate_iterative(task_b1, cfg_aag)
    assert calls[-1] == "split"
    assert res4 == ["split_result"]


def test_accent_proxy_pair_and_set_persistence(tmp_path: Path):
    torch = pytest.importorskip("torch")

    from subdub_titler.infrastructure.accent_proxy import AccentProxyPair, AccentProxySet

    torch.manual_seed(123)
    p1 = AccentProxyPair(
        id="proxy_female",
        gender="female",
        src_lang="en",
        src_text="Hello world.",
        src_audio_tokens=torch.randint(0, 1024, (8, 50)),
        tgt_lang="es",
        tgt_text="Hola mundo.",
        tgt_audio_tokens=torch.randint(0, 1024, (8, 48)),
        src_rms=0.12,
        tgt_rms=0.11,
    )
    p2 = AccentProxyPair(
        id="proxy_male",
        gender="male",
        src_lang="en",
        src_text="Good morning.",
        src_audio_tokens=torch.randint(0, 1024, (8, 60)),
        tgt_lang="es",
        tgt_text="Buenos días.",
        tgt_audio_tokens=torch.randint(0, 1024, (8, 58)),
        src_rms=0.09,
        tgt_rms=0.10,
    )
    proxy_set = AccentProxySet(
        version="1.0",
        source_language="en",
        target_language="es",
        pairs=(p1, p2),
    )

    # Filtrado por género
    assert proxy_set.for_gender("female") == (p1,)
    assert proxy_set.for_gender("male") == (p2,)
    assert proxy_set.for_gender("auto") == (p1, p2)
    assert proxy_set.for_gender("non_existent") == (p1, p2)

    # Persistencia
    out_dir = tmp_path / "proxies_v1"
    proxy_set.save(out_dir)
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "proxy_female_src.pt").is_file()
    assert (out_dir / "proxy_male_tgt.pt").is_file()

    loaded = AccentProxySet.load(out_dir)
    assert loaded.version == "1.0"
    assert len(loaded.pairs) == 2
    l1 = loaded.pairs[0]
    assert l1.id == "proxy_female"
    assert l1.gender == "female"
    assert l1.src_audio_tokens.shape == (8, 50)
    assert torch.equal(l1.src_audio_tokens, p1.src_audio_tokens)


def test_aag_cfg_log_probs_formula_and_normalization():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from subdub_titler.infrastructure.omnivoice_fast import aag_cfg_log_probs, dual_cfg_log_probs

    torch.manual_seed(99)
    u_logits = torch.randn(1, 1, 3, 16)
    c_logits = torch.randn(1, 1, 3, 16)
    t_logits = torch.randn(1, 1, 3, 16)
    ps1 = torch.randn(1, 1, 3, 16)
    pt1 = torch.randn(1, 1, 3, 16)
    ps2 = torch.randn(1, 1, 3, 16)
    pt2 = torch.randn(1, 1, 3, 16)

    # Con beta=0: exactamente igual a dual_cfg_log_probs
    base_res = dual_cfg_log_probs(u_logits, c_logits, t_logits, at=5.0, as_=2.0)
    aag_beta_0 = aag_cfg_log_probs(
        u_logits, c_logits, t_logits, at=5.0, as_=2.0,
        proxy_src_logits=[ps1], proxy_tgt_logits=[pt1], beta=0.0
    )
    assert torch.allclose(base_res, aag_beta_0, atol=1e-5)

    # Con beta=3.0 y M=2 proxies
    u_log = F.log_softmax(u_logits, dim=-1)
    c_log = F.log_softmax(c_logits, dim=-1)
    t_log = F.log_softmax(t_logits, dim=-1)
    base_guided = u_log + 5.0 * (t_log - u_log) + 2.0 * (c_log - t_log)

    bias1 = F.log_softmax(ps1, dim=-1) - F.log_softmax(pt1, dim=-1)
    bias2 = F.log_softmax(ps2, dim=-1) - F.log_softmax(pt2, dim=-1)
    accent_bias = bias1 + bias2
    manual_aag = base_guided - (3.0 / 2.0) * accent_bias
    expected = F.log_softmax(manual_aag, dim=-1)

    result = aag_cfg_log_probs(
        u_logits, c_logits, t_logits, at=5.0, as_=2.0,
        proxy_src_logits=[ps1, ps2], proxy_tgt_logits=[pt1, pt2], beta=3.0
    )
    assert torch.allclose(expected, result, atol=1e-5)
    # Suma exp = 1.0
    probs = torch.exp(result)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5)


def test_aag_requires_proxies_when_beta_positive():
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from subdub_titler.infrastructure.omnivoice_fast import _generate_split

    dummy_self = SimpleNamespace(
        config=SimpleNamespace(
            num_audio_codebook=8,
            audio_vocab_size=1024,
            audio_mask_id=1024,
        ),
        device="cpu",
    )
    task = SimpleNamespace(batch_size=1, target_lens=[10], texts=["Hola"])
    gen_config = SimpleNamespace(
        guidance_scale=2.0,
        accent_guidance_beta=3.0,
        accent_proxies=None,
    )
    with pytest.raises(ValueError, match="Modo AAG activo"):
        _generate_split(dummy_self, task, gen_config)


def test_dub_request_accent_guidance_validation(tmp_path: Path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"x")
    req = DubRequest(source=source, target_language="es", accent_guidance=5.0)
    assert req.accent_guidance == 5.0

    with pytest.raises(ValueError, match="accent_guidance"):
        DubRequest(source=source, target_language="es", accent_guidance=-0.1)
    with pytest.raises(ValueError, match="accent_guidance"):
        DubRequest(source=source, target_language="es", accent_guidance=55.0)


