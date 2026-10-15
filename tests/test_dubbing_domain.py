from pathlib import Path

import pytest

from subdub_titler.domain.dubbing import (
    Candidate,
    DubbedCue,
    DubCue,
    DubRequest,
    FitAction,
    QualitySummary,
    TimingPolicy,
    choose_candidate,
    fit_decision,
    pick_reference,
    plan_slots,
    quality_gates,
    sync_metrics,
)

POLICY = TimingPolicy(max_borrow=0.5, max_advance=0.25, min_gap=0.05)


def cue(index, start, end, text="Hola."):
    return DubCue(index, start, end, text)


def test_slots_without_estimates_borrow_up_to_limit_and_never_overlap():
    cues = [cue(0, 1.0, 2.0), cue(1, 2.2, 3.0), cue(2, 5.0, 6.0)]
    slots = plan_slots(cues, POLICY, media_end=6.2)
    assert [s.end for s in slots] == [2.15, 3.5, 6.2]
    assert all(a.end + POLICY.min_gap <= b.start + 1e-9
               for a, b in zip(slots, slots[1:], strict=False))


def test_slots_split_gap_by_need_previous_first_then_advance_next():
    cues = [cue(0, 1.0, 2.0), cue(1, 3.0, 4.0)]
    # El primero necesita 0.3 s más (los toma al final); el segundo 0.6 s: 0.5 al final
    # (máximo) y los 0.1 restantes adelantándose.
    slots = plan_slots(cues, POLICY, media_end=10, natural=[1.3, 1.6])
    assert slots[0].end == 2.3
    assert slots[1].start == 2.9
    assert slots[1].end == 4.5


def test_slots_respect_media_bounds_and_cue_minimum():
    cues = [cue(0, 0.1, 1.0)]
    slot = plan_slots(cues, POLICY, media_end=1.1, natural=[5.0])[0]
    assert slot.start == 0.0 and slot.end == 1.1
    shorter = plan_slots(cues, POLICY, natural=[0.2])[0]
    assert (shorter.start, shorter.end) == (0.1, 1.0)  # nunca más corta que el cue


def test_fit_keeps_natural_pace_compresses_or_flags():
    ok = fit_decision(1.0, 2.0)
    assert ok.target_duration == 1.0 and ok.speed == 1.0 and ok.action is FitAction.OK
    squeezed = fit_decision(2.2, 2.0)
    assert squeezed.target_duration == 2.0 and squeezed.action is FitAction.OK
    flagged = fit_decision(3.0, 2.0)
    assert flagged.action is FitAction.SHORTEN and flagged.speed == 1.5
    filled = fit_decision(1.0, 2.0, TimingPolicy(fill_slot=True, min_speed=0.8))
    assert filled.target_duration == 1.25


def test_candidate_prefers_best_score_that_fits_else_fastest_to_say():
    options = [Candidate("larga", -0.1), Candidate("corta", -0.5), Candidate("media", -0.3)]
    assert choose_candidate(options, [3.0, 1.0, 2.0], slot_seconds=2.0) == 2
    assert choose_candidate(options, [5.0, 3.0, 4.0], slot_seconds=2.0) == 1
    with pytest.raises(ValueError):
        choose_candidate(options, [1.0], 2.0)


def test_reference_prefers_clean_sentence_boundaries_with_margin():
    cues = [cue(0, 0.0, 4.0, "Una frase completa."), cue(1, 4.1, 9.0, "que sigue sin"),
            cue(2, 9.05, 10.0, "terminar"), cue(3, 12.0, 16.5, "Otra frase entera.")]
    reference = pick_reference(cues, min_seconds=3, max_seconds=10)
    assert reference.text == "Otra frase entera."
    assert reference.start == pytest.approx(11.8) and reference.end == pytest.approx(16.7)


def test_reference_falls_back_to_longest_cue():
    reference = pick_reference([cue(0, 0, 1, "Hola"), cue(1, 5, 7, "Adiós")], min_seconds=3)
    assert reference.text == "Adiós"
    with pytest.raises(ValueError):
        pick_reference([])


def dubbed(index, start, target, generated, speed=1.0, action=FitAction.OK):
    return DubbedCue(index, start, "src", "dst", target, target, target, generated, speed,
                     action)


def test_sync_metrics_detects_errors_overlaps_and_speed():
    track = [dubbed(0, 0.0, 1.0, 1.0), dubbed(1, 1.1, 1.0, 1.5, 1.4, FitAction.SHORTEN),
             dubbed(2, 2.0, 2.0, 2.05)]
    report = sync_metrics(track)
    assert report.cues == 3
    assert report.within_tolerance == pytest.approx(2 / 3)
    assert report.overlaps == 1  # el cue 1 termina en 2.6 > 2.0
    assert report.speed_in_range == pytest.approx(2 / 3)
    assert report.shortened == 1
    assert sync_metrics([]).within_tolerance == 1.0


def test_quality_gates_relative_similarity_and_language_specific_cer():
    sync = sync_metrics([dubbed(0, 0, 1, 1)])
    quality = QualitySummary(cer=0.05, similarity=0.5, similarity_original=0.55, mos=3.0,
                             mos_original=3.05, background_wpm=0.0, lufs=-16.4,
                             true_peak=-1.5)
    latin = {g.name: g.passed for g in quality_gates("es", sync, quality)}
    assert latin["cer_ida_vuelta"] is False and latin["similitud_voz_relativa"] is True
    assert latin["utmos_vs_original"] is True and latin["error_lufs"] is True
    cjk = {g.name: g.passed for g in quality_gates("ja", sync, quality)}
    assert cjk["cer_ida_vuelta"] is True
    unknown = {g.name: g.passed for g in quality_gates("es", sync, QualitySummary())}
    assert unknown["similitud_voz_relativa"] is None


def test_dub_request_validation(tmp_path: Path):
    with pytest.raises(ValueError):
        DubRequest(tmp_path / "a.mp4", target_language="fr")
    with pytest.raises(ValueError):
        DubRequest(tmp_path / "a.mp4", target_language="es", num_step=0)
    assert DubRequest(tmp_path / "a.mp4", target_language="es").separator == "htdemucs"
