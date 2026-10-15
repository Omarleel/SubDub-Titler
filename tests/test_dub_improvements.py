"""Fusión de fragmentos, sondeo de referencias y reintento verificado (con puertos falsos)."""
from dataclasses import replace
from pathlib import Path

from dub_fakes import FakeFactory, no_transcripts

from subdub_titler.application.dubbing import DubUseCase
from subdub_titler.domain.dubbing import (
    AudioClip,
    DubCue,
    DubRequest,
    MergePolicy,
    VerifyPolicy,
    merge_fragments,
    probe_indices,
    reference_candidates,
)
from subdub_titler.domain.models import Segment, Transcript


def test_merge_joins_fragments_and_short_cues_only_when_close():
    cues = [DubCue(0, 0.0, 2.0, "Tenemos que irnos"), DubCue(1, 2.2, 4.0, "antes de la tormenta."),
            DubCue(2, 4.1, 4.6, "Vale."), DubCue(3, 6.0, 8.0, "Otra frase completa."),
            DubCue(4, 8.1, 10.0, "De otro hablante.", speaker="S1")]
    merged = merge_fragments(cues)
    assert [c.source_text for c in merged] == [
        "Tenemos que irnos antes de la tormenta. Vale.", "Otra frase completa.",
        "De otro hablante."]
    assert [c.index for c in merged] == [0, 1, 2] and merged[0].end == 4.6
    # Límite de duración: no se une si el resultado sería demasiado largo.
    assert len(merge_fragments(cues, MergePolicy(max_seconds=3.0))) == 4  # sin el primer cue


def test_reference_candidates_do_not_overlap_and_probe_spreads():
    cues = [DubCue(i, i * 5.0, i * 5.0 + 4.0, f"Frase {i}.") for i in range(6)]
    spans = reference_candidates(cues, min_seconds=3, limit=3)
    assert len(spans) == 3
    assert all(a.end <= b.start or b.end <= a.start
               for i, a in enumerate(spans) for b in spans[i + 1:])
    assert probe_indices([2.0] * 20, VerifyPolicy(probe_cues=8)) == [0, 2, 5, 7, 10, 12, 15, 17]
    assert probe_indices([0.5, 2.0, 10.0]) == [1]


class Verifier:
    """Entiende el texto sintetizado salvo la variante 0 de los textos marcados con '!'."""

    def __init__(self, cloner) -> None:
        self.cloner = cloner

    def transcribe_clips(self, clips, language):
        return [self.cloner.spoken[id(clip.samples)] for clip in clips]


class TaggingCloner:
    """Marca las muestras: 0.0 = variante 0 de un texto con '!', 0.1 en el resto."""
    sample_rate = 100

    def __init__(self) -> None:
        self.variants = []
        self.voices = []
        self.spoken = {}

    def prepare_voice(self, reference, text):
        self.voices.append(text)
        return text

    def natural_seconds(self, texts, language, voice):
        return [1.0 for _ in texts]

    def synthesize(self, texts, durations, language, voice, variant=0):
        self.variants.append(variant)
        clips = []
        for text, duration in zip(texts, durations, strict=True):
            samples = [0.1] * int(duration * 100)
            self.spoken[id(samples)] = "mal" if "!" in text and variant == 0 else text
            clips.append(AudioClip(samples, 100))
        return clips


class Encoder:
    """Prefiere la segunda referencia candidata (la voz preparada con "Frase 1")."""

    def __init__(self, cloner) -> None:
        self.cloner = cloner

    def similarity(self, reference, clips):
        return [0.9 if self.cloner.voices and "Frase 1" in self.cloner.voices[-1] else 0.2
                for _ in clips]


def test_verify_retries_only_failing_cues_and_probe_picks_best_reference(tmp_path: Path):
    source = tmp_path / "v.mp4"
    source.write_bytes(b"x")
    segments = tuple(Segment(i * 6.0, i * 6.0 + 4.0, f"Frase {i}{'!' if i == 1 else ''}.")
                     for i in range(4))
    factory = FakeFactory()
    cloner = TaggingCloner()
    verifier = Verifier(cloner)
    original_build = factory.build
    factory.build = lambda request: replace(original_build(request), cloner=cloner,
                                            translator=_Identity(), verifier=verifier,
                                            speaker_encoder=Encoder(cloner))
    report = DubUseCase(no_transcripts, factory).execute(
        DubRequest(source, "es", source_language="es", reference_candidates=2,
                   verify_attempts=2), tmp_path / "w",
        transcript=Transcript(segments, "es", "fake", 30.0))
    assert [c.attempts for c in report.track] == [0, 1, 0, 0]  # solo el cue con '!'
    assert 1 in cloner.variants and 2 not in cloner.variants  # el primer reintento bastó
    assert "Frase 1" in report.reference.text  # la referencia elegida por el sondeo


class _Identity:
    def translate(self, texts, source, target, n_best=1):
        from subdub_titler.domain.dubbing import Candidate
        return [[Candidate(t, 0.0)] for t in texts]
