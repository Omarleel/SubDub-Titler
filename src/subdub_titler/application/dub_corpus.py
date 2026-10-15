"""Corpus dorado: regresión reproducible de la síntesis y de la traducción.

- Síntesis: cada elemento fija voz de referencia, texto y duración; se mide CER ida y vuelta,
  parecido de voz, UTMOS, error de duración y RTF. Con la misma semilla y formas, dos
  ejecuciones dan la misma salida: una diferencia frente a la línea base es real.
- Traducción: chrF frente a traducciones humanas de referencia.
Formato JSONL (una línea por elemento); ver `docs/PLAN-DOBLAJE.md`.
"""
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from subdub_titler.application.dub_quality import DubEvaluator, corpus_cer
from subdub_titler.application.quality import chrf
from subdub_titler.domain.dubbing import DubbedCue, DubReport, FitAction
from subdub_titler.domain.ports import AudioStore, Translator, VoiceCloner

# No-regresión frente a la línea base (positivo = cuánto puede empeorar).
TOLERANCES = {"cer": 0.01, "similarity": 0.02, "mos": 0.10, "mean_abs_duration_error_ms": 20.0,
              "chrf": 1.0}
LOWER_IS_BETTER = {"cer", "mean_abs_duration_error_ms"}


@dataclass(frozen=True, slots=True)
class TtsItem:
    id: str
    ref_audio: str
    ref_text: str
    text: str
    language: str
    duration: float | None = None
    ref_start: float | None = None
    ref_end: float | None = None


@dataclass(frozen=True, slots=True)
class TranslationItem:
    source: str
    reference: str
    source_language: str
    target_language: str


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(rows: Sequence[object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def items_from_report(report: DubReport, voice_source: Path, language: str,
                      limit: int | None = None) -> list[TtsItem]:
    """Corpus a partir de un doblaje real: misma referencia, textos y duraciones pedidas."""
    ref = report.reference
    return [TtsItem(f"cue-{c.index:04d}", str(voice_source), ref.text, c.target_text, language,
                    c.target_seconds, ref.start, ref.end)
            for c in report.track[:limit]]


def regressions(new: dict, base: dict) -> list[str]:
    found = []
    for key, tolerance in TOLERANCES.items():
        if new.get(key) is None or base.get(key) is None:
            continue
        worse = new[key] - base[key] if key in LOWER_IS_BETTER else base[key] - new[key]
        if worse > tolerance:
            found.append(f"{key}: {base[key]:.4g} → {new[key]:.4g}")
    return found


class TtsCorpusEvaluation:
    def __init__(self, cloner: VoiceCloner, audio: AudioStore, evaluator: DubEvaluator,
                 batch: int = 8) -> None:
        self._cloner = cloner
        self._audio = audio
        self._evaluator = evaluator
        self._batch = batch

    def run(self, items: Sequence[TtsItem]) -> dict:
        if not items:
            raise ValueError("El corpus está vacío")
        voices: dict[tuple, tuple[object, object]] = {}
        clips, requested, synth_seconds = {}, {}, 0.0
        for i, item in enumerate(items):
            key = (item.ref_audio, item.ref_start, item.ref_end, item.ref_text)
            if key not in voices:
                reference = self._audio.load(Path(item.ref_audio), self._cloner.sample_rate,
                                             item.ref_start, item.ref_end)
                voices[key] = (reference, self._cloner.prepare_voice(reference, item.ref_text))
            requested[i] = item.duration or self._cloner.natural_seconds(
                [item.text], item.language, voices[key][1])[0]
        groups: dict[tuple, list[int]] = {}
        for i, item in enumerate(items):
            key = (item.ref_audio, item.ref_start, item.ref_end, item.ref_text, item.language)
            groups.setdefault(key, []).append(i)
        for key, indices in groups.items():
            voice = voices[key[:4]][1]
            ordered = sorted(indices, key=lambda i: requested[i])  # cubetas por duración
            for first in range(0, len(ordered), self._batch):
                batch = ordered[first:first + self._batch]
                start = time.perf_counter()
                generated = self._cloner.synthesize([items[i].text for i in batch],
                                                    [requested[i] for i in batch], key[4], voice)
                synth_seconds += time.perf_counter() - start
                clips.update(zip(batch, generated, strict=True))
        rows, qualities = [], []
        for key, indices in groups.items():
            reference = voices[key[:4]][0]
            track = [DubbedCue(i, 0.0, "", items[i].text, requested[i], requested[i],
                               requested[i], clips[i].duration, 1.0, FitAction.OK)
                     for i in indices]
            for cue, quality in zip(track, self._evaluator.evaluate_cues(
                    track, clips, reference, key[4]), strict=True):
                qualities.append(quality)
                rows.append({"id": items[cue.index].id, "text": cue.target_text,
                             "requested_seconds": round(cue.target_seconds, 3),
                             "generated_seconds": round(cue.generated_seconds, 3),
                             **{k: v for k, v in asdict(quality).items() if k != "index"}})
        errors = [abs(r["generated_seconds"] - r["requested_seconds"]) * 1000 for r in rows]
        audio = sum(r["generated_seconds"] for r in rows)
        weights = [r["generated_seconds"] for r in rows if r["similarity"] is not None]
        sims = [r["similarity"] for r in rows if r["similarity"] is not None]
        mos = [r["mos"] for r in rows if r["mos"] is not None]
        native = [r["language_prob"] for r in rows if r["language_prob"] is not None]
        summary = {
            "items": len(rows), "cer": corpus_cer(qualities),
            "similarity": round(sum(s * w for s, w in zip(sims, weights, strict=True))
                                / sum(weights), 4) if weights else None,
            "mos": round(sum(mos) / len(mos), 4) if mos else None,
            "language_prob": round(sum(native) / len(native), 4) if native else None,
            "mean_abs_duration_error_ms": round(sum(errors) / len(errors), 1),
            "max_abs_duration_error_ms": round(max(errors), 1),
            "synthesis_seconds": round(synth_seconds, 3), "audio_seconds": round(audio, 3),
            "tts_rtf": round(synth_seconds / audio, 4) if audio else None,
        }
        return {"summary": summary, "items": rows}


def evaluate_translations(translator: Translator, items: Sequence[TranslationItem]) -> dict:
    """chrF por par de idiomas y global, más la relación de longitud (afecta al doblaje)."""
    if not items:
        raise ValueError("El corpus está vacío")
    pairs: dict[tuple[str, str], list[int]] = {}
    for i, item in enumerate(items):
        pairs.setdefault((item.source_language, item.target_language), []).append(i)
    hypotheses: dict[int, str] = {}
    for (source, target), indices in pairs.items():
        output = translator.translate([items[i].source for i in indices], source, target, 1)
        hypotheses.update({i: candidates[0].text for i, candidates
                           in zip(indices, output, strict=True)})
    ordered = range(len(items))
    by_pair = {f"{s}->{t}": round(chrf([hypotheses[i] for i in idx],
                                       [items[i].reference for i in idx]), 2)
               for (s, t), idx in pairs.items()}
    length = sum(len(hypotheses[i]) for i in ordered) / max(1, sum(len(items[i].reference)
                                                                   for i in ordered))
    return {
        "summary": {"items": len(items),
                    "chrf": round(chrf([hypotheses[i] for i in ordered],
                                       [items[i].reference for i in ordered]), 2),
                    "chrf_by_pair": by_pair, "length_ratio_vs_reference": round(length, 3)},
        "items": [{"source": items[i].source, "reference": items[i].reference,
                   "hypothesis": hypotheses[i]} for i in ordered],
    }
