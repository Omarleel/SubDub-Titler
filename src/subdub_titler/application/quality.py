"""Métricas reproducibles de texto. No confundir CER/WER con confianza ASR.

Cálculo exacto de sustituciones, eliminaciones e inserciones mediante distancia de
Levenshtein con dos filas. Memoria O(len(predicción)); tiempo O(N*M): evaluar
fragmentos/datasets controlados, no documentos gigantes sin segmentación.
"""
from dataclasses import asdict, dataclass
import re
import unicodedata


@dataclass(frozen=True, slots=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def rate(self) -> float | None:
        return (self.substitutions + self.deletions + self.insertions) / self.reference_units \
            if self.reference_units else None

    def to_dict(self) -> dict:
        return {**asdict(self), "rate": self.rate}


def normalize(text: str, language: str) -> str:
    """Política v1; preserva Kanji/Han/Kana, quita puntuación, casefold Latin."""
    if language not in ("en", "es", "ja", "zh"):
        raise ValueError("Idioma de evaluación inválido")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_counts(reference: list[str], hypothesis: list[str]) -> ErrorCounts:
    # Cada celda: (coste, sustituciones, eliminaciones, inserciones).
    # Elegir primero sustitución, luego eliminación/inserción en los empates.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_token in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_token in enumerate(hypothesis, 1):
            if ref_token == hyp_token:
                current.append(previous[j - 1])
                continue
            s = previous[j - 1]
            d = previous[j]
            ins = current[j - 1]
            # costo de edición, orden estable para desglosar errores
            candidates = (
                (s[0] + 1, s[1] + 1, s[2], s[3]),
                (d[0] + 1, d[1], d[2] + 1, d[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
            )
            current.append(min(candidates, key=lambda a: a[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return ErrorCounts(substitutions, deletions, insertions, len(reference))


def _unitize(text: str, language: str, metric: str) -> list[str]:
    if metric == "cer":
        # Espaciado chino/japonés libre: CER normalizado ignora espacios.
        return list(text.replace(" ", "") if language in ("ja", "zh") else text)
    if metric == "wer":
        return text.split()
    raise ValueError(f"Métrica inválida: {metric}")


def evaluate(reference: str, hypothesis: str, language: str) -> dict:
    if not reference.strip():
        raise ValueError("La referencia humana no puede estar vacía")
    ref_normalized = normalize(reference, language)
    hyp_normalized = normalize(hypothesis, language)
    output = {
        "language": language,
        "normalization": "nfkc-casefold-punctuation-v1",
        "raw": {}, "normalized": {},
        "primary_metric": "cer" if language in ("ja", "zh") else "wer",
    }
    for variant, (ref, hyp) in {
        "raw": (reference, hypothesis),
        "normalized": (ref_normalized, hyp_normalized),
    }.items():
        for metric in ("cer", "wer"):
            if metric == "wer" and language in ("ja", "zh"):
                # No comparar palabras sin tokenizador lingüístico explícito.
                output[variant][metric] = None
                continue
            counts = _edit_counts(_unitize(ref, language, metric),
                                  _unitize(hyp, language, metric))
            output[variant][metric] = counts.to_dict()
    return output
