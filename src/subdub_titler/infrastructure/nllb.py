"""Traducción NLLB-200 sobre CTranslate2: mismo runtime que el ASR, int8 en GPU y n-best.

Pesos de Meta con licencia CC-BY-NC. La primera vez convierte el checkpoint de Hugging Face
(necesita transformers + torch del extra `dub`) y lo guarda en caché; después solo usa CT2.
"""
import re
import shutil
import threading
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

from subdub_titler.domain.dubbing import Candidate
from subdub_titler.infrastructure.gpu_runtime import ensure_cuda_libraries
from subdub_titler.infrastructure.profiling import active_profiler

NLLB_CODES = {"en": "eng_Latn", "es": "spa_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans"}
NLLB_MODELS = {
    "nllb-600m": "facebook/nllb-200-distilled-600M",
    "nllb-1.3b": "facebook/nllb-200-distilled-1.3B",
}


_SENTENCE = re.compile(r"(?<=[.!?…])\s+|(?<=[。？！])")


def split_sentences(text: str) -> list[str]:
    """NLLB traduce frase a frase: con varias en un mismo cue tiende a omitir alguna."""
    parts = [p.strip() for p in _SENTENCE.split(text.strip())]
    return [p for p in parts if p] or [text]


_CJK = r"぀-ヿ㐀-䶿一-鿿"
_FULL_WIDTH = {"zh": {",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；"},
               "ja": {",": "、", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；"}}


def cjk_punctuation(text: str, language: str) -> str:
    """NLLB emite puntuación ASCII en chino/japonés; los subtítulos usan la de ancho completo.
    Solo se cambia la que sigue a un carácter CJK (no toca números como 3.5 ni siglas)."""
    table = _FULL_WIDTH.get(language)
    if not table:
        return text
    text = re.sub(rf"(?<=[{_CJK}”」』）\"])\s*([,.?!:;])", lambda m: table[m.group(1)], text)
    return re.sub(rf"(?<=[{_CJK}，、。？！：；])\s+(?=[{_CJK}])", "", text)


def default_cache() -> Path:
    return Path.home() / ".cache" / "subdub-titler" / "ct2"


class NllbTranslator:
    def __init__(self, model_name: str, device: str = "cuda",
                 compute_type: str = "int8_float16", cache_dir: Path | None = None,
                 beam_size: int = 4, max_batch_size: int = 32,
                 length_penalty: float = 1.0) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type if device == "cuda" else "int8"
        self.cache_dir = cache_dir or default_cache()
        self.beam_size = beam_size
        self.length_penalty = length_penalty  # <1 favorece hipótesis más cortas
        self.max_batch_size = max_batch_size
        self._translator = None
        self._tokenizer = None
        self._lock = threading.Lock()

    @property
    def converted_dir(self) -> Path:
        quantization = self.compute_type.split("_")[0]  # int8 / float16
        return self.cache_dir / f"{self.model_name.replace('/', '--')}-{quantization}"

    def _convert(self) -> None:
        from ctranslate2.converters import TransformersConverter
        target = self.converted_dir
        partial = target.with_name(target.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        TransformersConverter(self.model_name).convert(
            str(partial), quantization=self.compute_type.split("_")[0], force=True)
        partial.rename(target)

    def release(self) -> None:
        """Descarga los pesos de la GPU; la siguiente traducción los recarga (~0.5 s)."""
        with self._lock:
            if self._translator is not None and self._translator.model_is_loaded:
                self._translator.unload_model()

    def _load(self):
        with self._lock:
            if self._translator is not None and not self._translator.model_is_loaded:
                self._translator.load_model()
            if self._translator is None:
                ensure_cuda_libraries()
                import ctranslate2
                from transformers import AutoTokenizer
                profiler = active_profiler()
                with profiler.measure("dub.translate.load") if profiler else nullcontext():
                    if not (self.converted_dir / "model.bin").is_file():
                        self._convert()
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self._translator = ctranslate2.Translator(
                        str(self.converted_dir), device=self.device,
                        compute_type=self.compute_type)
            return self._translator, self._tokenizer

    def translate(self, texts: Sequence[str], source: str, target: str,
                  n_best: int = 1) -> list[list[Candidate]]:
        if source not in NLLB_CODES or target not in NLLB_CODES:
            raise ValueError(f"Par de idiomas no soportado por NLLB: {source}->{target}")
        sentences = [split_sentences(t) for t in texts]
        flat = self._translate_sentences([p for parts in sentences for p in parts], source,
                                         target, n_best)
        joiner = "" if target in ("ja", "zh") else " "
        output, position = [], 0
        for original, parts in zip(texts, sentences, strict=True):
            options = flat[position:position + len(parts)]
            position += len(parts)
            # Candidato k = k-ésima alternativa de cada frase (o la mejor si no hay tantas).
            seen: dict[str, Candidate] = {}
            for k in range(max(len(o) for o in options)):
                chosen = [o[min(k, len(o) - 1)] for o in options]
                text = cjk_punctuation(joiner.join(c.text for c in chosen).strip(), target)
                if text and text not in seen:
                    seen[text] = Candidate(text, round(sum(c.score for c in chosen) / len(chosen),
                                                       4))
            # Sin traducción utilizable se conserva el original antes que sintetizar vacío.
            output.append(list(seen.values()) or [Candidate(original, -1e9)])
        return output

    def _translate_sentences(self, texts: Sequence[str], source: str, target: str,
                             n_best: int) -> list[list[Candidate]]:
        translator, tokenizer = self._load()
        tokenizer.src_lang = NLLB_CODES[source]
        prefix = NLLB_CODES[target]
        tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(t)) for t in texts]
        results = translator.translate_batch(
            tokens, target_prefix=[[prefix]] * len(tokens), beam_size=max(self.beam_size, n_best),
            num_hypotheses=n_best, return_scores=True, max_batch_size=self.max_batch_size,
            # Se traduce frase a frase: repetir n-gramas dentro de una frase es alucinación.
            no_repeat_ngram_size=4,
            length_penalty=self.length_penalty,
        )
        output = []
        for source_text, result in zip(texts, results, strict=True):
            sentences = len(split_sentences(source_text))
            seen: dict[str, Candidate] = {}
            invented: list[Candidate] = []
            for hypothesis, score in zip(result.hypotheses, result.scores, strict=True):
                ids = tokenizer.convert_tokens_to_ids(hypothesis[1:])  # sin la etiqueta destino
                text = tokenizer.decode(ids, skip_special_tokens=True).strip().lstrip("-– ")
                if not text or text in seen:
                    continue
                # Más frases que el original = alucinación típica con textos cortos
                # ("I don't know." -> "- No lo sé. - No lo sé.").
                target = invented if len(split_sentences(text)) > sentences else seen
                if target is seen:
                    seen[text] = Candidate(text, float(score))
                else:  # se conserva solo la primera frase como último recurso
                    invented.append(Candidate(split_sentences(text)[0], float(score)))
            output.append(list(seen.values()) or invented[:1] or [Candidate("", -1e9)])
        return output
