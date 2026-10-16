"""Alineador forzado a nivel de fonema/carácter basado en CTC con Meta MMS-FA (Wav2Vec2).

Proporciona precisión milimétrica en los timestamps de inicio y fin de palabras (y por
consecuencia en los subtítulos resegmentados), corrigiendo desviaciones acústicas o cortes
prematuros que los modelos basados en atención cruzada como Whisper suelen presentar.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import replace
from pathlib import Path

from subdub_titler.domain.models import Segment, Transcript, Word
from subdub_titler.domain.ports import ForcedAligner, Releasable
from subdub_titler.infrastructure.audio import decode

logger = logging.getLogger(__name__)

# Mapeo explícito para caracteres latinos con diacríticos hacia el alfabeto base de MMS-FA
_DIACRITICS_MAP: dict[str, str] = {
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
    "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
    "â": "a", "ê": "e", "î": "i", "ô": "o", "û": "u",
    "ä": "a", "ë": "e", "ï": "i", "ö": "o",
    "ç": "c", "ß": "ss",
}


def clean_word_for_ctc(word: str) -> str:
    """Normaliza una palabra a minúsculas y caracteres admitidos en el alfabeto MMS-FA (a-z, ')."""
    w = word.lower().strip()
    for accented, plain in _DIACRITICS_MAP.items():
        w = w.replace(accented, plain)
    # Descomposición NFKD como respaldo para caracteres menos comunes
    normalized = unicodedata.normalize("NFKD", w)
    filtered = "".join(c for c in normalized if not unicodedata.combining(c))
    return re.sub(r"[^a-z']", "", filtered)


class MmsForcedAligner(ForcedAligner, Releasable):
    """Adaptador de alineación forzada CTC con MMS-FA (Wav2Vec2 300M, 16 kHz)."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._model = None
        self._dictionary: dict[str, int] | None = None
        self._blank_id: int = 0
        self._star_id: int = 1

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchaudio

        bundle = torchaudio.pipelines.MMS_FA
        self._dictionary = bundle.get_dict()
        self._blank_id = self._dictionary["-"]
        self._star_id = self._dictionary["*"]
        model = bundle.get_model()
        use_cuda = torch.cuda.is_available() and self.device == "cuda"
        target_device = torch.device(self.device if use_cuda else "cpu")
        self._model = model.to(target_device).eval()

    def release(self) -> None:
        """Libera los pesos del modelo de la GPU para mitigar presión de VRAM."""
        self._model = None
        self._dictionary = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def align(self, audio_source: Path, transcript: Transcript) -> Transcript:
        if not transcript.segments:
            return transcript

        self._ensure_model()
        assert self._model is not None
        assert self._dictionary is not None

        import torch
        from torchaudio.functional import forced_align, merge_tokens

        # Decodificación de audio uniforme mono 16 kHz
        try:
            pcm = decode(audio_source)
        except Exception as exc:
            logger.warning(
                "No se pudo decodificar audio %s para alineación forzada (%s); "
                "conservando transcripción original",
                audio_source, exc,
            )
            return transcript

        if len(pcm) == 0:
            return transcript

        model_device = next(self._model.parameters()).device
        waveform = torch.from_numpy(pcm).float()
        total_audio_duration = len(pcm) / 16000.0

        aligned_segments: list[Segment] = []

        for segment in transcript.segments:
            if not segment.words:
                aligned_segments.append(segment)
                continue

            # 1. Construir secuencia de tokens CTC para las palabras del segmento
            tokens: list[int] = []
            word_token_ranges: list[tuple[int, int, int]] = []  # (w_idx, start_tok, end_tok)

            for w_idx, word in enumerate(segment.words):
                clean = clean_word_for_ctc(word.text)
                if not clean:
                    continue
                word_toks = [self._dictionary[c] for c in clean if c in self._dictionary]
                if not word_toks:
                    continue
                if tokens:
                    tokens.append(self._star_id)
                tok_start = len(tokens)
                tokens.extend(word_toks)
                tok_end = len(tokens)
                word_token_ranges.append((w_idx, tok_start, tok_end))

            if not tokens or not word_token_ranges:
                aligned_segments.append(segment)
                continue

            # 2. Extraer fragmento de audio correspondiente con margen de seguridad
            pad_before = 0.05
            pad_after = 0.15
            seg_start_sec = max(0.0, segment.start - pad_before)
            seg_end_sec = min(total_audio_duration, segment.end + pad_after)
            start_sample = int(seg_start_sec * 16000)
            end_sample = int(seg_end_sec * 16000)

            chunk = waveform[start_sample:end_sample].unsqueeze(0)
            if chunk.shape[1] < 320:  # Menos de 20 ms, no procesable
                aligned_segments.append(segment)
                continue

            # 3. Emisiones acústicas con Wav2Vec2 MMS-FA
            try:
                with torch.no_grad():
                    emissions, _ = self._model(chunk.to(model_device))
                    log_probs = torch.log_softmax(emissions, dim=-1)

                num_frames = log_probs.shape[1]
                if num_frames < len(tokens):
                    logger.debug(
                        "Segmento '%s' tiene menos frames (%d) que tokens (%d); "
                        "conservando tiempos originales",
                        segment.text, num_frames, len(tokens),
                    )
                    aligned_segments.append(segment)
                    continue

                targets = torch.tensor([tokens], dtype=torch.int32, device=model_device)
                aligned_tokens, scores = forced_align(log_probs, targets, blank=self._blank_id)
                spans = merge_tokens(aligned_tokens[0], scores[0], blank=self._blank_id)

                if len(spans) != len(tokens):
                    aligned_segments.append(segment)
                    continue

                chunk_duration = (end_sample - start_sample) / 16000.0
                frame_duration = chunk_duration / num_frames

                # 4. Proyectar los spans a cada palabra
                new_words: list[Word] = list(segment.words)
                last_end = seg_start_sec

                for w_idx, tok_s, tok_e in word_token_ranges:
                    w_spans = spans[tok_s:tok_e]
                    w_start = seg_start_sec + (w_spans[0].start * frame_duration)
                    w_end = seg_start_sec + (w_spans[-1].end * frame_duration)

                    # Garantizar monotonicidad y duración mínima no nula (20 ms)
                    w_start = max(last_end, w_start)
                    w_end = max(w_start + 0.020, w_end)

                    orig = segment.words[w_idx]
                    new_words[w_idx] = Word(
                        start=round(w_start, 3), end=round(w_end, 3), text=orig.text,
                    )
                    last_end = w_end

                # Si alguna palabra no tenía caracteres fonéticos (ej. sólo puntuación), interpolar
                for i, w in enumerate(new_words):
                    if i > 0 and w.start < new_words[i - 1].end:
                        adj_start = new_words[i - 1].end
                        adj_end = max(adj_start + 0.020, w.end)
                        new_words[i] = Word(
                            start=round(adj_start, 3), end=round(adj_end, 3), text=w.text,
                        )

                new_seg_start = new_words[0].start
                new_seg_end = new_words[-1].end
                aligned_segments.append(Segment(
                    start=round(new_seg_start, 3),
                    end=round(new_seg_end, 3),
                    text=segment.text,
                    words=tuple(new_words),
                ))

            except Exception as exc:
                logger.warning(
                    "Alineación forzada falló en segmento '%s': %s; conservando tiempos originales",
                    segment.text, exc,
                )
                aligned_segments.append(segment)

        return replace(transcript, segments=tuple(aligned_segments))


class LazyMmsForcedAligner(ForcedAligner, Releasable):
    """Proxy diferido para MmsForcedAligner: no importa PyTorch/TorchAudio hasta que se invoque."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._aligner: MmsForcedAligner | None = None

    def align(self, audio_source: Path, transcript: Transcript) -> Transcript:
        if self._aligner is None:
            self._aligner = MmsForcedAligner(device=self.device)
        return self._aligner.align(audio_source, transcript)

    def release(self) -> None:
        if self._aligner is not None:
            self._aligner.release()
            self._aligner = None

