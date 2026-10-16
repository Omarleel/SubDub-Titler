"""Gestión de proxies y Analogía de Acento (AAG) para OmniVoice.

Un proxy AAG consiste en un par de referencias acústicas (src_audio_tokens, tgt_audio_tokens)
con la misma identidad de voz hablando el idioma de origen (inglés) y el de destino (español).
Permite estimar la dirección del sesgo de acento en el espacio de log-probabilidades
(Cho & Lee, 2026: Accent Analogy Guidance).
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AccentProxyPair:
    id: str
    gender: str  # "female" | "male" | "neutral"
    src_lang: str
    src_text: str
    src_audio_tokens: object  # Tensor 2D (C, T)
    tgt_lang: str
    tgt_text: str
    tgt_audio_tokens: object  # Tensor 2D (C, T)
    src_rms: float = 0.1
    tgt_rms: float = 0.1


@dataclass(frozen=True, slots=True)
class AccentProxySet:
    version: str
    source_language: str
    target_language: str
    pairs: tuple[AccentProxyPair, ...]

    def for_gender(self, gender: str | None = None) -> tuple[AccentProxyPair, ...]:
        if not gender or gender in ("auto", "none"):
            return self.pairs
        matching = tuple(p for p in self.pairs if p.gender.lower() == gender.lower())
        return matching or self.pairs

    def save(self, directory: Path) -> None:
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        manifest_pairs = []
        for pair in self.pairs:
            src_file = f"{pair.id}_src.pt"
            tgt_file = f"{pair.id}_tgt.pt"
            src_tensor = (pair.src_audio_tokens.cpu()
                          if hasattr(pair.src_audio_tokens, "cpu")
                          else pair.src_audio_tokens)
            tgt_tensor = (pair.tgt_audio_tokens.cpu()
                          if hasattr(pair.tgt_audio_tokens, "cpu")
                          else pair.tgt_audio_tokens)
            torch.save(src_tensor, directory / src_file)
            torch.save(tgt_tensor, directory / tgt_file)
            manifest_pairs.append({
                "id": pair.id,
                "gender": pair.gender,
                "src_lang": pair.src_lang,
                "src_text": pair.src_text,
                "src_file": src_file,
                "src_rms": pair.src_rms,
                "tgt_lang": pair.tgt_lang,
                "tgt_text": pair.tgt_text,
                "tgt_file": tgt_file,
                "tgt_rms": pair.tgt_rms,
            })
        manifest = {
            "version": self.version,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "pairs": manifest_pairs,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> "AccentProxySet":
        import torch

        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest de proxies no encontrado en {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pairs = []
        for item in manifest.get("pairs", []):
            src_tokens = torch.load(directory / item["src_file"], map_location=device,
                                    weights_only=True)
            tgt_tokens = torch.load(directory / item["tgt_file"], map_location=device,
                                    weights_only=True)
            # Asegurar forma 2D (C, T)
            if src_tokens.dim() == 3:
                src_tokens = src_tokens.squeeze(0)
            if tgt_tokens.dim() == 3:
                tgt_tokens = tgt_tokens.squeeze(0)
            pairs.append(AccentProxyPair(
                id=item["id"],
                gender=item.get("gender", "neutral"),
                src_lang=item["src_lang"],
                src_text=item["src_text"],
                src_audio_tokens=src_tokens,
                tgt_lang=item["tgt_lang"],
                tgt_text=item["tgt_text"],
                tgt_audio_tokens=tgt_tokens,
                src_rms=float(item.get("src_rms", 0.1)),
                tgt_rms=float(item.get("tgt_rms", 0.1)),
            ))
        return cls(
            version=manifest.get("version", "1.0"),
            source_language=manifest.get("source_language", "en"),
            target_language=manifest.get("target_language", "es"),
            pairs=tuple(pairs),
        )


def build_synthetic_proxy_pair(
    model,
    pair_id: str,
    gender: str,
    tgt_text: str,
    src_text: str,
    tgt_lang: str = "es",
    src_lang: str = "en",
    instruct: str | None = None,
) -> AccentProxyPair:
    """Genera un par de proxies sintéticos (Cho & Lee, 2026):
    1. Sintetiza audio en el idioma de destino (español) con acento nativo vía instruct/auto.
    2. Clona esa misma voz para generar el audio correspondiente en inglés.
    3. Extrae los tokens acústicos para cada idioma.
    """
    import torch

    # 1. Síntesis nativa en idioma de destino (español)
    with torch.inference_mode():
        tgt_audios = model.generate(
            text=[tgt_text],
            language=tgt_lang,
            instruct=instruct or f"A clear, natural sounding {gender} speaker.",
        )
        tgt_wav = torch.as_tensor(tgt_audios[0], dtype=torch.float32)
        tgt_prompt = model.create_voice_clone_prompt(
            (tgt_wav, 24_000), ref_text=tgt_text, preprocess_prompt=True
        )

        # 2. Síntesis clonada en idioma de origen (inglés) manteniendo la identidad
        src_audios = model.generate(
            text=[src_text],
            language=src_lang,
            voice_clone_prompt=[tgt_prompt],
        )
        src_wav = torch.as_tensor(src_audios[0], dtype=torch.float32)
        src_prompt = model.create_voice_clone_prompt(
            (src_wav, 24_000), ref_text=src_text, preprocess_prompt=True
        )

    src_tokens = src_prompt.ref_audio_tokens.squeeze(0).cpu()
    tgt_tokens = tgt_prompt.ref_audio_tokens.squeeze(0).cpu()

    return AccentProxyPair(
        id=pair_id,
        gender=gender,
        src_lang=src_lang,
        src_text=src_text,
        src_audio_tokens=src_tokens,
        tgt_lang=tgt_lang,
        tgt_text=tgt_text,
        tgt_audio_tokens=tgt_tokens,
        src_rms=float(src_prompt.ref_rms),
        tgt_rms=float(tgt_prompt.ref_rms),
    )


def check_accent_premise(
    model,
    proxy_pair: AccentProxyPair,
    test_phrases: Sequence[str],
    evaluator=None,
) -> dict:
    """Prueba de premisa (Fase 2):
    Sintetiza frases en español clonando la versión inglesa del proxy y después la española.
    Evalúa si el acento (language_prob y/o escucha) cambia de forma medible.
    """
    import torch
    from omnivoice.models.omnivoice import VoiceClonePrompt

    device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruir VoiceClonePrompt para cada rama
    src_tokens = proxy_pair.src_audio_tokens
    if not isinstance(src_tokens, torch.Tensor):
        src_tokens = torch.as_tensor(src_tokens)
    if src_tokens.dim() == 2:
        src_tokens = src_tokens.unsqueeze(0)

    tgt_tokens = proxy_pair.tgt_audio_tokens
    if not isinstance(tgt_tokens, torch.Tensor):
        tgt_tokens = torch.as_tensor(tgt_tokens)
    if tgt_tokens.dim() == 2:
        tgt_tokens = tgt_tokens.unsqueeze(0)

    src_prompt = VoiceClonePrompt(
        ref_audio_tokens=src_tokens.to(device),
        ref_text=proxy_pair.src_text,
        ref_rms=proxy_pair.src_rms,
    )
    tgt_prompt = VoiceClonePrompt(
        ref_audio_tokens=tgt_tokens.to(device),
        ref_text=proxy_pair.tgt_text,
        ref_rms=proxy_pair.tgt_rms,
    )

    with torch.inference_mode():
        # Síntesis con proxy en inglés (origen de la clonación)
        src_outputs = model.generate(
            text=list(test_phrases),
            language="es",
            voice_clone_prompt=[src_prompt] * len(test_phrases),
        )
        # Síntesis con proxy en español (destino nativo)
        tgt_outputs = model.generate(
            text=list(test_phrases),
            language="es",
            voice_clone_prompt=[tgt_prompt] * len(test_phrases),
        )

    results = {
        "proxy_id": proxy_pair.id,
        "phrases": list(test_phrases),
        "source_cloned": len(src_outputs),
        "target_cloned": len(tgt_outputs),
    }

    if evaluator is not None and hasattr(evaluator, "language_id"):
        # Evaluar probabilidad de idioma si está disponible
        from subdub_titler.domain.dubbing import AudioClip
        src_probs = []
        tgt_probs = []
        for wav in src_outputs:
            clip = AudioClip(wav, 24_000)
            src_probs.append(evaluator.language_id.language_probability(clip, "es"))
        for wav in tgt_outputs:
            clip = AudioClip(wav, 24_000)
            tgt_probs.append(evaluator.language_id.language_probability(clip, "es"))
        results["src_proxy_language_prob_mean"] = (
            sum(src_probs) / len(src_probs) if src_probs else None
        )
        results["tgt_proxy_language_prob_mean"] = (
            sum(tgt_probs) / len(tgt_probs) if tgt_probs else None
        )

    return results
