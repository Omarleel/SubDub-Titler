"""Panel de escucha a ciegas: las mismas frases de varias variantes de un doblaje, con la voz
original de cada ranura, para puntuar acento y parecido de voz.

La detección de idioma de Whisper es poco fiable como medida de acento cuando el destino es
español; el panel es la medida de referencia para adoptar cambios de voz. Los audios se guardan
con nombres aleatorios para no revelar la variante.
"""
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subdub_titler.application.dubbing import Workspace, load_report
from subdub_titler.domain.dubbing import DubbedCue, DubReport
from subdub_titler.domain.ports import AudioStore

SAMPLE_RATE = 24_000


@dataclass(frozen=True, slots=True)
class PanelClip:
    variant: str
    file: str  # relativo a la carpeta del panel


@dataclass(frozen=True, slots=True)
class PanelItem:
    index: int
    text: str
    original: str
    clips: tuple[PanelClip, ...]  # en orden aleatorio


def shared_cues(reports: Sequence[DubReport], limit: int, min_seconds: float,
                seed: int) -> list[DubbedCue]:
    """Muestra reproducible de frases presentes con el mismo texto en todas las variantes.
    Las muy cortas se descartan: en 1 s apenas se oye el acento."""
    others = [{c.index: c.target_text for c in r.track} for r in reports[1:]]
    pool = [c for c in reports[0].track if c.generated_seconds >= min_seconds
            and all(o.get(c.index) == c.target_text for o in others)]
    chosen = random.Random(seed).sample(pool, min(limit, len(pool)))
    return sorted(chosen, key=lambda c: c.index)


class ListeningPanel:
    def __init__(self, audio: AudioStore) -> None:
        self._audio = audio

    def build(self, variants: Mapping[str, Path], out: Path, limit: int = 20,
              min_seconds: float = 1.5, seed: int = 0) -> list[PanelItem]:
        """`variants`: nombre → carpeta de un doblaje (report.json + cues/). La voz original
        sale de la primera variante (todas son del mismo video)."""
        if not variants:
            raise ValueError("Indica al menos una variante")
        spaces = {name: Workspace(path) for name, path in variants.items()}
        reports = [load_report(space.report) for space in spaces.values()]
        cues = shared_cues(reports, limit, min_seconds, seed)
        if not cues:
            raise ValueError("Las variantes no comparten frases con el mismo texto")
        voice = next(iter(spaces.values())).stems / "voice.flac"
        rng = random.Random(seed + 1)
        used: set[str] = set()

        def name() -> str:
            while (token := f"{rng.getrandbits(48):012x}.wav") in used:
                pass
            used.add(token)
            return f"audio/{token}"

        items = []
        for cue in cues:
            original = name()
            self._audio.save(self._audio.load(voice, SAMPLE_RATE, cue.start,
                                              cue.start + cue.slot_seconds), out / original)
            clips = []
            for variant, space in spaces.items():
                file = name()
                self._audio.save(self._audio.load(space.cue(cue.index), SAMPLE_RATE), out / file)
                clips.append(PanelClip(variant, file))
            rng.shuffle(clips)
            items.append(PanelItem(cue.index, cue.target_text, original, tuple(clips)))
        return items
