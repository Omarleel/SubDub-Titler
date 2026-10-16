"""Diarización por embeddings de hablante (WavLM-ECAPA, el mismo modelo que mide el parecido
de voz) y clustering aglomerativo por coseno. Sin pyannote ni token de Hugging Face.

Los clips cortos dan embeddings poco fiables: forman los grupos solo los de al menos
`min_seconds`; el resto se asigna al centroide más parecido. Los grupos con muy poca voz
(risas, ruido) se funden con el más cercano.
"""
from collections.abc import Sequence
from pathlib import Path

from subdub_titler.domain.ports import AudioStore, Releasable, SpeakerDiarizer
from subdub_titler.infrastructure.dub_eval import EVAL_RATE, WavLmSpeakerEncoder


class EmbeddingDiarizer:
    def __init__(self, encoder: WavLmSpeakerEncoder, audio: AudioStore,
                 threshold: float = 0.6, min_seconds: float = 1.0,
                 min_share: float = 0.05) -> None:
        self._encoder = encoder
        self._audio = audio
        self.threshold = threshold  # distancia coseno máxima dentro de un mismo hablante
        self.min_seconds = min_seconds
        self.min_share = min_share  # fracción mínima de voz para ser un hablante propio

    def release(self) -> None:
        self._encoder.release()

    def assign(self, source: Path, spans: Sequence[tuple[float, float]],
               max_speakers: int) -> list[str]:
        import numpy as np
        if not spans:
            return []
        clips = [self._audio.load(source, EVAL_RATE, start, end) for start, end in spans]
        vectors = np.asarray(self._encoder.embeddings(clips), dtype=np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        durations = np.array([end - start for start, end in spans])
        reliable = np.flatnonzero(durations >= self.min_seconds)
        if len(reliable) < 2 or max_speakers == 1:
            return ["S0"] * len(spans)
        labels = self._cluster(vectors[reliable], max_speakers)
        centroids = self._centroids(vectors[reliable], labels)
        groups = self._merge_small(centroids, labels, durations[reliable])
        centroids = np.stack([centroids[g].mean(axis=0) for g in groups])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        assigned = np.argmax(vectors @ centroids.T, axis=1)
        # Etiquetas estables: S0, S1… por orden de aparición en el medio.
        order: dict[int, str] = {}
        for group in assigned:
            order.setdefault(int(group), f"S{len(order)}")
        return [order[int(g)] for g in assigned]

    def _cluster(self, vectors, max_speakers: int):
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                        distance_threshold=self.threshold).fit(vectors)
        if model.n_clusters_ > max_speakers:
            model = AgglomerativeClustering(n_clusters=max_speakers, metric="cosine",
                                            linkage="average").fit(vectors)
        return model.labels_

    @staticmethod
    def _centroids(vectors, labels):
        import numpy as np
        centroids = np.stack([vectors[labels == k].mean(axis=0) for k in range(labels.max() + 1)])
        return centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    def _merge_small(self, centroids, labels, durations) -> list[list[int]]:
        """Grupos de clusters: los que no llegan a `min_share` de la voz se unen al más
        parecido de los grandes."""
        import numpy as np
        share = np.array([durations[labels == k].sum() for k in range(len(centroids))])
        share = share / share.sum()
        big = [k for k in range(len(centroids)) if share[k] >= self.min_share] or \
            [int(share.argmax())]
        groups = {k: [k] for k in big}
        for k in range(len(centroids)):
            if k not in groups:
                nearest = max(big, key=lambda b: float(centroids[k] @ centroids[b]))
                groups[nearest].append(k)
        return list(groups.values())


class LazySpeakerDiarizer(SpeakerDiarizer, Releasable):
    """Proxy diferido para EmbeddingDiarizer: no importa PyTorch ni modelos hasta su uso."""

    def __init__(self, device: str = "cuda", threshold: float = 0.6,
                 min_seconds: float = 1.0, min_share: float = 0.05) -> None:
        self.device = device
        self.threshold = threshold
        self.min_seconds = min_seconds
        self.min_share = min_share
        self._diarizer: EmbeddingDiarizer | None = None

    def assign(self, source: Path, spans: Sequence[tuple[float, float]],
               max_speakers: int) -> list[str]:
        if self._diarizer is None:
            from subdub_titler.infrastructure.media import FfmpegAudioStore
            self._diarizer = EmbeddingDiarizer(
                WavLmSpeakerEncoder(self.device),
                FfmpegAudioStore(),
                threshold=self.threshold,
                min_seconds=self.min_seconds,
                min_share=self.min_share,
            )
        return self._diarizer.assign(source, spans, max_speakers=max_speakers)

    def release(self) -> None:
        if self._diarizer is not None:
            self._diarizer.release()
            self._diarizer = None
