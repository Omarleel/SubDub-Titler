"""Política de cues de subtítulo: corta por pausa, duración y longitud de texto.

Whisper devuelve segmentos de decodificación (hasta decenas de segundos y cruzando
silencios); un subtítulo legible necesita cues cortos alineados con la voz y conformes
a estándares de lectura profesional (Netflix, BBC, EBU-TT).
"""
import re
from dataclasses import dataclass, replace

from .models import Segment, Transcript, Word


@dataclass(frozen=True, slots=True)
class CuePolicy:
    max_duration: float = 6.0
    max_chars: int = 84  # dos líneas de 42 caracteres
    max_pause: float = 0.6
    min_duration: float = 0.05
    pad_end: float = 0.12  # margen de 120 ms tras la última palabra para evitar cortes
    merge_subsequent: bool = True
    merge_max_pause: float = 1.2
    # Estándares profesionales audiovisuales (Norma Netflix / BBC):
    max_cpl: int = 42
    max_lines: int = 2
    target_cps: float = 17.0
    min_subtitle_duration: float = 0.833  # 5/6 de segundo
    min_gap: float = 0.080  # 80 ms (~2 frames)
    smart_line_break: bool = False
    conform_cps: bool = False


DEFAULT_POLICY = CuePolicy()
NETFLIX_POLICY = CuePolicy(
    max_duration=7.0,
    max_chars=84,
    max_cpl=42,
    max_lines=2,
    target_cps=17.0,
    min_subtitle_duration=0.833,
    min_gap=0.080,
    smart_line_break=True,
    conform_cps=True,
)

TERMINAL_PUNCTUATION = ('.', '?', '!', '…', ':', ';')
CONNECTORS = {
    # Español
    "y", "e", "o", "u", "pero", "sino", "porque", "que", "cual", "quien",
    "donde", "cuando", "si", "entonces", "como", "para", "por", "con", "de",
    "en", "hacia", "hasta", "ni", "aunque", "mas",
    # Inglés
    "and", "or", "but", "so", "because", "that", "which", "who", "whom",
    "where", "when", "if", "then", "as", "than", "to", "for", "with",
    "at", "by", "from", "in", "on", "into", "onto", "nor", "yet",
}

ARTICLES = {
    # Español
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas",
    "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra", "nuestros", "nuestras",
    # Inglés
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
}

PREPOSITIONS = {
    # Español
    "de", "del", "al", "a", "en", "con", "por", "para", "sin", "sobre", "tras", "hacia",
    "hasta", "desde", "entre", "bajo", "ante", "según",
    # Inglés
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "into", "onto",
    "about", "through", "after", "before", "between", "under", "over",
}

CLITICS = {
    "me", "te", "se", "nos", "os", "le", "les", "lo", "la", "los", "las",
}

CONJUNCTIONS = {
    "pero", "sino", "aunque", "porque", "pues", "mas",
    "but", "because", "although", "since", "while", "whereas",
}


def _is_continuation(word_text: str) -> bool:
    clean = word_text.strip()
    if not clean:
        return False
    if clean[0].islower():
        return True
    return clean.lower().rstrip(".,?!;:…") in CONNECTORS


def balance_line_breaks(text: str, max_cpl: int = 42, max_lines: int = 2) -> str:
    """Divide un texto de subtítulo en hasta max_lines respetando el límite de caracteres
    por línea (max_cpl) y la cohesión sintáctica (Norma Netflix / BBC)."""
    clean_text = " ".join(text.strip().split())
    if len(clean_text) <= max_cpl or max_lines < 2:
        return clean_text

    words = clean_text.split()
    if len(words) <= 1:
        return clean_text

    best_split = None
    min_penalty = float("inf")

    for k in range(1, len(words)):
        l1 = " ".join(words[:k])
        l2 = " ".join(words[k:])

        penalty = 0.0
        # 1. Penalización dura por exceder límite
        if len(l1) > max_cpl:
            penalty += (len(l1) - max_cpl) * 50.0
        if len(l2) > max_cpl:
            penalty += (len(l2) - max_cpl) * 50.0

        # 2. Balance visual de longitud
        diff = abs(len(l1) - len(l2))
        penalty += diff * 1.2

        if len(l1) <= len(l2):
            penalty -= 2.0  # Ligero bono por estética piramidal/regular

        if len(l1) < 8:
            penalty += 20.0
        if len(l2) < 8:
            penalty += 25.0

        # 3. Penalizaciones sintácticas
        prev_word = words[k - 1]
        next_word = words[k]

        prev_clean = prev_word.lower().rstrip(".,?!;:…\"'»)")
        next_clean = next_word.lower().lstrip("¿¡\"'«(")

        if prev_word.endswith((",", ";", ":", "—", "-")):
            penalty -= 15.0
        elif prev_word.endswith((".", "?", "!")):
            penalty -= 20.0

        if prev_clean in ARTICLES:
            penalty += 35.0

        if prev_clean in PREPOSITIONS:
            penalty += 25.0

        if prev_clean in CLITICS:
            penalty += 30.0

        if next_clean in CONJUNCTIONS:
            penalty -= 8.0

        if len(next_clean) <= 1:
            penalty += 15.0

        if penalty < min_penalty:
            min_penalty = penalty
            best_split = (l1, l2)

    if best_split is not None:
        return f"{best_split[0]}\n{best_split[1]}"
    return clean_text


def conform_timing_netflix(
    segments: tuple[Segment, ...],
    policy: CuePolicy,
) -> tuple[Segment, ...]:
    """Ajusta los tiempos visuales de cada cue para cumplir los límites de velocidad de lectura
    (máx. target_cps, por defecto 17.0), duración mínima (0.833 s) y separación mínima entre
    subtítulos contiguos (min_gap, por defecto 80 ms), sin solapamientos."""
    if not segments:
        return segments

    adjusted: list[Segment] = []
    n = len(segments)

    for i, s in enumerate(segments):
        start = s.start
        end = s.end
        next_start = segments[i + 1].start if i + 1 < n else None

        clean_text = re.sub(r"\[.*?\]\s*", "", s.text).replace("\n", " ").strip()
        chars = len(clean_text) if clean_text else len(s.text.strip())

        # 1. Duración mínima requerida por norma (5/6 seg = 0.833s)
        current_dur = end - start
        target_dur = max(current_dur, policy.min_subtitle_duration)

        # 2. Extensión elástica por velocidad de lectura si CPS excede el límite
        if policy.target_cps > 0 and chars > 0:
            cps = chars / current_dur if current_dur > 0 else float("inf")
            if cps > policy.target_cps:
                needed_dur = chars / policy.target_cps
                target_dur = max(target_dur, needed_dur)

        target_dur = min(target_dur, policy.max_duration)

        # 3. Límite superior por espacio hasta el siguiente subtítulo respetando el min_gap
        if next_start is not None:
            max_end = next_start - policy.min_gap
            if max_end > start + policy.min_duration:
                end = min(start + target_dur, max_end)
            else:
                end = max(start + 0.05, next_start - 0.02)
        else:
            end = start + target_dur

        adjusted.append(replace(s, end=round(end, 3)))

    return tuple(adjusted)


def evaluate_audiovisual_compliance(
    transcript: Transcript,
    max_cps: float = 17.0,
    max_cpl: int = 42,
    min_duration: float = 0.833,
    min_gap: float = 0.080,
) -> dict:
    """Evalúa el cumplimiento de estándares audiovisuales profesionales (Norma Netflix/BBC)."""
    cues = transcript.segments
    if not cues:
        return {
            "total_cues": 0, "total_lines": 0, "mean_cps": 0.0, "max_cps": 0.0,
            "cps_compliance_rate": 100.0, "cpl_compliance_rate": 100.0,
            "lines_compliance_rate": 100.0, "min_duration_compliance_rate": 100.0,
            "gap_compliance_rate": 100.0, "mean_line_balance": 1.0, "is_compliant": True,
        }

    total_lines = 0
    cps_list = []
    compliant_cps = 0
    compliant_cpl = 0
    compliant_lines = 0
    compliant_dur = 0
    compliant_gaps = 0
    balances = []

    for i, c in enumerate(cues):
        dur = max(0.001, c.end - c.start)
        clean = re.sub(r"\[.*?\]\s*", "", c.text).strip()
        chars = len(clean.replace("\n", " "))
        cps = chars / dur
        cps_list.append(cps)
        if cps <= max_cps:
            compliant_cps += 1
        if dur >= min_duration - 0.05:  # tolerancia de 50ms
            compliant_dur += 1

        lines = [line.strip() for line in clean.split("\n") if line.strip()]
        total_lines += len(lines)
        if len(lines) <= 2:
            compliant_lines += 1
        for line_item in lines:
            if len(line_item) <= max_cpl:
                compliant_cpl += 1
        if len(lines) == 2:
            l1, l2 = len(lines[0]), len(lines[1])
            balance = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
            balances.append(balance)

        if i + 1 < len(cues):
            gap = cues[i + 1].start - c.end
            if gap >= min_gap - 0.01:
                compliant_gaps += 1

    total_cues = len(cues)
    gap_count = max(1, total_cues - 1)

    cps_rate = round((compliant_cps / total_cues) * 100, 2)
    cpl_rate = round((compliant_cpl / max(1, total_lines)) * 100, 2)
    lines_rate = round((compliant_lines / total_cues) * 100, 2)
    dur_rate = round((compliant_dur / total_cues) * 100, 2)
    gap_rate = round((compliant_gaps / gap_count) * 100, 2)
    mean_balance = round(sum(balances) / len(balances), 3) if balances else 1.0

    return {
        "total_cues": total_cues,
        "total_lines": total_lines,
        "mean_cps": round(sum(cps_list) / len(cps_list), 2),
        "max_cps": round(max(cps_list), 2),
        "cps_compliance_rate": cps_rate,
        "cpl_compliance_rate": cpl_rate,
        "lines_compliance_rate": lines_rate,
        "min_duration_compliance_rate": dur_rate,
        "gap_compliance_rate": gap_rate,
        "mean_line_balance": mean_balance,
        "is_compliant": (cps_rate >= 90.0 and cpl_rate >= 95.0 and lines_rate == 100.0),
    }


def _cue(
    words: list[Word],
    policy: CuePolicy,
    next_start: float | None = None,
    speaker: str | None = None,
    avg_logprob: float | None = None,
    no_speech_prob: float | None = None,
) -> Segment:
    text = "".join(w.text for w in words).strip()
    start = words[0].start
    end = words[-1].end + policy.pad_end
    if next_start is not None and end > next_start:
        end = next_start
    # Se conservan las palabras: el doblaje parte cues por cambio de hablante con ellas.
    return Segment(
        start, max(end, start + policy.min_duration), text, tuple(words),
        speaker=speaker, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob,
    )


def resegment(transcript: Transcript, policy: CuePolicy = DEFAULT_POLICY) -> Transcript:
    """Sin palabras con tiempos (motor sin soporte) devuelve el transcript intacto."""
    words = [w for s in transcript.segments for w in s.words if w.text.strip()]
    if not words:
        return transcript

    word_to_segment = {id(w): s for s in transcript.segments for w in s.words}

    raw_cues: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            text = "".join(w.text for w in current + [word]).strip()
            if (word.start - current[-1].end > policy.max_pause
                    or word.end - current[0].start > policy.max_duration
                    or len(text) > policy.max_chars):
                raw_cues.append(current)
                current = []
        current.append(word)
    if current:
        raw_cues.append(current)

    # Fusión sintáctica: une fragmentos partidos por pausas menores a merge_max_pause
    # si la cláusula gramatical no concluyó y cabe en las restricciones de lectura.
    merged: list[list[Word]] = []
    for cue_words in raw_cues:
        if not merged or not policy.merge_subsequent:
            merged.append(cue_words)
            continue
        prev = merged[-1]
        prev_text = "".join(w.text for w in prev).strip()
        curr_text = "".join(w.text for w in cue_words).strip()
        combined_text = f"{prev_text} {curr_text}".strip()
        combined_duration = cue_words[-1].end - prev[0].start
        pause = cue_words[0].start - prev[-1].end

        if (pause <= policy.merge_max_pause
                and not prev_text.endswith(TERMINAL_PUNCTUATION)
                and _is_continuation(cue_words[0].text)
                and len(combined_text) <= policy.max_chars
                and combined_duration <= policy.max_duration):
            merged[-1] = prev + cue_words
        else:
            merged.append(cue_words)

    cues: list[Segment] = []
    for i, cue_words in enumerate(merged):
        next_start = merged[i + 1][0].start if i + 1 < len(merged) else None
        orig_segs = [word_to_segment[id(w)] for w in cue_words if id(w) in word_to_segment]
        cue_speaker = None
        cue_logprob = None
        cue_nsp = None
        if orig_segs:
            speakers = [s.speaker for s in orig_segs if s.speaker is not None]
            if speakers:
                cue_speaker = max(set(speakers), key=speakers.count)
            logprobs = [s.avg_logprob for s in orig_segs if s.avg_logprob is not None]
            if logprobs:
                cue_logprob = sum(logprobs) / len(logprobs)
            nsps = [s.no_speech_prob for s in orig_segs if s.no_speech_prob is not None]
            if nsps:
                cue_nsp = sum(nsps) / len(nsps)

        seg = _cue(
            cue_words, policy, next_start=next_start,
            speaker=cue_speaker, avg_logprob=cue_logprob, no_speech_prob=cue_nsp,
        )
        if policy.smart_line_break:
            seg = replace(seg, text=balance_line_breaks(seg.text, max_cpl=policy.max_cpl,
                                                        max_lines=policy.max_lines))
        cues.append(seg)

    result_segments = tuple(cues)
    if policy.conform_cps:
        result_segments = conform_timing_netflix(result_segments, policy)

    return replace(transcript, segments=result_segments)
