from subdub_titler.domain.models import Segment, Transcript


def timestamp(seconds: float, separator: str = ",") -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milli = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milli:03d}"


def _format_text(s: Segment) -> str:
    lines = [" ".join(line.split()) for line in s.text.strip().split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    if not s.speaker:
        return "\n".join(lines)
    label = s.speaker
    if label.startswith("S") and label[1:].isdigit():
        label = f"Hablante {int(label[1:]) + 1}"
    lines[0] = f"[{label}] {lines[0]}"
    return "\n".join(lines)


class SrtRenderer:
    def render(self, transcript: Transcript) -> str:
        return "\n".join(
            f"{idx}\n{timestamp(s.start)} --> {timestamp(s.end)}\n{_format_text(s)}\n"
            for idx, s in enumerate(transcript.segments, start=1)
            if s.text.strip()
        )


class VttRenderer:
    def render(self, transcript: Transcript) -> str:
        return "WEBVTT\n\n" + "\n".join(
            f"{timestamp(s.start, '.')} --> {timestamp(s.end, '.')}\n{_format_text(s)}\n"
            for s in transcript.segments if s.text.strip()
        )
