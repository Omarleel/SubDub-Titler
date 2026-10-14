from subdub_titler.domain.models import Transcript


def timestamp(seconds: float, separator: str = ",") -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milli = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milli:03d}"


def _clean(text: str) -> str:
    return " ".join(text.strip().split())


class SrtRenderer:
    def render(self, transcript: Transcript) -> str:
        return "\n".join(
            f"{idx}\n{timestamp(s.start)} --> {timestamp(s.end)}\n{_clean(s.text)}\n"
            for idx, s in enumerate(transcript.segments, start=1)
            if s.text.strip()
        )


class VttRenderer:
    def render(self, transcript: Transcript) -> str:
        return "WEBVTT\n\n" + "\n".join(
            f"{timestamp(s.start, '.')} --> {timestamp(s.end, '.')}\n{_clean(s.text)}\n"
            for s in transcript.segments if s.text.strip()
        )
