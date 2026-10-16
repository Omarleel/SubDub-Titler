import pytest

from subdub_titler.domain.models import Segment, Transcript
from subdub_titler.infrastructure.subtitles import SrtRenderer, VttRenderer, timestamp


def test_timestamps_round_and_no_float_drift():
    assert timestamp(3661.234) == "01:01:01,234"
    assert timestamp(1.5, ".") == "00:00:01.500"


def test_render_srt_and_vtt():
    transcript = Transcript((Segment(0.0, 1.2, "Hola mundo"),), "es", "small")
    assert "00:00:00,000 --> 00:00:01,200" in SrtRenderer().render(transcript)
    assert VttRenderer().render(transcript).startswith("WEBVTT\n\n")


def test_invalid_segment():
    with pytest.raises(ValueError):
        Segment(2, 1, "Error")
