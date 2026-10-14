import pytest
from subdub_titler.application.quality import evaluate


def test_wer_one_substitution():
    result = evaluate("I visit Japan", "I visit China", "en")
    assert result["normalized"]["wer"]["substitutions"] == 1
    assert result["normalized"]["wer"]["rate"] == pytest.approx(1 / 3)


def test_cjk_uses_cer_not_whitespace_wer():
    result = evaluate("私は東京に行きます。", "私は京都に行きます", "ja")
    assert result["primary_metric"] == "cer"
    assert result["normalized"]["wer"] is None
    assert result["normalized"]["cer"]["rate"] > 0


def test_normalization_punctuation_case():
    result = evaluate("Hello, WORLD!", "hello world", "en")
    assert result["normalized"]["wer"]["rate"] == 0.0
    assert result["raw"]["wer"]["rate"] > 0


def test_empty_reference_is_invalid():
    with pytest.raises(ValueError):
        evaluate("", "hello", "en")


def test_deletion_and_insertion():
    missing = evaluate("we are here", "we here", "en")
    added = evaluate("we are here", "we are really here", "en")
    assert missing["normalized"]["wer"]["deletions"] == 1
    assert added["normalized"]["wer"]["insertions"] == 1
