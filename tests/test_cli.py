from typer.testing import CliRunner

from subdub_titler.interfaces.cli import app


def test_help_and_models():
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "large-v3" in result.stdout


def test_missing_media_fails_cleanly():
    result = CliRunner().invoke(app, ["transcribe", "/does-not-exist.wav"])
    assert result.exit_code == 1
    assert "Archivo no encontrado" in result.output


def test_transcribe_help_contains_new_flags():
    result = CliRunner(env={"COLUMNS": "160"}).invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--suppress-hallucinations" in result.stdout
    assert "--two-pass" in result.stdout
    assert "--netflix" in result.stdout
    assert "--diarize" in result.stdout
    assert "--align" in result.stdout

