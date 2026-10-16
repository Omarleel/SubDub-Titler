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
