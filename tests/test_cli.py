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


def test_transcribe_options_registered():
    import typer
    cmd = typer.main.get_command(app).commands["transcribe"]
    opts = {opt for p in cmd.params if hasattr(p, "opts") for opt in p.opts}
    sec = {opt for p in cmd.params if hasattr(p, "secondary_opts") for opt in p.secondary_opts}
    all_flags = opts | sec
    assert "--align" in all_flags
    assert "--no-align" in all_flags
    assert "--diarize" in all_flags
    assert "--no-diarize" in all_flags
    assert "--netflix" in all_flags
    assert "--no-netflix" in all_flags
    assert "--suppress-hallucinations" in all_flags
    assert "--no-suppress-hallucinations" in all_flags
    assert "--two-pass" in all_flags
    assert "--no-two-pass" in all_flags


def test_transcribe_help():
    result = CliRunner().invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.stdout.lower()

