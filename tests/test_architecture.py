from pathlib import Path
from subdub_titler.domain.models import ModelDescriptor, Segment, Transcript, TranscriptionRequest
from subdub_titler.infrastructure.registry import ModelRegistry
from subdub_titler.application.transcribe import TranscribeUseCase


class DummyEngine:
    def transcribe(self, request):
        return Transcript((Segment(0, 1, "Prueba"),), "es", request.model_id, 1)


def test_registry_is_open_for_extension(tmp_path: Path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"placeholder")
    registry = ModelRegistry()
    registry.register(ModelDescriptor("dummy", "fake"), lambda request: DummyEngine())
    result = TranscribeUseCase(registry).execute(
        TranscriptionRequest(audio, "dummy", device="cpu", compute_type="int8")
    )
    assert result.transcript.text == "Prueba"
    assert result.events[-1].name == "asr.transcribe_complete"


def test_cache_separates_model_device_and_precision(tmp_path: Path):
    registry = ModelRegistry()
    calls = []

    def factory(request):
        calls.append(request.compute_type)
        return DummyEngine()

    registry.register(ModelDescriptor("dummy", "fake"), factory)
    file = tmp_path / "x.wav"
    file.write_bytes(b"a")
    for compute in ("int8", "int8", "float32"):
        registry.engine_for(TranscriptionRequest(file, "dummy", "es", "cpu", compute))
    assert calls == ["int8", "float32"]


def test_domain_has_no_infrastructure_framework_imports():
    """Regla básica de dependencias de Clean Architecture."""
    import ast
    from pathlib import Path
    domain = Path(__file__).resolve().parents[1] / 'src' / 'subdub_titler' / 'domain'
    forbidden = {'fastapi', 'faster_whisper', 'ctranslate2', 'sqlalchemy', 'torch', 'typer'}
    for module_file in domain.glob('*.py'):
        parsed = ast.parse(module_file.read_text(encoding='utf-8'))
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            assert not forbidden.intersection(name.split('.')[0] for name in names), module_file
