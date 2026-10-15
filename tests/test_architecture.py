from pathlib import Path

from subdub_titler.application.transcribe import TranscribeUseCase
from subdub_titler.domain.models import ModelDescriptor, Segment, Transcript, TranscriptionRequest
from subdub_titler.infrastructure.registry import ModelRegistry


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
    assert [e.name for e in result.events if e.depth == 0] == [
        "registry.engine_for", "asr.transcribe_complete", "subtitles.resegment"]


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


PACKAGE = Path(__file__).resolve().parents[1] / 'src' / 'subdub_titler'
# Bibliotecas de motores/IO: solo pueden aparecer en adaptadores de infraestructura.
HEAVY = {'faster_whisper', 'ctranslate2', 'torch', 'torchaudio', 'omnivoice', 'demucs',
         'numpy', 'soundfile', 'soxr', 'transformers', 'huggingface_hub'}


def _imports(module_file: Path) -> set[str]:
    import ast
    parsed = ast.parse(module_file.read_text(encoding='utf-8'))
    names: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module or '')
    return {name.split('.')[0] for name in names} | {name for name in names}


def test_domain_has_no_infrastructure_framework_imports():
    """Regla básica de dependencias de Clean Architecture: dominio solo stdlib."""
    forbidden = HEAVY | {'fastapi', 'sqlalchemy', 'typer', 'pydantic', 'rich'}
    for module_file in (PACKAGE / 'domain').glob('*.py'):
        imports = _imports(module_file)
        assert not forbidden & imports, module_file
        assert not any(name.startswith('subdub_titler.') and
                       not name.startswith('subdub_titler.domain') for name in imports), \
            module_file


def test_dubbing_use_cases_depend_on_ports_not_engines():
    """Los casos de uso de doblaje no conocen motores, IO de audio ni la API."""
    for name in ('dubbing.py', 'dub_quality.py', 'dub_bench.py', 'dub_corpus.py'):
        imports = _imports(PACKAGE / 'application' / name)
        assert not HEAVY & imports, name
        assert not any(i.startswith(('subdub_titler.infrastructure',
                                     'subdub_titler.interfaces')) for i in imports), name


def test_engine_libraries_are_imported_lazily():
    """Importar la API o la CLI no debe cargar torch/OmniVoice/Demucs (arranque rápido)."""
    import subprocess
    import sys
    code = ("import sys, subdub_titler.interfaces.api, subdub_titler.interfaces.cli;"
            "print(sorted(m for m in ('torch','omnivoice','demucs','ctranslate2') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         check=True, env={**__import__('os').environ,
                                          'PYTHONPATH': str(PACKAGE.parent)}).stdout
    assert out.strip() == '[]'


def test_omnivoice_schedule_unmasks_every_token_once():
    from subdub_titler.infrastructure.omnivoice_fast import _schedules
    steps = [i / 16 for i in range(17)]
    for length in (1, 7, 75):
        schedule = _schedules(steps, [length], 8, 16)[0]
        assert sum(schedule) == length * 8 and len(schedule) == 16


def test_nllb_sentence_split():
    from subdub_titler.infrastructure.nllb import split_sentences
    assert split_sentences("What else? What else?") == ["What else?", "What else?"]
    assert split_sentences("Hola") == ["Hola"]
    assert split_sentences("今日は。元気？") == ["今日は。", "元気？"]
