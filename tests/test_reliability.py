"""Regresiones de integridad, configuración y persistencia; sin modelos reales."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from dub_fakes import FakeFactory, no_transcripts, transcript

from subdub_titler.application.dubbing import DubUseCase, load_report, report_to_dict
from subdub_titler.application.jobs import dub_request
from subdub_titler.application.transcribe import TranscribeUseCase
from subdub_titler.domain.dubbing import DubRequest, cues_from_transcript, merge_fragments
from subdub_titler.domain.models import ModelDescriptor, Segment, Transcript, TranscriptionRequest
from subdub_titler.domain.ports import Stems
from subdub_titler.infrastructure.demucs_separator import DemucsSeparator
from subdub_titler.infrastructure.registry import ModelRegistry
from subdub_titler.infrastructure.storage import SqliteJobs


def test_speakers_survive_conversion_and_fragment_merge():
    data = Transcript((Segment(0, 0.5, "Hola", speaker="Ana"),
                       Segment(0.6, 1.1, "Adiós", speaker="Luis")), "es", "fake")
    cues = merge_fragments(cues_from_transcript(data))
    assert [c.speaker for c in cues] == ["Ana", "Luis"]
    assert len(cues) == 2


def test_job_honors_reference_candidates(tmp_path):
    source = tmp_path / "a.wav"
    source.write_bytes(b"a")
    request = dub_request({"file_path": str(source), "language": "en", "model_id": "small",
                           "device": "cpu", "params": {"target_language": "es",
                           "reference_candidates": 4, "verify_attempts": 2}})
    assert request.reference_candidates == 4
    assert request.verify_attempts == 2


@pytest.mark.parametrize("failure", ["mix", "report"])
def test_failed_edit_preserves_active_report_and_audio(tmp_path, monkeypatch, failure):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"video")
    request = DubRequest(source, "es")
    factory = FakeFactory()
    use_case = DubUseCase(no_transcripts, factory)
    root = tmp_path / "work"
    first = use_case.execute(request, root, transcript=transcript())
    files = [root / "report.json", root / "cues" / "0000.wav", Path(first.outputs["video"])]
    before = [f.read_bytes() for f in files]
    released = []
    factory.cloner.release = lambda: released.append(True)

    def fail(*args, **kwargs):
        raise OSError("fallo simulado")

    if failure == "mix":
        monkeypatch.setattr(factory.mixer, "render", fail)
    else:
        monkeypatch.setattr("subdub_titler.application.dubbing.os.replace", fail)
    with pytest.raises(OSError, match="simulado"):
        use_case.regenerate(request, root, 0, "Otra frase")
    assert [f.read_bytes() for f in files] == before
    assert load_report(root / "report.json") == first
    assert released == [True]


def test_edits_are_independent_and_undo_restores_audio_and_ratings(tmp_path):
    source = tmp_path / "a.mp4"
    source.write_bytes(b"video")
    request = DubRequest(source, "es", evaluate=True)
    use_case = DubUseCase(no_transcripts, FakeFactory())
    root = tmp_path / "work"
    first = use_case.execute(request, root, transcript=transcript())
    original_cue = (root / "cues" / "0000.wav").read_bytes()
    second = use_case.regenerate(request, root, 0, "Nueva frase", report=first, publish=False)
    third = use_case.regenerate(request, root, 1, "Segunda edición", report=second, publish=False)
    assert third.track[0] == second.track[0]
    assert third.track[1].target_text == "Segunda edición"
    assert third.cue_quality[1].similarity_slot is not None
    assert len({r.outputs["mix"] for r in (first, second, third)}) == 3
    assert (root / "cues" / "0000.wav").read_bytes() == original_cue
    assert load_report(root / "report.json") == first
    store = SqliteJobs(tmp_path / "db.sqlite")
    job = store.create(source, source.name, "small", "en", "cpu", "int8", kind="dub")
    store.update(job["id"], status="completed", result_json=json.dumps(report_to_dict(first)))
    store.rate(job["id"], 0, 5)
    store.activate_revision(job["id"], report_to_dict(second), cue_index=0)
    assert store.ratings(job["id"]) == {}
    store.activate_revision(job["id"], report_to_dict(third), cue_index=1)
    assert store.revision_count(job["id"]) == 2
    assert store.undo_revision(job["id"])["track"][0]["target_text"] == "Nueva frase"
    assert store.undo_revision(job["id"])["outputs"] == first.outputs
    assert store.ratings(job["id"])[0]["score"] == 5
    assert store.get(job["id"])["output_path"] == first.outputs["video"]


def test_history_filters_before_pagination_and_summaries_omit_payloads(tmp_path):
    store = SqliteJobs(tmp_path / "jobs.sqlite")
    visible = []
    for i in range(3):
        job = store.create(tmp_path / "a", str(i), "small", "en", "cpu", "int8")
        store.update(job["id"], transcript_json=json.dumps({"text": "large"}))
        visible.append(job["id"])
    for _ in range(105):
        store.create(tmp_path / "a", "internal", "small", "en", "cpu", "int8",
                     kind="dub-cue", parent_id=visible[0])
    page = store.list(limit=2, visible=True, summary=True)
    assert [j["id"] for j in page] == list(reversed(visible))[:2]
    assert page[0]["transcript"] is None
    assert "file_path" not in page[0]
    assert store.list(limit=2, offset=2, visible=True)[0]["id"] == visible[0]
    assert store.get(visible[0])["transcript"]["text"] == "large"


def test_homonymous_inputs_use_different_separation_directories(tmp_path, monkeypatch):
    monkeypatch.setattr("subdub_titler.application.transcribe.tempfile.gettempdir",
                        lambda: str(tmp_path))
    directories = []

    class Separator:
        def separate(self, source, workdir):
            directories.append(workdir)
            return Stems(source, source)

    class Engine:
        def transcribe(self, request):
            return Transcript((Segment(0, 1, "Hola"),), "es", "fake", 1)

    registry = ModelRegistry()
    registry.register(ModelDescriptor("fake", "fake"), lambda request: Engine())
    case = TranscribeUseCase(registry, separator=Separator())
    for name in ("a", "b"):
        source = tmp_path / name / "video.mp4"
        source.parent.mkdir()
        source.write_bytes(name.encode())
        case.execute(TranscriptionRequest(source, "fake", device="cpu", isolate_dialogue=True))
    assert directories[0] != directories[1]


def test_demucs_reuses_only_manifest_with_matching_content_and_settings(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("voice.flac", "background.flac"):
        (cache / name).write_bytes(b"stem")
    identity = {"schema": 1, "sha256": hashlib.sha256(b"audio").hexdigest(), "model": "htdemucs",
                "device": "cpu", "shifts": 0, "overlap": 0.25}
    (cache / "source.json").write_text(json.dumps(identity))
    separator = DemucsSeparator(device="cpu")
    assert separator.separate(source, cache).voice == cache / "voice.flac"
    assert separator._model is None


def test_diarization_preserves_confidence_and_rejects_missing_labels(tmp_path):
    class Diarizer:
        def assign(self, *args, **kwargs):
            return ["A"]

    case = TranscribeUseCase(ModelRegistry(), diarizer=Diarizer())
    data = Transcript((Segment(0, 1, "Hola", avg_logprob=-0.2, no_speech_prob=0.1),),
                      "es", "fake")
    result = case._assign_speakers(tmp_path / "a", data, 2)
    assert result.segments[0] == replace(data.segments[0], speaker="A")
    with pytest.raises(ValueError):
        case._assign_speakers(tmp_path / "a", replace(data, segments=data.segments * 2), 2)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_api_propagates_device_to_every_auxiliary(tmp_path, monkeypatch, device):
    from fastapi.testclient import TestClient
    from test_dub_api import FakeEngine, wait

    from subdub_titler.interfaces import api as api_module

    seen = []

    class Auxiliary:
        def __init__(self, *, device):
            seen.append(device)

        def separate(self, source, workdir):
            return Stems(source, source)

        def align(self, source, data):
            return data

        def assign(self, source, spans, max_speakers):
            return ["Ana"] * len(spans)

    for name in ("LazyDemucsSeparator", "LazyMmsForcedAligner", "LazySpeakerDiarizer"):
        module = {"LazyDemucsSeparator": "demucs_separator",
                  "LazyMmsForcedAligner": "ctc_aligner",
                  "LazySpeakerDiarizer": "diarization"}[name]
        monkeypatch.setattr(f"subdub_titler.infrastructure.{module}.{name}", Auxiliary)
    registry = ModelRegistry()
    registry.register(ModelDescriptor("fake", "fake"), lambda request: FakeEngine())
    app = api_module.create_app(tmp_path, registry, FakeFactory(), no_transcripts)
    with TestClient(app) as client:
        response = client.post("/api/v1/jobs", data={
            "model_id": "fake", "device": device, "isolate_dialogue": True,
            "forced_alignment": True, "diarize": True,
        }, files={"file": ("a.wav", b"audio", "audio/wav")})
        assert response.status_code == 202
        done = wait(client, response.json()["id"])
        assert done["status"] == "completed", done["error"]
        assert seen == [device] * 3
        assert all(s["speaker"] == "Ana" for s in done["transcript"]["segments"])


def test_worker_cancellation_and_retry_api(tmp_path):
    import threading

    from fastapi.testclient import TestClient
    from test_dub_api import app, wait

    application = app(tmp_path)
    entered, resume = threading.Event(), threading.Event()
    original = application.state.worker.handlers["transcribe"]

    class SlowHandler:
        def run(self, job, progress):
            entered.set()
            assert resume.wait(5)
            progress("frontera de etapa", 0.5)
            raise AssertionError("No debe continuar tras cancelar")

    application.state.worker.handlers["transcribe"] = SlowHandler()
    with TestClient(application) as client:
        response = client.post("/api/v1/jobs", data={"model_id": "fake", "device": "cpu"},
                               files={"file": ("a.wav", b"audio", "audio/wav")})
        identifier = response.json()["id"]
        try:
            assert entered.wait(5)
            assert client.post(f"/api/v1/jobs/{identifier}/cancel").status_code == 200
        finally:
            resume.set()
        for _ in range(200):
            if application.state.jobs.get(identifier)["status"] == "cancelled":
                break
            threading.Event().wait(0.01)
        assert application.state.jobs.get(identifier)["status"] == "cancelled"
        application.state.worker.handlers["transcribe"] = original
        retry = client.post(f"/api/v1/jobs/{identifier}/retry")
        assert retry.status_code == 202
        assert retry.json()["id"] != identifier
        assert wait(client, retry.json()["id"])["status"] == "completed"
        assert client.post(f"/api/v1/jobs/{retry.json()['id']}/cancel").status_code == 409


def test_queued_cancel_and_completion_race(tmp_path):
    from subdub_titler.application.jobs import LocalJobWorker

    store = SqliteJobs(tmp_path / "jobs.sqlite")
    job = store.create(tmp_path / "a", "a", "small", "en", "cpu", "int8")
    assert store.cancel(job["id"])
    LocalJobWorker(store, {})._execute(job["id"])
    assert store.get(job["id"])["status"] == "cancelled"
    running = store.create(tmp_path / "a", "a", "small", "en", "cpu", "int8")
    store.update(running["id"], status="running")
    assert store.cancel(running["id"])
    store.complete(running["id"], {})
    assert store.get(running["id"])["status"] == "cancelled"


def test_api_undo_restores_active_artifacts(tmp_path):
    from fastapi.testclient import TestClient
    from test_dub_api import app, wait

    with TestClient(app(tmp_path)) as client:
        upload = client.post("/api/v1/jobs", data={"model_id": "fake", "device": "cpu"},
                             files={"file": ("a.mp4", b"video", "video/mp4")}).json()
        parent = wait(client, upload["id"])
        dub = wait(client, client.post(f"/api/v1/jobs/{parent['id']}/dub",
                   json={"target_language": "es"}).json()["id"])
        identifier = dub["id"]
        previous = client.get(f"/api/v1/jobs/{identifier}/dub/cues/1/audio").content
        edit = client.post(f"/api/v1/jobs/{identifier}/dub/cues/1",
                           json={"text": "Texto nuevo"}).json()
        assert wait(client, edit["id"])["status"] == "completed"
        assert client.get(f"/api/v1/jobs/{identifier}/dub/revisions").json()["count"] == 1
        undo = client.post(f"/api/v1/jobs/{identifier}/dub/undo")
        assert undo.status_code == 202
        assert wait(client, undo.json()["id"])["status"] == "completed"
        restored = client.get(f"/api/v1/jobs/{identifier}").json()
        assert restored["result"] == dub["result"]
        assert client.get(f"/api/v1/jobs/{identifier}/dub/cues/1/audio").content == previous
        assert client.post(f"/api/v1/jobs/{identifier}/dub/cues/99/rating",
                           json={"score": 4}).status_code == 404
        assert client.post(f"/api/v1/jobs/{identifier}/dub/undo").status_code == 409
        page = client.get("/api/v1/jobs?visible=true&summary=true&limit=1").json()
        assert len(page) == 1 and page[0]["kind"] == "dub"
        assert page[0]["result"] is None
        assert client.get("/api/v1/jobs?limit=0").status_code == 422


@pytest.mark.parametrize("change", ["source", "model", "overlap"])
def test_demucs_invalidates_stale_manifest(tmp_path, monkeypatch, change):
    source = tmp_path / "a.wav"
    source.write_bytes(b"original")
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("voice.flac", "background.flac"):
        (cache / name).write_bytes(b"stem")
    manifest = cache / "source.json"
    identity = {"schema": 1, "sha256": hashlib.sha256(b"original").hexdigest(),
                "model": "htdemucs", "device": "cpu", "shifts": 0, "overlap": 0.25}
    manifest.write_text(json.dumps(identity))
    separator = DemucsSeparator(device="cpu")
    if change == "source":
        source.write_bytes(b"different")
    elif change == "model":
        separator.model_name = "htdemucs_ft"
    else:
        separator.overlap = 0.5

    def fail_loading():
        raise RuntimeError("Debe recalcular; no cargar pesos en el test")

    monkeypatch.setattr(separator, "_load", fail_loading)
    with pytest.raises(RuntimeError, match="Debe recalcular"):
        separator.separate(source, cache)
    assert not manifest.exists()
