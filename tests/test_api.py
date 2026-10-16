from fastapi.testclient import TestClient

from subdub_titler.interfaces.api import create_app


def test_api_health_and_models(tmp_path):
    with TestClient(create_app(home=tmp_path)) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"
        assert client.get("/api/v1/models").json()[0]["backend"] == "faster-whisper"
        result = client.post("/api/v1/evaluate", json={
            "reference": "Hola mundo", "hypothesis": "hola Mundo!", "language": "es"
        })
        assert result.status_code == 200
        assert result.json()["normalized"]["wer"]["rate"] == 0


def test_reject_untrusted_upload(tmp_path):
    with TestClient(create_app(home=tmp_path)) as client:
        result = client.post("/api/v1/jobs", data={"model_id": "small"},
                             files={"file": ("script.py", b"print(1)", "text/plain")})
        assert result.status_code == 415


def test_upload_exceeds_max_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBDUB_MAX_UPLOAD_BYTES", "10")
    with TestClient(create_app(home=tmp_path)) as client:
        result = client.post("/api/v1/jobs", data={
            "model_id": "small", "device": "cpu", "compute_type": "int8", "language": "es"
        }, files={"file": ("video.mp4", b"0123456789extra_bytes", "video/mp4")})
        assert result.status_code == 413
        assert "excede" in result.text


def test_404_job(tmp_path):
    with TestClient(create_app(home=tmp_path)) as client:
        assert client.get("/api/v1/jobs/unknown").status_code == 404


def test_complete_job_via_adapter_contract(tmp_path):
    """E2E real API/SQLite/worker/SRT con backend falso, sin GPU ni pesos."""
    import time

    from subdub_titler.domain.models import ModelDescriptor, Segment, Transcript
    from subdub_titler.infrastructure.registry import ModelRegistry

    class FakeEngine:
        def transcribe(self, request):
            return Transcript((Segment(0.1, 1.3, "Hola"),), "es", request.model_id, 1.3)

    registry = ModelRegistry()
    registry.register(ModelDescriptor("fake", "test"), lambda request: FakeEngine())
    with TestClient(create_app(home=tmp_path, registry=registry)) as client:
        response = client.post("/api/v1/jobs", data={
            "model_id": "fake", "device": "cpu", "compute_type": "int8",
            "language": "es",
        }, files={"file": ("sample.wav", b"not-a-wav", "audio/wav")})
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        for _ in range(100):
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(0.01)
        assert job["status"] == "completed", job
        assert job["transcript"]["text"] == "Hola"
        subtitle = client.get(f"/api/v1/jobs/{job_id}/subtitles")
        assert subtitle.status_code == 200
        assert "00:00:00,100 --> 00:00:01,300" in subtitle.text
        assert any(event["name"] == "asr.transcribe_complete" for event in job["profiling"])
        media = client.get(f"/api/v1/jobs/{job_id}/media", headers={"Range": "bytes=0-2"})
        assert media.status_code == 206 and media.content == b"not"
        assert media.headers["content-type"] == "audio/wav"
        vtt = client.get(f"/api/v1/jobs/{job_id}/subtitles.vtt")
        assert vtt.headers["content-type"].startswith("text/vtt")
        assert "00:00:00.100 --> 00:00:01.300\nHola" in vtt.text
        assert client.get("/api/v1/jobs/unknown/media").status_code == 404


def test_frontend_assets_served_as_javascript(tmp_path):
    """Windows puede registrar .js como text/plain; los módulos ES lo rechazan."""
    from pathlib import Path

    import pytest
    assets = Path(__file__).resolve().parents[1] / "apps" / "web" / "dist" / "assets"
    bundles = list(assets.glob("*.js")) if assets.is_dir() else []
    if not bundles:
        pytest.skip("Frontend no compilado")
    with TestClient(create_app(home=tmp_path)) as client:
        response = client.get(f"/assets/{bundles[0].name}")
        assert response.headers["content-type"].startswith("text/javascript")


def test_spa_routes_survive_reload(tmp_path):
    """Recargar /precision o /rendimiento debe devolver la UI, no 404; /api sigue siendo JSON."""
    from pathlib import Path

    import pytest
    if not (Path(__file__).resolve().parents[1] / "apps" / "web" / "dist").is_dir():
        pytest.skip("Frontend no compilado")
    with TestClient(create_app(home=tmp_path)) as client:
        for path in ("/", "/precision", "/rendimiento?trabajo=abc", "/transcribir"):
            response = client.get(path)
            assert response.status_code == 200
            assert '<div id="root">' in response.text
        assert client.get("/api/v1/inexistente").status_code == 404
        assert client.get("/../pyproject.toml").status_code in (200, 404)
        assert "[project]" not in client.get("/..%2F..%2F..%2Fpyproject.toml").text


def test_upload_job_with_vad_and_preprocessing_params(tmp_path):
    with TestClient(create_app(home=tmp_path)) as client:
        response = client.post("/api/v1/jobs", data={
            "model_id": "small",
            "device": "cpu",
            "compute_type": "int8",
            "language": "es",
            "batch_size": "8",
            "vad_threshold": "0.30",
            "vad_speech_pad_ms": "1000",
            "vad_min_silence_ms": "2500",
            "no_speech_threshold": "0.35",
            "normalize_loudness": "true",
            "isolate_dialogue": "true",
            "forced_alignment": "true",
            "diarize": "true",
            "max_speakers": "3",
            "conform_netflix": "true",
            "suppress_hallucinations": "false",
            "adaptive_two_pass": "true",
        }, files={"file": ("sample.wav", b"not-a-wav", "audio/wav")})
        assert response.status_code == 202
        job = response.json()
        assert job["params"]["vad_threshold"] == 0.30
        assert job["params"]["vad_speech_pad_ms"] == 1000
        assert job["params"]["vad_min_silence_ms"] == 2500
        assert job["params"]["no_speech_threshold"] == 0.35
        assert job["params"]["normalize_loudness"] is True
        assert job["params"]["isolate_dialogue"] is True
        assert job["params"]["forced_alignment"] is True
        assert job["params"]["diarize"] is True
        assert job["params"]["max_speakers"] == 3
        assert job["params"]["conform_netflix"] is True
        assert job["params"]["suppress_hallucinations"] is False
        assert job["params"]["adaptive_two_pass"] is True

