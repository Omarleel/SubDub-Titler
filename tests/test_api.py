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
