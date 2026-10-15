import sqlite3
import time

from dub_fakes import FakeFactory, no_transcripts
from fastapi.testclient import TestClient

from subdub_titler.domain.models import ModelDescriptor, Segment, Transcript, Word
from subdub_titler.infrastructure.registry import ModelRegistry
from subdub_titler.infrastructure.storage import SqliteJobs
from subdub_titler.interfaces.api import create_app


class FakeEngine:
    def transcribe(self, request):
        return Transcript((
            Segment(1.0, 2.0, "Hello there.", (Word(1.0, 2.0, " Hello there."),)),
            Segment(3.0, 7.0, "This is a long and complete sentence.",
                    (Word(3.0, 7.0, " This is a long and complete sentence."),)),
        ), "en", request.model_id, 20.0)


def wait(client, job_id):
    for _ in range(300):
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.01)
    raise AssertionError("El trabajo no terminó")


def app(tmp_path):
    registry = ModelRegistry()
    registry.register(ModelDescriptor("fake", "test"), lambda request: FakeEngine())
    return create_app(home=tmp_path, registry=registry, dub_factory=FakeFactory(),
                      transcripts=no_transcripts)


def test_dub_job_flow_artifacts_regeneration_and_ratings(tmp_path):
    with TestClient(app(tmp_path)) as client:
        upload = client.post("/api/v1/jobs", data={
            "model_id": "fake", "device": "cpu", "compute_type": "int8", "language": "en",
            "batch_size": "4"}, files={"file": ("clip.mp4", b"video", "video/mp4")})
        parent = wait(client, upload.json()["id"])
        assert parent["status"] == "completed" and parent["kind"] == "transcribe"
        assert parent["params"] == {"batch_size": 4}

        assert client.post(f"/api/v1/jobs/{parent['id']}/dub",
                           json={"target_language": "fr"}).status_code == 422
        response = client.post(f"/api/v1/jobs/{parent['id']}/dub",
                               json={"target_language": "es", "tts_batch": 2})
        assert response.status_code == 202, response.text
        dub = wait(client, response.json()["id"])
        assert dub["status"] == "completed", dub["error"]
        assert dub["kind"] == "dub" and dub["parent_id"] == parent["id"]
        assert dub["stage"] == "listo" and dub["progress"] == 1.0
        assert dub["params"]["target_language"] == "es"
        result = dub["result"]
        assert len(result["track"]) == 2 and result["sync"]["overlaps"] == 0
        assert {g["name"] for g in result["gates"]} >= {"sincronia", "cer_ida_vuelta"}
        assert any(e["name"] == "dub.tts.generate" for e in dub["profiling"])
        assert [j["id"] for j in client.get("/api/v1/jobs?kind=dub").json()] == [dub["id"]]

        video = client.get(f"/api/v1/jobs/{dub['id']}/dub/video")
        assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
        assert client.get(f"/api/v1/jobs/{dub['id']}/dub/mix").headers[
            "content-type"] == "audio/flac"
        assert client.get(f"/api/v1/jobs/{dub['id']}/dub/otro").status_code == 404
        cue = client.get(f"/api/v1/jobs/{dub['id']}/dub/cues/1/audio")
        assert cue.status_code == 200 and cue.content.startswith(b"RIFF")
        # Un trabajo de transcripción no es un doblaje.
        assert client.get(f"/api/v1/jobs/{parent['id']}/dub/video").status_code == 404

        edit = client.post(f"/api/v1/jobs/{dub['id']}/dub/cues/1", json={"text": "Hola"})
        assert edit.status_code == 202
        done = wait(client, edit.json()["id"])
        assert done["status"] == "completed", done["error"]
        assert done["result"]["cue"]["target_text"] == "Hola"
        refreshed = client.get(f"/api/v1/jobs/{dub['id']}").json()["result"]
        assert refreshed["track"][1]["target_text"] == "Hola"
        assert client.post(f"/api/v1/jobs/{dub['id']}/dub/cues/9",
                           json={"text": "x"}).status_code == 404

        assert client.post(f"/api/v1/jobs/{dub['id']}/dub/cues/0/rating",
                           json={"score": 6}).status_code == 422
        client.post(f"/api/v1/jobs/{dub['id']}/dub/cues/0/rating", json={"score": 4})
        rated = client.post(f"/api/v1/jobs/{dub['id']}/dub/cues/1/rating",
                            json={"score": 5, "note": "natural"}).json()
        assert rated["count"] == 2 and rated["mean"] == 4.5
        assert rated["ratings"]["1"] == {"score": 5, "note": "natural"}
        components = client.get("/api/v1/dub/components").json()
        assert components["languages"] == ["en", "es", "ja", "zh"]


def test_dub_requires_completed_transcription(tmp_path):
    with TestClient(app(tmp_path)) as client:
        assert client.post("/api/v1/jobs/nope/dub",
                           json={"target_language": "es"}).status_code == 404


def test_database_from_v010_is_migrated(tmp_path):
    """Una base creada por v0.1.0 (sin kind/params/result) sigue funcionando."""
    db = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE jobs (id TEXT PRIMARY KEY, file_path TEXT NOT NULL,
            original_name TEXT NOT NULL, model_id TEXT NOT NULL, language TEXT NOT NULL,
            device TEXT NOT NULL, compute_type TEXT NOT NULL, status TEXT NOT NULL,
            output_path TEXT, transcript_json TEXT, profiling_json TEXT, error TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        conn.execute("INSERT INTO jobs (id,file_path,original_name,model_id,language,device,"
                     "compute_type,status,created_at,updated_at) VALUES "
                     "('old','a.mp4','a.mp4','small','es','cuda','float16','completed','t','t')")
    jobs = SqliteJobs(db)
    old = jobs.get("old")
    assert old["kind"] == "transcribe" and old["params"] == {} and old["result"] is None
    new = jobs.create(tmp_path / "b.mp4", "b.mp4", "small", "es", "cuda", "float16",
                      kind="dub", params={"x": 1}, parent_id="old")
    assert new["params"] == {"x": 1} and new["parent_id"] == "old"
    assert SqliteJobs(db).get("old")["kind"] == "transcribe"  # migración idempotente
