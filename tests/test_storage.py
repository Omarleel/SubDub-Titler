from subdub_titler.infrastructure.storage import SqliteJobs


def test_job_state_and_recovery(tmp_path):
    store = SqliteJobs(tmp_path / "jobs.sqlite3")
    source = tmp_path / "a.wav"
    source.touch()
    job = store.create(source, "a.wav", "small", "es", "cpu", "int8")
    store.update(job["id"], status="running")
    assert store.get(job["id"])["status"] == "running"
    store.recover()
    assert store.get(job["id"])["status"] == "failed"
