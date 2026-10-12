import time

from dir_language_pipeline.job_store import AWAITING, JobStore


def make_store(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create("j1", "hi", "auto", tmp_path / "in.mp4", "in.mp4", "combine")
    return store


def test_await_then_confirm_requeues_with_the_choice(tmp_path):
    store = make_store(tmp_path)
    claimed = store.claim()
    assert claimed["face_choice"] is None
    store.await_face("j1")
    assert store.get("j1")["status"] == AWAITING
    assert store.claim() is None  # a waiting job is not picked up by the worker
    assert store.confirm_face("j1", "1") == "queued"
    resumed = store.claim()
    assert resumed["job_id"] == "j1" and resumed["face_choice"] == "1"


def test_confirming_twice_is_idempotent(tmp_path):
    store = make_store(tmp_path)
    store.claim()
    store.await_face("j1")
    assert store.confirm_face("j1", "0") == "queued"
    assert store.confirm_face("j1", "0") == "already_confirmed"
    assert store.get("j1")["status"] == "queued"


def test_confirm_with_a_different_choice_after_resume_is_rejected(tmp_path):
    store = make_store(tmp_path)
    store.claim()
    store.await_face("j1")
    store.confirm_face("j1", "0")
    assert store.confirm_face("j1", "1") == "not_awaiting"
    assert store.get("j1")["face_choice"] == "0"


def test_confirm_on_a_job_that_is_not_waiting_or_unknown(tmp_path):
    store = make_store(tmp_path)
    assert store.confirm_face("j1", "0") == "not_awaiting"  # still queued
    assert store.confirm_face("nope", "0") == "unknown_job"


def test_waiting_job_survives_a_restart_but_running_job_is_requeued(tmp_path):
    store = make_store(tmp_path)
    store.create("j2", "hi", "auto", tmp_path / "in2.mp4", "in2.mp4", "combine")
    store.claim()
    store.await_face("j1")
    store.claim()  # j2 running
    reopened = JobStore(tmp_path / "jobs.sqlite3")
    assert reopened.get("j1")["status"] == AWAITING
    assert reopened.get("j2")["status"] == "queued"


def test_expiry_fails_only_stale_waiting_jobs(tmp_path):
    store = make_store(tmp_path)
    store.claim()
    store.await_face("j1")
    assert store.expire_stale(max_age_seconds=3600) == []
    time.sleep(0.05)
    assert store.expire_stale(max_age_seconds=0) == ["j1"]
    record = store.get("j1")
    assert record["status"] == "failed" and record["stage"] == "expired"
    assert "24 hours" in record["error"]


def test_old_databases_are_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE jobs (job_id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
            target_language TEXT NOT NULL, source_language TEXT NOT NULL, operation TEXT NOT NULL DEFAULT 'combine',
            input_path TEXT NOT NULL, input_name TEXT NOT NULL, artifacts TEXT NOT NULL DEFAULT '{}',
            error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    store = JobStore(path)
    store.create("j1", "hi", "auto", tmp_path / "i.mp4", "i.mp4")
    assert store.get("j1")["face_choice"] is None
    assert store.get("j1")["masking_mode"] == "new"


def test_masking_mode_is_persisted(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create("j1", "hi", "auto", tmp_path / "i.mp4", "i.mp4", "masking", "overlay")

    assert store.get("j1")["masking_mode"] == "overlay"
    assert store.claim()["masking_mode"] == "overlay"


def test_only_queued_jobs_can_be_removed(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create("running", "hi", "auto", tmp_path / "r.mp4", "r.mp4")
    store.create("queued", "hi", "auto", tmp_path / "q.mp4", "q.mp4")
    store.claim()

    assert store.remove_queued("running") == "not_queued"
    assert store.remove_queued("queued") == "removed"
    assert store.remove_queued("queued") == "unknown_job"


def test_list_returns_persisted_jobs_newest_first(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create("j1", "hi", "auto", tmp_path / "one.mp4", "one.mp4")
    time.sleep(0.01)
    store.create("j2", "ta", "en", tmp_path / "two.mp4", "two.mp4", "dubbing")

    assert [job["job_id"] for job in store.list()] == ["j2", "j1"]
    reopened = JobStore(tmp_path / "jobs.sqlite3")
    assert [job["input_name"] for job in reopened.list()] == ["two.mp4", "one.mp4"]
