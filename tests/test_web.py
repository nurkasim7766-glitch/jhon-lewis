"""Web API tests: project lifecycle, uploads, validation, jobs, persistence across restarts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from facepipe.web.app import create_app
from scripts.make_synthetic_source import make_transcript


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as c:
        yield c
    app.state.jobs.shutdown()
    app.state.db.close()


def _transcript_bytes() -> bytes:
    return json.dumps(make_transcript(minutes=3.0, hook_at_s=95.0)).encode()


def test_health_and_defaults(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["ffmpeg"] and ".mp4" in h["video_formats"]
    assert [s["id"] for s in h["steps"]][:3] == ["ingest", "transcript", "moments"]
    d = client.get("/api/settings/defaults").json()
    assert d["scorer"] == "heuristic" and d["asd_backend"] == "heuristic" and d["allow_hosted_llm"] is False


def test_project_crud_and_validation(client):
    p = client.post("/api/projects", json={"name": "Test"}).json()
    assert p["status"] == "new"
    assert client.patch(f"/api/projects/{p['id']}", json={"name": "Renamed"}).json()["name"] == "Renamed"
    r = client.patch(f"/api/projects/{p['id']}", json={"settings": {"min_clip_s": 90, "max_clip_s": 60}})
    assert r.status_code == 400 and "clip length" in r.json()["detail"]
    r = client.post(f"/api/projects/{p['id']}/video", files={"file": ("x.txt", b"nope", "text/plain")})
    assert r.status_code == 400 and "unsupported video format" in r.json()["detail"]
    r = client.post(f"/api/projects/{p['id']}/video", files={"file": ("x.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400 and "could not be decoded" in r.json()["detail"]
    r = client.post(f"/api/projects/{p['id']}/transcript", json={"text": "{not json"})
    assert r.status_code == 400
    r = client.post(f"/api/projects/{p['id']}/analyze")
    assert r.status_code == 400
    assert client.delete(f"/api/projects/{p['id']}").status_code == 204
    assert client.get(f"/api/projects/{p['id']}").status_code == 404


@pytest.mark.integration
def test_upload_analyze_persist(tmp_path):
    video = Path("assets/podcast/source.mp4")
    if not video.exists():
        pytest.skip("synthetic source not built; run scripts/make_synthetic_source.py")
    data_dir = tmp_path / "data"
    app = create_app(data_dir)
    with TestClient(app) as c:
        p = c.post("/api/projects", json={"name": "Persist"}).json()
        pid = p["id"]
        with video.open("rb") as fh:
            r = c.post(f"/api/projects/{pid}/video", files={"file": ("source.mp4", fh, "video/mp4")})
        assert r.status_code == 200 and r.json()["status"] == "uploaded"
        assert r.json()["source_info"]["duration"] == pytest.approx(180, abs=1)
        r = c.post(f"/api/projects/{pid}/transcript", files={"file": ("t.json", _transcript_bytes(), "application/json")})
        assert r.status_code == 200 and r.json()["status"] == "ready"
        c.patch(f"/api/projects/{pid}", json={"settings": {"max_clips": 3}})
        job = c.post(f"/api/projects/{pid}/analyze").json()
        assert job["status"] == "queued"
        # a second job on the same project is refused while one is active
        assert c.post(f"/api/projects/{pid}/analyze").status_code == 400
        for _ in range(120):
            job = c.get(f"/api/jobs/{job['id']}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.5)
        assert job["status"] == "done", job
        assert {k: v["status"] for k, v in job["progress"]["steps"].items()} == {"ingest": "done", "transcript": "done", "moments": "done"}
        clips = c.get(f"/api/projects/{pid}/clips").json()
        assert len(clips) == 3 and clips[0]["score"] >= clips[1]["score"] >= clips[2]["score"]
        assert all(c_["files"]["thumbnail"] for c_ in clips)
        # non-overlapping sentence ranges
        for a in clips:
            for b in clips:
                if a is not b:
                    assert a["last_sentence"] < b["first_sentence"] or a["first_sentence"] > b["last_sentence"]
        assert c.get(f"/api/projects/{pid}/clips/{clips[0]['id']}/thumbnail").status_code == 200
        r = c.get(f"/api/projects/{pid}/source", headers={"Range": "bytes=0-99"})
        assert r.status_code == 206 and len(r.content) == 100
        assert c.get(f"/api/projects/{pid}/clips/{clips[0]['id']}/tracks").status_code == 404  # not processed yet
        stats = c.get("/api/dashboard").json()["stats"]
        assert stats["projects"] == 1 and stats["candidates"] == 3
    app.state.jobs.shutdown()
    app.state.db.close()

    # restart: everything is still there
    app2 = create_app(data_dir)
    with TestClient(app2) as c:
        p = c.get(f"/api/projects/{pid}").json()
        assert p["name"] == "Persist" and p["status"] == "analyzed" and len(p["clips"]) == 3
        assert p["settings"]["max_clips"] == 3
        assert c.get(f"/api/projects/{pid}/clips/{p['clips'][0]['id']}/thumbnail").status_code == 200
        assert p["jobs"][0]["status"] == "done"
    app2.state.jobs.shutdown()
    app2.state.db.close()


def test_interrupted_jobs_are_marked_on_restart(tmp_path):
    data_dir = tmp_path / "data"
    app = create_app(data_dir)
    p = app.state.store.create("Crash")
    app.state.db.execute("INSERT INTO jobs(id,project_id,kind,status,params,progress,created_at,started_at) VALUES (?,?,?,?,?,?,?,?)",
                         ("deadbeef0000", p["id"], "process", "running", "{}", "{}", time.time(), time.time()))
    app.state.store.touch(p["id"], status="processing")
    app.state.jobs.shutdown()
    app.state.db.close()
    app2 = create_app(data_dir)
    job = app2.state.jobs.get("deadbeef0000")
    assert job["status"] == "interrupted"
    assert app2.state.store.get(p["id"])["status"] == "ready"
    app2.state.jobs.shutdown()
    app2.state.db.close()
