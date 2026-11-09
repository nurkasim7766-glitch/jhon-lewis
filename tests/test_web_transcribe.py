"""Video-only uploads, analysis readiness, and local transcription."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from facepipe.web.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "data")
    with TestClient(app) as c:
        yield c
    app.state.jobs.shutdown()
    app.state.db.close()


SOURCE = Path("assets/podcast/source.mp4")


def _upload_video(c, pid, path=SOURCE):
    with path.open("rb") as fh:
        return c.post(f"/api/projects/{pid}/video", files={"file": (path.name, fh, "video/mp4")})


def test_video_only_project_is_ready_and_explains_transcript_mode(client, monkeypatch):
    if not SOURCE.exists():
        pytest.skip("synthetic source not built")
    import facepipe.web.projects as projects_mod

    pid = client.post("/api/projects", json={"name": "Video only"}).json()["id"]
    assert client.get(f"/api/projects/{pid}").json()["readiness"] == {
        "can_analyze": False, "transcript_mode": "none", "message": "Upload a video to enable analysis.",
        "transcription_available": client.get("/api/health").json()["transcription_available"]}

    r = _upload_video(client, pid)
    assert r.status_code == 200
    p = r.json()
    assert p["status"] == "ready" and p["transcript_filename"] is None

    # with local transcription installed -> analysis allowed, mode "auto"
    monkeypatch.setattr("facepipe.transcribe.transcription_available", lambda: True)
    p = client.get(f"/api/projects/{pid}").json()
    assert p["readiness"]["can_analyze"] is True and p["readiness"]["transcript_mode"] == "auto"
    assert "transcribed locally" in p["readiness"]["message"]

    # without it -> button must stay disabled *with* an explicit message, and the API refuses clearly
    monkeypatch.setattr("facepipe.transcribe.transcription_available", lambda: False)
    p = client.get(f"/api/projects/{pid}").json()
    assert p["readiness"]["can_analyze"] is False and p["readiness"]["transcript_mode"] == "missing"
    assert "transcript is required" in p["readiness"]["message"] and "[transcribe]" in p["readiness"]["message"]
    r = client.post(f"/api/projects/{pid}/analyze")
    assert r.status_code == 400 and "transcript is required" in r.json()["detail"]

    # uploading a transcript makes it ready regardless
    with open("assets/podcast/transcript.json", "rb") as fh:
        p = client.post(f"/api/projects/{pid}/transcript", files={"file": ("t.json", fh, "application/json")}).json()
    assert p["readiness"] == {"can_analyze": True, "transcript_mode": "uploaded", "transcription_available": False,
                              "message": "Transcript uploaded. Ready to analyze."}
    assert projects_mod  # imported for monkeypatch target clarity


def test_silent_video_cannot_be_auto_transcribed(client, tmp_path, monkeypatch):
    if not SOURCE.exists():
        pytest.skip("synthetic source not built")
    from facepipe.ffmpeg import ffmpeg_exe

    silent = tmp_path / "silent.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-t", "8", "-i", str(SOURCE), "-an", "-c:v", "copy", str(silent)], check=True)
    monkeypatch.setattr("facepipe.transcribe.transcription_available", lambda: True)
    pid = client.post("/api/projects", json={"name": "Silent"}).json()["id"]
    p = _upload_video(client, pid, silent).json()
    assert p["source_info"]["has_audio"] is False
    assert p["readiness"]["can_analyze"] is False and "no audio track" in p["readiness"]["message"]


def test_analyze_job_transcribes_when_no_transcript(client, monkeypatch):
    """The analyze job calls the local transcriber (mocked here) and continues to moment selection."""
    if not SOURCE.exists():
        pytest.skip("synthetic source not built")
    from scripts.make_synthetic_source import make_transcript

    calls = []

    def fake_transcribe(video, out_json, model_size="base", language=None, progress=None, cancel_check=None):
        calls.append((Path(video).name, model_size, language))
        if progress:
            progress(90.0, 180.0)
        doc = make_transcript(minutes=3.0, hook_at_s=95.0)
        Path(out_json).write_text(json.dumps(doc))
        return doc

    monkeypatch.setattr("facepipe.transcribe.transcription_available", lambda: True)
    monkeypatch.setattr("facepipe.transcribe.transcribe_video", fake_transcribe)
    pid = client.post("/api/projects", json={"name": "Auto"}).json()["id"]
    _upload_video(client, pid)
    client.patch(f"/api/projects/{pid}", json={"settings": {"max_clips": 2, "transcribe_model": "tiny", "transcribe_language": "en"}})
    job = client.post(f"/api/projects/{pid}/analyze").json()
    for _ in range(120):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    assert calls == [("source.mp4", "tiny", "en")]
    steps = {k: v["status"] for k, v in job["progress"]["steps"].items()}
    assert steps == {"ingest": "done", "transcribe": "done", "transcript": "done", "moments": "done"}
    p = client.get(f"/api/projects/{pid}").json()
    assert p["transcript_info"]["generated"] is True and p["transcript_filename"] == "transcript.generated.json"
    assert len(p["clips"]) == 2
    assert client.get(f"/api/projects/{pid}/transcript").status_code == 200
    # a user-supplied transcript is reused, not re-transcribed
    job2 = client.post(f"/api/projects/{pid}/analyze").json()
    for _ in range(120):
        job2 = client.get(f"/api/jobs/{job2['id']}").json()
        if job2["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert job2["progress"]["steps"]["transcribe"]["status"] == "reused" and len(calls) == 1


@pytest.mark.integration
def test_real_local_transcription_of_synthesised_speech(tmp_path):
    """End-to-end with the real faster-whisper model on espeak-generated speech (needs [transcribe] + espeak-ng)."""
    pytest.importorskip("faster_whisper")
    import shutil

    if not shutil.which("espeak-ng") or not SOURCE.exists():
        pytest.skip("espeak-ng or synthetic source missing")
    from facepipe.ffmpeg import ffmpeg_exe
    from facepipe.transcribe import transcribe_video
    from facepipe.transcript import load_transcript

    text = "Why did the first company fail? The biggest mistake was that nobody talked to a customer. That is the lesson."
    wav = tmp_path / "speech.wav"
    subprocess.run(["espeak-ng", "-s", "150", "-w", str(wav), text], check=True, capture_output=True)
    video = tmp_path / "speech.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-t", "12", "-i", str(SOURCE), "-i", str(wav), "-map", "0:v:0",
                    "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(video)], check=True)
    out = tmp_path / "transcript.json"
    doc = transcribe_video(video, out, model_size="tiny", language="en")
    assert doc["source"]["backend"] == "faster-whisper" and doc["segments"]
    words = " ".join(s["text"] for s in doc["segments"]).lower()
    assert "company" in words and "customer" in words, words
    t = load_transcript(out)
    assert t.sentences and all(s.words for s in t.sentences)  # word timings present for sentence snapping
