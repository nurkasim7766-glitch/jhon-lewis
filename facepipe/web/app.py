"""FastAPI application. Run with `facepipe-web` or `python -m facepipe.web`."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from facepipe.web.db import Database
from facepipe.web.jobs import STEPS, JobManager
from facepipe.web.projects import TRANSCRIPT_EXTENSIONS, VIDEO_EXTENSIONS, ProjectSettings, ProjectStore

log = logging.getLogger("facepipe.web")
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = int(os.environ.get("FACEPIPE_MAX_UPLOAD_MB", "8192")) * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("FACEPIPE_DATA_DIR") or Path.cwd() / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(data_dir / "facepipe.db")
    store = ProjectStore(db, data_dir)
    jobs = JobManager(db, store)

    app = FastAPI(title="facepipe", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.db, app.state.store, app.state.jobs, app.state.data_dir = db, store, jobs, data_dir

    @app.on_event("shutdown")
    def _shutdown() -> None:
        jobs.shutdown()
        db.close()

    @app.exception_handler(KeyError)
    async def _not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"detail": f"not found: {exc.args[0] if exc.args else ''}"}, status_code=404)

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # -- meta --------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        from facepipe.ffmpeg import ffmpeg_exe, ffprobe_exe

        try:
            import torch  # noqa: F401
            lrasd = True
        except ImportError:
            lrasd = False
        return {"ok": True, "ffmpeg": ffmpeg_exe(), "ffprobe": ffprobe_exe(), "lrasd_available": lrasd,
                "video_formats": sorted(VIDEO_EXTENSIONS), "transcript_formats": sorted(TRANSCRIPT_EXTENSIONS),
                "max_upload_bytes": MAX_UPLOAD_BYTES, "steps": [{"id": s, "label": l} for s, l in STEPS],
                "data_dir": str(data_dir)}

    @app.get("/api/dashboard")
    def dashboard() -> dict[str, Any]:
        return {"stats": store.stats(), "recent_projects": store.list()[:6], "activity": store.activity(20),
                "jobs": jobs.list(limit=10)}

    # -- projects ----------------------------------------------------------------
    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        return store.list()

    @app.post("/api/projects", status_code=201)
    def create_project(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        return store.create(str(payload.get("name") or "Untitled project")[:120])

    @app.get("/api/projects/{pid}")
    def get_project(pid: str) -> dict[str, Any]:
        p = store.get(pid)
        p["clips"] = store.clips(pid)
        p["jobs"] = jobs.list(pid, limit=20)
        return p

    @app.patch("/api/projects/{pid}")
    def patch_project(pid: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if "name" in payload:
            store.rename(pid, str(payload["name"])[:120])
        if "settings" in payload:
            store.update_settings(pid, dict(payload["settings"]))
        return store.get(pid)

    @app.delete("/api/projects/{pid}", status_code=204)
    def delete_project(pid: str) -> None:
        active = db.one("SELECT id FROM jobs WHERE project_id=? AND status IN ('queued','running')", (pid,))
        if active:
            raise ValueError("cancel the running job before deleting the project")
        store.delete(pid)

    @app.get("/api/settings/defaults")
    def default_settings() -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(ProjectSettings())

    # -- uploads -----------------------------------------------------------------
    async def _save_upload(upload: UploadFile, limit: int, dest: Path) -> int:
        size = 0
        with dest.open("wb") as fh:
            while chunk := await upload.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > limit:
                    fh.close()
                    dest.unlink(missing_ok=True)
                    raise ValueError(f"file exceeds the {limit // (1024 * 1024)} MB upload limit")
                fh.write(chunk)
        return size

    @app.post("/api/projects/{pid}/video")
    async def upload_video(pid: str, file: UploadFile = File(...)) -> dict[str, Any]:
        store.get(pid)
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported video format {ext or '(none)'}; use one of {', '.join(sorted(VIDEO_EXTENSIONS))}")
        tmp = Path(tempfile.mkstemp(suffix=ext, dir=store.project_dir(pid))[1])
        try:
            await _save_upload(file, MAX_UPLOAD_BYTES, tmp)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return store.attach_source(pid, file.filename or f"video{ext}", tmp)

    @app.post("/api/projects/{pid}/transcript")
    async def upload_transcript(pid: str, file: UploadFile | None = File(default=None),
                                payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        store.get(pid)
        if file is not None:
            content = await file.read(MAX_TRANSCRIPT_BYTES + 1)
            if len(content) > MAX_TRANSCRIPT_BYTES:
                raise ValueError("transcript too large")
            name = file.filename or "transcript.json"
        elif payload and payload.get("text"):
            content = str(payload["text"]).encode("utf-8")
            name = "pasted.json"
        else:
            raise ValueError("provide a transcript file or JSON text")
        return store.attach_transcript(pid, name, content)

    # -- jobs --------------------------------------------------------------------
    @app.post("/api/projects/{pid}/analyze", status_code=202)
    def analyze(pid: str) -> dict[str, Any]:
        p = store.get(pid)
        if not store.source_path(p) or not store.transcript_path(pid).exists():
            raise ValueError("upload a video and a transcript first")
        return jobs.submit(pid, "analyze")

    @app.post("/api/projects/{pid}/process", status_code=202)
    def process(pid: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        store.get(pid)
        ids = payload.get("clip_ids")
        if ids is not None:
            for cid in ids:
                store.clip(pid, cid)
        force = payload.get("force") or []
        return jobs.submit(pid, "process", {"clip_ids": ids, "force": force})

    @app.post("/api/projects/{pid}/rerender", status_code=202)
    def rerender(pid: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        store.get(pid)
        return jobs.submit(pid, "render", {"clip_ids": payload.get("clip_ids")})

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return jobs.list(limit=100)

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str) -> dict[str, Any]:
        return jobs.get(jid)

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str) -> dict[str, Any]:
        return jobs.cancel(jid)

    # -- clips -------------------------------------------------------------------
    @app.get("/api/projects/{pid}/clips")
    def list_clips(pid: str) -> list[dict[str, Any]]:
        store.get(pid)
        return store.clips(pid)

    @app.get("/api/projects/{pid}/clips/{cid}")
    def get_clip(pid: str, cid: str) -> dict[str, Any]:
        clip = store.clip(pid, cid)
        d = store.clip_dir(pid, cid)
        for name in ("speakers.json",):
            f = d / "phase2" / name
            if f.exists():
                doc = json.loads(f.read_text(encoding="utf-8"))
                clip["segments"] = doc["segments"]
                clip["asd"] = doc["asd"]
        plan = d / "phase2" / "crop_plan.json"
        if plan.exists():
            pd = json.loads(plan.read_text(encoding="utf-8"))
            clip["plan"] = {"source": pd["source"], "output": pd["output"], "segments": pd["segments"]}
        run = d / "run.json"
        if run.exists():
            clip["run"] = json.loads(run.read_text(encoding="utf-8"))
        return clip

    @app.patch("/api/projects/{pid}/clips/{cid}")
    def patch_clip(pid: str, cid: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if "selected" in payload:
            return store.set_selected(pid, cid, bool(payload["selected"]))
        return store.clip(pid, cid)

    @app.get("/api/projects/{pid}/clips/{cid}/tracks")
    def clip_tracks(pid: str, cid: str) -> dict[str, Any]:
        """Compact per-frame tracking data for the editor overlay, read from the pipeline files."""
        from facepipe.contract import iter_frames_jsonl

        store.clip(pid, cid)
        d = store.clip_dir(pid, cid)
        manifest_p, speakers_p, plan_p = d / "phase1" / "manifest.json", d / "phase2" / "speakers.json", d / "phase2" / "crop_plan.json"
        if not (manifest_p.exists() and speakers_p.exists() and plan_p.exists()):
            raise KeyError("tracking data not available yet; process the clip first")
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        speakers = json.loads(speakers_p.read_text(encoding="utf-8"))
        plan = json.loads(plan_p.read_text(encoding="utf-8"))
        frames = []
        with (d / "phase1" / manifest["frames_file"]).open(encoding="utf-8") as fh:
            for f, sf, pf in zip(iter_frames_jsonl(fh), speakers["frames"], plan["frames"]):
                frames.append({
                    "t": f["timestamp_ms"],
                    "faces": [{"id": face["face_id"], "b": [round(face["bbox"][k], 4) for k in ("x_min", "y_min", "x_max", "y_max")],
                               "s": sf["scores"].get(str(face["face_id"]))} for face in f["faces"]],
                    "a": sf["active_face_id"],
                    "c": [pf["crop"]["x"], pf["crop"]["y"], pf["crop"]["w"], pf["crop"]["h"]],
                })
        return {"video": manifest["video"], "threshold": speakers["asd"]["speaking_threshold"],
                "backend": speakers["asd"]["backend"], "segments": speakers["segments"],
                "output": plan["output"], "frames": frames}

    # -- media -------------------------------------------------------------------
    def _file(path: Path, media_type: str, download_name: str | None = None) -> FileResponse:
        if not path.exists():
            raise KeyError(path.name)
        headers = {"Cache-Control": "no-cache"}
        if download_name:
            headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
        return FileResponse(str(path), media_type=media_type, headers=headers)

    @app.get("/api/projects/{pid}/source")
    def source_video(pid: str) -> FileResponse:
        p = store.get(pid)
        src = store.source_path(p)
        if not src:
            raise KeyError("source video")
        return _file(src, "video/mp4")

    @app.get("/api/projects/{pid}/clips/{cid}/thumbnail")
    def thumbnail(pid: str, cid: str) -> FileResponse:
        store.clip(pid, cid)
        return _file(store.thumb_path(pid, cid), "image/jpeg")

    @app.get("/api/projects/{pid}/clips/{cid}/media/{kind}")
    def clip_media(pid: str, cid: str, kind: str, download: bool = False) -> FileResponse:
        clip = store.clip(pid, cid)
        names = {"clip": "clip.mp4", "output": "output.mp4", "debug": "debug.mp4"}
        if kind not in names:
            raise KeyError(kind)
        path = store.clip_dir(pid, cid) / names[kind]
        project = store.get(pid)
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in project["name"]).strip() or "clip"
        dl = f"{safe} - clip {clip['rank'] + 1} ({kind}).mp4" if download else None
        return _file(path, "video/mp4", dl)

    @app.get("/api/projects/{pid}/clips/{cid}/artifact/{name}")
    def clip_artifact(pid: str, cid: str, name: str) -> FileResponse:
        store.clip(pid, cid)
        allowed = {"moment": "moment.json", "speakers": "phase2/speakers.json", "crop_plan": "phase2/crop_plan.json",
                   "manifest": "phase1/manifest.json", "run": "run.json"}
        if name not in allowed:
            raise KeyError(name)
        return _file(store.clip_dir(pid, cid) / allowed[name], "application/json", f"{cid}-{allowed[name].split('/')[-1]}")

    @app.post("/api/projects/{pid}/export")
    def export_zip(pid: str, payload: dict[str, Any] = Body(default={})) -> StreamingResponse:
        project = store.get(pid)
        ids = payload.get("clip_ids") or [c["id"] for c in store.clips(pid) if c["status"] == "rendered"]
        files = []
        for cid in ids:
            clip = store.clip(pid, cid)
            out = store.clip_dir(pid, cid) / "output.mp4"
            if out.exists():
                files.append((f"clip_{clip['rank'] + 1:02d}_{int(clip['start_ms'] // 1000)}s.mp4", out))
        if not files:
            raise ValueError("none of the requested clips has a rendered output")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:  # mp4 is already compressed
            for name, path in files:
                zf.write(path, name)
        buf.seek(0)
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in project["name"]).strip() or "clips"
        return StreamingResponse(buf, media_type="application/zip",
                                 headers={"Content-Disposition": f'attachment; filename="{safe} - clips.zip"'})

    # -- static SPA --------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
    def spa(full_path: str) -> HTMLResponse:
        if full_path.startswith("api/"):
            raise HTTPException(404)
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    return app


def main() -> None:
    import argparse
    import threading
    import webbrowser

    import uvicorn

    ap = argparse.ArgumentParser(prog="facepipe-web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("FACEPIPE_PORT", "8765")))
    ap.add_argument("--data-dir", default=os.environ.get("FACEPIPE_DATA_DIR"))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(args.data_dir)
    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  facepipe web  →  {url}\n  data dir      →  {app.state.data_dir}\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
