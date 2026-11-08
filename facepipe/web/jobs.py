"""Background job runner: one worker thread, a FIFO queue, cooperative cancellation.

Job kinds
  analyze  transcript -> ranked non-overlapping moments -> clips table + thumbnails
  process  for each selected clip: cut, detect, speakers (asd/segmentation/cropplan),
           render, debug — via facepipe.pipeline.Pipeline, so finished stages are reused
  render   force re-render (render + debug) of given clips; detection untouched

Progress is a JSON snapshot persisted on the job row so the UI can poll it and it
survives restarts. Every number in it comes from the engine's own callbacks.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from facepipe.pipeline import Pipeline, PipelineCancelled
from facepipe.web.db import Database, dumps, loads, new_id
from facepipe.web.projects import ProjectSettings, ProjectStore

log = logging.getLogger("facepipe.web.jobs")

# UI step catalogue: (id, label). Steps map 1:1 onto engine callbacks; none is synthetic.
STEPS: list[tuple[str, str]] = [
    ("ingest", "Video ingestion"),
    ("transcript", "Transcript processing"),
    ("moments", "AI moment selection"),
    ("clip", "Clip extraction"),
    ("detect", "Face detection & landmark tracking"),
    ("asd", "Active-speaker detection"),
    ("segmentation", "Speaker segmentation"),
    ("cropplan", "Camera crop planning"),
    ("render", "Clip rendering"),
    ("debug", "Debug overlay"),
    ("output", "Final output"),
]
STEP_IDS = [s for s, _ in STEPS]


class JobCancelled(Exception):
    pass


class JobManager:
    def __init__(self, db: Database, store: ProjectStore):
        self.db = db
        self.store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self._current: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="facepipe-worker", daemon=True)
        self._recover()
        self._thread.start()

    # -- lifecycle ---------------------------------------------------------------
    def _recover(self) -> None:
        """Jobs left 'running' by a crash cannot be resumed mid-stage; mark and re-queue queued ones."""
        now = time.time()
        for j in self.db.query("SELECT id, project_id FROM jobs WHERE status='running'"):
            self.db.execute("UPDATE jobs SET status='interrupted', finished_at=?, error=? WHERE id=?",
                            (now, "application restarted while the job was running; finished stages are kept", j["id"]))
            self.db.execute("UPDATE clips SET status='candidate' WHERE project_id=? AND status='processing'", (j["project_id"],))
            self.db.execute("UPDATE projects SET status=CASE WHEN status IN ('analyzing','processing') THEN 'ready' ELSE status END WHERE id=?",
                            (j["project_id"],))
            self.db.log(j["project_id"], "job", "Job interrupted by restart")
        for j in self.db.query("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at"):
            self._queue.put(j["id"])

    def shutdown(self) -> None:
        self._stop.set()

    # -- API ---------------------------------------------------------------------
    def submit(self, project_id: str, kind: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        active = self.db.one("SELECT id FROM jobs WHERE project_id=? AND status IN ('queued','running')", (project_id,))
        if active:
            raise ValueError("this project already has a job queued or running")
        jid = new_id()
        self.db.execute("INSERT INTO jobs(id,project_id,kind,status,params,progress,created_at) VALUES (?,?,?,?,?,?,?)",
                        (jid, project_id, kind, "queued", dumps(params or {}), dumps({"steps": {}, "message": "queued"}),
                         time.time()))
        self.db.log(project_id, "job", f"Queued {kind} job")
        self._queue.put(jid)
        return self.get(jid)

    def get(self, job_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not row:
            raise KeyError(job_id)
        row["params"] = loads(row["params"])
        row["progress"] = loads(row["progress"])
        return row

    def list(self, project_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if project_id:
            rows = self.db.query("SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT ?", (project_id, limit))
        else:
            rows = self.db.query("SELECT j.*, p.name AS project_name FROM jobs j LEFT JOIN projects p ON p.id=j.project_id "
                                 "ORDER BY j.created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["params"] = loads(r["params"])
            r["progress"] = loads(r["progress"])
        return rows

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] == "queued":
            self.db.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
                            (time.time(), job_id))
            self._restore_project(job["project_id"])
            self.db.log(job["project_id"], "job", "Cancelled queued job")
        elif job["status"] == "running":
            with self._lock:
                self._cancel.add(job_id)
        return self.get(job_id)

    def _cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel or self._stop.is_set()

    # -- worker ------------------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                jid = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            job = self.db.one("SELECT * FROM jobs WHERE id=?", (jid,))
            if not job or job["status"] != "queued":
                continue
            self._current = jid
            self.db.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (time.time(), jid))
            try:
                handler = {"analyze": self._run_analyze, "process": self._run_process, "render": self._run_render}[job["kind"]]
                handler(jid, job["project_id"], loads(job["params"]))
                self.db.execute("UPDATE jobs SET status='done', finished_at=? WHERE id=?", (time.time(), jid))
                self.db.log(job["project_id"], "job", f"Finished {job['kind']} job")
            except (JobCancelled, PipelineCancelled):
                self.db.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?", (time.time(), jid))
                self.db.execute("UPDATE clips SET status='candidate' WHERE project_id=? AND status='processing'", (job["project_id"],))
                self._restore_project(job["project_id"])
                self.db.log(job["project_id"], "job", f"Cancelled {job['kind']} job")
            except Exception as exc:  # noqa: BLE001 - surfaced to the user as job error
                log.exception("job %s failed", jid)
                msg = f"{type(exc).__name__}: {exc}"
                self.db.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?", (time.time(), msg, jid))
                self.db.execute("UPDATE clips SET status='error', error=? WHERE project_id=? AND status='processing'",
                                (msg, job["project_id"]))
                self.store.touch(job["project_id"], status="error", error=msg)
                self.db.log(job["project_id"], "error", f"{job['kind']} failed: {msg[:200]}")
                self._set_progress(jid, message=msg, traceback=traceback.format_exc()[-3000:])
            finally:
                with self._lock:
                    self._cancel.discard(jid)
                self._current = None

    def _restore_project(self, project_id: str) -> None:
        has_clips = self.db.one("SELECT COUNT(*) AS n FROM clips WHERE project_id=?", (project_id,))["n"]
        self.store.touch(project_id, status="analyzed" if has_clips else "ready")

    # -- progress ----------------------------------------------------------------
    def _set_progress(self, job_id: str, **patch: Any) -> dict[str, Any]:
        job = self.get(job_id)
        prog = {**job["progress"], **patch, "updated_at": time.time()}
        self.db.execute("UPDATE jobs SET progress=? WHERE id=?", (dumps(prog), job_id))
        return prog

    def _step(self, job_id: str, step: str, status: str, **extra: Any) -> None:
        prog = self.get(job_id)["progress"]
        steps = prog.get("steps", {})
        entry = {**steps.get(step, {}), "status": status, **extra}
        if status == "running" and "started_at" not in entry:
            entry["started_at"] = time.time()
        if status in ("done", "reused", "failed"):
            entry["finished_at"] = time.time()
        steps[step] = entry
        self._set_progress(job_id, steps=steps, current_step=step if status == "running" else prog.get("current_step"))

    # -- handlers ----------------------------------------------------------------
    def _run_analyze(self, job_id: str, project_id: str, params: dict[str, Any]) -> None:
        from facepipe.moments import get_scorer
        from facepipe.moments.select import select_moments
        from facepipe.transcript import load_transcript

        project = self.store.get(project_id)
        src = self.store.source_path(project)
        tpath = self.store.transcript_path(project_id)
        if not src or not tpath.exists():
            raise ValueError("project needs both a video and a transcript before analysis")
        settings = ProjectSettings.from_dict(project["settings"])
        self.store.touch(project_id, status="analyzing", error=None)

        self._step(job_id, "ingest", "running", message=f"{project['source_filename']}")
        info = project["source_info"] or {}
        self._step(job_id, "ingest", "done", message=f"{info.get('width')}x{info.get('height')}, {info.get('duration', 0):.0f}s")

        self._step(job_id, "transcript", "running")
        transcript = load_transcript(tpath)
        self._step(job_id, "transcript", "done", message=f"{len(transcript.sentences)} sentences")
        if self._cancelled(job_id):
            raise JobCancelled()

        self._step(job_id, "moments", "running", message=f"scorer={settings.scorer}")
        scorer = get_scorer(settings.scorer, **settings.llm_kwargs())
        moments = select_moments(transcript, src, settings.selection_config(), scorer, max_clips=settings.max_clips)
        clips = self.store.replace_clips(project_id, moments)
        self._step(job_id, "moments", "done", message=f"{len(clips)} non-overlapping moments")

        for clip in clips:
            if self._cancelled(job_id):
                raise JobCancelled()
            self.store.make_thumbnail(project, clip)
        self.store.touch(project_id, status="analyzed")
        self._set_progress(job_id, message=f"found {len(clips)} moments", clips=[c["id"] for c in clips])
        self.db.log(project_id, "analysis", f"Discovered {len(clips)} candidate moments")

    def _clip_pipeline(self, job_id: str, project: dict[str, Any], clip: dict[str, Any], force: set[str]) -> Pipeline:
        from facepipe.moments.select import write_moment

        settings = ProjectSettings.from_dict(project["settings"])
        src = self.store.source_path(project)
        workdir = self.store.clip_dir(project["id"], clip["id"])
        workdir.mkdir(parents=True, exist_ok=True)
        moment_path = workdir / "moment.json"
        if not moment_path.exists():
            write_moment(self.store.clip_moment_doc(project["id"], clip["id"]), moment_path)
        cfg = settings.pipeline_config(src, self.store.transcript_path(project["id"]), workdir)
        cid = clip["id"]

        def on_stage(stage: str, event: str) -> None:
            if stage == "moment":
                return
            status = {"start": "running", "done": "done", "reused": "reused"}[event]
            self._step(job_id, stage, status, clip=cid)

        def progress(stage: str, done: int, total: int) -> None:
            if done == total or done % 10 == 0:
                self._step(job_id, stage, "running", clip=cid, done=done, total=total)

        def on_step(step: str) -> None:
            # asd -> segmentation -> cropplan run inside the single "speakers" stage
            order = ["asd", "segmentation", "cropplan"]
            for prev in order[: order.index(step)]:
                self._step(job_id, prev, "done", clip=cid)
            self._step(job_id, step, "running", clip=cid)

        return Pipeline(cfg, force=force, on_stage=on_stage, progress=progress, on_step=on_step,
                        cancel_check=lambda: self._cancelled(job_id))

    def _run_clips(self, job_id: str, project_id: str, clip_ids: list[str], force: set[str]) -> None:
        project = self.store.get(project_id)
        settings = ProjectSettings.from_dict(project["settings"])
        self.store.touch(project_id, status="processing", error=None)
        total = len(clip_ids)
        for n, cid in enumerate(clip_ids, start=1):
            if self._cancelled(job_id):
                raise JobCancelled()
            clip = self.store.clip(project_id, cid)
            self.store.set_clip_status(cid, "processing")
            # reset per-clip step statuses so the UI shows this clip's real state
            prog = self.get(job_id)["progress"]
            steps = {k: v for k, v in prog.get("steps", {}).items() if k in ("ingest", "transcript", "moments")}
            self._set_progress(job_id, steps=steps, message=f"clip {n}/{total}", clip_index=n, clip_total=total, clip=cid)
            try:
                pipeline = self._clip_pipeline(job_id, project, clip, force)
                pipeline.run(until="debug" if settings.debug_overlay else "render")
                # speakers stage callbacks leave the last sub-step running
                for s in ("asd", "segmentation", "cropplan"):
                    self._step(job_id, s, "done", clip=cid)
                if not settings.debug_overlay:
                    self._step(job_id, "debug", "reused", clip=cid, message="disabled in settings")
                self._step(job_id, "output", "done", clip=cid, message="output.mp4 ready")
                self.store.set_clip_status(cid, "rendered")
                self.db.log(project_id, "render", f"Rendered clip {n}/{total} ({clip['duration_ms'] / 1000:.0f}s)")
            except (JobCancelled, PipelineCancelled):
                raise
            except Exception as exc:
                self.store.set_clip_status(cid, "error", f"{type(exc).__name__}: {exc}")
                raise
        self.store.touch(project_id, status="analyzed")

    def _run_process(self, job_id: str, project_id: str, params: dict[str, Any]) -> None:
        ids = params.get("clip_ids") or [c["id"] for c in self.store.clips(project_id) if c["selected"]]
        if not ids:
            raise ValueError("no clips selected")
        self._run_clips(job_id, project_id, ids, force=set(params.get("force") or ()))

    def _run_render(self, job_id: str, project_id: str, params: dict[str, Any]) -> None:
        ids = params.get("clip_ids") or [c["id"] for c in self.store.clips(project_id) if c["status"] == "rendered"]
        if not ids:
            raise ValueError("no rendered clips to re-render")
        self._run_clips(job_id, project_id, ids, force={"render"})
