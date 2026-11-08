"""Project model, file layout, settings, and the engine-facing operations.

Layout: <data_dir>/projects/<project_id>/
    source.<ext>            uploaded video
    transcript.json         uploaded/pasted transcript (5.4)
    thumbs/<clip_id>.jpg    candidate thumbnails
    clips/<clip_id>/        one Pipeline work directory per clip:
        moment.json, clip.mp4, phase1/, phase2/, output.mp4, debug.mp4, run.json
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from facepipe.cropplan import CropConfig
from facepipe.detector import DetectorConfig
from facepipe.ffmpeg import ffmpeg_exe, probe_media
from facepipe.moments import SelectionConfig
from facepipe.pipeline import PipelineConfig
from facepipe.segmentation import SegmentationConfig
from facepipe.web.db import Database, dumps, loads, new_id

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".wmv"}
TRANSCRIPT_EXTENSIONS = {".json"}


@dataclass
class ProjectSettings:
    """User-facing knobs; everything else uses engine defaults."""

    min_clip_s: float = 30.0
    max_clip_s: float = 60.0
    max_clips: int = 5
    scorer: str = "heuristic"  # heuristic | llm
    llm_endpoint: str = "http://127.0.0.1:11434/v1/chat/completions"
    llm_model: str = "llama3.1"
    allow_hosted_llm: bool = False
    asd_backend: str = "heuristic"  # heuristic | lrasd
    min_segment_ms: float = 1500.0
    output_width: int = 1080
    output_height: int = 1920
    render_crf: int = 18
    render_preset: str = "medium"
    debug_overlay: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProjectSettings":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        s = cls(**known)
        s.validate()
        return s

    def validate(self) -> None:
        if not (5 <= self.min_clip_s < self.max_clip_s <= 600):
            raise ValueError("clip length range must satisfy 5s <= min < max <= 600s")
        if not (1 <= self.max_clips <= 20):
            raise ValueError("max_clips must be between 1 and 20")
        if self.scorer not in ("heuristic", "llm"):
            raise ValueError("scorer must be heuristic or llm")
        if self.asd_backend not in ("heuristic", "lrasd"):
            raise ValueError("asd_backend must be heuristic or lrasd")
        if self.output_width % 2 or self.output_height % 2 or not (240 <= self.output_width <= 4096 and 240 <= self.output_height <= 4096):
            raise ValueError("output size must be even and between 240 and 4096")
        if not (0 <= self.render_crf <= 51):
            raise ValueError("render_crf must be 0-51")
        if self.render_preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
            raise ValueError("unknown x264 preset")

    def selection_config(self) -> SelectionConfig:
        return SelectionConfig(min_ms=self.min_clip_s * 1000, max_ms=self.max_clip_s * 1000, scorer=self.scorer,
                               top_n=10)

    def llm_kwargs(self) -> dict[str, Any]:
        if self.scorer != "llm":
            return {}
        return {"endpoint": self.llm_endpoint, "model": self.llm_model, "allow_hosted": self.allow_hosted_llm}

    def pipeline_config(self, video: Path, transcript: Path, workdir: Path) -> PipelineConfig:
        return PipelineConfig(
            video=str(video), transcript=str(transcript), workdir=str(workdir),
            selection=self.selection_config(), detector=DetectorConfig(), asd_backend=self.asd_backend,
            segmentation=SegmentationConfig(min_segment_ms=self.min_segment_ms),
            crop=CropConfig(output_width=self.output_width, output_height=self.output_height),
            llm=self.llm_kwargs(), render_crf=self.render_crf, render_preset=self.render_preset,
            debug=self.debug_overlay,
        )


class ProjectStore:
    def __init__(self, db: Database, data_dir: Path):
        self.db = db
        self.data_dir = Path(data_dir)
        (self.data_dir / "projects").mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------------
    def project_dir(self, project_id: str) -> Path:
        return self.data_dir / "projects" / project_id

    def source_path(self, project: dict[str, Any]) -> Path | None:
        if not project.get("source_filename"):
            return None
        ext = Path(project["source_filename"]).suffix.lower()
        p = self.project_dir(project["id"]) / f"source{ext}"
        return p if p.exists() else None

    def transcript_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "transcript.json"

    def clip_dir(self, project_id: str, clip_id: str) -> Path:
        return self.project_dir(project_id) / "clips" / clip_id

    def thumb_path(self, project_id: str, clip_id: str) -> Path:
        return self.project_dir(project_id) / "thumbs" / f"{clip_id}.jpg"

    # -- projects ----------------------------------------------------------------
    def create(self, name: str) -> dict[str, Any]:
        pid = new_id()
        now = time.time()
        self.db.execute("INSERT INTO projects(id,name,created_at,updated_at,status,settings) VALUES (?,?,?,?,?,?)",
                        (pid, name.strip() or "Untitled project", now, now, "new", dumps(asdict(ProjectSettings()))))
        self.project_dir(pid).mkdir(parents=True, exist_ok=True)
        self.db.log(pid, "project", f"Created project “{name}”")
        return self.get(pid)

    def get(self, project_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM projects WHERE id=?", (project_id,))
        if not row:
            raise KeyError(project_id)
        return self._hydrate(row)

    def list(self) -> list[dict[str, Any]]:
        return [self._hydrate(r) for r in self.db.query("SELECT * FROM projects ORDER BY updated_at DESC")]

    def _hydrate(self, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row["source_info"] = loads(row.get("source_info"), None)
        row["transcript_info"] = loads(row.get("transcript_info"), None)
        row["settings"] = asdict(ProjectSettings.from_dict(loads(row.get("settings"))))
        clips = self.db.query("SELECT status, selected FROM clips WHERE project_id=?", (row["id"],))
        row["clip_counts"] = {
            "total": len(clips),
            "selected": sum(1 for c in clips if c["selected"]),
            "rendered": sum(1 for c in clips if c["status"] == "rendered"),
        }
        job = self.db.one("SELECT id, kind, status, progress FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1",
                          (row["id"],))
        row["last_job"] = {**job, "progress": loads(job["progress"])} if job else None
        return row

    def touch(self, project_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE projects SET {cols} WHERE id=?", (*fields.values(), project_id))

    def rename(self, project_id: str, name: str) -> dict[str, Any]:
        self.get(project_id)
        self.touch(project_id, name=name.strip() or "Untitled project")
        self.db.log(project_id, "project", f"Renamed to “{name}”")
        return self.get(project_id)

    def update_settings(self, project_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        current = self.get(project_id)["settings"]
        merged = ProjectSettings.from_dict({**current, **patch})
        self.touch(project_id, settings=dumps(asdict(merged)))
        return self.get(project_id)

    def delete(self, project_id: str) -> None:
        self.get(project_id)
        self.db.execute("DELETE FROM projects WHERE id=?", (project_id,))
        shutil.rmtree(self.project_dir(project_id), ignore_errors=True)
        self.db.log(None, "project", f"Deleted project {project_id}")

    # -- uploads -----------------------------------------------------------------
    def attach_source(self, project_id: str, filename: str, tmp_path: Path) -> dict[str, Any]:
        ext = Path(filename).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"unsupported video format {ext or '(none)'}; use one of {', '.join(sorted(VIDEO_EXTENSIONS))}")
        try:
            info = probe_media(tmp_path)
        except (ValueError, FileNotFoundError) as exc:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"{filename} could not be decoded as a video: {exc}") from exc
        if not info["duration"] or info["duration"] < 5:
            tmp_path.unlink(missing_ok=True)
            raise ValueError("video is shorter than 5 seconds")
        # Replacing the source invalidates every clip derived from the old one.
        self._clear_clips(project_id)
        for old in self.project_dir(project_id).glob("source.*"):
            old.unlink(missing_ok=True)
        dest = self.project_dir(project_id) / f"source{ext}"
        shutil.move(str(tmp_path), dest)
        info["size_bytes"] = dest.stat().st_size
        project = self.get(project_id)
        status = "ready" if project.get("transcript_filename") else "uploaded"
        self.touch(project_id, source_filename=filename, source_info=dumps(info), status=status, error=None)
        self.db.log(project_id, "upload", f"Uploaded video {filename} ({info['duration']:.0f}s)")
        return self.get(project_id)

    def attach_transcript(self, project_id: str, filename: str, content: bytes) -> dict[str, Any]:
        from facepipe.transcript import TranscriptError, load_transcript

        dest = self.transcript_path(project_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp.json")
        tmp.write_bytes(content)
        try:
            t = load_transcript(tmp)
        except TranscriptError as exc:
            tmp.unlink(missing_ok=True)
            raise ValueError(str(exc)) from exc
        self._clear_clips(project_id)
        tmp.replace(dest)
        info = {"sentences": len(t.sentences), "duration_ms": t.duration_ms, "speakers": t.speakers,
                "words": sum(len(s.words) for s in t.sentences)}
        project = self.get(project_id)
        status = "ready" if project.get("source_filename") else project["status"]
        self.touch(project_id, transcript_filename=filename, transcript_info=dumps(info), status=status, error=None)
        self.db.log(project_id, "upload", f"Added transcript ({len(t.sentences)} sentences)")
        return self.get(project_id)

    def _clear_clips(self, project_id: str) -> None:
        self.db.execute("DELETE FROM clips WHERE project_id=?", (project_id,))
        shutil.rmtree(self.project_dir(project_id) / "clips", ignore_errors=True)
        shutil.rmtree(self.project_dir(project_id) / "thumbs", ignore_errors=True)

    # -- clips -------------------------------------------------------------------
    def replace_clips(self, project_id: str, moments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._clear_clips(project_id)
        now = time.time()
        with self.db.tx() as c:
            for rank, doc in enumerate(moments):
                c.execute("INSERT INTO clips(id,project_id,rank,selected,status,moment,created_at) VALUES (?,?,?,?,?,?,?)",
                          (new_id(), project_id, rank, 1 if rank < 3 else 0, "candidate", dumps(doc), now))
        return self.clips(project_id)

    def clips(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM clips WHERE project_id=? ORDER BY rank", (project_id,))
        return [self._hydrate_clip(r) for r in rows]

    def clip(self, project_id: str, clip_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM clips WHERE project_id=? AND id=?", (project_id, clip_id))
        if not row:
            raise KeyError(clip_id)
        return self._hydrate_clip(row)

    def _hydrate_clip(self, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        doc = loads(row.pop("moment"))
        m = doc["moment"]
        d = self.clip_dir(row["project_id"], row["id"])
        row.update({
            "start_ms": m["start_ms"], "end_ms": m["end_ms"], "duration_ms": m["duration_ms"],
            "sentence_start_ms": m["sentence_start_ms"], "sentence_end_ms": m["sentence_end_ms"],
            "first_sentence": m["first_sentence"], "last_sentence": m["last_sentence"],
            "score": m["score"], "features": m.get("features", {}), "text": m["text"],
            "scorer": doc["selection"]["scorer"].get("name"),
            "selected": bool(row["selected"]),
            "files": {
                "thumbnail": self.thumb_path(row["project_id"], row["id"]).exists(),
                "clip": (d / "clip.mp4").exists(),
                "phase1": (d / "phase1" / "manifest.json").exists(),
                "phase2": (d / "phase2" / "crop_plan.json").exists(),
                "output": (d / "output.mp4").exists(),
                "debug": (d / "debug.mp4").exists(),
            },
        })
        if row["files"]["output"]:
            row["output_size_bytes"] = (d / "output.mp4").stat().st_size
        speakers_file = d / "phase2" / "speakers.json"
        if speakers_file.exists():
            try:
                sp = json.loads(speakers_file.read_text(encoding="utf-8"))
                row["speaker_segments"] = len(sp["segments"])
                row["speaker_faces"] = sorted({s["speaker_face_id"] for s in sp["segments"] if s["speaker_face_id"] is not None})
            except (json.JSONDecodeError, KeyError):
                pass
        return row

    def clip_moment_doc(self, project_id: str, clip_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT moment FROM clips WHERE project_id=? AND id=?", (project_id, clip_id))
        if not row:
            raise KeyError(clip_id)
        return loads(row["moment"])

    def set_selected(self, project_id: str, clip_id: str, selected: bool) -> dict[str, Any]:
        self.clip(project_id, clip_id)
        self.db.execute("UPDATE clips SET selected=? WHERE id=?", (1 if selected else 0, clip_id))
        return self.clip(project_id, clip_id)

    def set_clip_status(self, clip_id: str, status: str, error: str | None = None) -> None:
        fields = {"status": status, "error": error}
        if status == "rendered":
            fields["rendered_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE clips SET {cols} WHERE id=?", (*fields.values(), clip_id))

    def make_thumbnail(self, project: dict[str, Any], clip: dict[str, Any]) -> Path | None:
        src = self.source_path(project)
        if not src:
            return None
        out = self.thumb_path(project["id"], clip["id"])
        out.parent.mkdir(parents=True, exist_ok=True)
        mid = (clip["start_ms"] + clip["end_ms"]) / 2000.0
        proc = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{mid:.3f}", "-i", str(src),
                               "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(out)],
                              capture_output=True, text=True)
        return out if proc.returncode == 0 and out.exists() else None

    # -- dashboard ---------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        projects = self.db.one("SELECT COUNT(*) AS n FROM projects")["n"]
        videos = self.db.one("SELECT COUNT(*) AS n FROM projects WHERE source_filename IS NOT NULL")["n"]
        clips = self.db.one("SELECT COUNT(*) AS n FROM clips WHERE status='rendered'")["n"]
        candidates = self.db.one("SELECT COUNT(*) AS n FROM clips")["n"]
        running = self.db.query("SELECT id, project_id, kind, status, progress FROM jobs WHERE status IN ('queued','running')")
        return {
            "projects": projects, "videos": videos, "clips_rendered": clips, "candidates": candidates,
            "active_jobs": [{**j, "progress": loads(j["progress"])} for j in running],
        }

    def activity(self, limit: int = 30) -> list[dict[str, Any]]:
        return self.db.query("SELECT a.*, p.name AS project_name FROM activity a LEFT JOIN projects p ON p.id=a.project_id "
                             "ORDER BY a.at DESC LIMIT ?", (limit,))
