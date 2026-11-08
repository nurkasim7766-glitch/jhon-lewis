"""Phase 3 renderer: crop_plan.json + source clip -> output video with original audio."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import cv2

from facepipe.contract_phase2 import CropPlan, validate_crop_plan
from facepipe.ffmpeg import ffmpeg_exe, probe_media


class RenderError(RuntimeError):
    pass


def load_plan(path: str | Path) -> CropPlan:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_crop_plan(d)
    return CropPlan.from_dict(d)


def has_audio_stream(video: Path) -> bool:
    try:
        return bool(probe_media(video)["has_audio"])
    except (ValueError, FileNotFoundError):
        return False


def mux(video_only: Path, audio_source: Path, out_path: Path, crf: int = 18, preset: str = "medium") -> None:
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video_only)]
    if has_audio_stream(audio_source):
        cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "160k", "-shortest"]
    else:
        cmd += ["-map", "0:v:0", "-an"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"ffmpeg mux failed: {proc.stderr.strip()}")


def render(plan_path: str | Path, clip_path: str | Path, out_path: str | Path, crf: int = 18,
           preset: str = "medium", progress: Callable[[int, int], None] | None = None) -> Path:
    """Apply the plan frame by frame. The plan's `source.path` is informational; `clip_path` is used."""
    plan = load_plan(plan_path)
    clip_path, out_path = Path(clip_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ow, oh = plan.output.width, plan.output.height
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RenderError(f"cannot open {clip_path}")
    fps = float(plan.source["fps"])

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "video_only.mp4"
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        n = len(plan.frames)
        for i, f in enumerate(plan.frames):
            ok, bgr = cap.read()
            if not ok:
                raise RenderError(f"clip ended at frame {i}; plan has {n} frames")
            c = f.crop
            crop = bgr[c.y: c.y + c.h, c.x: c.x + c.w]
            writer.write(cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_AREA if c.w > ow else cv2.INTER_CUBIC))
            if progress:
                progress(i + 1, n)
        cap.release()
        writer.release()
        mux(tmp, clip_path, out_path, crf=crf, preset=preset)
    return out_path


def probe(video: Path) -> dict:
    """ffprobe-shaped summary ({streams:[...], format:{duration}}) built from probe_media."""
    info = probe_media(video)
    streams = [{"codec_type": "video", "width": info["width"], "height": info["height"],
                "nb_frames": str(info["frames"] or 0)}]
    if info["has_audio"]:
        streams.append({"codec_type": "audio"})
    return {"streams": streams, "format": {"duration": str(info["duration"] or 0.0)}}
