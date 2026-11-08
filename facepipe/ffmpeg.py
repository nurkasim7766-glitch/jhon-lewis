"""Locate ffmpeg/ffprobe: system binaries first, then the imageio-ffmpeg bundled ffmpeg.

imageio-ffmpeg (BSD-2) ships ffmpeg but not ffprobe, so callers that need probing use
`probe_media`, which falls back to parsing `ffmpeg -i` output plus OpenCV.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ffmpeg not found on PATH and imageio-ffmpeg is unavailable; install ffmpeg") from exc


@lru_cache(maxsize=1)
def ffprobe_exe() -> str | None:
    return shutil.which("ffprobe")


def run_ffmpeg(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-1500:]}")
    return proc


def probe_media(path: str | Path) -> dict:
    """Return {duration, width, height, fps, frames, has_audio, video_codec} for a media file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    info: dict = {"duration": None, "width": None, "height": None, "fps": None, "frames": None,
                  "has_audio": False, "video_codec": None}
    probe = ffprobe_exe()
    if probe:
        proc = subprocess.run([probe, "-v", "error", "-show_entries",
                               "stream=codec_type,codec_name,width,height,nb_frames,r_frame_rate:format=duration",
                               "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode == 0:
            doc = json.loads(proc.stdout or "{}")
            for s in doc.get("streams", []):
                if s.get("codec_type") == "video" and info["width"] is None:
                    info["width"], info["height"] = int(s["width"]), int(s["height"])
                    info["video_codec"] = s.get("codec_name")
                    num, _, den = str(s.get("r_frame_rate", "0/1")).partition("/")
                    info["fps"] = float(num) / float(den or 1) if float(den or 1) else None
                    if s.get("nb_frames", "").isdigit():
                        info["frames"] = int(s["nb_frames"])
                elif s.get("codec_type") == "audio":
                    info["has_audio"] = True
            if doc.get("format", {}).get("duration"):
                info["duration"] = float(doc["format"]["duration"])
    if info["width"] is None:
        proc = run_ffmpeg(["-i", str(path)], check=False)
        err = proc.stderr
        info["has_audio"] = "Audio:" in err
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", err)
        if m:
            info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        m = re.search(r"Video: (\w+).*?, (\d{2,5})x(\d{2,5})", err)
        if m:
            info["video_codec"], info["width"], info["height"] = m.group(1), int(m.group(2)), int(m.group(3))
        m = re.search(r"(\d+\.?\d*) fps", err)
        if m:
            info["fps"] = float(m.group(1))
    if info["width"] is None:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            info["width"], info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["fps"] = cap.get(cv2.CAP_PROP_FPS) or None
            info["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            if info["fps"] and info["frames"] and not info["duration"]:
                info["duration"] = info["frames"] / info["fps"]
        cap.release()
    if info["width"] is None:
        raise ValueError(f"{path.name} is not a decodable video")
    return info
