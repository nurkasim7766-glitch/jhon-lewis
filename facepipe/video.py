"""Video decoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    reported_frame_count: int


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoMeta]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0  # containers without timing info; timestamps become nominal
    meta = VideoMeta(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(fps),
        reported_frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, meta


def iter_frames_rgb(cap: cv2.VideoCapture) -> Iterator[tuple[int, np.ndarray]]:
    index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            return
        yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        index += 1
