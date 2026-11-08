"""Synthetic Phase 1 data with scripted speech, for Phase 2 tests and verification.

Faces are placed left/right in a 1920x1080 frame. A face "talks" by oscillating its
mouth aperture (landmarks 13/14) while its position jitters slightly; a quiet face has
a nearly constant aperture. The ground-truth speaker timeline is returned alongside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from facepipe.contract import (
    LEFT_MOUTH_CORNER,
    LOWER_LIP_CENTER,
    NUM_LANDMARKS,
    RIGHT_MOUTH_CORNER,
    UPPER_LIP_CENTER,
    BBox,
    FaceRecord,
    FrameRecord,
)
from facepipe.phase1_io import Phase1Data


@dataclass
class SpeechSpan:
    face_id: int
    start_ms: float
    end_ms: float


def _face(face_id: int, cx: float, cy: float, size: float, aperture: float, W: int, H: int) -> FaceRecord:
    # Face box in normalized coords; landmarks scattered inside it, mouth points explicit.
    half_w, half_h = size / 2 / W, size * 1.3 / 2 / H
    rng = np.random.default_rng(face_id)
    pts = np.column_stack([
        rng.uniform(cx / W - half_w, cx / W + half_w, NUM_LANDMARKS),
        rng.uniform(cy / H - half_h, cy / H + half_h, NUM_LANDMARKS),
        np.zeros(NUM_LANDMARKS),
    ])
    mouth_y = cy / H + half_h * 0.45
    mouth_w = size * 0.45
    pts[LEFT_MOUTH_CORNER] = (cx / W - mouth_w / 2 / W, mouth_y, 0)
    pts[RIGHT_MOUTH_CORNER] = (cx / W + mouth_w / 2 / W, mouth_y, 0)
    pts[UPPER_LIP_CENTER] = (cx / W, mouth_y - aperture * mouth_w / 2 / H, 0)
    pts[LOWER_LIP_CENTER] = (cx / W, mouth_y + aperture * mouth_w / 2 / H, 0)
    bbox = BBox(float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
    return FaceRecord(face_id=face_id, bbox=bbox, landmarks=pts.round(6).tolist())


def make_two_speaker_data(
    duration_ms: float = 10_000.0,
    fps: float = 30.0,
    spans: list[SpeechSpan] | None = None,
    jitter_px: float = 3.0,
    drift_px: float = 60.0,
    seed: int = 0,
    W: int = 1920,
    H: int = 1080,
) -> tuple[Phase1Data, list[SpeechSpan]]:
    spans = spans if spans is not None else [
        SpeechSpan(0, 0, 4000),
        SpeechSpan(1, 2000, 2400),  # short interjection: must NOT become a segment
        SpeechSpan(1, 4000, 8000),
        SpeechSpan(0, 8000, 10_000),
    ]
    rng = np.random.default_rng(seed)
    n = int(round(duration_ms / 1000.0 * fps))
    frames: list[FrameRecord] = []
    base = {0: (W * 0.28, H * 0.5), 1: (W * 0.72, H * 0.5)}
    for i in range(n):
        t_ms = i * 1000.0 / fps
        faces = []
        for fid in (0, 1):
            talking = any(s.face_id == fid and s.start_ms <= t_ms < s.end_ms for s in spans)
            # 4 Hz mouth oscillation while talking, closed-ish otherwise.
            aperture = 0.25 + 0.25 * np.sin(2 * np.pi * 4 * t_ms / 1000.0 + fid) if talking else 0.05
            cx = base[fid][0] + drift_px * np.sin(2 * np.pi * t_ms / 6000.0 + fid) + rng.normal(0, jitter_px)
            cy = base[fid][1] + rng.normal(0, jitter_px)
            faces.append(_face(fid, cx, cy, 260.0, aperture, W, H))
        frames.append(FrameRecord(frame_index=i, timestamp_ms=round(t_ms, 3), faces=faces))
    manifest = {
        "schema_version": "1.0",
        "video": {"path": "synthetic://two_speakers", "width": W, "height": H, "fps": fps,
                  "frame_count": n, "duration_ms": round(n * 1000.0 / fps, 3)},
        "detector": {"backend": "mediapipe", "task": "synthetic"},
        "num_landmarks": NUM_LANDMARKS,
        "lip_landmark_indices": list(__import__("facepipe.contract", fromlist=["LIP_LANDMARK_INDICES"]).LIP_LANDMARK_INDICES),
        "frames_with_faces": n,
        "face_ids": [0, 1],
        "frames_file": "frames.jsonl",
    }
    return Phase1Data(manifest=manifest, frames=frames), spans


def truth_timeline(spans: list[SpeechSpan], n: int, fps: float, min_ms: float) -> list[int]:
    """Expected dominant speaker per frame, ignoring spans shorter than min_ms."""
    out = []
    for i in range(n):
        t = i * 1000.0 / fps
        active = [s.face_id for s in spans if s.start_ms <= t < s.end_ms and s.end_ms - s.start_ms >= min_ms]
        out.append(active[0] if active else out[-1] if out else 0)
    return out
