"""Default ASD: lip-motion heuristic from Phase 1 landmarks (no audio, no ML).

Mouth aperture = |upper lip centre - lower lip centre| / |left corner - right corner|
in pixel space, so it is invariant to face size and camera distance. Talking shows up
as sustained variability of the aperture, so the score is the rolling standard
deviation of the (median-filtered) aperture over a short centred window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from facepipe.asd.base import ASDBackend, ASDResult
from facepipe.contract import LEFT_MOUTH_CORNER, LOWER_LIP_CENTER, RIGHT_MOUTH_CORNER, UPPER_LIP_CENTER, FaceRecord
from facepipe.phase1_io import Phase1Data, nan_series


def mouth_aperture(face: FaceRecord, width: int, height: int) -> float:
    lm = face.landmarks
    up, lo = lm[UPPER_LIP_CENTER], lm[LOWER_LIP_CENTER]
    left, right = lm[LEFT_MOUTH_CORNER], lm[RIGHT_MOUTH_CORNER]
    opening = np.hypot((lo[0] - up[0]) * width, (lo[1] - up[1]) * height)
    mouth_w = np.hypot((right[0] - left[0]) * width, (right[1] - left[1]) * height)
    return float(opening / mouth_w) if mouth_w > 1e-6 else 0.0


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling std that ignores NaNs; NaN where the centre sample is NaN."""
    n = len(x)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        if np.isnan(x[i]):
            continue
        seg = x[max(0, i - half): min(n, i + half + 1)]
        seg = seg[~np.isnan(seg)]
        out[i] = seg.std() if len(seg) >= 3 else 0.0
    return out


class LipMotionASD(ASDBackend):
    name = "heuristic"

    def __init__(self, window_ms: float = 500.0, median_ms: float = 100.0, speaking_threshold: float = 0.03):
        self.window_ms = window_ms
        self.median_ms = median_ms
        self.speaking_threshold = speaking_threshold

    def score(self, data: Phase1Data, video_path: Path | None = None) -> ASDResult:
        fps = data.fps
        window = max(3, int(round(self.window_ms / 1000.0 * fps)) | 1)
        med = max(1, int(round(self.median_ms / 1000.0 * fps)) | 1)
        scores: dict[int, np.ndarray] = {}
        for fid in data.face_ids:
            ap = nan_series(data, fid, lambda f: mouth_aperture(f, data.width, data.height))
            if med > 1:
                # Fill gaps for filtering, then restore NaNs so absence stays absence.
                mask = np.isnan(ap)
                if mask.all():
                    scores[fid] = ap
                    continue
                filled = ap.copy()
                idx = np.arange(len(ap))
                filled[mask] = np.interp(idx[mask], idx[~mask], ap[~mask])
                filtered = median_filter(filled, size=med, mode="nearest")
                filtered[mask] = np.nan
                ap = filtered
            scores[fid] = rolling_std(ap, window)
        return ASDResult(
            backend=self.name,
            params={"window_ms": self.window_ms, "median_ms": self.median_ms, "metric": "rolling_std(mouth_aperture)"},
            speaking_threshold=self.speaking_threshold,
            scores=scores,
        )
