"""Crop planning: a smoothed, in-bounds crop window that follows the active speaker.

Within a segment the window glides: the raw target (speaker face centre) passes
through a dead-zone (small moves are ignored), then a first-order follower with a
time constant, then a per-frame speed clamp. The first frame of a segment snaps to
the median target of its opening moments (a cut, not a pan). Finally a light
centred moving average removes residual jitter without crossing segment edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from facepipe.contract_phase2 import CropFrame, CropPlan, CropRect, CropSegment, OutputSpec, Segment
from facepipe.phase1_io import Phase1Data


@dataclass
class CropConfig:
    output_width: int = 1080
    output_height: int = 1920
    deadzone_frac: float = 0.06  # of crop width; target moves smaller than this are ignored
    time_constant_ms: float = 400.0  # follower response time
    max_speed_frac: float = 0.025  # of crop width per frame
    settle_ms: float = 400.0  # window whose median target sets the opening position
    post_smooth_ms: float = 160.0
    face_y_anchor: float = 0.42  # where the face centre sits vertically in the crop (0=top)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def crop_size(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int]:
    """Largest crop with the output aspect that fits inside the source."""
    aspect = out_w / out_h
    if src_w / src_h >= aspect:
        h = src_h
        w = int(round(h * aspect))
    else:
        w = src_w
        h = int(round(w / aspect))
    return max(1, min(w, src_w)), max(1, min(h, src_h))


def _targets(data: Phase1Data, seg: Segment, cw: int, ch: int, cfg: CropConfig) -> np.ndarray:
    """Desired crop top-left per frame of the segment (float, unclamped)."""
    n = seg.num_frames
    t = np.full((n, 2), np.nan)
    for k, fi in enumerate(range(seg.start_frame, seg.end_frame + 1)):
        if seg.speaker_face_id is None:
            continue
        c = data.face_center_px(fi, seg.speaker_face_id)
        if c is not None:
            t[k] = (c[0] - cw / 2, c[1] - ch * cfg.face_y_anchor)
    if np.isnan(t[:, 0]).all():
        t[:] = ((data.width - cw) / 2, (data.height - ch) / 2)
    else:  # hold the last known position across detection gaps, backfill the start
        for d in range(2):
            col = t[:, d]
            idx = np.arange(n)
            good = ~np.isnan(col)
            t[:, d] = np.interp(idx, idx[good], col[good])
    return t


def _clamp(pos: np.ndarray, data: Phase1Data, cw: int, ch: int) -> np.ndarray:
    pos = pos.copy()
    pos[:, 0] = np.clip(pos[:, 0], 0, data.width - cw)
    pos[:, 1] = np.clip(pos[:, 1], 0, data.height - ch)
    return pos


def smooth_path(targets: np.ndarray, fps: float, cw: int, cfg: CropConfig) -> np.ndarray:
    n = len(targets)
    if n == 0:
        return targets
    settle = max(1, int(round(cfg.settle_ms / 1000.0 * fps)))
    deadzone = cfg.deadzone_frac * cw
    alpha = 1.0 - np.exp(-1000.0 / (cfg.time_constant_ms * fps)) if cfg.time_constant_ms > 0 else 1.0
    max_step = cfg.max_speed_frac * cw

    pos = np.empty_like(targets)
    committed = np.median(targets[:settle], axis=0)
    pos[0] = committed
    for i in range(1, n):
        if np.linalg.norm(targets[i] - committed) > deadzone:
            committed = targets[i]
        step = alpha * (committed - pos[i - 1])
        norm = np.linalg.norm(step)
        if norm > max_step:
            step *= max_step / norm
        pos[i] = pos[i - 1] + step

    k = int(round(cfg.post_smooth_ms / 1000.0 * fps)) | 1
    if k >= 3 and n >= k:
        pad = k // 2
        padded = np.pad(pos, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(k) / k
        pos = np.stack([np.convolve(padded[:, d], kernel, mode="valid") for d in range(2)], axis=1)
    return pos


def plan_crops(data: Phase1Data, segments: list[Segment], cfg: CropConfig | None = None) -> CropPlan:
    cfg = cfg or CropConfig()
    cw, ch = crop_size(data.width, data.height, cfg.output_width, cfg.output_height)
    frames: list[CropFrame] = []
    crop_segments: list[CropSegment] = []
    for seg in segments:
        targets = _targets(data, seg, cw, ch, cfg)
        path = _clamp(smooth_path(_clamp(targets, data, cw, ch), data.fps, cw, cfg), data, cw, ch)
        for k, fi in enumerate(range(seg.start_frame, seg.end_frame + 1)):
            x, y = int(round(path[k, 0])), int(round(path[k, 1]))
            x, y = min(max(x, 0), data.width - cw), min(max(y, 0), data.height - ch)
            frames.append(CropFrame(frame_index=fi, timestamp_ms=data.frames[fi].timestamp_ms,
                                    segment_id=seg.segment_id, crop=CropRect(x, y, cw, ch)))
        crop_segments.append(CropSegment(seg.segment_id, seg.start_frame, seg.end_frame, seg.start_ms, seg.end_ms,
                                         seg.speaker_face_id, "cut"))
    v = data.manifest["video"]
    return CropPlan(
        source={k: v[k] for k in ("path", "width", "height", "fps", "frame_count", "duration_ms")},
        output=OutputSpec(cfg.output_width, cfg.output_height, _aspect_label(cfg.output_width, cfg.output_height)),
        smoothing=cfg.to_dict(),
        segments=crop_segments,
        frames=frames,
    )


def _aspect_label(w: int, h: int) -> str:
    from math import gcd

    g = gcd(w, h)
    return f"{w // g}:{h // g}"
