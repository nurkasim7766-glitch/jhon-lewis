"""Active-speaker decision with hysteresis and speaker-change segmentation.

Flicker control, in order:
1. Hysteresis: the current speaker keeps the floor while its score stays above
   `hold_ratio * threshold`; a challenger must beat the threshold *and* the current
   speaker's score for `switch_confirm_ms` consecutive frames to take over.
2. Minimum duration: segments shorter than `min_segment_ms` are absorbed into a
   neighbour (the longer neighbour by default, so a brief interjection disappears
   rather than splitting the surrounding speech).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from facepipe.asd.base import ASDResult
from facepipe.contract_phase2 import Segment
from facepipe.phase1_io import Phase1Data


@dataclass
class SegmentationConfig:
    min_segment_ms: float = 1500.0
    switch_confirm_ms: float = 300.0
    hold_ratio: float = 0.6
    fallback: str = "largest_face"  # speaker when nobody is speaking and none is held

    def to_dict(self) -> dict:
        return {"min_segment_ms": self.min_segment_ms, "switch_confirm_ms": self.switch_confirm_ms,
                "hold_ratio": self.hold_ratio, "fallback": self.fallback}


@dataclass
class _State:
    current: int | None = None
    challenger: int | None = None
    challenger_frames: int = 0
    history: list = field(default_factory=list)


def decide_active(data: Phase1Data, asd: ASDResult, cfg: SegmentationConfig) -> list[int | None]:
    """Per-frame active speaker before minimum-duration enforcement."""
    thr = asd.speaking_threshold
    confirm = max(1, int(round(cfg.switch_confirm_ms / 1000.0 * data.fps)))
    st = _State()
    out: list[int | None] = []
    for i, fr in enumerate(data.frames):
        visible = [f.face_id for f in fr.faces]
        sc = {fid: asd.scores[fid][i] for fid in visible if not np.isnan(asd.scores[fid][i])}
        speaking = {fid: s for fid, s in sc.items() if s >= thr}

        cur_score = sc.get(st.current, np.nan) if st.current is not None else np.nan
        cur_visible = st.current in visible
        cur_holding = cur_visible and (np.isnan(cur_score) or cur_score >= cfg.hold_ratio * thr)

        best = max(speaking, key=speaking.get) if speaking else None
        if best is not None and best != st.current and (not cur_holding or speaking[best] > cur_score):
            if best == st.challenger:
                st.challenger_frames += 1
            else:
                st.challenger, st.challenger_frames = best, 1
            if st.challenger_frames >= confirm or not cur_visible:
                st.current, st.challenger, st.challenger_frames = best, None, 0
        else:
            st.challenger, st.challenger_frames = None, 0

        if st.current is None or not cur_visible and st.current not in visible:
            if best is not None:
                st.current = best
            elif visible and cfg.fallback == "largest_face":
                st.current = max(visible, key=lambda fid: data.face_area(i, fid))
            elif visible:
                st.current = visible[0]
            else:
                st.current = None
        out.append(st.current)
    return out


def _runs(active: list[int | None]) -> list[tuple[int, int, int | None]]:
    runs = []
    start = 0
    for i in range(1, len(active) + 1):
        if i == len(active) or active[i] != active[start]:
            runs.append((start, i - 1, active[start]))
            start = i
    return runs


def enforce_min_duration(active: list[int | None], fps: float, min_segment_ms: float) -> list[int | None]:
    min_frames = int(np.ceil(min_segment_ms / 1000.0 * fps))
    runs = _runs(active)
    while len(runs) > 1:
        lengths = [e - s + 1 for s, e, _ in runs]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_frames:
            break
        # Absorb the shortest run into its longer neighbour, then merge equal neighbours.
        if k == 0:
            target = 1
        elif k == len(runs) - 1:
            target = k - 1
        else:
            target = k - 1 if lengths[k - 1] >= lengths[k + 1] else k + 1
        s, e, _ = runs[k]
        runs[k] = (s, e, runs[target][2])
        merged = []
        for r in runs:
            if merged and merged[-1][2] == r[2]:
                merged[-1] = (merged[-1][0], r[1], r[2])
            else:
                merged.append(r)
        runs = merged
    out: list[int | None] = []
    for s, e, fid in runs:
        out.extend([fid] * (e - s + 1))
    return out


def build_segments(active: list[int | None], fps: float) -> list[Segment]:
    segments = []
    for sid, (s, e, fid) in enumerate(_runs(active)):
        segments.append(Segment(
            segment_id=sid, start_frame=s, end_frame=e,
            start_ms=round(s * 1000.0 / fps, 3), end_ms=round((e + 1) * 1000.0 / fps, 3),
            speaker_face_id=fid,
        ))
    return segments


def segment_speakers(data: Phase1Data, asd: ASDResult, cfg: SegmentationConfig) -> tuple[list[int | None], list[Segment]]:
    raw = decide_active(data, asd, cfg)
    active = enforce_min_duration(raw, data.fps, cfg.min_segment_ms) if raw else []
    return active, build_segments(active, data.fps)
