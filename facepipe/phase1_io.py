"""Phase 1 output loading and face-track utilities shared by Phase 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facepipe.contract import FrameRecord, iter_frames_jsonl, validate_frame, validate_manifest


@dataclass
class Phase1Data:
    manifest: dict
    frames: list[FrameRecord]

    @property
    def width(self) -> int:
        return int(self.manifest["video"]["width"])

    @property
    def height(self) -> int:
        return int(self.manifest["video"]["height"])

    @property
    def fps(self) -> float:
        return float(self.manifest["video"]["fps"])

    @property
    def face_ids(self) -> list[int]:
        return [int(i) for i in self.manifest["face_ids"]]

    def face_center_px(self, frame_index: int, face_id: int) -> tuple[float, float] | None:
        for face in self.frames[frame_index].faces:
            if face.face_id == face_id:
                b = face.bbox
                return ((b.x_min + b.x_max) / 2 * self.width, (b.y_min + b.y_max) / 2 * self.height)
        return None

    def face_bbox_px(self, frame_index: int, face_id: int) -> tuple[float, float, float, float] | None:
        for face in self.frames[frame_index].faces:
            if face.face_id == face_id:
                b = face.bbox
                return (b.x_min * self.width, b.y_min * self.height, b.x_max * self.width, b.y_max * self.height)
        return None

    def face_area(self, frame_index: int, face_id: int) -> float:
        bb = self.face_bbox_px(frame_index, face_id)
        return 0.0 if bb is None else (bb[2] - bb[0]) * (bb[3] - bb[1])


def load_phase1(out_dir: str | Path) -> Phase1Data:
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    frames: list[FrameRecord] = []
    with (out_dir / manifest["frames_file"]).open(encoding="utf-8") as fh:
        for i, f in enumerate(iter_frames_jsonl(fh)):
            validate_frame(f, expected_index=i)
            frames.append(FrameRecord.from_dict(f))
    if len(frames) != manifest["video"]["frame_count"]:
        raise ValueError("frames.jsonl length disagrees with manifest frame_count")
    return Phase1Data(manifest=manifest, frames=frames)


def nan_series(data: Phase1Data, face_id: int, fn) -> np.ndarray:
    """Evaluate fn(face) per frame for one face_id; NaN where the face is absent."""
    out = np.full(len(data.frames), np.nan, dtype=np.float64)
    for i, fr in enumerate(data.frames):
        for face in fr.faces:
            if face.face_id == face_id:
                out[i] = fn(face)
                break
    return out
