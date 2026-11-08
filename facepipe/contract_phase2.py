"""Phase 2 data contracts (sections 5.2 and 5.3).

5.2  speakers.json   per-frame speaking scores, the chosen active speaker, and the
                     speaker segments derived from speaker changes.
5.3  crop_plan.json  the complete rendering instruction set for Phase 3: given only this
                     file and the source video, the renderer crops each frame to
                     frames[i].crop (source pixels) and scales it to output.width x
                     output.height. Consecutive frames within a segment glide; segment
                     boundaries are hard cuts.

Both files carry schema_version PHASE2_SCHEMA_VERSION; bump it on any change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PHASE2_SCHEMA_VERSION = "1.0"


# ----------------------------------------------------------------------------- 5.2
@dataclass
class SpeakerFrame:
    frame_index: int
    timestamp_ms: float
    scores: dict[int, float]  # face_id -> speaking score (backend-specific scale)
    active_face_id: int | None  # after segmentation; null when no face is visible

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "scores": {str(k): v for k, v in self.scores.items()},
            "active_face_id": self.active_face_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerFrame":
        return cls(
            frame_index=int(d["frame_index"]),
            timestamp_ms=float(d["timestamp_ms"]),
            scores={int(k): float(v) for k, v in d["scores"].items()},
            active_face_id=None if d["active_face_id"] is None else int(d["active_face_id"]),
        )


@dataclass
class Segment:
    segment_id: int
    start_frame: int  # inclusive
    end_frame: int  # inclusive
    start_ms: float
    end_ms: float  # end of the last frame (start of the next segment)
    speaker_face_id: int | None

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        return cls(
            segment_id=int(d["segment_id"]),
            start_frame=int(d["start_frame"]),
            end_frame=int(d["end_frame"]),
            start_ms=float(d["start_ms"]),
            end_ms=float(d["end_ms"]),
            speaker_face_id=None if d["speaker_face_id"] is None else int(d["speaker_face_id"]),
        )


@dataclass
class SpeakersDoc:
    video: dict[str, Any]  # copied from the Phase 1 manifest
    asd: dict[str, Any]  # backend name + parameters + speaking_threshold
    segmentation: dict[str, Any]  # min_segment_ms, switch_confirm_ms, ...
    frames: list[SpeakerFrame]
    segments: list[Segment]
    schema_version: str = PHASE2_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video": self.video,
            "asd": self.asd,
            "segmentation": self.segmentation,
            "frames": [f.to_dict() for f in self.frames],
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakersDoc":
        return cls(
            video=d["video"],
            asd=d["asd"],
            segmentation=d["segmentation"],
            frames=[SpeakerFrame.from_dict(f) for f in d["frames"]],
            segments=[Segment.from_dict(s) for s in d["segments"]],
            schema_version=d["schema_version"],
        )


# ----------------------------------------------------------------------------- 5.3
@dataclass(frozen=True)
class CropRect:
    """Integer crop rectangle in source pixels; w/h are constant across the plan."""

    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CropRect":
        return cls(int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))


@dataclass
class CropFrame:
    frame_index: int
    timestamp_ms: float
    segment_id: int
    crop: CropRect

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "segment_id": self.segment_id,
            "crop": self.crop.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CropFrame":
        return cls(int(d["frame_index"]), float(d["timestamp_ms"]), int(d["segment_id"]), CropRect.from_dict(d["crop"]))


@dataclass
class CropSegment:
    segment_id: int
    start_frame: int
    end_frame: int
    start_ms: float
    end_ms: float
    speaker_face_id: int | None
    transition_in: str = "cut"  # how the renderer enters this segment

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CropSegment":
        return cls(**{k: d[k] for k in ("segment_id", "start_frame", "end_frame", "start_ms", "end_ms",
                                        "speaker_face_id", "transition_in")})


@dataclass
class OutputSpec:
    width: int
    height: int
    aspect_ratio: str  # "W:H"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CropPlan:
    source: dict[str, Any]  # path, width, height, fps, frame_count, duration_ms
    output: OutputSpec
    smoothing: dict[str, Any]  # parameters used, for provenance
    segments: list[CropSegment]
    frames: list[CropFrame]
    schema_version: str = PHASE2_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "output": self.output.to_dict(),
            "smoothing": self.smoothing,
            "segments": [s.to_dict() for s in self.segments],
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CropPlan":
        return cls(
            source=d["source"],
            output=OutputSpec(**d["output"]),
            smoothing=d["smoothing"],
            segments=[CropSegment.from_dict(s) for s in d["segments"]],
            frames=[CropFrame.from_dict(f) for f in d["frames"]],
            schema_version=d["schema_version"],
        )


# ----------------------------------------------------------------------- validators
class Phase2ContractError(ValueError):
    pass


def _check_segments_cover(segments: list[dict[str, Any]], frame_count: int, fps: float, min_segment_ms: float | None) -> None:
    if frame_count == 0:
        if segments:
            raise Phase2ContractError("segments present for an empty video")
        return
    if not segments:
        raise Phase2ContractError("no segments")
    expected_start = 0
    for i, s in enumerate(segments):
        if s["segment_id"] != i:
            raise Phase2ContractError(f"segment_id must be sequential, got {s['segment_id']} at {i}")
        if s["start_frame"] != expected_start:
            raise Phase2ContractError(f"segment {i} starts at {s['start_frame']}, expected {expected_start}")
        if s["end_frame"] < s["start_frame"]:
            raise Phase2ContractError(f"segment {i} ends before it starts")
        if i > 0 and segments[i - 1]["speaker_face_id"] == s["speaker_face_id"]:
            raise Phase2ContractError(f"segments {i - 1} and {i} have the same speaker; segment on speaker changes only")
        n = s["end_frame"] - s["start_frame"] + 1
        if min_segment_ms is not None and len(segments) > 1 and n * 1000.0 / fps < min_segment_ms - 1e-6:
            raise Phase2ContractError(f"segment {i} is {n * 1000.0 / fps:.0f} ms, below min_segment_ms={min_segment_ms}")
        expected_start = s["end_frame"] + 1
    if expected_start != frame_count:
        raise Phase2ContractError(f"segments cover {expected_start} frames, video has {frame_count}")


def validate_speakers(doc: dict[str, Any]) -> None:
    if doc.get("schema_version") != PHASE2_SCHEMA_VERSION:
        raise Phase2ContractError("speakers.json schema_version mismatch")
    frame_count = int(doc["video"]["frame_count"])
    frames = doc["frames"]
    if len(frames) != frame_count:
        raise Phase2ContractError(f"speakers.json has {len(frames)} frames, video has {frame_count}")
    for i, f in enumerate(frames):
        if f["frame_index"] != i:
            raise Phase2ContractError(f"frame {i} has frame_index {f['frame_index']}")
        if f["active_face_id"] is not None and str(f["active_face_id"]) not in f["scores"]:
            raise Phase2ContractError(f"frame {i}: active_face_id {f['active_face_id']} not among scored faces")
    _check_segments_cover(doc["segments"], frame_count, float(doc["video"]["fps"]), doc["segmentation"].get("min_segment_ms"))
    for s in doc["segments"]:
        for f in frames[s["start_frame"]: s["end_frame"] + 1]:
            if f["active_face_id"] != s["speaker_face_id"]:
                raise Phase2ContractError(f"frame {f['frame_index']} active_face_id disagrees with its segment")


def validate_crop_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PHASE2_SCHEMA_VERSION:
        raise Phase2ContractError("crop_plan.json schema_version mismatch")
    src = plan["source"]
    W, H, frame_count = int(src["width"]), int(src["height"]), int(src["frame_count"])
    out = plan["output"]
    if out["width"] <= 0 or out["height"] <= 0:
        raise Phase2ContractError("output size must be positive")
    frames = plan["frames"]
    if len(frames) != frame_count:
        raise Phase2ContractError(f"crop_plan has {len(frames)} frames, source has {frame_count}")
    _check_segments_cover(plan["segments"], frame_count, float(src["fps"]), None)
    seg_of_frame = {}
    for s in plan["segments"]:
        for fi in range(s["start_frame"], s["end_frame"] + 1):
            seg_of_frame[fi] = s["segment_id"]
    size = None
    for i, f in enumerate(frames):
        if f["frame_index"] != i:
            raise Phase2ContractError(f"frame {i} has frame_index {f['frame_index']}")
        if f["segment_id"] != seg_of_frame.get(i):
            raise Phase2ContractError(f"frame {i} segment_id {f['segment_id']} does not match segments")
        c = f["crop"]
        if size is None:
            size = (c["w"], c["h"])
        elif (c["w"], c["h"]) != size:
            raise Phase2ContractError(f"frame {i}: crop size changed within plan")
        if c["x"] < 0 or c["y"] < 0 or c["x"] + c["w"] > W or c["y"] + c["h"] > H:
            raise Phase2ContractError(f"frame {i}: crop {c} outside source {W}x{H}")
    if size is not None:
        want = out["width"] / out["height"]
        got = size[0] / size[1]
        if abs(got - want) > 0.02:
            raise Phase2ContractError(f"crop aspect {got:.3f} differs from output aspect {want:.3f}")


def to_json_safe(d: dict[str, Any]) -> dict[str, Any]:
    return d


__all__ = [
    "PHASE2_SCHEMA_VERSION",
    "SpeakerFrame", "Segment", "SpeakersDoc",
    "CropRect", "CropFrame", "CropSegment", "OutputSpec", "CropPlan",
    "Phase2ContractError", "validate_speakers", "validate_crop_plan",
]
