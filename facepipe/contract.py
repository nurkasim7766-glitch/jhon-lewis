"""Phase 1 output data contract.

Phase 2 (lip activity) and Phase 3 consume these records, so changes here must bump
SCHEMA_VERSION. On disk, a Phase 1 run produces a directory containing:

    manifest.json   one Manifest object
    frames.jsonl    one FrameRecord object per line, in frame_index order, including
                    frames with zero faces (faces == [])

All coordinates are normalized to the frame: x in [0, 1] of width, y in [0, 1] of
height. Landmark z is MediaPipe's relative depth (same scale as x, origin at the
face centroid; negative is toward the camera). The 478 landmarks follow the
MediaPipe Face Mesh topology; LIP_LANDMARK_INDICES selects the 40 lip points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Sequence

SCHEMA_VERSION = "1.0"
NUM_LANDMARKS = 478

# MediaPipe Face Mesh lip contour (FACE_LANDMARKS_LIPS). Outer ring then inner ring,
# each ordered clockwise starting at the left mouth corner.
OUTER_LIP_INDICES: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
)
INNER_LIP_INDICES: tuple[int, ...] = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
)
LIP_LANDMARK_INDICES: tuple[int, ...] = OUTER_LIP_INDICES + INNER_LIP_INDICES

# Convenience anchors for Phase 2 mouth-aperture metrics.
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in normalized coordinates, derived from the landmark extent."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BBox":
        return cls(float(d["x_min"]), float(d["y_min"]), float(d["x_max"]), float(d["y_max"]))


@dataclass
class FaceRecord:
    face_id: int  # stable track id across frames (>= 0)
    bbox: BBox
    landmarks: list[list[float]]  # NUM_LANDMARKS x [x, y, z]

    def to_dict(self) -> dict[str, Any]:
        return {"face_id": self.face_id, "bbox": self.bbox.to_dict(), "landmarks": self.landmarks}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FaceRecord":
        return cls(
            face_id=int(d["face_id"]),
            bbox=BBox.from_dict(d["bbox"]),
            landmarks=[[float(v) for v in pt] for pt in d["landmarks"]],
        )


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_ms: float
    faces: list[FaceRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "faces": [f.to_dict() for f in self.faces],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrameRecord":
        return cls(
            frame_index=int(d["frame_index"]),
            timestamp_ms=float(d["timestamp_ms"]),
            faces=[FaceRecord.from_dict(f) for f in d["faces"]],
        )


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int  # frames actually decoded and written to frames.jsonl
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Manifest:
    video: VideoInfo
    detector: dict[str, Any]  # backend, model, thresholds, max_faces
    num_landmarks: int = NUM_LANDMARKS
    lip_landmark_indices: list[int] = field(default_factory=lambda: list(LIP_LANDMARK_INDICES))
    frames_with_faces: int = 0
    face_ids: list[int] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    frames_file: str = "frames.jsonl"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["video"] = self.video.to_dict()
        return d


def lip_landmarks(face: FaceRecord) -> list[list[float]]:
    """Return the 40 lip landmarks of a face in LIP_LANDMARK_INDICES order."""
    return [face.landmarks[i] for i in LIP_LANDMARK_INDICES]


class ContractError(ValueError):
    pass


def validate_face(face: dict[str, Any], where: str = "") -> None:
    if not isinstance(face.get("face_id"), int) or face["face_id"] < 0:
        raise ContractError(f"{where}face_id must be a non-negative int")
    bbox = face.get("bbox")
    if not isinstance(bbox, dict) or set(bbox) != {"x_min", "y_min", "x_max", "y_max"}:
        raise ContractError(f"{where}bbox must have x_min,y_min,x_max,y_max")
    if not (bbox["x_min"] <= bbox["x_max"] and bbox["y_min"] <= bbox["y_max"]):
        raise ContractError(f"{where}bbox min must not exceed max")
    lm = face.get("landmarks")
    if not isinstance(lm, list) or len(lm) != NUM_LANDMARKS:
        raise ContractError(f"{where}expected {NUM_LANDMARKS} landmarks, got {len(lm) if isinstance(lm, list) else type(lm)}")
    for i, pt in enumerate(lm):
        if not isinstance(pt, list) or len(pt) != 3:
            raise ContractError(f"{where}landmark {i} must be [x, y, z]")


def validate_frame(frame: dict[str, Any], expected_index: int | None = None) -> None:
    where = f"frame {frame.get('frame_index')}: "
    if not isinstance(frame.get("frame_index"), int):
        raise ContractError(f"{where}frame_index must be int")
    if expected_index is not None and frame["frame_index"] != expected_index:
        raise ContractError(f"{where}expected frame_index {expected_index}")
    if not isinstance(frame.get("timestamp_ms"), (int, float)):
        raise ContractError(f"{where}timestamp_ms must be numeric")
    faces = frame.get("faces")
    if not isinstance(faces, list):
        raise ContractError(f"{where}faces must be a list")
    ids = [f.get("face_id") for f in faces]
    if len(ids) != len(set(ids)):
        raise ContractError(f"{where}duplicate face_id within frame")
    for f in faces:
        validate_face(f, where)


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"schema_version {manifest.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    if manifest.get("num_landmarks") != NUM_LANDMARKS:
        raise ContractError("num_landmarks mismatch")
    if list(manifest.get("lip_landmark_indices", [])) != list(LIP_LANDMARK_INDICES):
        raise ContractError("lip_landmark_indices mismatch")
    for key in ("path", "width", "height", "fps", "frame_count", "duration_ms"):
        if key not in manifest.get("video", {}):
            raise ContractError(f"manifest.video missing {key}")
    if manifest.get("detector", {}).get("backend") != "mediapipe":
        raise ContractError("detector.backend must be 'mediapipe'")


def iter_frames_jsonl(lines: Sequence[str] | Iterator[str]) -> Iterator[dict[str, Any]]:
    import json

    for line in lines:
        line = line.strip()
        if line:
            yield json.loads(line)
