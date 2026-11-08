"""MediaPipe Face Landmarker wrapper producing bounding boxes + 478 landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facepipe.contract import NUM_LANDMARKS, BBox
from facepipe.models import ensure_face_landmarker


@dataclass
class Detection:
    bbox: BBox
    landmarks: np.ndarray  # (NUM_LANDMARKS, 3) float32, normalized


@dataclass
class DetectorConfig:
    max_faces: int = 4
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: Path | None = None

    def to_manifest(self, model_path: Path) -> dict:
        return {
            "backend": "mediapipe",
            "task": "face_landmarker",
            "model_file": model_path.name,
            "running_mode": "VIDEO",
            "max_faces": self.max_faces,
            "min_face_detection_confidence": self.min_face_detection_confidence,
            "min_face_presence_confidence": self.min_face_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "bbox_source": "landmark_extent",
        }


class FaceLandmarkDetector:
    """Stateful VIDEO-mode detector; call detect() with strictly increasing timestamps."""

    def __init__(self, config: DetectorConfig | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self.config = config or DetectorConfig()
        self.model_path = self.config.model_path or ensure_face_landmarker()
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.config.max_faces,
            min_face_detection_confidence=self.config.min_face_detection_confidence,
            min_face_presence_confidence=self.config.min_face_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts: int | None = None

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[Detection]:
        # MediaPipe VIDEO mode rejects non-increasing timestamps.
        if self._last_ts is not None and timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        detections: list[Detection] = []
        for face in result.face_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in face], dtype=np.float32)
            if pts.shape != (NUM_LANDMARKS, 3):
                raise RuntimeError(f"unexpected landmark shape {pts.shape}")
            x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
            x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
            clip = lambda v: round(float(np.clip(v, 0.0, 1.0)), 6)  # noqa: E731
            bbox = BBox(x_min=clip(x_min), y_min=clip(y_min), x_max=clip(x_max), y_max=clip(y_max))
            detections.append(Detection(bbox=bbox, landmarks=pts))
        return detections

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
