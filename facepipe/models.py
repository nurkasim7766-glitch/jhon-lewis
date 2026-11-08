"""Model asset management for the MediaPipe Face Landmarker (Apache-2.0)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_FILENAME = "face_landmarker.task"
# Files much smaller than this are download errors (HTML pages), not models.
_MIN_MODEL_BYTES = 1_000_000


def default_model_dir() -> Path:
    env = os.environ.get("FACEPIPE_MODEL_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "models"


def ensure_face_landmarker(model_dir: Path | None = None, download: bool = True) -> Path:
    """Return the path to face_landmarker.task, downloading it if allowed and missing."""
    model_dir = model_dir or default_model_dir()
    path = model_dir / FACE_LANDMARKER_FILENAME
    if path.is_file() and path.stat().st_size >= _MIN_MODEL_BYTES:
        return path
    if not download:
        raise FileNotFoundError(f"Face Landmarker model not found at {path}")
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)  # noqa: S310 - fixed https URL
    if tmp.stat().st_size < _MIN_MODEL_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded model is too small; download of {FACE_LANDMARKER_URL} failed")
    tmp.replace(path)
    return path
