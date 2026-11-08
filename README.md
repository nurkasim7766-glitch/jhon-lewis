# facepipe — Phase 1: face detection + landmarks

Phase 1 turns a video into per-frame face bounding boxes and 478-point face landmarks
(including the 40 lip landmarks Phase 2 needs), using MediaPipe's Face Landmarker.

Dependency policy: permissive licenses only (MediaPipe is Apache-2.0). No Ultralytics
YOLO or other AGPL-3.0 packages.

## Setup

```bash
# MediaPipe's native library links against EGL/GLES even on CPU (Debian/Ubuntu):
sudo apt-get install -y libegl1 libgles2 libgl1 libglib2.0-0
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
```

The `face_landmarker.task` model is downloaded to `./models/` on first use
(override with `FACEPIPE_MODEL_DIR` or `--model`).

## Usage

```bash
facepipe extract input.mp4 output/run1        # -> manifest.json + frames.jsonl
facepipe validate output/run1                 # contract check
python scripts/verify_phase1.py               # checkpoint verification on a synthetic clip
pytest                                        # unit + integration tests
```

## Output contract (schema 1.0)

See `facepipe/contract.py` for the normative definition.

`manifest.json`
```json
{
  "schema_version": "1.0",
  "video": {"path": "...", "width": 640, "height": 480, "fps": 30.0, "frame_count": 60, "duration_ms": 2000.0},
  "detector": {"backend": "mediapipe", "task": "face_landmarker", "running_mode": "VIDEO", "max_faces": 4, "...": "..."},
  "num_landmarks": 478,
  "lip_landmark_indices": [61, 146, "...", 191],
  "frames_with_faces": 60,
  "face_ids": [0],
  "frames_file": "frames.jsonl"
}
```

`frames.jsonl` — one object per decoded frame, in order, including frames with no faces:
```json
{"frame_index": 0, "timestamp_ms": 0.0,
 "faces": [{"face_id": 0,
            "bbox": {"x_min": 0.31, "y_min": 0.12, "x_max": 0.62, "y_max": 0.71},
            "landmarks": [[x, y, z], "... 478 entries"]}]}
```

- `x`, `y` are normalized to frame width/height; `z` is MediaPipe relative depth.
- `face_id` is a stable track id (greedy IoU matching, 15-frame gap tolerance).
- `bbox` is the extent of the landmarks, clipped to [0, 1].
- Lip landmarks: `facepipe.contract.lip_landmarks(face)` returns the 40 points in
  `LIP_LANDMARK_INDICES` order (outer ring, then inner ring). Mouth centre anchors:
  upper lip 13, lower lip 14, corners 61 / 291.
