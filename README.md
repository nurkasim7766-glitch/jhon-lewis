# facepipe — face landmarks, active speaker, crop planning

- **Phase 1** turns a video into per-frame face bounding boxes and 478-point face
  landmarks (including the 40 lip landmarks) using MediaPipe's Face Landmarker.
- **Phase 2** scores who is speaking, segments the video on speaker changes, and plans
  a smoothed crop window that follows the active speaker (`crop_plan.json`, the sole
  input Phase 3 needs besides the source video).

Dependency policy: permissive licenses only (MediaPipe Apache-2.0, LR-ASD MIT, torch
BSD). No Ultralytics YOLO or other AGPL-3.0 packages.

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
facepipe extract input.mp4 output/run1        # Phase 1 -> manifest.json + frames.jsonl
facepipe validate output/run1                 # 5.1 contract check
facepipe speakers output/run1 output/run1_p2  # Phase 2 -> speakers.json + crop_plan.json
facepipe speakers output/run1 output/run1_p2 --backend lrasd --video input.mp4
facepipe validate-phase2 output/run1_p2       # 5.2 / 5.3 contract check
python scripts/verify_phase1.py               # Phase 1 checkpoint
python scripts/verify_phase2.py               # Phase 2 checkpoint
pytest                                        # unit + integration tests
```

### ASD backends

| backend | default | needs | score scale |
|---|---|---|---|
| `heuristic` | yes | Phase 1 landmarks only | rolling std of normalized mouth aperture; speaking ≥ 0.03 |
| `lrasd` | no | `pip install -e ".[lrasd]"` (+ torch CPU index), ffmpeg | LR-ASD logit; speaking > 0 |

Both implement `facepipe.asd.ASDBackend.score(data, video_path) -> ASDResult`
(per-face, per-frame scores + threshold). LR-ASD model code is vendored under
`facepipe/asd/lrasd_model/` (MIT, Liao et al.); weights auto-download to `./models/`.

### Segmentation (anti-flicker)

1. Hysteresis: a challenger must out-score the current speaker above threshold for
   `switch_confirm_ms` (300 ms) consecutive frames.
2. `min_segment_ms` (1500 ms): shorter runs are absorbed into the longer neighbour,
   so short interjections never produce a segment. Adjacent segments always have
   different speakers.
3. When nobody speaks, the current speaker is held; with no history the largest face
   is used.

### Crop smoothing

Within a segment the crop target passes through a dead-zone (6% of crop width), a
first-order follower (400 ms time constant), a per-frame speed cap (2.5% of crop width)
and a 160 ms moving average. Segment starts are hard cuts positioned at the median
target of the first 400 ms.

## Output contract (schema 1.0)

### 5.1 Phase 1 — `manifest.json` + `frames.jsonl`

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

### 5.2 Phase 2 — `speakers.json`

Normative definition: `facepipe/contract_phase2.py`.
```json
{
  "schema_version": "1.0",
  "video": {"...": "copied from the Phase 1 manifest"},
  "asd": {"backend": "heuristic", "params": {"...": "..."}, "speaking_threshold": 0.03},
  "segmentation": {"min_segment_ms": 1500.0, "switch_confirm_ms": 300.0, "hold_ratio": 0.6, "fallback": "largest_face"},
  "frames": [{"frame_index": 0, "timestamp_ms": 0.0, "scores": {"0": 0.071, "1": 0.004}, "active_face_id": 0}],
  "segments": [{"segment_id": 0, "start_frame": 0, "end_frame": 127, "start_ms": 0.0, "end_ms": 4266.667, "speaker_face_id": 0}]
}
```
`frames` has one entry per source frame; `active_face_id` equals the enclosing segment's
`speaker_face_id` (null only when no face is visible). Segments are contiguous, cover
every frame, change speaker at every boundary, and each lasts ≥ `min_segment_ms`.

### 5.3 Phase 2 — `crop_plan.json`

```json
{
  "schema_version": "1.0",
  "source": {"path": "input.mp4", "width": 1920, "height": 1080, "fps": 30.0, "frame_count": 300, "duration_ms": 10000.0},
  "output": {"width": 1080, "height": 1920, "aspect_ratio": "9:16"},
  "smoothing": {"...": "parameters used (provenance only)"},
  "segments": [{"segment_id": 0, "start_frame": 0, "end_frame": 127, "start_ms": 0.0, "end_ms": 4266.667,
                "speaker_face_id": 0, "transition_in": "cut"}],
  "frames": [{"frame_index": 0, "timestamp_ms": 0.0, "segment_id": 0, "crop": {"x": 233, "y": 0, "w": 608, "h": 1080}}]
}
```
Renderer contract (Phase 3): for each frame `i`, read source frame `frames[i].frame_index`,
take the rectangle `crop` (integer source pixels, always in bounds, constant `w`×`h`
matching the output aspect), and scale it to `output.width`×`output.height`. Nothing
else is required.
