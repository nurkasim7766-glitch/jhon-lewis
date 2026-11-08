# facepipe

Turns a long-form talking-head video plus its transcript into a short vertical clip
that follows whoever is speaking, with a debug overlay showing every decision the
pipeline made.

```
transcript.json ─┐
                 ├─► select moment ─► cut clip ─► detect faces ─► decide speaker ─► plan crop ─► render
source.mp4 ──────┘        (5.5)                     (5.1)           (5.2)            (5.3)      output.mp4
                                                                                                debug.mp4
```

Everything runs locally. Dependency policy: permissive licenses only (MediaPipe
Apache-2.0, LR-ASD MIT, PyTorch BSD). No Ultralytics / YOLO or other AGPL-3.0
packages, and no paid API calls in the processing chain (the optional LLM scorer refuses
non-local endpoints unless you explicitly opt in).

---

## 1. Setup from scratch

Requirements: Linux or macOS, Python ≥ 3.10, `ffmpeg`/`ffprobe` on `PATH`.

```bash
# Debian/Ubuntu system libraries (MediaPipe links against EGL/GLES even on CPU; ffmpeg for audio/video)
sudo apt-get install -y ffmpeg libegl1 libgles2 libgl1 libglib2.0-0

# Python environment
uv venv .venv && . .venv/bin/activate          # or: python -m venv .venv
uv pip install -e ".[dev]"                     # core + pytest

# Optional: LR-ASD neural active-speaker backend (CPU torch is enough)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[lrasd]"
```

Models are downloaded on first use into `./models/` (override with `FACEPIPE_MODEL_DIR`):

| file | size | source | license |
|---|---|---|---|
| `face_landmarker.task` | 3.8 MB | MediaPipe Face Landmarker | Apache-2.0 |
| `lrasd_pretrain_AVA.model` | 3.4 MB | LR-ASD (Liao et al., IJCV 2025) | MIT |

Verify the install:

```bash
pytest                                  # 24 unit/integration tests
python scripts/make_synthetic_source.py # builds assets/podcast/{source.mp4,transcript.json}
python scripts/verify_phase3.py         # full end-to-end checkpoint
```

---

## 2. Running the pipeline

One command, from the full-length source and its transcript:

```bash
facepipe run --video podcast.mp4 --transcript podcast.json --workdir output/ep42
```

Produces `output/ep42/output.mp4` (1080×1920 by default) and `output/ep42/debug.mp4`.

### Reusing intermediates

Every stage writes its outputs into the work directory and is **skipped on the next run
if those outputs exist**. Forcing a stage re-runs it *and everything downstream*
(later artifacts would otherwise describe stale inputs).

```bash
facepipe run ... --workdir output/ep42                     # second run: all six stages reused, ~0.2 s
facepipe run ... --workdir output/ep42 --force render      # re-render only (+ debug); detection untouched
facepipe run ... --workdir output/ep42 --force speakers    # new speaker decisions + crop + render + debug
facepipe run ... --workdir output/ep42 --force moment      # start over from selection
facepipe run ... --workdir output/ep42 --until speakers    # stop after crop planning (no render)
```

Stages, in order: `moment` → `clip` → `detect` → `speakers` → `render` → `debug`.

### Individual stage commands

Each stage is also a standalone command reading/writing the same files:

```bash
facepipe select  transcript.json source.mp4 work/moment.json      # deliverable 0
facepipe cut     source.mp4 work/moment.json work/clip.mp4
facepipe extract work/clip.mp4 work/phase1                        # Phase 1
facepipe speakers work/phase1 work/phase2 --video work/clip.mp4   # Phase 2
facepipe render  work/phase2/crop_plan.json work/clip.mp4 work/output.mp4
facepipe debug   work/clip.mp4 work/phase1 work/phase2 work/debug.mp4
facepipe validate work/phase1 ; facepipe validate-phase2 work/phase2
```

### Config file

`facepipe run --config cfg.json` accepts a JSON `PipelineConfig`; `--video`,
`--transcript`, `--workdir` flags override the file. `run.json` in every work directory
contains the exact config that produced it and can be reused as a config file.

```json
{
  "video": "podcast.mp4", "transcript": "podcast.json", "workdir": "output/ep42",
  "selection":    {"min_ms": 30000, "max_ms": 60000, "pad_in_ms": 250, "pad_out_ms": 400, "top_n": 10, "scorer": "heuristic"},
  "detector":     {"max_faces": 4, "min_face_detection_confidence": 0.5, "min_face_presence_confidence": 0.5, "min_tracking_confidence": 0.5, "model_path": null},
  "asd_backend":  "heuristic",
  "segmentation": {"min_segment_ms": 1500, "switch_confirm_ms": 300, "hold_ratio": 0.6, "fallback": "largest_face"},
  "crop":         {"output_width": 1080, "output_height": 1920, "deadzone_frac": 0.06, "time_constant_ms": 400, "max_speed_frac": 0.025, "settle_ms": 400, "post_smooth_ms": 160, "face_y_anchor": 0.42},
  "llm":          {"endpoint": "http://127.0.0.1:11434/v1/chat/completions", "model": "llama3.1", "shortlist": 8, "allow_hosted": false},
  "render_crf": 18, "render_preset": "medium", "debug": true
}
```

---

## 3. Every configuration option

### Moment selection (`selection`, `facepipe select`)

| option | CLI flag | default | meaning |
|---|---|---|---|
| `scorer` | `--scorer` | `heuristic` | `heuristic` (local, default) or `llm` (re-ranks the heuristic shortlist) |
| `min_ms` | `--min-ms` | 30000 | shortest allowed sentence span |
| `max_ms` | `--max-ms` | 60000 | longest allowed sentence span |
| `pad_in_ms` | `--pad-in-ms` | 250 | lead-in before the first sentence; clamped so it never enters the previous sentence |
| `pad_out_ms` | `--pad-out-ms` | 400 | tail after the last sentence; clamped to the next sentence start |
| `top_n` | `--top-n` | 10 | number of ranked candidates kept in `moment.json` |
| `llm.endpoint` | `--llm-endpoint` | `http://127.0.0.1:11434/v1/chat/completions` | OpenAI-compatible chat endpoint (Ollama default) |
| `llm.model` | `--llm-model` | `llama3.1` | model name sent to the endpoint |
| `llm.shortlist` | — | 8 | how many heuristic-ranked candidates the LLM sees |
| `llm.allow_hosted` | `--allow-hosted-llm` | false | **required** to use any endpoint that is not loopback / private-network |
| — | env `FACEPIPE_LLM_API_KEY` | unset | bearer token, only sent if set |

**Hard filters (not weights):** a candidate must start exactly at a sentence start and end
exactly at a sentence end, and its sentence span must be within `[min_ms, max_ms]`. Only
then is it scored. Padding is applied after selection and cannot cross a sentence.

Heuristic features (each ≈ 0–1, weights in `facepipe/moments/heuristic.py`): hook words,
intensity words, question in the opening sentences, numbers, lexical density, speech rate,
speaker alternation, closeness to a 45 s target, strong opener (not "and/so/but…"),
filler penalty, clean ending punctuation.

### Face detection (`detector`, `facepipe extract`)

| option | CLI flag | default | meaning |
|---|---|---|---|
| `max_faces` | `--max-faces` | 4 | faces tracked per frame |
| `min_face_detection_confidence` | `--min-detection-confidence` | 0.5 | MediaPipe detector threshold |
| `min_face_presence_confidence` | — | 0.5 | landmark presence threshold |
| `min_tracking_confidence` | — | 0.5 | inter-frame tracking threshold |
| `model_path` | `--model` | auto | path to `face_landmarker.task` |

### Active speaker + segmentation (`asd_backend`, `segmentation`, `facepipe speakers`)

| option | CLI flag | default | meaning |
|---|---|---|---|
| `asd_backend` | `--backend` | `heuristic` | `heuristic` = lip-motion from landmarks (no audio); `lrasd` = audio-visual neural model |
| `min_segment_ms` | `--min-segment-ms` | 1500 | runs shorter than this are absorbed into the longer neighbour |
| `switch_confirm_ms` | `--switch-confirm-ms` | 300 | a challenger must out-score the holder for this long before a switch |
| `hold_ratio` | — | 0.6 | holder keeps the floor while its score ≥ `hold_ratio × threshold` |
| `fallback` | — | `largest_face` | speaker when nobody is speaking and there is no history |

Scores: heuristic = rolling std of normalized mouth aperture (speaking ≥ 0.03);
LR-ASD = mean logit (speaking > 0). Segments change *only* on speaker changes.

### Crop planning (`crop`)

| option | CLI flag | default | meaning |
|---|---|---|---|
| `output_width/height` | `--output-width/--output-height` | 1080 / 1920 | output size; the crop is the largest source window with this aspect |
| `deadzone_frac` | — | 0.06 | target moves smaller than this × crop width are ignored |
| `time_constant_ms` | — | 400 | first-order follower response time |
| `max_speed_frac` | — | 0.025 | per-frame pan limit as a fraction of crop width (the glide) |
| `settle_ms` | — | 400 | segment opening position = median target over this window |
| `post_smooth_ms` | — | 160 | centred moving average after the follower |
| `face_y_anchor` | — | 0.42 | where the face centre sits vertically in the crop (0 = top) |

### Rendering

| option | CLI flag | default | meaning |
|---|---|---|---|
| `render_crf` | `--crf` | 18 | x264 quality (lower = better/larger) |
| `render_preset` | `--preset` | `medium` | x264 speed preset (`veryfast` for iteration) |
| `debug` | `--no-debug` | true | also produce `debug.mp4` |

---

## 4. Input format: transcript (section 5.4)

A JSON file with a `segments` list — the shape written by Whisper `verbose_json`,
faster-whisper and WhisperX. Word timings are used when present; otherwise sentence
boundaries are interpolated within each segment by character position.

```json
{"segments": [
  {"start": 12.40, "end": 15.92, "text": "Why did the first company fail?", "speaker": "A",
   "words": [{"word": "Why", "start": 12.40, "end": 12.61}, {"word": "did", "start": 12.61, "end": 12.74}, "..."]}
]}
```

Sentences end at `. ! ?` (abbreviations like "Dr." excluded) or at a speaker change.

---

## 5. Output files (work directory)

| file | stage | contents |
|---|---|---|
| `moment.json` | moment | **5.5** – chosen window with sentence indices, padded/unpadded times, score, feature breakdown, and the top-N ranked candidates |
| `clip.mp4` | clip | frame-accurate re-encode of `[start_ms, end_ms)` with audio |
| `phase1/manifest.json`, `phase1/frames.jsonl` | detect | **5.1** – per-frame face boxes + 478 landmarks (normalized), stable `face_id`s |
| `phase2/speakers.json` | speakers | **5.2** – per-frame per-face scores, `active_face_id`, speaker segments |
| `phase2/crop_plan.json` | speakers | **5.3** – per-frame integer crop rectangle + segments; the complete render instruction set |
| `output.mp4` | render | final clip: each frame = `crop` of the clip frame scaled to the output size, original audio |
| `debug.mp4` | debug | the clip with the pipeline's actual data drawn on it (see below) |
| `run.json` | — | config used, forced stages, and per-stage ran/reused status + timings |

### `moment.json`

```json
{
  "schema_version": "1.0",
  "source": {"video": "podcast.mp4", "transcript": "podcast.json", "transcript_duration_ms": 180427.0},
  "selection": {"min_ms": 30000.0, "max_ms": 60000.0, "pad_in_ms": 250.0, "pad_out_ms": 400.0, "top_n": 10,
                "scorer": {"name": "heuristic", "target_ms": 45000.0, "weights": {"...": 0}, "local_only": true},
                "num_candidates": 231},
  "moment": {"start_ms": 95105.0, "end_ms": 144610.0, "duration_ms": 49505.0,
             "first_sentence": 23, "last_sentence": 33,
             "sentence_start_ms": 95355.0, "sentence_end_ms": 144210.0,
             "text": "Why did the first company fail ...", "score": 6.57,
             "features": {"hook": 1.0, "question": 1.0, "...": 0}},
  "candidates": [{"first_sentence": 23, "last_sentence": 33, "start_ms": 95355.0, "end_ms": 144210.0,
                  "duration_ms": 48855.0, "score": 6.57, "features": {"...": 0}, "text": "..."}]
}
```
`start_ms/end_ms` are what gets cut (padding included); `sentence_start_ms/sentence_end_ms`
are the exact sentence boundaries. Invariants enforced by `validate_moment`: sentence span
within `[min_ms, max_ms]`, padding encloses the span and never crosses a neighbouring sentence.

### `speakers.json` (5.2)

```json
{"schema_version": "1.0",
 "video": {"path": "clip.mp4", "width": 1280, "height": 720, "fps": 30.0, "frame_count": 1486, "duration_ms": 49533.3},
 "asd": {"backend": "heuristic", "params": {"...": 0}, "speaking_threshold": 0.03},
 "segmentation": {"min_segment_ms": 1500.0, "switch_confirm_ms": 300.0, "hold_ratio": 0.6, "fallback": "largest_face"},
 "frames": [{"frame_index": 0, "timestamp_ms": 0.0, "scores": {"0": 0.005, "1": 0.006}, "active_face_id": 1}],
 "segments": [{"segment_id": 0, "start_frame": 0, "end_frame": 1485, "start_ms": 0.0, "end_ms": 49533.3, "speaker_face_id": 1}]}
```
Segments are contiguous, cover every frame, differ in speaker at every boundary, and each
lasts ≥ `min_segment_ms`. `active_face_id` always equals its segment's `speaker_face_id`.

### `crop_plan.json` (5.3)

```json
{"schema_version": "1.0",
 "source": {"path": "clip.mp4", "width": 1280, "height": 720, "fps": 30.0, "frame_count": 1486, "duration_ms": 49533.3},
 "output": {"width": 1080, "height": 1920, "aspect_ratio": "9:16"},
 "smoothing": {"...": "crop config used"},
 "segments": [{"segment_id": 0, "start_frame": 0, "end_frame": 1485, "start_ms": 0.0, "end_ms": 49533.3, "speaker_face_id": 1, "transition_in": "cut"}],
 "frames": [{"frame_index": 0, "timestamp_ms": 0.0, "segment_id": 0, "crop": {"x": 147, "y": 0, "w": 405, "h": 720}}]}
```
Renderer contract: for frame *i*, take `crop` (integer source pixels, always in bounds,
constant `w×h` matching the output aspect) from clip frame `frame_index`, scale to
`output.width × output.height`. Nothing else is needed.

### `manifest.json` + `frames.jsonl` (5.1)

`frames.jsonl` has one JSON object per decoded frame (including frames with no faces):
`{"frame_index", "timestamp_ms", "faces": [{"face_id", "bbox": {x_min,y_min,x_max,y_max}, "landmarks": [[x,y,z] × 478]}]}`.
Coordinates are normalized to the frame; `z` is MediaPipe relative depth. `face_id` is a
stable track id. `facepipe.contract.lip_landmarks(face)` returns the 40 lip points
(`LIP_LANDMARK_INDICES`); mouth anchors are 13 (upper lip), 14 (lower lip), 61/291 (corners).

### `debug.mp4`

The clip at source resolution plus a 48 px strip, drawn **only from the files above**:

- every tracked face: box coloured by `face_id` (from `frames.jsonl`), its 40 lip landmarks
- the active speaker (`speakers.json` `active_face_id`): thick box + `SPEAKING`
- a score bar per face with a white tick at `speaking_threshold`, numeric score
- the exact crop rectangle the renderer used for that frame (`crop_plan.json`, green)
- strip: frame index, time, segment id, speaker, ASD backend; timeline of segments coloured
  by speaker with a playhead

If the overlay and `output.mp4` disagree, one of the artifacts is stale — re-run with `--force`.

---

## 6. Verification scripts

| script | what it checks |
|---|---|
| `scripts/verify_phase1.py` | detection on a synthetic clip: all frames written, stable id, 40 lip landmarks inside the box |
| `scripts/verify_phase2.py` | scripted two-speaker fixture: segment order, interjection suppressed, boundaries, glide cap; LR-ASD smoke test |
| `scripts/verify_phase3.py` | single-command end-to-end on `assets/podcast`: moment alignment, hook passage chosen, clip/output/debug frame counts, render fidelity vs `crop_plan.json`, overlay rectangle position, reuse and `--force` behaviour, hosted-LLM refusal |
| `scripts/make_synthetic_source.py` | builds the 3-minute synthetic podcast (two faces, tone audio) + transcript with a deliberately strong passage at 95 s |

---

## 7. Troubleshooting

- `libEGL.so.1 / libGLESv2.so.2: cannot open shared object file` → install the system libs in §1.
- No faces detected → faces are too small; MediaPipe needs roughly ≥ 100 px face height. Check `phase1/manifest.json` → `frames_with_faces`.
- `HostedEndpointRefused` → the LLM endpoint is not local. Run a local server (Ollama, llama.cpp, vLLM) or pass `--allow-hosted-llm` deliberately.
- `no sentence-aligned window between 30s and 60s exists` → transcript sentences are too long/short for the range; adjust `--min-ms/--max-ms`.
- Stale results after changing parameters → outputs are reused by existence; pass `--force <stage>`.
