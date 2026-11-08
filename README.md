# facepipe — AI video clipping

Turn a long talking-head video (podcast, interview, panel) plus its transcript into
short vertical clips that follow whoever is speaking — through a browser, or from the
command line. Everything runs **locally on your machine**: no cloud AI, no paid APIs.

```
transcript ─┐
            ├─► moment selection ─► cut ─► face + landmark tracking ─► active speaker ─► speaker segments ─► crop plan ─► render
video ──────┘      (local)                      (MediaPipe)             (local)                                          output.mp4
                                                                                                                         debug.mp4
```

**Licence policy:** permissive only — MediaPipe (Apache-2.0), LR-ASD (MIT), PyTorch
(BSD), FastAPI/uvicorn (BSD/MIT), imageio-ffmpeg (BSD-2). No Ultralytics / YOLO, no
AGPL-3.0 code. The optional LLM re-ranker refuses non-local endpoints unless you opt in.

---

## Contents

1. [Requirements](#1-requirements)
2. [Windows setup (start.bat)](#2-windows-setup)
3. [macOS / Linux setup](#3-macos--linux-setup)
4. [Starting the application](#4-starting-the-application)
5. [Using the web app](#5-using-the-web-app) — dashboard, upload, processing, clip discovery, editor, export
6. [Configuration](#6-configuration)
7. [Command-line interface](#7-command-line-interface)
8. [Output files and data contracts](#8-output-files-and-data-contracts)
9. [Verification scripts & tests](#9-verification-scripts--tests)
10. [Troubleshooting](#10-troubleshooting)
11. [Architecture](#11-architecture)

---

## 1. Requirements

| | minimum | notes |
|---|---|---|
| OS | Windows 10/11 (64-bit), macOS 12+, Linux | tested on Linux + Windows-style launcher |
| Python | 3.10 – 3.12 | 3.12 recommended; MediaPipe wheels exist for 3.9–3.12 |
| CPU / RAM | 4 cores, 8 GB | processing is CPU-only; GPU not required |
| Disk | ~2 GB + your videos | the venv is ≈1.2 GB |
| ffmpeg | bundled | `imageio-ffmpeg` ships ffmpeg; a system `ffmpeg`/`ffprobe` is used if present |
| Browser | Chrome, Edge, Firefox, Safari (current) | |

Processing speed on a 4-core laptop: roughly 1.5–2× real time for a 50 s clip
(face tracking ≈ 45 fps, rendering ≈ 40 fps at 1080×1920).

Models (downloaded automatically into `./models/` on first use):

| file | size | purpose | licence |
|---|---|---|---|
| `face_landmarker.task` | 3.8 MB | MediaPipe face detection + 478 landmarks | Apache-2.0 |
| `lrasd_pretrain_AVA.model` | 3.4 MB | optional LR-ASD neural active-speaker model | MIT |

---

## 2. Windows setup

1. Install **Python 3.12** from <https://www.python.org/downloads/windows/>.
   In the installer tick **“Add python.exe to PATH”**.
2. Download or `git clone` this repository to a folder, e.g. `C:\facepipe`.
3. Double-click **`start.bat`**.

The first run creates a virtual environment in `.venv\`, installs the dependencies
(a few minutes) and then opens <http://127.0.0.1:8765/> in your browser. Later runs
start in a couple of seconds. Close the console window to stop the server.

Optional:

- **Firewall prompt** — allow access on private networks only if you want to open the
  app from another device (`start.bat --host 0.0.0.0`). By default it listens on
  localhost only.
- **Different port** — `set FACEPIPE_PORT=9000` before running `start.bat`.
- **Different data folder** — `set FACEPIPE_DATA_DIR=D:\facepipe-data`.
- **LR-ASD neural speaker detection** — in the console:
  `.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu`
  then `.venv\Scripts\pip install -e ".[lrasd]"`.

---

## 3. macOS / Linux setup

```bash
# Debian/Ubuntu only: MediaPipe's native library needs these even on CPU
sudo apt-get install -y libegl1 libgles2 libgl1 libglib2.0-0

./start.sh          # creates .venv, installs, starts the server and opens the browser
```

Manual equivalent:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
facepipe-web                       # or: python -m facepipe.web
```

---

## 4. Starting the application

| action | Windows | macOS / Linux |
|---|---|---|
| start | double-click `start.bat` | `./start.sh` |
| stop | close the console window / Ctrl+C | Ctrl+C |
| URL | <http://127.0.0.1:8765/> | same |
| API docs | <http://127.0.0.1:8765/api/docs> | same |

Flags (append to the launcher or `python -m facepipe.web`): `--port N`, `--host ADDR`,
`--data-dir PATH`, `--no-browser`.

All state lives in the data directory (default `./data/`):

```
data/
  facepipe.db                 SQLite: projects, clips, jobs, activity
  projects/<id>/
    source.mp4                uploaded video
    transcript.json           transcript (5.4)
    thumbs/<clip>.jpg         candidate thumbnails
    clips/<clip>/             one pipeline work directory per clip (see §8)
```

Projects, results and rendered clips survive restarts. A job that was running when the
app was closed is marked **interrupted**; its finished stages are kept and reused.

---

## 5. Using the web app

### Dashboard
Totals (projects, videos, rendered clips, candidates), active processing, recent
projects and a recent-activity feed. **Quick upload** starts a new project.

### Upload
1. Name the project.
2. Drag a video onto the drop zone (or click to browse). Supported:
   `.mp4 .mov .m4v .mkv .webm .avi .mpg .mpeg .wmv`, up to 8 GB
   (`FACEPIPE_MAX_UPLOAD_MB`). The server probes the file; undecodable or
   < 5 s videos are rejected with a clear message. Progress is shown and the upload
   can be cancelled.
3. Add the transcript: drop a JSON file or paste JSON. Format: Whisper /
   faster-whisper / WhisperX `segments` (see §8, 5.4). Word timings are optional.
4. **Start AI analysis**.

Replacing the video or transcript clears the clips derived from the old one.

### Processing workspace
Live view of the real pipeline, polled from the job record:

| # | step | engine source of the status |
|---|---|---|
| 1 | Video ingestion | ffprobe/OpenCV probe of the upload |
| 2 | Transcript processing | sentence segmentation |
| 3 | AI moment selection | candidate generation + scoring |
| 4 | Clip extraction | frame-accurate ffmpeg cut |
| 5 | Face detection & landmark tracking | MediaPipe, per-frame progress bar |
| 6 | Active-speaker detection | ASD backend |
| 7 | Speaker segmentation | hysteresis + min-duration |
| 8 | Camera crop planning | smoothed crop path |
| 9 | Clip rendering | per-frame progress bar |
| 10 | Debug overlay | per-frame progress bar |
| 11 | Final output | output.mp4 verified on disk |

Steps 1–3 run once per project; 4–11 run per selected clip. Stages whose outputs
already exist show **reused**. **Cancel job** stops at the next frame/stage boundary.
Failed jobs show the error and a traceback.

### AI clip discovery (Clips tab)
Each discovered moment shows thumbnail, start/end, duration, AI score, the reasons
behind the score (feature chips), the transcript excerpt, speaker-tracking summary after
processing, and a **Preview** that seeks the source video to the exact range.

- **Select / deselect** which clips to process (the top 3 are pre-selected).
- **Compare** two or more candidates feature by feature.
- **Re-run selection** after changing settings (replaces current candidates).
- **Process & export selected** runs steps 4–11 for the selected clips.

All candidates are sentence-aligned and non-overlapping — a clip never starts
mid-sentence.

### Editor
- Playback with play/pause, timeline (coloured by speaker segment) and click-to-seek.
- Overlay layers drawn **from the pipeline's own files**: face boxes (`frames.jsonl`),
  the active-speaker decision (`speakers.json`), per-face scores with the threshold
  tick, and the crop window (`crop_plan.json`).
- **Vertical preview**: a live 9:16 crop of the current frame using the exact per-frame
  rectangle the renderer used.
- **Debug overlay video** toggles to the rendered `debug.mp4`; **Rendered output**
  plays `output.mp4`.
- Download MP4, re-render (reuses detection), and raw artifact JSON links.

### Export center
All rendered clips across projects: download a single MP4, download checked clips as a
ZIP, download the debug video, re-render, and see export status/history. Downloads work
for every past export as long as the project exists.

### Projects
Create, rename (click the title), open, delete. Deleting removes the project's files.

---

## 6. Configuration

### Per-project settings (Settings tab)

| setting | default | effect |
|---|---|---|
| Min / max clip length | 30 / 60 s | hard filter on the sentence span of a candidate |
| Moments to discover | 5 | number of non-overlapping candidates |
| Scorer | heuristic | `heuristic` (local) or `llm` (re-ranks the heuristic shortlist via an OpenAI-compatible endpoint) |
| LLM endpoint / model | `http://127.0.0.1:11434/v1/chat/completions` / `llama3.1` | Ollama defaults |
| Allow hosted LLM | off | **must** be on to use any non-loopback/private endpoint |
| Active-speaker model | heuristic | lip-motion heuristic (landmarks only) or `lrasd` (audio-visual neural, needs extras) |
| Min speaker segment | 1500 ms | shorter speaker runs are absorbed (anti-flicker) |
| Output size | 1080 × 1920 | any even size; crop keeps this aspect |
| Quality (CRF) / encoder speed | 18 / medium | x264 settings; `veryfast` for iteration |
| Debug overlay | on | also render `debug.mp4` per clip |

### Environment variables

| variable | default | meaning |
|---|---|---|
| `FACEPIPE_PORT` | 8765 | web server port |
| `FACEPIPE_DATA_DIR` | `./data` | projects, database, uploads |
| `FACEPIPE_MODEL_DIR` | `./models` | model downloads |
| `FACEPIPE_MAX_UPLOAD_MB` | 8192 | upload size limit |
| `FACEPIPE_LLM_API_KEY` | unset | bearer token for the LLM endpoint (only sent if set) |

### Engine-level options (CLI / `PipelineConfig`)

Detector thresholds, hysteresis, crop smoothing constants, padding etc. are documented in
§7 and in `facepipe/cropplan.py`, `facepipe/segmentation.py`, `facepipe/moments/`.

---

## 7. Command-line interface

The CLI and the web app share the same engine and file layout.

```bash
facepipe run --video podcast.mp4 --transcript podcast.json --workdir out/ep42   # full pipeline
facepipe run ... --force render          # re-render only (detection reused)
facepipe run ... --until speakers        # stop after crop planning
facepipe select transcript.json video.mp4 moment.json      # moment selection only
facepipe cut video.mp4 moment.json clip.mp4
facepipe extract clip.mp4 phase1/                          # faces + landmarks (5.1)
facepipe speakers phase1/ phase2/ --video clip.mp4         # speakers + crop plan (5.2/5.3)
facepipe render phase2/crop_plan.json clip.mp4 output.mp4
facepipe debug clip.mp4 phase1/ phase2/ debug.mp4
facepipe validate phase1/ ; facepipe validate-phase2 phase2/
```

Key flags: `--min-ms/--max-ms`, `--scorer heuristic|llm`, `--llm-endpoint`,
`--allow-hosted-llm`, `--backend heuristic|lrasd`, `--min-segment-ms`,
`--output-width/--output-height`, `--crf`, `--preset`, `--no-debug`,
`--force <stage>` (repeatable), `--until <stage>`, `--config cfg.json`.

---

## 8. Output files and data contracts

Per clip work directory (`data/projects/<id>/clips/<clip>/` or a CLI `--workdir`):

| file | stage | contents |
|---|---|---|
| `moment.json` | moment | **5.5** chosen window, sentence indices, score, features, top-N candidates |
| `clip.mp4` | clip | frame-accurate cut with audio |
| `phase1/manifest.json`, `phase1/frames.jsonl` | detect | **5.1** per-frame face boxes + 478 landmarks, stable `face_id`s |
| `phase2/speakers.json` | speakers | **5.2** per-frame per-face scores, `active_face_id`, speaker segments |
| `phase2/crop_plan.json` | speakers | **5.3** per-frame integer crop rectangle; the complete render instruction set |
| `output.mp4` | render | final vertical clip with original audio |
| `debug.mp4` | debug | source clip with boxes, lips, scores, active speaker, crop rectangle, timeline |
| `run.json` | — | config used, forced stages, per-stage ran/reused + timings |

### 5.4 Transcript input
```json
{"segments": [
  {"start": 12.40, "end": 15.92, "text": "Why did the first company fail?", "speaker": "A",
   "words": [{"word": "Why", "start": 12.40, "end": 12.61}, "..."]}
]}
```
Sentences end at `. ! ?` (abbreviations excluded) or at a speaker change. Without word
timings, boundaries are interpolated by character position within the segment.

### 5.5 `moment.json`
```json
{"schema_version": "1.0",
 "source": {"video": "...", "transcript": "...", "transcript_duration_ms": 180427.0},
 "selection": {"min_ms": 30000.0, "max_ms": 60000.0, "pad_in_ms": 250.0, "pad_out_ms": 400.0, "top_n": 10,
               "scorer": {"name": "heuristic", "local_only": true, "...": 0}, "num_candidates": 231},
 "moment": {"start_ms": 95105.0, "end_ms": 144610.0, "duration_ms": 49505.0, "first_sentence": 23, "last_sentence": 33,
            "sentence_start_ms": 95355.0, "sentence_end_ms": 144210.0, "text": "...", "score": 6.57, "features": {"...": 0}},
 "candidates": [{"first_sentence": 23, "last_sentence": 33, "start_ms": 95355.0, "end_ms": 144210.0, "duration_ms": 48855.0, "score": 6.57, "features": {}, "text": "..."}]}
```

### 5.1 / 5.2 / 5.3
```json
// frames.jsonl (one line per frame)
{"frame_index": 0, "timestamp_ms": 0.0, "faces": [{"face_id": 0, "bbox": {"x_min": 0.31, "y_min": 0.12, "x_max": 0.62, "y_max": 0.71}, "landmarks": [[x, y, z], "...478"]}]}
// speakers.json
{"schema_version": "1.0", "video": {...}, "asd": {"backend": "heuristic", "speaking_threshold": 0.03, "params": {}},
 "segmentation": {"min_segment_ms": 1500.0, "switch_confirm_ms": 300.0, "hold_ratio": 0.6, "fallback": "largest_face"},
 "frames": [{"frame_index": 0, "timestamp_ms": 0.0, "scores": {"0": 0.071}, "active_face_id": 0}],
 "segments": [{"segment_id": 0, "start_frame": 0, "end_frame": 1485, "start_ms": 0.0, "end_ms": 49533.3, "speaker_face_id": 0}]}
// crop_plan.json
{"schema_version": "1.0", "source": {...}, "output": {"width": 1080, "height": 1920, "aspect_ratio": "9:16"}, "smoothing": {...},
 "segments": [{"segment_id": 0, "start_frame": 0, "end_frame": 1485, "start_ms": 0.0, "end_ms": 49533.3, "speaker_face_id": 0, "transition_in": "cut"}],
 "frames": [{"frame_index": 0, "timestamp_ms": 0.0, "segment_id": 0, "crop": {"x": 147, "y": 0, "w": 405, "h": 720}}]}
```
Normative definitions and validators: `facepipe/contract.py`, `facepipe/contract_phase2.py`,
`facepipe/moments/select.py`, `facepipe/transcript.py`.

---

## 9. Verification scripts & tests

```bash
pytest                                  # 28 tests: contracts, tracker, segmentation, crop, moments, overlay, pipeline, web API
python scripts/make_synthetic_source.py # 3-minute synthetic two-face podcast + transcript -> assets/podcast
python scripts/verify_phase1.py         # detection checkpoint
python scripts/verify_phase2.py         # speaker/segmentation/crop checkpoint (+ LR-ASD smoke test)
python scripts/verify_phase3.py         # single-command end-to-end + render/overlay fidelity + reuse
python scripts/ui_walkthrough.py        # headless Chrome: upload -> analyze -> process -> editor -> download (needs playwright + Chrome)
```

---

## 10. Troubleshooting

| symptom | fix |
|---|---|
| `start.bat` says Python not found | install Python 3.10+ and tick “Add python.exe to PATH”; reopen the console |
| `libEGL.so.1` / `libGLESv2.so.2` missing (Linux) | `sudo apt-get install libegl1 libgles2 libgl1 libglib2.0-0` |
| “could not be decoded as a video” | the file is corrupt or an unsupported codec; re-export as H.264 MP4 |
| No faces detected (clip renders centred) | faces are too small; MediaPipe needs ≈ ≥ 100 px face height. Check `phase1/manifest.json → frames_with_faces` |
| “no sentence-aligned window between 30s and 60s” | the transcript's sentences don't fit; widen min/max clip length in Settings |
| Analysis says “needs both a video and a transcript” | upload both on the Upload tab |
| `HostedEndpointRefused` | the LLM endpoint isn't local; run Ollama/llama.cpp/vLLM locally or enable “Allow hosted LLM” deliberately |
| LR-ASD option disabled | install the extras: `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install -e ".[lrasd]"` |
| Job shows **interrupted** | the app was closed mid-job; run Process again — finished stages are reused |
| Port already in use | `set FACEPIPE_PORT=9000` (Windows) / `FACEPIPE_PORT=9000 ./start.sh` |
| Stale results after changing settings | use **Re-run selection** (new candidates) or **Re-render** (new output size/quality) |
| Slow processing | set encoder speed to `veryfast`, reduce output size, or process fewer clips at once |

Logs print to the console the launcher opened. The API is documented at `/api/docs`.

---

## 11. Architecture

```
facepipe/
  contract*.py          data contracts + validators (5.1–5.3)
  detector.py tracker.py extract.py        Phase 1 — MediaPipe Face Landmarker, IoU tracking
  asd/                  Phase 2 — ASDBackend interface; heuristic (default) and lrasd backends
  segmentation.py cropplan.py phase2.py    Phase 2 — speaker decision, segmentation, crop smoothing
  transcript.py moments/                   Phase 3 — sentence splitting; MomentScorer interface; heuristic (default) + llm
  render.py overlay.py pipeline.py cli.py  Phase 3 — renderer, debug overlay, cached 6-stage pipeline, CLI
  ffmpeg.py             ffmpeg/ffprobe resolution (system, else bundled imageio-ffmpeg)
  web/
    app.py              FastAPI routes (projects, uploads, jobs, clips, media with HTTP range, ZIP export, SPA)
    db.py               SQLite (stdlib)
    projects.py         project files, settings, clip records, thumbnails, dashboard stats
    jobs.py             worker thread + queue; analyze / process / render jobs; cooperative cancellation; restart recovery
    static/             dependency-free SPA (index.html, styles.css, app.js)
```

The web app never re-implements processing: `jobs.py` calls `select_moments`, then runs
`facepipe.pipeline.Pipeline` per clip with the same hooks the CLI uses, so stage reuse,
contracts and outputs are identical whether a clip was made from the browser or the
terminal.
