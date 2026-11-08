"""Debug overlay: draws the pipeline's actual intermediate data on the source clip.

Every element comes from a Phase 1/2 artifact, never recomputed:
  frames.jsonl    face boxes (coloured by face_id) + lip landmarks
  speakers.json   per-face speaking score bars, threshold tick, the active_face_id
                  decision (thick box + "SPEAKING") and the current segment
  crop_plan.json  the exact crop rectangle the renderer used for this frame
A timeline strip at the bottom shows segments coloured by speaker with a playhead.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from facepipe.contract import FrameRecord, iter_frames_jsonl, lip_landmarks
from facepipe.contract_phase2 import CropPlan, SpeakersDoc
from facepipe.render import mux

PALETTE = [(80, 200, 255), (255, 150, 60), (120, 255, 120), (255, 110, 230), (255, 255, 90), (170, 130, 255)]
CROP_COLOR = (0, 255, 0)
TIMELINE_H = 48


def face_color(face_id: int) -> tuple[int, int, int]:
    return PALETTE[face_id % len(PALETTE)]


def draw_overlay(bgr: np.ndarray, frame: FrameRecord, speakers: SpeakersDoc, plan: CropPlan, index: int) -> np.ndarray:
    H, W = bgr.shape[:2]
    out = bgr.copy()
    sf = speakers.frames[index]
    seg = next(s for s in speakers.segments if s.start_frame <= index <= s.end_frame)
    thr = float(speakers.asd["speaking_threshold"])
    scores = sf.scores

    for face in frame.faces:
        b = face.bbox
        x0, y0, x1, y1 = int(b.x_min * W), int(b.y_min * H), int(b.x_max * W), int(b.y_max * H)
        color = face_color(face.face_id)
        active = sf.active_face_id == face.face_id
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 4 if active else 1)
        for x, y, _ in lip_landmarks(face):
            cv2.circle(out, (int(x * W), int(y * H)), 2, color, -1)
        label = f"id {face.face_id}" + ("  SPEAKING" if active else "")
        cv2.putText(out, label, (x0, max(14, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # score bar, scaled so the threshold sits at 40% of the bar
        score = scores.get(face.face_id)
        if score is not None:
            bar_w, bar_h = max(60, x1 - x0), 8
            by = min(H - bar_h - 2, y1 + 6)
            cv2.rectangle(out, (x0, by), (x0 + bar_w, by + bar_h), (60, 60, 60), -1)
            scale = 0.4 / thr if thr > 0 else 0.1
            frac = float(np.clip(0.4 + (score - thr) * scale, 0.0, 1.0)) if thr > 0 else float(np.clip(0.4 + score * 0.1, 0, 1))
            cv2.rectangle(out, (x0, by), (x0 + int(bar_w * frac), by + bar_h), color, -1)
            tx = x0 + int(bar_w * 0.4)
            cv2.line(out, (tx, by - 2), (tx, by + bar_h + 2), (255, 255, 255), 1)
            cv2.putText(out, f"{score:.3f}", (x0 + bar_w + 4, by + bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    c = plan.frames[index].crop
    cv2.rectangle(out, (c.x, c.y), (c.x + c.w - 1, c.y + c.h - 1), CROP_COLOR, 2)
    cv2.putText(out, f"crop {c.x},{c.y} {c.w}x{c.h}", (c.x + 4, min(H - 6, c.y + c.h - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CROP_COLOR, 1)

    hud = (f"frame {index}  t={sf.timestamp_ms / 1000:6.2f}s  seg {seg.segment_id} speaker {seg.speaker_face_id}  "
           f"asd={speakers.asd['backend']} thr={thr:g}")

    # HUD + timeline live in a strip below the frame so they never occlude drawn data.
    n = len(speakers.frames)
    strip = np.zeros((TIMELINE_H, W, 3), dtype=np.uint8)
    cv2.putText(strip, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    top, bottom = 24, TIMELINE_H - 4
    for s in speakers.segments:
        x0, x1 = int(s.start_frame / n * W), int((s.end_frame + 1) / n * W)
        col = face_color(s.speaker_face_id) if s.speaker_face_id is not None else (90, 90, 90)
        cv2.rectangle(strip, (x0, top), (max(x0, x1 - 1), bottom), col, -1)
    px = int(index / n * W)
    cv2.line(strip, (px, top - 2), (px, TIMELINE_H), (255, 255, 255), 2)
    return np.vstack([out, strip])


def load_phase_artifacts(phase1_dir: Path, phase2_dir: Path) -> tuple[list[FrameRecord], SpeakersDoc, CropPlan]:
    manifest = json.loads((phase1_dir / "manifest.json").read_text(encoding="utf-8"))
    with (phase1_dir / manifest["frames_file"]).open(encoding="utf-8") as fh:
        frames = [FrameRecord.from_dict(f) for f in iter_frames_jsonl(fh)]
    speakers = SpeakersDoc.from_dict(json.loads((phase2_dir / "speakers.json").read_text(encoding="utf-8")))
    plan = CropPlan.from_dict(json.loads((phase2_dir / "crop_plan.json").read_text(encoding="utf-8")))
    if not (len(frames) == len(speakers.frames) == len(plan.frames)):
        raise ValueError("Phase 1/2 artifacts disagree on frame count; re-run the stages together")
    return frames, speakers, plan


def render_debug(clip_path: Path, phase1_dir: Path, phase2_dir: Path, out_path: Path, crf: int = 23) -> Path:
    frames, speakers, plan = load_phase_artifacts(phase1_dir, phase2_dir)
    cap = cv2.VideoCapture(str(clip_path))
    fps = float(plan.source["fps"])
    W, H = int(plan.source["width"]), int(plan.source["height"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "debug_video_only.mp4"
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + TIMELINE_H))
        for i in range(len(frames)):
            ok, bgr = cap.read()
            if not ok:
                break
            writer.write(draw_overlay(bgr, frames[i], speakers, plan, i))
        cap.release()
        writer.release()
        mux(tmp, clip_path, out_path, crf=crf, preset="veryfast")
    return out_path
