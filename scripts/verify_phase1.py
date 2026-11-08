"""Phase 1 checkpoint verification.

Builds a short synthetic test video (a public MediaPipe sample portrait panned across
the frame), runs extraction, validates the output against the contract, checks that
lip landmarks are present and sane, and renders an annotated frame for inspection.

Usage: python scripts/verify_phase1.py [--video PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facepipe.cli import validate_output_dir  # noqa: E402
from facepipe.contract import (  # noqa: E402
    LIP_LANDMARK_INDICES,
    LOWER_LIP_CENTER,
    UPPER_LIP_CENTER,
    FrameRecord,
    iter_frames_jsonl,
    lip_landmarks,
)
from facepipe.extract import extract_video  # noqa: E402

SAMPLE_IMAGE_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"


def ensure_sample_video(work_dir: Path, frames: int = 60, fps: float = 30.0) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    img_path = work_dir / "portrait.jpg"
    if not img_path.is_file():
        urllib.request.urlretrieve(SAMPLE_IMAGE_URL, img_path)  # noqa: S310
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"could not read {img_path}")
    scale = 480 / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * scale), 480))
    h, w = img.shape[:2]
    canvas_w, canvas_h = w + 160, h + 80

    video_path = work_dir / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))
    for i in range(frames):
        canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)
        dx = int(80 + 60 * np.sin(2 * np.pi * i / frames))
        dy = int(40 + 30 * np.cos(2 * np.pi * i / frames))
        canvas[dy : dy + h, dx : dx + w] = img
        writer.write(canvas)
    writer.release()
    return video_path


def render_annotated(video_path: Path, out_dir: Path, frame_index: int) -> Path:
    with (out_dir / "frames.jsonl").open(encoding="utf-8") as fh:
        rec = next(FrameRecord.from_dict(f) for f in iter_frames_jsonl(fh) if f["frame_index"] == frame_index)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read frame for annotation")
    h, w = bgr.shape[:2]
    for face in rec.faces:
        b = face.bbox
        cv2.rectangle(bgr, (int(b.x_min * w), int(b.y_min * h)), (int(b.x_max * w), int(b.y_max * h)), (0, 255, 0), 2)
        for x, y, _ in face.landmarks:
            cv2.circle(bgr, (int(x * w), int(y * h)), 1, (255, 200, 0), -1)
        for x, y, _ in lip_landmarks(face):
            cv2.circle(bgr, (int(x * w), int(y * h)), 2, (0, 0, 255), -1)
        cv2.putText(bgr, f"id {face.face_id}", (int(b.x_min * w), int(b.y_min * h) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    path = out_dir / f"annotated_frame_{frame_index:04d}.png"
    cv2.imwrite(str(path), bgr)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="use this video instead of the synthetic sample")
    ap.add_argument("--out", default="output/phase1_verify")
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out)
    video = Path(args.video) if args.video else ensure_sample_video(Path("assets"), frames=args.frames)

    t0 = time.perf_counter()
    manifest = extract_video(video, out_dir)
    elapsed = time.perf_counter() - t0

    checks: list[tuple[str, bool, str]] = []
    stats = validate_output_dir(out_dir)
    checks.append(("output validates against contract", True, json.dumps(stats)))
    checks.append(("every decoded frame written", manifest.video.frame_count == stats["frames"],
                   f"{stats['frames']} frames"))
    face_frac = manifest.frames_with_faces / max(1, manifest.video.frame_count)
    checks.append(("faces detected in >= 90% of frames", face_frac >= 0.9, f"{face_frac:.1%}"))
    checks.append(("single stable face_id across the clip", manifest.face_ids == [0], f"face_ids={manifest.face_ids}"))

    lip_ok, mouth_spans, inside_bbox = True, [], True
    with (out_dir / "frames.jsonl").open(encoding="utf-8") as fh:
        for f in iter_frames_jsonl(fh):
            for face in FrameRecord.from_dict(f).faces:
                lips = lip_landmarks(face)
                lip_ok &= len(lips) == len(LIP_LANDMARK_INDICES) == 40
                xs, ys = [p[0] for p in lips], [p[1] for p in lips]
                b = face.bbox
                inside_bbox &= min(xs) >= b.x_min - 1e-6 and max(xs) <= b.x_max + 1e-6
                inside_bbox &= min(ys) >= b.y_min - 1e-6 and max(ys) <= b.y_max + 1e-6
                # Lips sit in the lower half of the face box.
                lip_ok &= np.mean(ys) > (b.y_min + b.y_max) / 2
                mouth_spans.append(face.landmarks[LOWER_LIP_CENTER][1] - face.landmarks[UPPER_LIP_CENTER][1])
    checks.append(("40 lip landmarks per face, located in lower face", lip_ok, ""))
    checks.append(("lip landmarks lie inside face bbox", inside_bbox, ""))
    checks.append(("mouth aperture (lm14.y - lm13.y) non-negative", all(s >= 0 for s in mouth_spans),
                   f"mean={np.mean(mouth_spans):.4f}" if mouth_spans else "no faces"))

    annotated = render_annotated(video, out_dir, manifest.video.frame_count // 2)
    fps_proc = manifest.video.frame_count / elapsed if elapsed else 0

    print(f"video: {video} ({manifest.video.width}x{manifest.video.height} @ {manifest.video.fps:g} fps)")
    print(f"output: {out_dir}  |  processed {manifest.video.frame_count} frames in {elapsed:.2f}s ({fps_proc:.1f} fps)")
    print(f"annotated frame: {annotated}")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    print("PHASE 1 CHECKPOINT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
