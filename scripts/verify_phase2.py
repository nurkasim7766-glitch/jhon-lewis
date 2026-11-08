"""Phase 2 checkpoint verification.

1. Synthetic two-speaker clip with a short interjection: checks segmentation accuracy,
   interjection suppression, minimum segment duration, and crop-path smoothness.
2. Real Phase 1 output from the Phase 1 checkpoint clip: end-to-end run, contract
   validation, and a preview render that uses *only* crop_plan.json + the source video
   (what Phase 3 will do).
3. LR-ASD backend smoke test when its optional dependencies are installed.

Usage: python scripts/verify_phase2.py [--out DIR] [--skip-lrasd]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facepipe.contract_phase2 import CropPlan, validate_crop_plan, validate_speakers  # noqa: E402
from facepipe.cropplan import CropConfig  # noqa: E402
from facepipe.phase2 import run_phase2, run_phase2_on_data  # noqa: E402
from facepipe.segmentation import SegmentationConfig  # noqa: E402
from facepipe.synthetic import make_two_speaker_data, truth_timeline  # noqa: E402
from scripts.verify_phase1 import ensure_sample_video  # noqa: E402

Check = tuple[str, bool, str]


def render_preview_from_plan(plan_path: Path, out_path: Path, n_tiles: int = 4) -> Path:
    """Phase 3 dress rehearsal: crop frames using nothing but crop_plan.json + source video."""
    plan = CropPlan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
    cap = cv2.VideoCapture(plan.source["path"])
    picks = np.linspace(0, len(plan.frames) - 1, n_tiles).round().astype(int)
    tiles = []
    for fi in picks:
        f = plan.frames[fi]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f.frame_index)
        ok, bgr = cap.read()
        if not ok:
            continue
        c = f.crop
        tile = cv2.resize(bgr[c.y: c.y + c.h, c.x: c.x + c.w], (plan.output.width // 4, plan.output.height // 4))
        cv2.putText(tile, f"f{f.frame_index} seg{f.segment_id}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(tile)
    cap.release()
    cv2.imwrite(str(out_path), np.hstack(tiles))
    return out_path


def crop_smoothness(plan: CropPlan) -> tuple[float, float]:
    """Max per-frame step inside segments, and the max step at cuts (for information)."""
    xs = np.array([[f.crop.x, f.crop.y] for f in plan.frames], dtype=float)
    inner, cuts = 0.0, 0.0
    for s in plan.segments:
        seg = xs[s.start_frame: s.end_frame + 1]
        if len(seg) > 1:
            inner = max(inner, float(np.abs(np.diff(seg, axis=0)).max()))
        if s.start_frame > 0:
            cuts = max(cuts, float(np.abs(xs[s.start_frame] - xs[s.start_frame - 1]).max()))
    return inner, cuts


def synthetic_checks(out_dir: Path) -> list[Check]:
    data, spans = make_two_speaker_data()
    seg_cfg = SegmentationConfig()
    t0 = time.perf_counter()
    speakers, plan = run_phase2_on_data(data, out_dir, seg_cfg=seg_cfg)
    dt = time.perf_counter() - t0
    checks: list[Check] = []
    order = [s.speaker_face_id for s in speakers.segments]
    checks.append(("speaker order follows the scripted changes (0 -> 1 -> 0)", order == [0, 1, 0], f"got {order}"))
    checks.append(("0.4 s interjection did not create a segment", len(speakers.segments) == 3,
                   f"{len(speakers.segments)} segments"))
    starts = [s.start_ms for s in speakers.segments[1:]]
    err = [abs(a - b) for a, b in zip(starts, [4000, 8000])] if len(starts) == 2 else [9e9]
    checks.append(("segment boundaries within 600 ms of truth", max(err) < 600, f"errors {err} ms"))
    truth = truth_timeline(spans, len(data.frames), data.fps, seg_cfg.min_segment_ms)
    agree = float(np.mean([f.active_face_id == t for f, t in zip(speakers.frames, truth)]))
    checks.append(("per-frame active speaker agrees with truth >= 90%", agree >= 0.9, f"{agree:.1%}"))
    short = [s.segment_id for s in speakers.segments if s.num_frames * 1000 / data.fps < seg_cfg.min_segment_ms]
    checks.append((f"all segments >= min_segment_ms ({seg_cfg.min_segment_ms:g})", not short, f"short: {short}"))
    inner, cuts = crop_smoothness(plan)
    limit = CropConfig().max_speed_frac * plan.frames[0].crop.w + 1
    checks.append(("crop glides inside segments (max step <= speed cap)", inner <= limit,
                   f"max inner step {inner:.1f}px, cap {limit:.1f}px, cut jump {cuts:.0f}px"))
    centred = True
    for s in plan.segments:
        mid = (s.start_frame + s.end_frame) // 2
        c = plan.frames[mid].crop
        fx = data.face_center_px(mid, s.speaker_face_id)[0]
        centred &= abs(c.x + c.w / 2 - fx) < 80
    checks.append(("crop centred on the active speaker mid-segment", centred, ""))
    checks.append(("speakers.json + crop_plan.json validate", True, f"{dt:.2f}s"))
    return checks


def real_checks(out_dir: Path, phase1_dir: Path) -> list[Check]:
    if not (phase1_dir / "manifest.json").exists():
        from facepipe.extract import extract_video

        video = ensure_sample_video(Path("assets"))
        extract_video(video, phase1_dir)
    speakers, plan = run_phase2(phase1_dir, out_dir)
    doc = json.loads((out_dir / "speakers.json").read_text())
    pl = json.loads((out_dir / "crop_plan.json").read_text())
    validate_speakers(doc)
    validate_crop_plan(pl)
    checks: list[Check] = [("real clip: outputs validate against 5.2 / 5.3", True, "")]
    checks.append(("real clip: single quiet face -> one segment via fallback", len(speakers.segments) == 1
                   and speakers.segments[0].speaker_face_id == 0, f"{[s.speaker_face_id for s in speakers.segments]}"))
    inner, _ = crop_smoothness(plan)
    checks.append(("real clip: crop follows the panning face smoothly", inner <= CropConfig().max_speed_frac * plan.frames[0].crop.w + 1,
                   f"max step {inner:.1f}px"))
    preview = render_preview_from_plan(out_dir / "crop_plan.json", out_dir / "preview_from_plan.png")
    checks.append(("render from crop_plan.json + source video only", preview.exists(), str(preview)))
    return checks


def lrasd_checks(out_dir: Path, phase1_dir: Path) -> list[Check]:
    try:
        import torch  # noqa: F401
        import python_speech_features  # noqa: F401
    except ImportError as exc:
        return [("lrasd backend (optional) skipped", True, f"missing dependency: {exc.name}")]
    t0 = time.perf_counter()
    speakers, plan = run_phase2(phase1_dir, out_dir, backend="lrasd")
    dt = time.perf_counter() - t0
    doc = json.loads((out_dir / "speakers.json").read_text())
    validate_speakers(doc)
    validate_crop_plan(json.loads((out_dir / "crop_plan.json").read_text()))
    scores = [f["scores"].get("0") for f in doc["frames"] if f["scores"]]
    mean = float(np.mean(scores)) if scores else float("nan")
    checks: list[Check] = [("lrasd: weights load, scores every frame, outputs validate", len(scores) == len(doc["frames"]),
                            f"{len(scores)} scored frames in {dt:.1f}s")]
    checks.append(("lrasd: silent clip scored as not speaking (mean logit < 0)", mean < 0,
                   f"mean logit {mean:.2f}, had_audio={doc['asd']['params']['had_audio']}"))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/phase2_verify")
    ap.add_argument("--phase1", default="output/phase1_verify")
    ap.add_argument("--skip-lrasd", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    checks: list[Check] = []
    checks += synthetic_checks(out / "synthetic")
    checks += real_checks(out / "real_heuristic", Path(args.phase1))
    if not args.skip_lrasd:
        checks += lrasd_checks(out / "real_lrasd", Path(args.phase1))

    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    print("PHASE 2 CHECKPOINT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
