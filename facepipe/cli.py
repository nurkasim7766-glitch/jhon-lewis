"""CLI: `facepipe extract VIDEO OUT_DIR` and `facepipe validate OUT_DIR`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from facepipe.contract import iter_frames_jsonl, validate_frame, validate_manifest
from facepipe.detector import DetectorConfig


def cmd_extract(args: argparse.Namespace) -> int:
    from facepipe.extract import extract_video

    config = DetectorConfig(
        max_faces=args.max_faces,
        min_face_detection_confidence=args.min_detection_confidence,
        model_path=Path(args.model) if args.model else None,
    )
    manifest = extract_video(args.video, args.out_dir, config)
    v = manifest.video
    print(
        f"wrote {v.frame_count} frames ({manifest.frames_with_faces} with faces, "
        f"{len(manifest.face_ids)} face ids) to {args.out_dir}"
    )
    return 0


def validate_output_dir(out_dir: Path) -> dict:
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    frames_with_faces = 0
    face_ids: set[int] = set()
    count = 0
    prev_ts = -1.0
    with (out_dir / manifest["frames_file"]).open(encoding="utf-8") as fh:
        for expected, frame in enumerate(iter_frames_jsonl(fh)):
            validate_frame(frame, expected_index=expected)
            if frame["timestamp_ms"] < prev_ts:
                raise ValueError(f"frame {expected}: timestamp_ms decreased")
            prev_ts = frame["timestamp_ms"]
            count += 1
            if frame["faces"]:
                frames_with_faces += 1
                face_ids.update(f["face_id"] for f in frame["faces"])
    if count != manifest["video"]["frame_count"]:
        raise ValueError(f"frames.jsonl has {count} frames, manifest says {manifest['video']['frame_count']}")
    if frames_with_faces != manifest["frames_with_faces"]:
        raise ValueError("frames_with_faces mismatch between manifest and frames.jsonl")
    if sorted(face_ids) != manifest["face_ids"]:
        raise ValueError("face_ids mismatch between manifest and frames.jsonl")
    return {"frames": count, "frames_with_faces": frames_with_faces, "face_ids": sorted(face_ids)}


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        stats = validate_output_dir(Path(args.out_dir))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {stats}")
    return 0


def cmd_speakers(args: argparse.Namespace) -> int:
    from facepipe.cropplan import CropConfig
    from facepipe.phase2 import run_phase2
    from facepipe.segmentation import SegmentationConfig

    seg_cfg = SegmentationConfig(min_segment_ms=args.min_segment_ms, switch_confirm_ms=args.switch_confirm_ms)
    crop_cfg = CropConfig(output_width=args.output_width, output_height=args.output_height)
    speakers, plan = run_phase2(args.phase1_dir, args.out_dir, backend=args.backend, seg_cfg=seg_cfg,
                                crop_cfg=crop_cfg, video_path=args.video)
    print(f"{args.backend}: {len(speakers.segments)} segments, speakers "
          f"{[s.speaker_face_id for s in speakers.segments]}; wrote speakers.json + crop_plan.json to {args.out_dir}")
    return 0


def cmd_validate_phase2(args: argparse.Namespace) -> int:
    from facepipe.contract_phase2 import Phase2ContractError, validate_crop_plan, validate_speakers

    out_dir = Path(args.out_dir)
    try:
        speakers = json.loads((out_dir / "speakers.json").read_text(encoding="utf-8"))
        plan = json.loads((out_dir / "crop_plan.json").read_text(encoding="utf-8"))
        validate_speakers(speakers)
        validate_crop_plan(plan)
    except (Phase2ContractError, FileNotFoundError, KeyError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(speakers['segments'])} segments, {len(plan['frames'])} crop frames, "
          f"output {plan['output']['width']}x{plan['output']['height']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="facepipe")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="Phase 1: run face detection + landmarks on a video")
    p.add_argument("video")
    p.add_argument("out_dir")
    p.add_argument("--max-faces", type=int, default=4)
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--model", help="path to face_landmarker.task (default: auto-download to ./models)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("validate", help="check a Phase 1 output directory against the contract")
    p.add_argument("out_dir")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("speakers", help="Phase 2: active speaker segments + crop plan from Phase 1 output")
    p.add_argument("phase1_dir")
    p.add_argument("out_dir")
    p.add_argument("--backend", choices=["heuristic", "lrasd"], default="heuristic")
    p.add_argument("--video", help="source video (needed by lrasd; defaults to manifest path)")
    p.add_argument("--min-segment-ms", type=float, default=1500.0)
    p.add_argument("--switch-confirm-ms", type=float, default=300.0)
    p.add_argument("--output-width", type=int, default=1080)
    p.add_argument("--output-height", type=int, default=1920)
    p.set_defaults(func=cmd_speakers)

    p = sub.add_parser("validate-phase2", help="check speakers.json and crop_plan.json against the contracts")
    p.add_argument("out_dir")
    p.set_defaults(func=cmd_validate_phase2)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
