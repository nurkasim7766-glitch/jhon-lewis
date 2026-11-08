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


def cmd_select(args: argparse.Namespace) -> int:
    from facepipe.moments import SelectionConfig, get_scorer, select_moment
    from facepipe.moments.select import write_moment
    from facepipe.transcript import load_transcript

    cfg = SelectionConfig(min_ms=args.min_ms, max_ms=args.max_ms, pad_in_ms=args.pad_in_ms,
                          pad_out_ms=args.pad_out_ms, top_n=args.top_n, scorer=args.scorer)
    scorer = get_scorer(args.scorer, **_llm_kwargs(args))
    doc = select_moment(load_transcript(args.transcript), args.video, cfg, scorer)
    write_moment(doc, args.out)
    m = doc["moment"]
    print(f"moment {m['start_ms'] / 1000:.2f}s-{m['end_ms'] / 1000:.2f}s ({m['duration_ms'] / 1000:.1f}s), "
          f"sentences {m['first_sentence']}-{m['last_sentence']}, score {m['score']:.3f} -> {args.out}")
    print(f"  \"{m['text'][:140]}{'...' if len(m['text']) > 140 else ''}\"")
    return 0


def cmd_cut(args: argparse.Namespace) -> int:
    from facepipe.pipeline import cut_clip

    m = json.loads(Path(args.moment).read_text(encoding="utf-8"))["moment"]
    cut_clip(Path(args.video), m["start_ms"], m["end_ms"], Path(args.out))
    print(f"cut {m['start_ms'] / 1000:.3f}s-{m['end_ms'] / 1000:.3f}s -> {args.out}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from facepipe.render import render

    out = render(args.crop_plan, args.clip, args.out, crf=args.crf, preset=args.preset)
    print(f"rendered {out}")
    return 0


def cmd_debug(args: argparse.Namespace) -> int:
    from facepipe.overlay import render_debug

    out = render_debug(Path(args.clip), Path(args.phase1_dir), Path(args.phase2_dir), Path(args.out))
    print(f"debug overlay {out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from facepipe.cropplan import CropConfig
    from facepipe.moments import SelectionConfig
    from facepipe.pipeline import Pipeline, PipelineConfig
    from facepipe.segmentation import SegmentationConfig

    if args.config:
        cfg = PipelineConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
        # explicit flags override the config file
        if args.video: cfg.video = args.video  # noqa: E701
        if args.transcript: cfg.transcript = args.transcript  # noqa: E701
        if args.workdir: cfg.workdir = args.workdir  # noqa: E701
    else:
        if not (args.video and args.transcript and args.workdir):
            print("run requires --video, --transcript and --workdir (or --config)", file=sys.stderr)
            return 2
        cfg = PipelineConfig(
            video=args.video, transcript=args.transcript, workdir=args.workdir,
            selection=SelectionConfig(min_ms=args.min_ms, max_ms=args.max_ms, pad_in_ms=args.pad_in_ms,
                                      pad_out_ms=args.pad_out_ms, top_n=args.top_n, scorer=args.scorer),
            asd_backend=args.backend,
            segmentation=SegmentationConfig(min_segment_ms=args.min_segment_ms, switch_confirm_ms=args.switch_confirm_ms),
            crop=CropConfig(output_width=args.output_width, output_height=args.output_height),
            llm=_llm_kwargs(args), render_crf=args.crf, render_preset=args.preset, debug=not args.no_debug,
        )
    pipeline = Pipeline(cfg, force=set(args.force or []), log=print)
    results = pipeline.run(until=args.until)
    ran = [r.name for r in results if r.status == "ran"]
    reused = [r.name for r in results if r.status == "reused"]
    print(f"finished: ran {ran or 'nothing'}, reused {reused or 'nothing'}; outputs in {cfg.workdir}")
    return 0


def _llm_kwargs(args: argparse.Namespace) -> dict:
    if getattr(args, "scorer", "heuristic") != "llm":
        return {}
    kw = {"allow_hosted": bool(args.allow_hosted_llm)}
    if args.llm_endpoint:
        kw["endpoint"] = args.llm_endpoint
    if args.llm_model:
        kw["model"] = args.llm_model
    return kw


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scorer", choices=["heuristic", "llm"], default="heuristic")
    p.add_argument("--min-ms", type=float, default=30_000.0, help="minimum sentence-span duration")
    p.add_argument("--max-ms", type=float, default=60_000.0, help="maximum sentence-span duration")
    p.add_argument("--pad-in-ms", type=float, default=250.0, help="lead-in before the first sentence (never crosses a sentence)")
    p.add_argument("--pad-out-ms", type=float, default=400.0, help="tail after the last sentence (never crosses a sentence)")
    p.add_argument("--top-n", type=int, default=10, help="candidates to keep in moment.json")
    p.add_argument("--llm-endpoint", help="OpenAI-compatible chat completions URL (local by default)")
    p.add_argument("--llm-model", help="model name for the LLM scorer")
    p.add_argument("--allow-hosted-llm", action="store_true", help="explicitly permit a non-local LLM endpoint")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="facepipe")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="full pipeline: transcript + source video -> output.mp4 (+ debug.mp4)")
    p.add_argument("--video")
    p.add_argument("--transcript")
    p.add_argument("--workdir")
    p.add_argument("--config", help="JSON file with a PipelineConfig (flags above override it)")
    _add_selection_args(p)
    p.add_argument("--backend", choices=["heuristic", "lrasd"], default="heuristic", help="active speaker detector")
    p.add_argument("--min-segment-ms", type=float, default=1500.0)
    p.add_argument("--switch-confirm-ms", type=float, default=300.0)
    p.add_argument("--output-width", type=int, default=1080)
    p.add_argument("--output-height", type=int, default=1920)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="medium")
    p.add_argument("--no-debug", action="store_true", help="skip the debug overlay video")
    p.add_argument("--force", action="append", choices=["all", "moment", "clip", "detect", "speakers", "render", "debug"],
                   help="re-run this stage and everything after it (repeatable)")
    p.add_argument("--until", choices=["moment", "clip", "detect", "speakers", "render", "debug"], default="debug",
                   help="stop after this stage")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("select", help="deliverable 0: choose the strongest sentence-aligned moment -> moment.json")
    p.add_argument("transcript")
    p.add_argument("video")
    p.add_argument("out")
    _add_selection_args(p)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("cut", help="cut the clip described by moment.json from the source video")
    p.add_argument("video")
    p.add_argument("moment")
    p.add_argument("out")
    p.set_defaults(func=cmd_cut)

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

    p = sub.add_parser("render", help="Phase 3: apply crop_plan.json to the clip -> output video with audio")
    p.add_argument("crop_plan")
    p.add_argument("clip")
    p.add_argument("out")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="medium")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("debug", help="Phase 3: debug overlay of tracking + speaker decisions + crop on the clip")
    p.add_argument("clip")
    p.add_argument("phase1_dir")
    p.add_argument("phase2_dir")
    p.add_argument("out")
    p.set_defaults(func=cmd_debug)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
