"""Phase 2 pipeline: Phase 1 output -> speakers.json (5.2) + crop_plan.json (5.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from facepipe.asd import get_backend
from facepipe.contract_phase2 import CropPlan, SpeakerFrame, SpeakersDoc, validate_crop_plan, validate_speakers
from facepipe.cropplan import CropConfig, plan_crops
from facepipe.phase1_io import Phase1Data, load_phase1
from facepipe.segmentation import SegmentationConfig, segment_speakers


def run_phase2(
    phase1_dir: str | Path,
    out_dir: str | Path,
    backend: str = "heuristic",
    backend_kwargs: dict[str, Any] | None = None,
    seg_cfg: SegmentationConfig | None = None,
    crop_cfg: CropConfig | None = None,
    video_path: str | Path | None = None,
    on_step: Callable[[str], None] | None = None,
) -> tuple[SpeakersDoc, CropPlan]:
    data = load_phase1(phase1_dir)
    return run_phase2_on_data(data, out_dir, backend, backend_kwargs, seg_cfg, crop_cfg, video_path, on_step)


def run_phase2_on_data(
    data: Phase1Data,
    out_dir: str | Path,
    backend: str = "heuristic",
    backend_kwargs: dict[str, Any] | None = None,
    seg_cfg: SegmentationConfig | None = None,
    crop_cfg: CropConfig | None = None,
    video_path: str | Path | None = None,
    on_step: Callable[[str], None] | None = None,
) -> tuple[SpeakersDoc, CropPlan]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_cfg = seg_cfg or SegmentationConfig()
    step = on_step or (lambda s: None)

    step("asd")
    asd = get_backend(backend, **(backend_kwargs or {})).score(data, Path(video_path) if video_path else None)
    step("segmentation")
    active, segments = segment_speakers(data, asd, seg_cfg)

    frames = []
    for i, fr in enumerate(data.frames):
        scores = {f.face_id: float(round(asd.scores[f.face_id][i], 5)) for f in fr.faces
                  if not np.isnan(asd.scores[f.face_id][i])}
        frames.append(SpeakerFrame(frame_index=i, timestamp_ms=fr.timestamp_ms, scores=scores, active_face_id=active[i]))
    speakers = SpeakersDoc(
        video=data.manifest["video"],
        asd={"backend": asd.backend, "params": asd.params, "speaking_threshold": asd.speaking_threshold},
        segmentation=seg_cfg.to_dict(),
        frames=frames,
        segments=segments,
    )
    step("cropplan")
    plan = plan_crops(data, segments, crop_cfg)

    speakers_dict, plan_dict = speakers.to_dict(), plan.to_dict()
    validate_speakers(speakers_dict)
    validate_crop_plan(plan_dict)
    (out_dir / "speakers.json").write_text(json.dumps(speakers_dict, separators=(",", ":")), encoding="utf-8")
    (out_dir / "crop_plan.json").write_text(json.dumps(plan_dict, separators=(",", ":")), encoding="utf-8")
    return speakers, plan
