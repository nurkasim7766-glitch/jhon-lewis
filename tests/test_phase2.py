import numpy as np
import pytest

from facepipe.asd.heuristic import LipMotionASD
from facepipe.contract_phase2 import Phase2ContractError, validate_crop_plan, validate_speakers
from facepipe.cropplan import CropConfig, crop_size, plan_crops, smooth_path
from facepipe.phase2 import run_phase2_on_data
from facepipe.segmentation import SegmentationConfig, enforce_min_duration, segment_speakers
from facepipe.synthetic import SpeechSpan, make_two_speaker_data, truth_timeline


@pytest.fixture(scope="module")
def two_speakers():
    return make_two_speaker_data()


def test_heuristic_scores_separate_talking_from_quiet(two_speakers):
    data, spans = two_speakers
    asd = LipMotionASD().score(data)
    fps = data.fps
    # face 0 talks 0-4s, face 1 quiet during 0-2s
    assert np.nanmean(asd.scores[0][int(0.5 * fps): int(1.8 * fps)]) > asd.speaking_threshold
    assert np.nanmean(asd.scores[1][int(0.5 * fps): int(1.8 * fps)]) < asd.speaking_threshold


def test_segments_follow_speaker_changes_and_ignore_interjection(two_speakers):
    data, spans = two_speakers
    cfg = SegmentationConfig(min_segment_ms=1500.0)
    active, segments = segment_speakers(data, LipMotionASD().score(data), cfg)
    assert [s.speaker_face_id for s in segments] == [0, 1, 0]
    truth = truth_timeline(spans, len(data.frames), data.fps, cfg.min_segment_ms)
    agreement = np.mean([a == t for a, t in zip(active, truth)])
    assert agreement > 0.9
    for s in segments:
        assert s.num_frames * 1000.0 / data.fps >= cfg.min_segment_ms
    # boundaries within 0.6 s of the truth (4 s and 8 s)
    assert abs(segments[1].start_ms - 4000) < 600 and abs(segments[2].start_ms - 8000) < 600


def test_enforce_min_duration_absorbs_short_runs():
    fps = 10.0
    active = [0] * 30 + [1] * 3 + [0] * 30 + [1] * 25 + [0] * 2
    out = enforce_min_duration(active, fps, min_segment_ms=1000.0)
    assert out[:63] == [0] * 63
    assert out[63:88] == [1] * 25
    assert out[88:] == [1] * 2  # trailing stub joins its only neighbour
    assert len(out) == len(active)


def test_crop_size_fits_source():
    assert crop_size(1920, 1080, 1080, 1920) == (608, 1080)
    assert crop_size(1080, 1920, 1080, 1920) == (1080, 1920)
    assert crop_size(1080, 1920, 1920, 1080) == (1080, 608)


def test_smooth_path_glides_not_snaps():
    fps, cw = 30.0, 600
    cfg = CropConfig()
    targets = np.zeros((90, 2))
    targets[45:, 0] = 400.0  # abrupt 400 px jump in the raw target mid-segment
    targets[:, 0] += np.random.default_rng(1).normal(0, 2.0, 90)  # jitter
    path = smooth_path(targets, fps, cw, cfg)
    steps = np.abs(np.diff(path[:, 0]))
    assert steps.max() <= cfg.max_speed_frac * cw + 1e-6
    assert path[-1, 0] > 300  # it does arrive
    assert np.abs(np.diff(path[:40, 0])).max() < 1.0  # jitter inside the dead-zone is ignored


def test_phase2_outputs_validate_and_crop_tracks_speaker(two_speakers, tmp_path):
    data, _ = two_speakers
    speakers, plan = run_phase2_on_data(data, tmp_path)
    validate_speakers(speakers.to_dict())
    validate_crop_plan(plan.to_dict())
    assert plan.output.aspect_ratio == "9:16"
    for seg in plan.segments:
        mid = (seg.start_frame + seg.end_frame) // 2
        crop = plan.frames[mid].crop
        cx = crop.x + crop.w / 2
        face_cx = data.face_center_px(mid, seg.speaker_face_id)[0]
        assert abs(cx - face_cx) < 80
    xs = np.array([f.crop.x for f in plan.frames])
    for seg in plan.segments:
        inner = np.abs(np.diff(xs[seg.start_frame: seg.end_frame + 1]))
        assert inner.max() <= CropConfig().max_speed_frac * plan.frames[0].crop.w + 1


def test_validators_reject_same_speaker_adjacent_and_out_of_bounds(two_speakers, tmp_path):
    data, _ = two_speakers
    speakers, plan = run_phase2_on_data(data, tmp_path)
    bad = speakers.to_dict()
    bad["segments"][1]["speaker_face_id"] = bad["segments"][0]["speaker_face_id"]
    with pytest.raises(Phase2ContractError, match="same speaker"):
        validate_speakers(bad)
    bad_plan = plan.to_dict()
    bad_plan["frames"][0]["crop"]["x"] = -1
    with pytest.raises(Phase2ContractError, match="outside source"):
        validate_crop_plan(bad_plan)


def test_single_quiet_face_yields_one_segment_with_fallback():
    data, _ = make_two_speaker_data(duration_ms=3000, spans=[])
    data.frames = [type(f)(f.frame_index, f.timestamp_ms, [f.faces[0]]) for f in data.frames]
    data.manifest["face_ids"] = [0]
    active, segments = segment_speakers(data, LipMotionASD().score(data), SegmentationConfig())
    assert len(segments) == 1 and segments[0].speaker_face_id == 0
