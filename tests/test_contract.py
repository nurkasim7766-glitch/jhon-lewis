import json

import pytest

from facepipe.contract import (
    INNER_LIP_INDICES,
    LIP_LANDMARK_INDICES,
    NUM_LANDMARKS,
    OUTER_LIP_INDICES,
    BBox,
    ContractError,
    FaceRecord,
    FrameRecord,
    lip_landmarks,
    validate_frame,
)
from facepipe.tracker import BBox as TrackerBBox
from facepipe.tracker import IoUTracker, iou


def _face(face_id=0):
    return FaceRecord(
        face_id=face_id,
        bbox=BBox(0.1, 0.1, 0.5, 0.6),
        landmarks=[[i / NUM_LANDMARKS, 0.5, 0.0] for i in range(NUM_LANDMARKS)],
    )


def test_lip_indices_match_mediapipe_connections():
    from mediapipe.tasks.python import vision

    mp_lips = {c.start for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS} | {
        c.end for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    }
    assert set(LIP_LANDMARK_INDICES) == mp_lips
    assert len(LIP_LANDMARK_INDICES) == 40 == len(set(LIP_LANDMARK_INDICES))
    assert len(OUTER_LIP_INDICES) == len(INNER_LIP_INDICES) == 20


def test_frame_round_trip_and_lip_selection():
    frame = FrameRecord(frame_index=3, timestamp_ms=100.0, faces=[_face()])
    restored = FrameRecord.from_dict(json.loads(json.dumps(frame.to_dict())))
    assert restored == frame
    lips = lip_landmarks(restored.faces[0])
    assert len(lips) == 40
    assert lips[0] == restored.faces[0].landmarks[61]


def test_validate_frame_rejects_bad_landmark_count():
    d = FrameRecord(frame_index=0, timestamp_ms=0.0, faces=[_face()]).to_dict()
    d["faces"][0]["landmarks"].pop()
    with pytest.raises(ContractError, match="478"):
        validate_frame(d)


def test_validate_frame_rejects_duplicate_ids_and_wrong_index():
    d = FrameRecord(frame_index=0, timestamp_ms=0.0, faces=[_face(1), _face(1)]).to_dict()
    with pytest.raises(ContractError, match="duplicate"):
        validate_frame(d)
    d = FrameRecord(frame_index=5, timestamp_ms=0.0).to_dict()
    with pytest.raises(ContractError, match="expected frame_index 4"):
        validate_frame(d, expected_index=4)


def test_tracker_keeps_ids_stable_and_allocates_new():
    t = IoUTracker(iou_threshold=0.3, max_missing_frames=2)
    a = TrackerBBox(0.1, 0.1, 0.3, 0.3)
    b = TrackerBBox(0.6, 0.6, 0.9, 0.9)
    assert t.update([a, b], 0) == [0, 1]
    a2 = TrackerBBox(0.12, 0.11, 0.32, 0.31)
    assert t.update([b, a2], 1) == [1, 0]  # order swap keeps identity
    assert t.update([], 2) == []
    assert t.update([a2], 3) == [0]  # within gap tolerance
    assert t.update([], 4) == [] and t.update([], 5) == [] and t.update([], 6) == []
    assert t.update([a2], 7) == [2]  # track expired -> new id
    assert iou(a, a) == pytest.approx(1.0) and iou(a, b) == 0.0
