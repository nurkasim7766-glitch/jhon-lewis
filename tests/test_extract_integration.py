from pathlib import Path

import pytest

from facepipe.cli import validate_output_dir
from facepipe.contract import NUM_LANDMARKS
from facepipe.extract import extract_video
from scripts.verify_phase1 import ensure_sample_video

pytestmark = pytest.mark.integration


def test_extract_sample_video(tmp_path: Path):
    video = ensure_sample_video(tmp_path, frames=12)
    manifest = extract_video(video, tmp_path / "out")
    assert manifest.video.frame_count == 12
    assert manifest.frames_with_faces == 12
    assert manifest.face_ids == [0]
    stats = validate_output_dir(tmp_path / "out")
    assert stats["frames"] == 12
    first = (tmp_path / "out" / "frames.jsonl").read_text().splitlines()[0]
    assert first.count("[") >= NUM_LANDMARKS
