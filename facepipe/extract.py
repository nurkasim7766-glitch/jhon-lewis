"""Phase 1 pipeline: video -> manifest.json + frames.jsonl (see contract.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from facepipe.contract import FaceRecord, FrameRecord, Manifest, VideoInfo
from facepipe.detector import DetectorConfig, FaceLandmarkDetector
from facepipe.tracker import IoUTracker
from facepipe.video import iter_frames_rgb, open_video


def extract_video(
    video_path: str | Path,
    out_dir: str | Path,
    config: DetectorConfig | None = None,
    progress: Callable[[int], None] | None = None,
) -> Manifest:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = config or DetectorConfig()

    cap, meta = open_video(video_path)
    tracker = IoUTracker()
    frames_written = 0
    frames_with_faces = 0
    face_ids: set[int] = set()
    last_ts = 0.0

    frames_path = out_dir / "frames.jsonl"
    with FaceLandmarkDetector(config) as detector, frames_path.open("w", encoding="utf-8") as fh:
        try:
            for index, rgb in iter_frames_rgb(cap):
                last_ts = index * 1000.0 / meta.fps
                detections = detector.detect(rgb, int(round(last_ts)))
                ids = tracker.update([d.bbox for d in detections], index)
                faces = [
                    FaceRecord(face_id=fid, bbox=d.bbox, landmarks=d.landmarks.astype(float).round(6).tolist())
                    for fid, d in zip(ids, detections)
                ]
                record = FrameRecord(frame_index=index, timestamp_ms=round(last_ts, 3), faces=faces)
                fh.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
                frames_written += 1
                if faces:
                    frames_with_faces += 1
                    face_ids.update(ids)
                if progress:
                    progress(index)
        finally:
            cap.release()

    manifest = Manifest(
        video=VideoInfo(
            path=str(video_path),
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            frame_count=frames_written,
            duration_ms=round(frames_written * 1000.0 / meta.fps, 3),
        ),
        detector=config.to_manifest(detector.model_path),
        frames_with_faces=frames_with_faces,
        face_ids=sorted(face_ids),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest
