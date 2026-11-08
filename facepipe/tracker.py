"""Greedy IoU tracker that assigns stable face_id values across frames."""

from __future__ import annotations

from dataclasses import dataclass

from facepipe.contract import BBox


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    iy = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    return inter / (area_a + area_b - inter)


@dataclass
class _Track:
    face_id: int
    bbox: BBox
    last_seen: int


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._tracks: list[_Track] = []
        self._next_id = 0

    def update(self, bboxes: list[BBox], frame_index: int) -> list[int]:
        """Return a face_id for each bbox, in input order."""
        self._tracks = [t for t in self._tracks if frame_index - t.last_seen <= self.max_missing_frames]

        pairs = sorted(
            ((iou(t.bbox, b), ti, bi) for ti, t in enumerate(self._tracks) for bi, b in enumerate(bboxes)),
            reverse=True,
        )
        assigned: dict[int, int] = {}
        used_tracks: set[int] = set()
        for score, ti, bi in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_tracks or bi in assigned:
                continue
            assigned[bi] = ti
            used_tracks.add(ti)

        ids: list[int] = []
        for bi, b in enumerate(bboxes):
            if bi in assigned:
                track = self._tracks[assigned[bi]]
                track.bbox, track.last_seen = b, frame_index
            else:
                track = _Track(face_id=self._next_id, bbox=b, last_seen=frame_index)
                self._next_id += 1
                self._tracks.append(track)
            ids.append(track.face_id)
        return ids
