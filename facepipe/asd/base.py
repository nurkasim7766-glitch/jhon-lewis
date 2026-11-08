"""Swappable active-speaker-detection interface.

A backend scores every (frame, face) pair. Scores live on a backend-specific scale;
`speaking_threshold` marks the value above which a face counts as speaking. Decision
making (hysteresis, segmentation) is shared and lives in facepipe.segmentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from facepipe.phase1_io import Phase1Data


@dataclass
class ASDResult:
    backend: str
    params: dict[str, Any]
    speaking_threshold: float
    scores: dict[int, np.ndarray]  # face_id -> per-frame score, NaN where face absent

    def visible(self, face_id: int, frame_index: int) -> bool:
        return not np.isnan(self.scores[face_id][frame_index])


class ASDBackend(ABC):
    name: str = "base"

    @abstractmethod
    def score(self, data: Phase1Data, video_path: Path | None = None) -> ASDResult:
        """Return a score series per face_id covering every frame in `data`."""


def get_backend(name: str, **kwargs: Any) -> ASDBackend:
    if name == "heuristic":
        from facepipe.asd.heuristic import LipMotionASD

        return LipMotionASD(**kwargs)
    if name == "lrasd":
        from facepipe.asd.lrasd import LRASD

        return LRASD(**kwargs)
    raise ValueError(f"unknown ASD backend {name!r}; choose 'heuristic' (default) or 'lrasd'")
