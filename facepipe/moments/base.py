"""Swappable moment-scoring interface.

Candidates are produced by facepipe.moments.select and already satisfy the hard
filters (sentence-aligned, duration in range). A scorer only ranks them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from facepipe.transcript import Transcript


@dataclass
class Candidate:
    first_sentence: int
    last_sentence: int
    start_ms: float
    end_ms: float
    text: str
    score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_sentence": self.first_sentence, "last_sentence": self.last_sentence,
            "start_ms": self.start_ms, "end_ms": self.end_ms, "duration_ms": round(self.duration_ms, 3),
            "score": round(self.score, 6), "features": {k: round(v, 6) for k, v in self.features.items()},
            "text": self.text,
        }


class MomentScorer(ABC):
    name: str = "base"
    local_only: bool = True

    @abstractmethod
    def score(self, candidates: list[Candidate], transcript: Transcript) -> list[Candidate]:
        """Fill in .score (higher is better) and return the candidates (any order)."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


def get_scorer(name: str, **kwargs: Any) -> MomentScorer:
    if name == "heuristic":
        from facepipe.moments.heuristic import HeuristicScorer

        return HeuristicScorer(**kwargs)
    if name == "llm":
        from facepipe.moments.llm import LLMScorer

        return LLMScorer(**kwargs)
    raise ValueError(f"unknown moment scorer {name!r}; choose 'heuristic' (default) or 'llm'")
