"""Default moment scorer: fully local lexical/prosodic heuristics, no network.

Each feature is scaled to roughly [0, 1] and combined with the weights below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from facepipe.moments.base import Candidate, MomentScorer
from facepipe.transcript import Transcript

HOOK_WORDS = {
    "why", "how", "what", "secret", "mistake", "never", "always", "nobody", "everyone", "truth", "actually",
    "realize", "realized", "biggest", "worst", "best", "first", "last", "only", "wrong", "right", "problem",
    "surprising", "shocking", "crazy", "insane", "honestly", "imagine", "because", "reason", "lesson", "learned",
    "money", "million", "billion", "percent", "years", "changed", "change", "important", "matter", "matters",
}
INTENSITY_WORDS = {
    "love", "hate", "amazing", "terrible", "incredible", "huge", "massive", "tiny", "fail", "failed", "failure",
    "win", "won", "lost", "lose", "afraid", "scared", "angry", "happy", "sad", "excited", "proud", "regret",
    "fight", "broke", "quit", "fired", "died", "killed", "risk", "danger", "impossible", "unbelievable",
}
FILLERS = {"um", "uh", "uhm", "hmm", "like,", "yeah,", "you know,", "i mean,", "sort of", "kind of"}
WEAK_OPENERS = {"and", "but", "so", "or", "because", "which", "that", "also", "then", "anyway", "yeah", "okay",
                "right", "um", "uh", "well"}
_WORD = re.compile(r"[a-z']+")


@dataclass
class HeuristicScorer(MomentScorer):
    name: str = "heuristic"
    target_ms: float = 45_000.0
    weights: dict[str, float] = field(default_factory=lambda: {
        "hook": 1.4, "intensity": 1.0, "question": 0.8, "numbers": 0.5, "density": 0.8, "speech_rate": 0.4,
        "dialogue": 0.6, "duration_fit": 0.7, "strong_opener": 1.0, "filler": -1.0, "clean_ending": 0.6,
    })

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "target_ms": self.target_ms, "weights": self.weights, "local_only": True}

    def features(self, c: Candidate, transcript: Transcript) -> dict[str, float]:
        sents = transcript.sentences[c.first_sentence: c.last_sentence + 1]
        text = c.text.lower()
        tokens = _WORD.findall(text)
        n = max(1, len(tokens))
        minutes = max(1e-3, c.duration_ms / 60_000)
        first_tokens = _WORD.findall(sents[0].text.lower())
        speakers = [s.speaker for s in sents if s.speaker]
        changes = sum(1 for a, b in zip(speakers, speakers[1:]) if a != b)
        rate = len(tokens) / minutes  # words per minute

        return {
            "hook": min(1.0, sum(t in HOOK_WORDS for t in tokens) / n * 12),
            "intensity": min(1.0, sum(t in INTENSITY_WORDS for t in tokens) / n * 15),
            "question": min(1.0, sum("?" in s.text for s in sents[:2]) + 0.3 * sum("?" in s.text for s in sents[2:])),
            "numbers": min(1.0, len(re.findall(r"\d", c.text)) / 6),
            "density": len(set(tokens)) / n if n >= 10 else 0.0,
            "speech_rate": float(np.clip((rate - 90) / 90, 0.0, 1.0)),
            "dialogue": min(1.0, changes / 4),
            "duration_fit": float(max(0.0, 1.0 - abs(c.duration_ms - self.target_ms) / self.target_ms)),
            "strong_opener": 0.0 if (first_tokens and first_tokens[0] in WEAK_OPENERS) else 1.0,
            "filler": min(1.0, sum(text.count(f) for f in FILLERS) / max(1, len(sents)) / 2),
            "clean_ending": 1.0 if sents[-1].text.rstrip().endswith((".", "!", "?")) else 0.0,
        }

    def score(self, candidates: list[Candidate], transcript: Transcript) -> list[Candidate]:
        for c in candidates:
            f = self.features(c, transcript)
            c.features = f
            c.score = float(sum(self.weights[k] * v for k, v in f.items()))
        return candidates
