"""Candidate generation (hard filters) and moment.json (section 5.5).

Hard filters, applied before any scoring:
  * a candidate starts exactly at a sentence start and ends exactly at a sentence end
  * min_ms <= duration <= max_ms
  * optional lead/tail padding is applied *after* selection and is clamped so it never
    crosses into a neighbouring sentence

moment.json:
{
  "schema_version": "1.0",
  "source": {"video": path, "transcript": path, "transcript_duration_ms": float},
  "selection": {"min_ms", "max_ms", "pad_in_ms", "pad_out_ms", "top_n", "scorer": {name, ...}, "num_candidates"},
  "moment": {"start_ms", "end_ms", "duration_ms", "first_sentence", "last_sentence",
             "sentence_start_ms", "sentence_end_ms", "text", "score", "features"},
  "candidates": [top-N candidate dicts, best first]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from facepipe.moments.base import Candidate, MomentScorer, get_scorer
from facepipe.transcript import Transcript

MOMENT_SCHEMA_VERSION = "1.0"


@dataclass
class SelectionConfig:
    min_ms: float = 30_000.0
    max_ms: float = 60_000.0
    pad_in_ms: float = 250.0
    pad_out_ms: float = 400.0
    top_n: int = 10
    scorer: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {"min_ms": self.min_ms, "max_ms": self.max_ms, "pad_in_ms": self.pad_in_ms,
                "pad_out_ms": self.pad_out_ms, "top_n": self.top_n, "scorer": self.scorer}


class SelectionError(ValueError):
    pass


def generate_candidates(transcript: Transcript, cfg: SelectionConfig) -> list[Candidate]:
    sents = transcript.sentences
    out: list[Candidate] = []
    for i, first in enumerate(sents):
        for j in range(i, len(sents)):
            dur = sents[j].end_ms - first.start_ms
            if dur > cfg.max_ms:
                break
            if dur >= cfg.min_ms:
                out.append(Candidate(i, j, first.start_ms, sents[j].end_ms, transcript.text_between(i, j)))
    return out


def _pad(c: Candidate, transcript: Transcript, cfg: SelectionConfig) -> tuple[float, float]:
    sents = transcript.sentences
    prev_end = sents[c.first_sentence - 1].end_ms if c.first_sentence > 0 else 0.0
    next_start = sents[c.last_sentence + 1].start_ms if c.last_sentence + 1 < len(sents) else float("inf")
    start = max(prev_end, c.start_ms - cfg.pad_in_ms, 0.0)
    end = min(next_start, c.end_ms + cfg.pad_out_ms)
    return round(start, 3), round(end, 3)


def select_moment(transcript: Transcript, video_path: str | Path, cfg: SelectionConfig | None = None,
                  scorer: MomentScorer | None = None) -> dict[str, Any]:
    cfg = cfg or SelectionConfig()
    scorer = scorer or get_scorer(cfg.scorer)
    candidates = generate_candidates(transcript, cfg)
    if not candidates:
        raise SelectionError(
            f"no sentence-aligned window between {cfg.min_ms / 1000:g}s and {cfg.max_ms / 1000:g}s exists in "
            f"{transcript.path} ({len(transcript.sentences)} sentences, {transcript.duration_ms / 1000:.1f}s)"
        )
    ranked = sorted(scorer.score(candidates, transcript), key=lambda c: (c.score, -c.start_ms), reverse=True)
    best = ranked[0]
    start, end = _pad(best, transcript, cfg)
    return {
        "schema_version": MOMENT_SCHEMA_VERSION,
        "source": {"video": str(video_path), "transcript": transcript.path,
                   "transcript_duration_ms": transcript.duration_ms},
        "selection": {**cfg.to_dict(), "scorer": scorer.describe(), "num_candidates": len(candidates)},
        "moment": {
            "start_ms": start, "end_ms": end, "duration_ms": round(end - start, 3),
            "first_sentence": best.first_sentence, "last_sentence": best.last_sentence,
            "sentence_start_ms": best.start_ms, "sentence_end_ms": best.end_ms,
            "text": best.text, "score": round(best.score, 6),
            "features": {k: round(v, 6) for k, v in best.features.items()},
        },
        "candidates": [c.to_dict() for c in ranked[: cfg.top_n]],
    }


def validate_moment(doc: dict[str, Any], transcript: Transcript | None = None) -> None:
    if doc.get("schema_version") != MOMENT_SCHEMA_VERSION:
        raise SelectionError("moment.json schema_version mismatch")
    m = doc["moment"]
    sel = doc["selection"]
    for key in ("start_ms", "end_ms", "duration_ms", "first_sentence", "last_sentence", "sentence_start_ms",
                "sentence_end_ms", "text", "score"):
        if key not in m:
            raise SelectionError(f"moment missing {key!r}")
    if not (m["start_ms"] <= m["sentence_start_ms"] < m["sentence_end_ms"] <= m["end_ms"]):
        raise SelectionError("moment padding must enclose the sentence span")
    sent_dur = m["sentence_end_ms"] - m["sentence_start_ms"]
    if not (sel["min_ms"] - 1e-6 <= sent_dur <= sel["max_ms"] + 1e-6):
        raise SelectionError(f"sentence span {sent_dur:.0f} ms outside [{sel['min_ms']}, {sel['max_ms']}]")
    if transcript is not None:
        s0 = transcript.sentences[m["first_sentence"]]
        s1 = transcript.sentences[m["last_sentence"]]
        if abs(s0.start_ms - m["sentence_start_ms"]) > 1e-3 or abs(s1.end_ms - m["sentence_end_ms"]) > 1e-3:
            raise SelectionError("moment is not aligned to sentence boundaries of the transcript")
        if m["first_sentence"] > 0 and m["start_ms"] < transcript.sentences[m["first_sentence"] - 1].end_ms - 1e-3:
            raise SelectionError("lead padding crosses into the previous sentence")
        if m["last_sentence"] + 1 < len(transcript.sentences) and \
                m["end_ms"] > transcript.sentences[m["last_sentence"] + 1].start_ms + 1e-3:
            raise SelectionError("tail padding crosses into the next sentence")


def write_moment(doc: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
