"""Transcript input contract (section 5.4) and sentence segmentation.

Accepted input: a JSON file with a top-level ``segments`` list (the shape produced by
Whisper ``verbose_json``, faster-whisper, and WhisperX). Each segment has:

    start   float seconds      end  float seconds      text  str
    speaker optional str       words optional list of {word, start, end[, speaker]}

Word timings are used when present; otherwise sentence boundaries are interpolated
proportionally to character offsets within the segment. Times are converted to ms.

Sentences are the atomic unit for moment selection: a moment always starts at a
sentence start and ends at a sentence end.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRANSCRIPT_SCHEMA = "whisper-segments"
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)]*$")
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "vs.", "e.g.", "i.e.", "etc.", "st.", "jr.", "sr."}


@dataclass
class Word:
    text: str
    start_ms: float
    end_ms: float
    speaker: str | None = None


@dataclass
class Sentence:
    index: int
    start_ms: float
    end_ms: float
    text: str
    speaker: str | None
    words: list[Word] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "start_ms": self.start_ms, "end_ms": self.end_ms, "text": self.text,
                "speaker": self.speaker}


@dataclass
class Transcript:
    path: str
    sentences: list[Sentence]
    duration_ms: float
    speakers: list[str]

    def text_between(self, first: int, last: int) -> str:
        return " ".join(s.text for s in self.sentences[first: last + 1])


class TranscriptError(ValueError):
    pass


def _segment_words(seg: dict[str, Any]) -> list[Word]:
    words = seg.get("words")
    if words:
        out = []
        for w in words:
            txt = str(w.get("word", w.get("text", ""))).strip()
            if not txt:
                continue
            if "start" not in w or "end" not in w or w["start"] is None or w["end"] is None:
                continue  # WhisperX leaves numerals untimed; they are re-attached below
            out.append(Word(txt, float(w["start"]) * 1000, float(w["end"]) * 1000, w.get("speaker", seg.get("speaker"))))
        if out:
            return out
    # No usable word timings: distribute tokens over the segment by character length.
    tokens = str(seg["text"]).split()
    if not tokens:
        return []
    s, e = float(seg["start"]) * 1000, float(seg["end"]) * 1000
    total = sum(len(t) + 1 for t in tokens)
    out, cursor = [], s
    for t in tokens:
        span = (e - s) * (len(t) + 1) / total
        out.append(Word(t, cursor, cursor + span, seg.get("speaker")))
        cursor += span
    return out


def _is_sentence_end(word: str) -> bool:
    w = word.lower()
    if w in _ABBREVIATIONS:
        return False
    return bool(_SENTENCE_END.search(word))


def load_transcript(path: str | Path) -> Transcript:
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptError(f"{path}: not valid JSON ({exc})") from exc
    segments = doc.get("segments") if isinstance(doc, dict) else doc
    if not isinstance(segments, list) or not segments:
        raise TranscriptError(f"{path}: expected a non-empty 'segments' list")
    for i, seg in enumerate(segments):
        for key in ("start", "end", "text"):
            if key not in seg:
                raise TranscriptError(f"{path}: segment {i} missing {key!r}")
        if float(seg["end"]) < float(seg["start"]):
            raise TranscriptError(f"{path}: segment {i} ends before it starts")

    words: list[Word] = []
    for seg in segments:
        words.extend(_segment_words(seg))
    if not words:
        raise TranscriptError(f"{path}: transcript has no words")

    sentences: list[Sentence] = []
    buf: list[Word] = []
    for i, w in enumerate(words):
        buf.append(w)
        speaker_change = i + 1 < len(words) and words[i + 1].speaker != w.speaker and w.speaker is not None
        if _is_sentence_end(w.text) or speaker_change or i == len(words) - 1:
            sentences.append(Sentence(
                index=len(sentences), start_ms=round(buf[0].start_ms, 3), end_ms=round(buf[-1].end_ms, 3),
                text=" ".join(x.text for x in buf), speaker=buf[0].speaker, words=buf,
            ))
            buf = []
    speakers = sorted({s.speaker for s in sentences if s.speaker})
    return Transcript(path=str(path), sentences=sentences, duration_ms=sentences[-1].end_ms, speakers=speakers)
