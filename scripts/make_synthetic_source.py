"""Build a synthetic full-length "podcast" source + transcript for end-to-end testing.

Video: two copies of a public MediaPipe sample portrait (two tracked faces) on a dark
background, slow drift, 1280x720 @ 30 fps, with a 440 Hz / 660 Hz tone alternating
per speaker so the muxed audio is audibly aligned with the transcript.
Transcript: generated sentences with word timings (section 5.4 shape), two speakers,
with one deliberately strong, hook-laden passage so moment selection is testable.

Usage: python scripts/make_synthetic_source.py --out assets/podcast --minutes 3
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_IMAGE_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"

FILLER_SENTENCES = [
    "So yeah, we were talking about that earlier.", "And then we moved on to the next thing.",
    "It was fine, nothing special really.", "Um, I think the weather was okay that week.",
    "We had lunch and, you know, kept going.", "Right, and that was pretty much the plan.",
    "Anyway, the meeting ran a little long.", "I guess that is how these things go sometimes.",
    "Yeah, we checked the schedule again after that.", "Okay, so the next part is about the office move.",
    "The parking situation was kind of annoying.", "They said the forms would be ready on Tuesday.",
    "Well, we waited around for a bit.", "And the coffee machine was broken again.",
    "It took a while to sort out the paperwork.", "So that was the morning, more or less.",
]
HOOK_PASSAGE = [
    ("A", "Why did the first company fail after we raised 3 million dollars?"),
    ("A", "Honestly, the biggest mistake was that nobody on the team had ever talked to a customer."),
    ("B", "That is shocking, because everyone assumed the product was the problem."),
    ("A", "No, the truth is we were wrong about who actually paid for it."),
    ("A", "I realized that 80 percent of the revenue came from 5 customers we never even called."),
    ("B", "So what changed when you finally picked up the phone?"),
    ("A", "Everything changed, because the reason they stayed had nothing to do with features."),
    ("A", "They stayed because we were the only team that answered at 2 in the morning."),
    ("B", "That is the lesson, right? The important thing was never the roadmap."),
    ("A", "Exactly, and that lesson is worth more than the 3 million we lost."),
]


def _words_for(sentence: str, start: float, wpm: float, rng: random.Random) -> tuple[list[dict], float]:
    words = sentence.split()
    t = start
    out = []
    for w in words:
        dur = 60.0 / wpm * (0.6 + 0.08 * len(w)) * rng.uniform(0.85, 1.15)
        out.append({"word": w, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur
    return out, t


def make_transcript(minutes: float, hook_at_s: float, seed: int = 7) -> dict:
    rng = random.Random(seed)
    segments = []
    t = 0.5
    total = minutes * 60
    hook_done = False
    while t < total - 2:
        if not hook_done and t >= hook_at_s:
            for spk, sent in HOOK_PASSAGE:
                words, t_end = _words_for(sent, t, 165, rng)
                segments.append({"start": words[0]["start"], "end": words[-1]["end"], "text": sent,
                                 "speaker": spk, "words": words})
                t = t_end + rng.uniform(0.2, 0.5)
            hook_done = True
            continue
        spk = rng.choice("AB")
        sent = rng.choice(FILLER_SENTENCES)
        words, t_end = _words_for(sent, t, 130, rng)
        segments.append({"start": words[0]["start"], "end": words[-1]["end"], "text": sent, "speaker": spk,
                         "words": words})
        t = t_end + rng.uniform(0.3, 1.2)
    return {"language": "en", "segments": segments}


def make_video(out_path: Path, transcript: dict, minutes: float, fps: float = 30.0, W: int = 1280, H: int = 720) -> None:
    img_path = out_path.parent / "portrait.jpg"
    if not img_path.exists():
        urllib.request.urlretrieve(SAMPLE_IMAGE_URL, img_path)  # noqa: S310
    face = cv2.imread(str(img_path))
    # Head-and-shoulders crop so faces are podcast-sized (~200 px tall at 720p).
    ph, pw = face.shape[:2]
    face = face[int(ph * 0.02): int(ph * 0.42), int(pw * 0.25): int(pw * 0.75)]
    face = cv2.resize(face, (int(face.shape[1] * 420 / face.shape[0]), 420))
    fh, fw = face.shape[:2]
    n = int(minutes * 60 * fps)
    tmp = out_path.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n):
        canvas = np.full((H, W, 3), 30, dtype=np.uint8)
        t = i / fps
        for k, cx in enumerate((W * 0.27, W * 0.73)):
            dx = int(cx - fw / 2 + 25 * np.sin(2 * np.pi * t / 9 + k))
            dy = int(H / 2 - fh / 2 + 15 * np.cos(2 * np.pi * t / 7 + k))
            canvas[dy: dy + fh, dx: dx + fw] = face
        writer.write(canvas)
    writer.release()

    # Audio: a tone per speaker during their sentences, silence elsewhere.
    sr = 16000
    audio = np.zeros(int(minutes * 60 * sr) + sr, dtype=np.float32)
    for seg in transcript["segments"]:
        f = 440.0 if seg["speaker"] == "A" else 660.0
        s, e = int(seg["start"] * sr), int(seg["end"] * sr)
        tt = np.arange(e - s) / sr
        audio[s:e] = 0.2 * np.sin(2 * np.pi * f * tt) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * tt))
    wav = out_path.with_suffix(".wav")
    from scipy.io import wavfile

    wavfile.write(wav, sr, (audio * 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-i", str(wav), "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(out_path)], check=True)
    tmp.unlink()
    wav.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/podcast")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--hook-at", type=float, default=95.0, help="seconds into the source where the strong passage starts")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    transcript = make_transcript(args.minutes, args.hook_at)
    (out / "transcript.json").write_text(json.dumps(transcript, indent=1), encoding="utf-8")
    make_video(out / "source.mp4", transcript, args.minutes)
    print(f"wrote {out / 'source.mp4'} and {out / 'transcript.json'} ({len(transcript['segments'])} segments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
