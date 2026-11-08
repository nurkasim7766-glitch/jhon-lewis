"""Phase 3 checkpoint verification.

Runs the whole pipeline with the single `facepipe run` command on the synthetic
full-length source (scripts/make_synthetic_source.py) and checks:

  moment     sentence-aligned, in range, lands on the hook passage, padding valid
  clip       duration matches moment.json, has audio
  output     1080x1920, frame count == crop_plan frames, has audio, pixels equal the
             crop of the source clip described by crop_plan.json (renderer fidelity)
  debug      exists, frame count matches, and its drawn crop rectangle sits exactly
             where crop_plan.json says (overlay fidelity)
  reuse      a second run re-uses every stage; --force render re-runs only render+debug
  llm guard  scorer refuses a hosted endpoint

Usage: python scripts/verify_phase3.py [--source assets/podcast] [--work output/phase3_verify]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facepipe.contract_phase2 import CropPlan  # noqa: E402
from facepipe.moments.llm import HostedEndpointRefused, LLMScorer  # noqa: E402
from facepipe.moments.select import validate_moment  # noqa: E402
from facepipe.overlay import CROP_COLOR, TIMELINE_H  # noqa: E402
from facepipe.render import probe  # noqa: E402
from facepipe.transcript import load_transcript  # noqa: E402
from scripts.make_synthetic_source import HOOK_PASSAGE  # noqa: E402

Check = tuple[str, bool, str]


def run_cli(*args: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "facepipe.cli", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"facepipe {' '.join(args)} failed:\n{proc.stderr[-2000:]}")
    return proc.stdout, time.perf_counter() - t0


def stream_info(path: Path) -> dict:
    p = probe(path)
    v = next(s for s in p["streams"] if s["codec_type"] == "video")
    return {"w": int(v["width"]), "h": int(v["height"]), "frames": int(v.get("nb_frames", 0)),
            "audio": any(s["codec_type"] == "audio" for s in p["streams"]),
            "duration": float(p["format"]["duration"])}


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {path}")
    return bgr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="assets/podcast")
    ap.add_argument("--work", default="output/phase3_verify")
    args = ap.parse_args()
    src = Path(args.source)
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    video, transcript_path = src / "source.mp4", src / "transcript.json"
    if not video.exists():
        subprocess.run([sys.executable, "scripts/make_synthetic_source.py", "--out", str(src)], check=True)

    checks: list[Check] = []

    # 1. single-command end-to-end run
    out1, t1 = run_cli("run", "--video", str(video), "--transcript", str(transcript_path), "--workdir", str(work),
                       "--preset", "veryfast")
    run_json = json.loads((work / "run.json").read_text())
    statuses = {s["name"]: s["status"] for s in run_json["stages"]}
    checks.append(("single command ran all six stages", all(v == "ran" for v in statuses.values()),
                   f"{t1:.0f}s total; {statuses}"))

    # 2. moment
    moment = json.loads((work / "moment.json").read_text())
    transcript = load_transcript(transcript_path)
    validate_moment(moment, transcript)
    m = moment["moment"]
    checks.append(("moment.json validates (sentence-aligned, in range, padding sane)", True,
                   f"{m['start_ms'] / 1000:.1f}s-{m['end_ms'] / 1000:.1f}s, sentences {m['first_sentence']}-{m['last_sentence']}"))
    first_words = transcript.sentences[m["first_sentence"]].text.split()[:3]
    checks.append(("clip opens on a sentence start", m["sentence_start_ms"] == transcript.sentences[m["first_sentence"]].start_ms,
                   f"opens with: {' '.join(first_words)} ..."))
    hook_hit = sum(h in m["text"] for _, h in HOOK_PASSAGE)
    checks.append(("heuristic selected the strong passage", hook_hit >= 6, f"{hook_hit}/{len(HOOK_PASSAGE)} hook sentences inside"))
    checks.append(("scorer is local-only", moment["selection"]["scorer"].get("local_only") is True,
                   moment["selection"]["scorer"]["name"]))

    # 3. clip
    clip = stream_info(work / "clip.mp4")
    checks.append(("clip duration matches moment", abs(clip["duration"] - m["duration_ms"] / 1000) < 0.15,
                   f"clip {clip['duration']:.2f}s vs moment {m['duration_ms'] / 1000:.2f}s; audio={clip['audio']}"))

    # 4. render fidelity
    plan = CropPlan.from_dict(json.loads((work / "phase2" / "crop_plan.json").read_text()))
    out = stream_info(work / "output.mp4")
    checks.append(("output is 1080x1920 with audio and one frame per plan frame",
                   (out["w"], out["h"]) == (1080, 1920) and out["audio"] and out["frames"] == len(plan.frames),
                   f"{out['w']}x{out['h']}, {out['frames']} frames (plan {len(plan.frames)}), audio={out['audio']}"))
    idx = len(plan.frames) // 2
    c = plan.frames[idx].crop
    expected = cv2.resize(read_frame(work / "clip.mp4", idx)[c.y: c.y + c.h, c.x: c.x + c.w], (1080, 1920), interpolation=cv2.INTER_AREA)
    got = read_frame(work / "output.mp4", idx)
    diff = float(np.abs(expected.astype(int) - got.astype(int)).mean())
    checks.append(("rendered frame equals crop_plan crop of the clip (mean abs diff < 6)", diff < 6.0, f"diff {diff:.2f}"))

    # 5. debug overlay fidelity
    dbg = stream_info(work / "debug.mp4")
    checks.append(("debug.mp4 exists with matching frame count", dbg["frames"] == len(plan.frames) and dbg["h"] == plan.source["height"] + TIMELINE_H,
                   f"{dbg['w']}x{dbg['h']}, {dbg['frames']} frames"))
    frame = read_frame(work / "debug.mp4", idx)
    # The crop rectangle is drawn in CROP_COLOR; sample the midpoint of each edge.
    def is_green(y: int, x: int) -> bool:
        b, g, r = frame[y, x].astype(int)
        return g > 150 and r < 110 and b < 110
    pts = [(c.y, c.x + c.w // 2), (c.y + c.h - 1, c.x + c.w // 2), (c.y + c.h // 2, c.x), (c.y + c.h // 2, c.x + c.w - 1)]
    hits = sum(any(is_green(min(max(y + dy, 0), frame.shape[0] - 1), min(max(x + dx, 0), frame.shape[1] - 1))
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)) for y, x in pts)
    checks.append(("overlay crop rectangle drawn at crop_plan coordinates", hits == 4, f"{hits}/4 edges found at {c.to_dict()}"))
    speakers = json.loads((work / "phase2" / "speakers.json").read_text())
    checks.append(("overlay and renderer consume the same speakers/crop_plan files",
                   speakers["video"]["frame_count"] == len(plan.frames) == out["frames"], ""))

    # 6. reuse
    out2, t2 = run_cli("run", "--video", str(video), "--transcript", str(transcript_path), "--workdir", str(work))
    st2 = {s["name"]: s["status"] for s in json.loads((work / "run.json").read_text())["stages"]}
    checks.append(("second run reuses every stage", all(v == "reused" for v in st2.values()), f"{t2:.1f}s"))
    mtime_detect = (work / "phase1" / "frames.jsonl").stat().st_mtime
    out3, t3 = run_cli("run", "--video", str(video), "--transcript", str(transcript_path), "--workdir", str(work),
                       "--force", "render", "--preset", "veryfast")
    st3 = {s["name"]: s["status"] for s in json.loads((work / "run.json").read_text())["stages"]}
    checks.append(("--force render re-runs only render + debug (detection untouched)",
                   st3 == {"moment": "reused", "clip": "reused", "detect": "reused", "speakers": "reused",
                           "render": "ran", "debug": "ran"} and (work / "phase1" / "frames.jsonl").stat().st_mtime == mtime_detect,
                   f"{st3}"))

    # 7. LLM guard
    try:
        LLMScorer(endpoint="https://api.openai.com/v1/chat/completions")
        refused = False
    except HostedEndpointRefused:
        refused = True
    checks.append(("llm scorer refuses hosted endpoint without --allow-hosted-llm", refused, ""))

    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    print("PHASE 3 CHECKPOINT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
