"""End-to-end pipeline with reusable intermediates.

Work directory layout (every stage reads only the files of earlier stages):

    moment.json        stage "moment"   transcript -> selected window          (5.5)
    clip.mp4           stage "clip"     source video -> frame-accurate cut
    phase1/            stage "detect"   clip -> manifest.json + frames.jsonl   (5.1)
    phase2/            stage "speakers" phase1 -> speakers.json + crop_plan.json (5.2/5.3)
    output.mp4         stage "render"   crop_plan + clip -> final vertical video
    debug.mp4          stage "debug"    overlay of phase1/phase2 data on the clip
    run.json           configuration + per-stage status of the last run

A stage is skipped when its outputs exist, unless it (or an earlier stage) is forced.
Forcing a stage invalidates everything downstream, because later artifacts would no
longer describe the same inputs.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from facepipe.cropplan import CropConfig
from facepipe.detector import DetectorConfig
from facepipe.moments import SelectionConfig
from facepipe.segmentation import SegmentationConfig

STAGES = ("moment", "clip", "detect", "speakers", "render", "debug")


@dataclass
class PipelineConfig:
    video: str
    transcript: str
    workdir: str
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    asd_backend: str = "heuristic"
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    llm: dict[str, Any] = field(default_factory=dict)  # endpoint, model, shortlist, allow_hosted
    render_crf: int = 18
    render_preset: str = "medium"
    debug: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["detector"]["model_path"] = str(self.detector.model_path) if self.detector.model_path else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        d = dict(d)
        d["selection"] = SelectionConfig(**d.get("selection", {}))
        det = dict(d.get("detector", {}))
        if det.get("model_path"):
            det["model_path"] = Path(det["model_path"])
        d["detector"] = DetectorConfig(**det)
        d["segmentation"] = SegmentationConfig(**d.get("segmentation", {}))
        d["crop"] = CropConfig(**d.get("crop", {}))
        return cls(**d)


@dataclass
class StageResult:
    name: str
    status: str  # "ran" | "reused" | "skipped"
    seconds: float
    outputs: list[str]


Log = Callable[[str], None]
StageHook = Callable[[str, str], None]  # (stage, event) with event in {"start", "done", "reused"}
Progress = Callable[[str, int, int], None]  # (stage, done, total)


class PipelineCancelled(RuntimeError):
    pass


def cut_clip(video: Path, start_ms: float, end_ms: float, out: Path, fps_hint: float | None = None) -> None:
    """Frame-accurate re-encode of [start, end) with audio; keyframe-copy cuts would drift."""
    from facepipe.ffmpeg import ffmpeg_exe

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start_ms / 1000:.3f}", "-to", f"{end_ms / 1000:.3f}",
           "-i", str(video), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {proc.stderr.strip()}")


class Pipeline:
    def __init__(self, cfg: PipelineConfig, force: set[str] | None = None, log: Log | None = None,
                 on_stage: StageHook | None = None, progress: Progress | None = None,
                 cancel_check: Callable[[], bool] | None = None, on_step: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.work = Path(cfg.workdir)
        self.log = log or (lambda s: None)
        self.on_stage = on_stage or (lambda stage, event: None)
        self.progress = progress or (lambda stage, done, total: None)
        self.cancel_check = cancel_check or (lambda: False)
        self.on_step = on_step or (lambda step: None)
        force = set(force or ())
        if "all" in force:
            force = set(STAGES)
        # forcing a stage forces everything downstream
        self.force: set[str] = set()
        for i, s in enumerate(STAGES):
            if s in force or any(prev in self.force for prev in STAGES[:i]):
                self.force.add(s)
        self.results: list[StageResult] = []

    # -- paths -------------------------------------------------------------------
    @property
    def moment_path(self) -> Path: return self.work / "moment.json"
    @property
    def clip_path(self) -> Path: return self.work / "clip.mp4"
    @property
    def phase1_dir(self) -> Path: return self.work / "phase1"
    @property
    def phase2_dir(self) -> Path: return self.work / "phase2"
    @property
    def output_path(self) -> Path: return self.work / "output.mp4"
    @property
    def debug_path(self) -> Path: return self.work / "debug.mp4"

    def _outputs(self, stage: str) -> list[Path]:
        return {
            "moment": [self.moment_path],
            "clip": [self.clip_path],
            "detect": [self.phase1_dir / "manifest.json", self.phase1_dir / "frames.jsonl"],
            "speakers": [self.phase2_dir / "speakers.json", self.phase2_dir / "crop_plan.json"],
            "render": [self.output_path],
            "debug": [self.debug_path],
        }[stage]

    def _check_cancel(self) -> None:
        if self.cancel_check():
            raise PipelineCancelled("cancelled")

    def _frame_progress(self, stage: str) -> Callable[[int, int], None]:
        def cb(done: int, total: int) -> None:
            self._check_cancel()
            self.progress(stage, done, total)
        return cb

    def _run_stage(self, stage: str, fn: Callable[[], None]) -> None:
        outs = self._outputs(stage)
        if stage not in self.force and all(p.exists() for p in outs):
            self.log(f"[{stage}] reused {', '.join(p.name for p in outs)}")
            self.results.append(StageResult(stage, "reused", 0.0, [str(p) for p in outs]))
            self.on_stage(stage, "reused")
            return
        self._check_cancel()
        self.on_stage(stage, "start")
        t0 = time.perf_counter()
        try:
            fn()
        except BaseException:
            # never leave half-written outputs that a later run would "reuse"
            for p in outs:
                if p.is_file():
                    p.unlink(missing_ok=True)
            raise
        dt = time.perf_counter() - t0
        self.log(f"[{stage}] done in {dt:.1f}s -> {', '.join(p.name for p in outs)}")
        self.results.append(StageResult(stage, "ran", round(dt, 3), [str(p) for p in outs]))
        self.on_stage(stage, "done")

    # -- stages ------------------------------------------------------------------
    def stage_moment(self) -> None:
        from facepipe.moments import get_scorer, select_moment
        from facepipe.moments.select import write_moment
        from facepipe.transcript import load_transcript

        transcript = load_transcript(self.cfg.transcript)
        scorer = get_scorer(self.cfg.selection.scorer, **(self.cfg.llm if self.cfg.selection.scorer == "llm" else {}))
        doc = select_moment(transcript, self.cfg.video, self.cfg.selection, scorer)
        write_moment(doc, self.moment_path)
        m = doc["moment"]
        self.log(f"[moment] {m['start_ms'] / 1000:.1f}s - {m['end_ms'] / 1000:.1f}s "
                 f"(sentences {m['first_sentence']}-{m['last_sentence']}, score {m['score']:.2f})")

    def stage_clip(self) -> None:
        m = json.loads(self.moment_path.read_text(encoding="utf-8"))["moment"]
        cut_clip(Path(self.cfg.video), m["start_ms"], m["end_ms"], self.clip_path)

    def stage_detect(self) -> None:
        from facepipe.extract import extract_video

        total = self._clip_frame_count()
        cb = self._frame_progress("detect")
        extract_video(self.clip_path, self.phase1_dir, self.cfg.detector, progress=lambda i: cb(i + 1, total))

    def _clip_frame_count(self) -> int:
        import cv2

        cap = cv2.VideoCapture(str(self.clip_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(n, 1)

    def stage_speakers(self) -> None:
        from facepipe.phase2 import run_phase2

        run_phase2(self.phase1_dir, self.phase2_dir, backend=self.cfg.asd_backend, seg_cfg=self.cfg.segmentation,
                   crop_cfg=self.cfg.crop, video_path=self.clip_path, on_step=self.on_step)

    def stage_render(self) -> None:
        from facepipe.render import render

        render(self.phase2_dir / "crop_plan.json", self.clip_path, self.output_path, crf=self.cfg.render_crf,
               preset=self.cfg.render_preset, progress=self._frame_progress("render"))

    def stage_debug(self) -> None:
        from facepipe.overlay import render_debug

        render_debug(self.clip_path, self.phase1_dir, self.phase2_dir, self.debug_path,
                     progress=self._frame_progress("debug"))

    def run(self, until: str = "debug") -> list[StageResult]:
        self.work.mkdir(parents=True, exist_ok=True)
        stop = STAGES.index(until)
        fns = {"moment": self.stage_moment, "clip": self.stage_clip, "detect": self.stage_detect,
               "speakers": self.stage_speakers, "render": self.stage_render, "debug": self.stage_debug}
        for stage in STAGES[: stop + 1]:
            if stage == "debug" and not self.cfg.debug:
                self.results.append(StageResult(stage, "skipped", 0.0, []))
                continue
            self._run_stage(stage, fns[stage])
        (self.work / "run.json").write_text(json.dumps({
            "config": self.cfg.to_dict(),
            "forced": sorted(self.force),
            "stages": [asdict(r) for r in self.results],
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2), encoding="utf-8")
        return self.results
