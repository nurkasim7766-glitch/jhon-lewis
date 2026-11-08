"""LR-ASD backend (Liao et al., IJCV 2025; MIT) — audio-visual active speaker detection.

Model code is vendored under facepipe/asd/lrasd_model (see its LICENSE). The
pretrained AVA weights are downloaded on first use. Requires the optional
`facepipe[lrasd]` extras (torch CPU build is fine) and ffmpeg for audio extraction.

Per face track the model consumes 25 fps grayscale 112x112 face crops and 100 Hz
13-coefficient MFCCs of the full-mix audio; it emits a logit per 25 fps tick where
> 0 means speaking. Ticks are mapped back onto the source frame rate.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from facepipe.asd.base import ASDBackend, ASDResult
from facepipe.models import default_model_dir
from facepipe.phase1_io import Phase1Data

WEIGHTS_URL = "https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/main/weight/pretrain_AVA.model"
WEIGHTS_FILENAME = "lrasd_pretrain_AVA.model"
MODEL_FPS = 25.0
AUDIO_HZ = 100.0
CROP_SCALE = 0.40
# Phase 1 boxes are tight landmark extents; LR-ASD was trained on detector boxes that
# include the forehead, so grow the box before cropping.
BBOX_GROW = 1.35
BBOX_SHIFT_UP = 0.12


def ensure_weights(model_dir: Path | None = None) -> Path:
    model_dir = model_dir or default_model_dir()
    path = model_dir / WEIGHTS_FILENAME
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)  # noqa: S310
    tmp.replace(path)
    return path


def extract_audio_16k(video_path: Path) -> np.ndarray:
    """Mono 16 kHz int16 PCM of the whole video; silence if the video has no audio."""
    from scipy.io import wavfile

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
               "-acodec", "pcm_s16le", str(wav)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not wav.exists():
            return np.zeros(0, dtype=np.int16)
        _, audio = wavfile.read(wav)
    return audio.astype(np.int16)


class LRASD(ASDBackend):
    name = "lrasd"

    def __init__(self, speaking_threshold: float = 0.0, weights_path: Path | None = None, device: str = "cpu",
                 durations: tuple[int, ...] = (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6)):
        self.speaking_threshold = speaking_threshold
        self.weights_path = weights_path
        self.device = device
        self.durations = durations
        self._model = None
        self._fc = None

    # -- model -----------------------------------------------------------------
    def _load(self):
        import torch

        from facepipe.asd.lrasd_model.Model import ASD_Model

        try:  # NNPACK is unavailable on many CPUs and only produces warning spam
            torch.backends.nnpack.set_flags(False)
        except Exception:  # noqa: BLE001
            pass
        weights = self.weights_path or ensure_weights()
        model = ASD_Model()
        fc = torch.nn.Linear(128, 2)
        state = torch.load(weights, map_location="cpu")
        model_state, fc_state = {}, {}
        for name, param in state.items():
            name = name.replace("module.", "")
            if name.startswith("model."):
                model_state[name[len("model."):]] = param
            elif name.startswith("lossAV.FC."):
                fc_state[name[len("lossAV.FC."):]] = param
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing or not fc_state:
            raise RuntimeError(f"LR-ASD weights incomplete: missing={missing[:3]} fc={bool(fc_state)}")
        fc.load_state_dict(fc_state)
        self._model = model.to(self.device).eval()
        self._fc = fc.to(self.device).eval()
        self.weights_path = weights

    # -- features ----------------------------------------------------------------
    @staticmethod
    def _face_crop(frame_bgr: np.ndarray, bbox_px: tuple[float, float, float, float]) -> np.ndarray:
        x0, y0, x1, y1 = bbox_px
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - (y1 - y0) * BBOX_SHIFT_UP
        bs = max(x1 - x0, y1 - y0) / 2 * BBOX_GROW
        bsi = int(bs * (1 + 2 * CROP_SCALE)) + 1
        padded = np.pad(frame_bgr, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
        my, mx = cy + bsi, cx + bsi
        face = padded[int(my - bs): int(my + bs * (1 + 2 * CROP_SCALE)), int(mx - bs * (1 + CROP_SCALE)): int(mx + bs * (1 + CROP_SCALE))]
        if face.size == 0:
            face = np.full((224, 224, 3), 110, dtype=np.uint8)
        gray = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY)
        return gray[56:168, 56:168]

    def _track_scores(self, audio_feat: np.ndarray, visual: np.ndarray, start_tick: int) -> np.ndarray:
        import torch

        n_ticks = visual.shape[0]
        a_start = int(start_tick * AUDIO_HZ / MODEL_FPS)
        audio = audio_feat[a_start: a_start + int(n_ticks * AUDIO_HZ / MODEL_FPS)]
        need = int(n_ticks * AUDIO_HZ / MODEL_FPS)
        if audio.shape[0] < need:  # pad audio (silence-like) to the visual length
            audio = np.pad(audio, ((0, need - audio.shape[0]), (0, 0)), "edge" if audio.shape[0] else "constant")
        all_scores = []
        with torch.no_grad():
            for duration in self.durations:
                scores = []
                for i in range(math.ceil(n_ticks / (duration * MODEL_FPS))):
                    v = visual[int(i * duration * MODEL_FPS): int((i + 1) * duration * MODEL_FPS)]
                    a = audio[int(i * duration * AUDIO_HZ): int((i + 1) * duration * AUDIO_HZ)]
                    a = a[: len(v) * 4]
                    if len(v) == 0:
                        continue
                    in_a = torch.FloatTensor(a).unsqueeze(0).to(self.device)
                    in_v = torch.FloatTensor(v).unsqueeze(0).to(self.device)
                    out = self._model.forward_audio_visual_backend(
                        self._model.forward_audio_frontend(in_a), self._model.forward_visual_frontend(in_v))
                    logits = self._fc(out.squeeze(1))
                    scores.extend(logits[:, 1].cpu().numpy().tolist())
                all_scores.append(np.array(scores[:n_ticks]))
        return np.mean(np.stack(all_scores), axis=0)

    # -- interface ----------------------------------------------------------------
    def score(self, data: Phase1Data, video_path: Path | None = None) -> ASDResult:
        import python_speech_features

        if video_path is None:
            video_path = Path(data.manifest["video"]["path"])
        if self._model is None:
            self._load()

        audio = extract_audio_16k(Path(video_path))
        had_audio = audio.size > 0
        if not had_audio:
            audio = np.zeros(int(16000 * len(data.frames) / data.fps) + 16000, dtype=np.int16)
        audio_feat = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

        # Decode frames once, at 25 fps ticks (nearest source frame per tick).
        n_frames = len(data.frames)
        n_ticks = int(math.floor(n_frames / data.fps * MODEL_FPS))
        tick_to_frame = np.minimum((np.arange(n_ticks) / MODEL_FPS * data.fps).round().astype(int), n_frames - 1)
        cap = cv2.VideoCapture(str(video_path))
        frames_bgr: dict[int, np.ndarray] = {}
        wanted = set(tick_to_frame.tolist())
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx in wanted:
                frames_bgr[idx] = bgr
            idx += 1
        cap.release()

        scores: dict[int, np.ndarray] = {}
        for fid in data.face_ids:
            per_frame = np.full(n_frames, np.nan)
            # Contiguous runs of ticks where the face is visible form tracks.
            tick_bbox = [data.face_bbox_px(int(f), fid) for f in tick_to_frame]
            t = 0
            while t < n_ticks:
                if tick_bbox[t] is None:
                    t += 1
                    continue
                start = t
                while t < n_ticks and tick_bbox[t] is not None:
                    t += 1
                crops = np.stack([self._face_crop(frames_bgr[int(tick_to_frame[k])], tick_bbox[k]) for k in range(start, t)])
                track_scores = self._track_scores(audio_feat, crops, start)
                for k, s in zip(range(start, t), track_scores):
                    per_frame[int(tick_to_frame[k])] = s
            # Frames between ticks inherit the nearest scored frame where the face is present.
            present = np.array([any(f.face_id == fid for f in fr.faces) for fr in data.frames])
            scored = ~np.isnan(per_frame)
            if scored.any():
                xi = np.arange(n_frames)
                filled = np.interp(xi, xi[scored], per_frame[scored])
                per_frame = np.where(present, filled, np.nan)
            scores[fid] = per_frame

        return ASDResult(
            backend=self.name,
            params={"weights": Path(self.weights_path).name, "durations": list(self.durations),
                    "had_audio": had_audio, "model_fps": MODEL_FPS, "metric": "mean logit (speaking > 0)"},
            speaking_threshold=self.speaking_threshold,
            scores=scores,
        )
