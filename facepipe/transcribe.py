"""Optional local speech-to-text producing a section-5.4 transcript.

Backend: faster-whisper (MIT) on CTranslate2 (MIT), CPU int8. Models (Systran, MIT)
are downloaded once into FACEPIPE_MODEL_DIR/whisper. Nothing leaves the machine.
Install with `pip install -e ".[transcribe]"`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from facepipe.models import default_model_dir

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")
DEFAULT_MODEL = "base"
INSTALL_HINT = 'pip install -e ".[transcribe]"'


class TranscriptionUnavailable(RuntimeError):
    pass


def transcription_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _decode_audio_16k(video: Path):
    """Mono 16 kHz float32 PCM via ffmpeg (what Whisper expects)."""
    import subprocess

    import numpy as np

    from facepipe.ffmpeg import ffmpeg_exe

    cmd = [ffmpeg_exe(), "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode audio: {proc.stderr.decode('utf-8', 'replace').strip()[-500:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe_video(
    video: str | Path,
    out_json: str | Path,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    progress: Callable[[float, float], None] | None = None,  # (seconds_done, total_seconds)
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Transcribe the audio track and write a Whisper-style `segments` JSON with word timings."""
    if not transcription_available():
        raise TranscriptionUnavailable(f"local transcription is not installed; run {INSTALL_HINT}")
    if model_size not in MODEL_SIZES:
        raise ValueError(f"unknown whisper model {model_size!r}; choose one of {', '.join(MODEL_SIZES)}")
    from faster_whisper import WhisperModel

    from facepipe.ffmpeg import probe_media

    video = Path(video)
    info = probe_media(video)
    if not info["has_audio"]:
        raise ValueError(f"{video.name} has no audio track, so it cannot be transcribed; upload a transcript instead")
    total = float(info["duration"] or 0.0)

    model_dir = default_model_dir() / "whisper"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=str(model_dir))
    # Decode with our ffmpeg instead of faster-whisper's PyAV path (version-sensitive, and
    # the bundled imageio-ffmpeg is what Windows users have).
    audio = _decode_audio_16k(video)
    segments_iter, meta = model.transcribe(
        audio, language=language or None, word_timestamps=True, vad_filter=True, beam_size=5,
    )
    segments: list[dict[str, Any]] = []
    for seg in segments_iter:
        if cancel_check and cancel_check():
            raise InterruptedError("transcription cancelled")
        text = seg.text.strip()
        if not text:
            continue
        words = [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
                 for w in (seg.words or []) if w.word.strip()]
        segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text, "words": words})
        if progress:
            progress(min(seg.end, total) if total else seg.end, total)
    if not segments:
        raise ValueError("no speech was detected in the audio; upload a transcript instead")
    doc = {
        "language": meta.language,
        "language_probability": round(float(meta.language_probability), 3),
        "source": {"backend": "faster-whisper", "model": model_size, "compute_type": "int8", "video": video.name},
        "segments": segments,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc
