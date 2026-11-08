import json
from pathlib import Path

import numpy as np
import pytest

from facepipe.contract_phase2 import CropPlan
from facepipe.overlay import CROP_COLOR, TIMELINE_H, draw_overlay, load_phase_artifacts
from facepipe.phase2 import run_phase2_on_data
from facepipe.pipeline import STAGES, Pipeline, PipelineConfig
from facepipe.synthetic import make_two_speaker_data


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Synthetic Phase 1 + real Phase 2 outputs, written to disk like the pipeline does."""
    root = tmp_path_factory.mktemp("p3")
    data, _ = make_two_speaker_data(duration_ms=4000)
    p1, p2 = root / "phase1", root / "phase2"
    p1.mkdir()
    (p1 / "manifest.json").write_text(json.dumps(data.manifest))
    with (p1 / "frames.jsonl").open("w") as fh:
        for f in data.frames:
            fh.write(json.dumps(f.to_dict()) + "\n")
    run_phase2_on_data(data, p2)
    return root, data


def test_overlay_draws_crop_rect_and_active_speaker_from_artifacts(artifacts):
    root, data = artifacts
    frames, speakers, plan = load_phase_artifacts(root / "phase1", root / "phase2")
    idx = len(frames) // 2
    bgr = np.zeros((data.height, data.width, 3), dtype=np.uint8)
    out = draw_overlay(bgr, frames[idx], speakers, plan, idx)
    assert out.shape == (data.height + TIMELINE_H, data.width, 3)
    c = plan.frames[idx].crop
    # crop rectangle edges are present in CROP_COLOR at the plan coordinates
    for y, x in [(c.y, c.x + c.w // 2), (c.y + c.h - 1, c.x + c.w // 2), (c.y + c.h // 2, c.x), (c.y + c.h // 2, c.x + c.w - 1)]:
        patch = out[max(0, y - 1): y + 2, max(0, x - 1): x + 2].reshape(-1, 3)
        assert any(tuple(int(v) for v in p) == CROP_COLOR for p in patch), (y, x)
    # active face box is drawn thick (4 px, centred on the edge) in its palette colour; the other face 1 px
    from facepipe.overlay import face_color

    active = speakers.frames[idx].active_face_id
    for f in frames[idx].faces:
        x0 = int(f.bbox.x_min * data.width)
        ymid = int((f.bbox.y_min + f.bbox.y_max) / 2 * data.height)
        row = out[ymid, x0 - 3: x0 + 4]
        hits = sum(tuple(int(v) for v in px) == face_color(f.face_id) for px in row)
        assert (hits >= 3) if f.face_id == active else (hits <= 2), (f.face_id, hits)


def test_pipeline_force_propagates_downstream(tmp_path):
    cfg = PipelineConfig(video="v.mp4", transcript="t.json", workdir=str(tmp_path))
    p = Pipeline(cfg, force={"speakers"})
    assert p.force == {"speakers", "render", "debug"}
    assert Pipeline(cfg, force={"all"}).force == set(STAGES)
    assert Pipeline(cfg).force == set()


def test_pipeline_reuses_existing_outputs(tmp_path, monkeypatch):
    cfg = PipelineConfig(video="v.mp4", transcript="t.json", workdir=str(tmp_path), debug=False)
    calls = []
    p = Pipeline(cfg, log=calls.append)
    # pretend moment + clip already exist; stub later stages
    (tmp_path / "moment.json").write_text("{}")
    (tmp_path / "clip.mp4").write_bytes(b"")
    ran = []
    for name in ("detect", "speakers", "render"):
        def fn(name=name):
            ran.append(name)
            for out in p._outputs(name):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("x")
        monkeypatch.setattr(p, f"stage_{name}", fn)
    results = p.run(until="render")
    assert [r.status for r in results] == ["reused", "reused", "ran", "ran", "ran"]
    assert ran == ["detect", "speakers", "render"]
    run_json = json.loads((tmp_path / "run.json").read_text())
    assert run_json["config"]["workdir"] == str(tmp_path)
    # second run: everything reused
    p2 = Pipeline(cfg)
    assert all(r.status == "reused" for r in p2.run(until="render"))
    assert CropPlan  # imported for type completeness
    assert Path(tmp_path / "run.json").exists()
