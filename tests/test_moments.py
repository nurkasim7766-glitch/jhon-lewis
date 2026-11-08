import json

import pytest

from facepipe.moments import SelectionConfig, get_scorer, select_moment
from facepipe.moments.heuristic import HeuristicScorer
from facepipe.moments.llm import HostedEndpointRefused, LLMScorer
from facepipe.moments.select import SelectionError, generate_candidates, validate_moment
from facepipe.transcript import TranscriptError, load_transcript
from scripts.make_synthetic_source import HOOK_PASSAGE, make_transcript


@pytest.fixture(scope="module")
def transcript(tmp_path_factory):
    path = tmp_path_factory.mktemp("t") / "transcript.json"
    path.write_text(json.dumps(make_transcript(minutes=3.0, hook_at_s=95.0)))
    return load_transcript(path)


def test_sentence_boundaries_follow_punctuation_and_speaker(transcript):
    assert all(s.text.rstrip().endswith((".", "?", "!")) for s in transcript.sentences)
    for a, b in zip(transcript.sentences, transcript.sentences[1:]):
        assert a.end_ms <= b.start_ms
    assert transcript.speakers == ["A", "B"]


def test_candidates_are_hard_filtered_to_sentence_bounds(transcript):
    cfg = SelectionConfig(min_ms=30_000, max_ms=60_000)
    cands = generate_candidates(transcript, cfg)
    assert cands
    for c in cands:
        assert c.start_ms == transcript.sentences[c.first_sentence].start_ms
        assert c.end_ms == transcript.sentences[c.last_sentence].end_ms
        assert cfg.min_ms <= c.duration_ms <= cfg.max_ms


def test_heuristic_picks_the_hook_passage(transcript):
    doc = select_moment(transcript, "video.mp4")
    validate_moment(doc, transcript)
    m = doc["moment"]
    hook_texts = [s for _, s in HOOK_PASSAGE]
    assert any(h in m["text"] for h in hook_texts[:3]), m["text"][:200]
    assert m["start_ms"] <= m["sentence_start_ms"] and m["end_ms"] >= m["sentence_end_ms"]
    # padding never crosses a neighbouring sentence
    prev_end = transcript.sentences[m["first_sentence"] - 1].end_ms
    assert m["start_ms"] >= prev_end
    assert doc["candidates"][0]["score"] == m["score"]
    assert doc["selection"]["scorer"]["local_only"] is True


def test_no_candidates_raises(transcript):
    with pytest.raises(SelectionError):
        select_moment(transcript, "v.mp4", SelectionConfig(min_ms=10_000_000, max_ms=20_000_000))


def test_transcript_without_word_timings(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"segments": [
        {"start": 0.0, "end": 4.0, "text": "Hello there. How are you?"},
        {"start": 4.5, "end": 6.0, "text": "Fine thanks."},
    ]}))
    t = load_transcript(p)
    assert [s.text for s in t.sentences] == ["Hello there.", "How are you?", "Fine thanks."]
    assert t.sentences[0].end_ms <= t.sentences[1].start_ms <= 4000
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"segments": [{"start": 1, "end": 0, "text": "x"}]}))
    with pytest.raises(TranscriptError):
        load_transcript(bad)


def test_llm_scorer_refuses_hosted_endpoint_unless_opted_in():
    with pytest.raises(HostedEndpointRefused):
        LLMScorer(endpoint="https://api.openai.com/v1/chat/completions")
    with pytest.raises(HostedEndpointRefused):
        get_scorer("llm", endpoint="https://api.example.com/v1/chat/completions")
    local = LLMScorer(endpoint="http://127.0.0.1:11434/v1/chat/completions")
    assert local.local_only is False and local.prefilter.local_only is True
    hosted = LLMScorer(endpoint="https://api.openai.com/v1/chat/completions", allow_hosted=True)
    assert hosted.allow_hosted


def test_llm_scorer_reranks_shortlist(transcript, monkeypatch):
    scorer = LLMScorer(endpoint="http://localhost:1/v1/chat/completions", shortlist=3)
    calls = []

    def fake_chat(messages):
        calls.append(messages)
        return "Ranking: [2, 0, 1]"

    monkeypatch.setattr(scorer, "_chat", fake_chat)
    cands = HeuristicScorer().score(generate_candidates(transcript, SelectionConfig()), transcript)
    pre = sorted(cands, key=lambda c: c.score, reverse=True)[:3]
    out = sorted(scorer.score(generate_candidates(transcript, SelectionConfig()), transcript),
                 key=lambda c: c.score, reverse=True)
    assert len(calls) == 1
    assert (out[0].first_sentence, out[0].last_sentence) == (pre[2].first_sentence, pre[2].last_sentence)
    assert out[0].features["llm_rank"] == 0.0
