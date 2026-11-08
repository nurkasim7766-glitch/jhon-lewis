"""Optional LLM re-ranker over heuristic-prefiltered candidates.

Talks to an OpenAI-compatible ``/v1/chat/completions`` endpoint using only the standard
library. Section 2.1 forbids paid API calls in the processing chain, so by default the
endpoint must resolve to this machine or a private network (Ollama, llama.cpp, vLLM,
LM Studio); a hosted endpoint is rejected unless ``allow_hosted=True`` is passed
explicitly (``--allow-hosted-llm``).

Candidates stay sentence-aligned: the model only ranks the shortlist it is given.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from facepipe.moments.base import Candidate, MomentScorer
from facepipe.moments.heuristic import HeuristicScorer
from facepipe.transcript import Transcript


class HostedEndpointRefused(RuntimeError):
    pass


def is_local_endpoint(url: str) -> bool:
    host = urlparse(url).hostname
    if not host:
        return False
    if host in ("localhost",) or host.endswith(".local") or host.endswith(".localhost"):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    addrs = {ipaddress.ip_address(i[4][0]) for i in infos}
    return bool(addrs) and all(a.is_private or a.is_loopback or a.is_link_local for a in addrs)


@dataclass
class LLMScorer(MomentScorer):
    name: str = "llm"
    local_only: bool = False
    endpoint: str = "http://127.0.0.1:11434/v1/chat/completions"
    model: str = "llama3.1"
    shortlist: int = 8
    allow_hosted: bool = False
    api_key_env: str = "FACEPIPE_LLM_API_KEY"
    timeout_s: float = 120.0
    prefilter: HeuristicScorer | None = None

    def __post_init__(self) -> None:
        self.prefilter = self.prefilter or HeuristicScorer()
        if not self.allow_hosted and not is_local_endpoint(self.endpoint):
            raise HostedEndpointRefused(
                f"LLM endpoint {self.endpoint!r} is not local/private. Section 2.1 forbids paid API calls; "
                "pass --allow-hosted-llm to opt in explicitly."
            )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "endpoint": self.endpoint, "model": self.model, "shortlist": self.shortlist,
                "allow_hosted": self.allow_hosted, "prefilter": self.prefilter.describe()}

    # -- transport --------------------------------------------------------------
    def _chat(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.0}).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - endpoint vetted above
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request to {self.endpoint} failed: {exc}") from exc
        return payload["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_ranking(text: str, n: int) -> list[int]:
        m = re.search(r"\[[\d,\s]+\]", text)
        if not m:
            raise RuntimeError(f"LLM returned no ranking list: {text[:200]!r}")
        ranking = [int(x) for x in re.findall(r"\d+", m.group(0))]
        seen, out = set(), []
        for r in ranking:
            if 0 <= r < n and r not in seen:
                out.append(r)
                seen.add(r)
        return out

    # -- interface ----------------------------------------------------------------
    def score(self, candidates: list[Candidate], transcript: Transcript) -> list[Candidate]:
        candidates = self.prefilter.score(candidates, transcript)
        ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
        short = ordered[: self.shortlist]
        if len(short) < 2:
            return candidates
        listing = "\n\n".join(f"[{i}] ({c.duration_ms / 1000:.0f}s) {c.text}" for i, c in enumerate(short))
        messages = [
            {"role": "system", "content": "You pick the most compelling self-contained short-form clip from a "
                                          "transcript. Prefer a strong opening hook, a complete thought, and a "
                                          "satisfying ending. Reply with only a JSON list of candidate indices, "
                                          "best first."},
            {"role": "user", "content": f"Candidates:\n\n{listing}\n\nRank them, best first, as a JSON list."},
        ]
        ranking = self._parse_ranking(self._chat(messages), len(short))
        # LLM rank dominates; heuristic score breaks ties and orders the unranked rest.
        base = max(c.score for c in candidates) + 1.0
        for rank, idx in enumerate(ranking):
            short[idx].score = base + (len(short) - rank) * 10.0 + short[idx].score / base
            short[idx].features["llm_rank"] = float(rank)
        return candidates
