"""Moment selection: pick the strongest sentence-aligned window from a transcript."""

from facepipe.moments.base import Candidate, MomentScorer, get_scorer
from facepipe.moments.select import SelectionConfig, select_moment

__all__ = ["Candidate", "MomentScorer", "get_scorer", "SelectionConfig", "select_moment"]
