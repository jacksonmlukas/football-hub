"""Proper scoring rules, and the reliability diagram that reads them.

These lived in `hub.publish` -- the module that writes `site/data/*.json` -- so
`hub.models.eval` imported its metrics from the site writer, and could not be read or
imported without dragging in `nflreadpy`, the survivor solver and the manifest machinery.
The depth was real (two callers, three test files); it was in the wrong file.

Named `scoring_rules` rather than `scoring` on purpose: `hub.models.components.SCORING` is
already the league's fantasy point weights, and this repo does not reuse a word for two
things (see `CONTEXT.md`).

`hub.models.margin` carried a second `log_loss` with `eps=1e-12` against this one's `1e-15`
-- two clipping constants, two test files, one concept. It now imports this.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import polars as pl


def log_loss(probs: Sequence[float] | np.ndarray,
             outcomes: Sequence[int] | np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log likelihood.

    Clipped at eps because a model that said 1.0 and was wrong would otherwise put an
    infinity on the page. Clipping bounds the penalty at ~34 per game, which is still
    ruinous and still renders.
    """
    # len() rather than truthiness: `if not probs` raises on a numpy array, which went
    # unnoticed while every caller passed lists. Vectorised because hub.models.eval
    # bootstraps this thousands of times per comparison.
    q = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    if q.size == 0:
        return float("nan")
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean(-(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))))


def brier(probs: Sequence[float] | np.ndarray,
          outcomes: Sequence[int] | np.ndarray) -> float:
    q = np.asarray(probs, dtype=float)
    if q.size == 0:
        return float("nan")
    return float(np.mean((q - np.asarray(outcomes, dtype=float)) ** 2))


def reliability(df: pl.DataFrame, n_bins: int = 10) -> list[dict[str, Any]]:
    """Reliability diagram: predicted versus actual, with the count in each bin.

    Counts are not decoration. `docs/track-record.md` asks for them because a bin holding
    four games says nothing, and a diagram that hides its bin sizes invites exactly the
    over-reading the page exists to prevent.
    """
    if df.is_empty():
        return []
    out = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        sel = df.filter((pl.col("home_win_prob") >= lo)
                        & (pl.col("home_win_prob") < (hi if i < n_bins - 1 else 1.01)))
        n = sel.height
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}", "n": n,
            "predicted": float(cast(float, sel["home_win_prob"].mean())) if n else None,
            "actual": float(cast(float, sel["home_won"].mean())) if n else None,
        })
    return out
