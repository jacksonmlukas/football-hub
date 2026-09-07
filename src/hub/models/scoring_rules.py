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

import itertools
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


def reliability_by(df: pl.DataFrame, edges: Sequence[float], *, on: str,
                   prob: str = "home_win_prob", outcome: str = "home_won",
                   places: int = 1) -> list[dict[str, Any]]:
    """Predicted versus actual, binned on `on` rather than on the probability itself.

    `reliability` bins a probability against its own value, which answers "when the model
    said 70%, did it happen 70% of the time". That is the right question when the model is
    the thing under test and the wrong one when the *input* is: a survivor pick is chosen by
    spread, so a miss concentrated in one spread range is invisible in a diagram whose bins
    are probabilities, because every game in a probability bin came from roughly one spread
    anyway. Binning on the input is how a miss gets attributed to the range it lives in.

    `edges` are bin boundaries, low to high; the last bin includes its upper edge so nothing
    at the top of the range falls out of the diagram. Empty bins are kept, for the reason
    `reliability` keeps them.

    `gap` is actual minus predicted, so a positive gap is a model that was *under*-confident.
    """
    if df.is_empty():
        return []
    out = []
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        last = i == len(edges) - 2
        sel = df.filter((pl.col(on) >= lo)
                        & ((pl.col(on) <= hi) if last else (pl.col(on) < hi)))
        n = sel.height
        p = float(cast(float, sel[prob].mean())) if n else None
        a = float(cast(float, sel[outcome].mean())) if n else None
        out.append({"bin": f"{lo:.{places}f}-{hi:.{places}f}", "n": n,
                    "predicted": p, "actual": a,
                    "gap": (a - p) if (p is not None and a is not None) else None})
    return out


def reliability(df: pl.DataFrame, n_bins: int = 10) -> list[dict[str, Any]]:
    """Reliability diagram: predicted versus actual, with the count in each bin.

    Counts are not decoration. `docs/track-record.md` asks for them because a bin holding
    four games says nothing, and a diagram that hides its bin sizes invites exactly the
    over-reading the page exists to prevent.

    The equal-width probability case of `reliability_by`, rather than a second copy of the
    binning loop. The bins and their labels are unchanged: the last one used to admit its
    upper edge by comparing against 1.01, which for a probability validated into [0, 1] is
    the same set of rows as including the edge.
    """
    edges = [i / n_bins for i in range(n_bins + 1)]
    return reliability_by(df, edges, on="home_win_prob")
