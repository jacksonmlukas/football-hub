"""Nightly: the component rebuild still reproduces the source's own pre-summed total.

This is the evidence that replaced a gate. Issue #1 proposed pre-registering an accuracy gate
before letting components build the Board's xFP; measurement said the two quantities differ by
about 0.017 points a player-week, so a gate would have returned "no detectable difference" by
construction after four seasons of harness work. What the change actually needs is proof the
plumbing is faithful -- that rebuilding the total from its parts, priced by this league's own
scoring, lands on the number the source already publishes.

Asserted here rather than measured once because it is the kind of claim that decays silently:
`ff_opportunity` renaming a column, or adding a scored component we do not map, would widen this
without breaking anything else. Marked `golden` and deselected by default for the reason the
rest of this directory is -- a third-party outage must not block a docs commit.

    uv run pytest -m golden -k component_equivalence
"""
import numpy as np
import polars as pl
import pytest

from hub.models import components

# Observed on 2025 over 6,054 player-weeks: mean absolute difference 0.017, max 0.213. The
# residual is upstream's own rounding, not a modelling disagreement. Set with headroom so a
# normal season does not trip it and a renamed column does.
MEAN_TOLERANCE = 0.05
MAX_TOLERANCE = 1.0

SEASONS = (2021, 2022, 2023, 2024, 2025)


@pytest.mark.golden
@pytest.mark.parametrize("season", SEASONS)
def test_the_rebuild_reproduces_the_sources_own_total(season):
    import nflreadpy as nfl

    o = nfl.load_ff_opportunity(seasons=[season], stat_type="weekly")
    if o.is_empty() or "total_fantasy_points_exp" not in o.columns:
        pytest.skip(f"ff_opportunity has no weekly total for {season}")

    expr = pl.lit(0.0)
    for stat, cols in components.EXPECTED.items():
        for c in cols:
            if c in o.columns:
                expr = expr + pl.col(c).fill_null(0.0) * components.SCORING[stat]

    d = (o.with_columns(expr.alias("rebuilt"))
          .select("total_fantasy_points_exp", "rebuilt")
          .drop_nulls("total_fantasy_points_exp"))
    diff = (d["rebuilt"] - d["total_fantasy_points_exp"]).to_numpy()

    assert float(np.abs(diff).mean()) < MEAN_TOLERANCE, (
        f"{season}: rebuilding from components drifted from the published total by "
        f"{np.abs(diff).mean():.4f} a player-week. A renamed or new expected column is the "
        f"first thing to check -- see components.EXPECTED.")
    assert float(np.abs(diff).max()) < MAX_TOLERANCE, (
        f"{season}: worst player-week differs by {np.abs(diff).max():.3f}")


@pytest.mark.golden
def test_every_mapped_column_still_exists_upstream():
    """The map naming a column the source has dropped would silently zero that component, and
    the equivalence above would absorb it as drift rather than name it."""
    import nflreadpy as nfl

    o = nfl.load_ff_opportunity(seasons=[max(SEASONS)], stat_type="weekly")
    gone = [c for c in components.expected_columns() if c not in o.columns]
    assert not gone, f"components.EXPECTED names columns ff_opportunity no longer has: {gone}"
