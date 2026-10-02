"""The two draft-side lineup scorers hold the same rule, so they must agree (#396).

The League's Slot/FLEX rule has three implementations in three shapes: `league.starting_lineup`
(selection indices), `draft.season.lineup_points` (vectorised over sims and weeks) and
`draft.evaluate.starter_points` (a scalar season total). The forms differ in shape on purpose
and are not being merged. `test_flex_arity.py` holds each to the League's flex count on one
hand-built roster; this holds the vectorised form and the scalar form to *each other* on
random ones, which is the locality a merge would have given.

A one-sim, one-week grid makes `lineup_points` a single lineup, and `starter_points` over a
frame whose `actual_points` are that week's scores is the same lineup: the two must return
the same number. The flex count is drawn too, because at the League's own count of one flex a
fill that takes one candidate where it should take `FLEX_SLOTS` is indistinguishable from the
right one -- the shape `test_flex_arity.py`'s docstring describes.

All offline.
"""
from unittest import mock

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from hub.draft import evaluate, season

# Half-point steps: every sum is exact in binary floating point, so the numpy and Python
# accumulations cannot disagree by rounding and `==` is the honest comparison.
points = st.integers(min_value=-40, max_value=120).map(lambda n: n / 2)
players = st.lists(st.tuples(st.sampled_from(["QB", "RB", "WR", "TE"]), points),
                   min_size=1, max_size=24)


@settings(max_examples=300, deadline=None)
@given(roster=players, flex_slots=st.integers(min_value=0, max_value=3))
def test_the_vectorised_and_the_scalar_lineup_score_the_same_roster_equally(roster, flex_slots):
    pos = [p for p, _ in roster]
    pts = [x for _, x in roster]
    scores = np.array(pts, dtype=float).reshape(1, 1, len(pts))
    frame = pl.DataFrame({"pos": pos, "actual_points": pts})
    with mock.patch.object(season, "FLEX_SLOTS", flex_slots), \
            mock.patch.object(evaluate, "FLEX_SLOTS", flex_slots):
        vectorised = season.lineup_points(scores, np.array(pos))[0, 0]
        scalar = evaluate.starter_points(frame)
    assert vectorised == scalar, (
        f"lineup_points={vectorised} but starter_points={scalar} for {roster} "
        f"with {flex_slots} flex slot(s)")
