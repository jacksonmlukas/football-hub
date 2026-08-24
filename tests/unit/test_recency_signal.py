"""A recency-weighted regression signal.

The season-total version treats week 1 and week 17 as equally informative about next
year. That is not obviously right: roles change, depth charts move, and a back who took
over in November is a different asset from one who lost the job in October, even when
their season totals match.

This is a different hypothesis, not a re-measurement of the old one. It gets the same
treatment: the uniform case must reproduce the existing signal exactly, so the comparison
is like-for-like, and the weighting must actually do what it claims on data where the
answer is known.
"""
import numpy as np
import polars as pl
import pytest

from hub.draft.projection import regression_signal, weighted_signal


def _weekly(rows):
    """rows: (player, position, week, xfp, fp)"""
    return pl.DataFrame({
        "full_name": [r[0] for r in rows], "position": [r[1] for r in rows],
        "week": [r[2] for r in rows], "xfp": [float(r[3]) for r in rows],
        "fp": [float(r[4]) for r in rows]})


def test_uniform_weighting_reproduces_the_season_signal():
    """The baseline has to be identical, or every comparison is confounded."""
    wk = _weekly([("a", "WR", w, 10.0, 10.0 + w) for w in range(1, 6)]
                 + [("b", "WR", w, 10.0, 10.0 - w) for w in range(1, 6)])
    season = (wk.group_by(["full_name", "position"])
              .agg(pl.col("xfp").sum(), pl.col("fp").sum()))
    want = regression_signal(season).sort("full_name")["z_regress"].to_list()
    got = weighted_signal(wk, half_life=None).sort("full_name")["z_regress"].to_list()
    assert got == pytest.approx(want)


def test_recent_weeks_dominate_a_short_half_life():
    """Same season totals, opposite trajectories. Uniform cannot tell them apart."""
    early = [("early", "RB", w, 10.0, 20.0) for w in range(1, 5)] \
          + [("early", "RB", w, 10.0, 0.0) for w in range(5, 9)]
    late = [("late", "RB", w, 10.0, 0.0) for w in range(1, 5)] \
         + [("late", "RB", w, 10.0, 20.0) for w in range(5, 9)]
    # Filler at the same position so there is dispersion to standardise against; with
    # only two identical values the z-score is undefined, not zero.
    filler = [(f"f{i}", "RB", w, 10.0, 10.0 + i * 3) for i in range(4)
              for w in range(1, 9)]
    wk = _weekly(early + late + filler)

    uniform = weighted_signal(wk, half_life=None)
    zu = dict(zip(uniform["full_name"].to_list(), uniform["z_regress"].to_list()))
    assert zu["early"] == pytest.approx(zu["late"]), \
        "identical season totals must give an identical uniform signal"

    recent = weighted_signal(wk, half_life=2.0).sort("full_name")
    z = dict(zip(recent["full_name"].to_list(), recent["z_regress"].to_list()))
    # "late" overperformed recently, so its underperformance signal is the lower one
    assert z["late"] < z["early"]


def test_scale_is_preserved_so_lambda_means_the_same_thing():
    """Weighting must not silently rescale the signal, or a lambda tuned on one is not
    comparable to a lambda tuned on the other."""
    wk = _weekly([(f"p{i}", "WR", w, 10.0, 10.0 + i) for i in range(8)
                  for w in range(1, 10)])
    for hl in (None, 2.0, 6.0):
        z = weighted_signal(wk, half_life=hl)["z_regress"].to_numpy()
        assert abs(float(np.nanstd(z)) - 1.0) < 0.2, "z stays standardised"


def test_a_player_with_one_week_is_not_given_a_confident_signal():
    wk = _weekly([("solo", "TE", 9, 5.0, 40.0)]
                 + [(f"p{i}", "TE", w, 10.0, 10.0 + i) for i in range(6)
                    for w in range(1, 10)])
    got = weighted_signal(wk, half_life=4.0, min_weeks=4)
    assert got.filter(pl.col("full_name") == "solo")["z_regress"][0] is None


def test_standardisation_is_within_position():
    wk = _weekly([("wr", "WR", w, 10.0, 30.0) for w in range(1, 6)]
                 + [("te", "TE", w, 10.0, 30.0) for w in range(1, 6)])
    got = weighted_signal(wk, half_life=None)
    # each is the only player at its position, so each is its own mean -> z = 0 or null
    assert all(v is None or abs(v) < 1e-9 for v in got["z_regress"].to_list())


def test_missing_weeks_do_not_crash():
    wk = _weekly([("a", "WR", 1, 10.0, 12.0), ("a", "WR", 9, 10.0, 8.0),
                  ("b", "WR", 3, 10.0, 15.0), ("b", "WR", 4, 10.0, 5.0)])
    assert weighted_signal(wk, half_life=3.0).height == 2
