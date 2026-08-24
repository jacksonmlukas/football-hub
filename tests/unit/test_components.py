"""Fantasy points built from component stats rather than projected directly.

The direction, and the reason for it: points are an aggregate of counts, and modelling the
aggregate throws away everything the counts know. Three things fall out of doing it
properly, measured in `docs/component-projection.md`:

- **Touchdowns should be regressed and volume should not.** Swapping a player's own
  touchdown rate for his position's, applied to his own yardage, beats carrying his points
  forward: RMSE 3.30 against 3.38 over 1,049 player-season pairs, 99.7% on a paired
  bootstrap. The optimal shrink is 1.0 -- a player's own touchdown rate carries no
  information beyond his yards.
- **The distribution comes out right for free.** Weekly scoring is right-skewed (1.04) with
  a median at 0.90 of the mean. A normal says 0.00 and 1.00. Sampling counts reproduces the
  skew instead of assuming it away.
- **The spread law is derived, not fitted.** Counts have variance growing with their mean,
  so their sum has spread growing with its square root -- which is what `docs/weekly-spread.md`
  measured at 0.498 +/- 0.012.

The honest caveat, kept in view because it bounds what this is worth: the gain on the
*mean* projection is about 2%, and it is concentrated in RB and WR. The payoff is the
distribution.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import components as C


# --- the aggregation is exact ---------------------------------------------

def test_points_reproduce_full_ppr():
    """Not approximately. This is arithmetic, and if it drifts every number downstream is
    quietly wrong. Checked against nflverse's own fantasy_points_ppr for 99.4% of
    player-weeks in docs/component-projection.md."""
    got = C.points({"receptions": 6, "receiving_yards": 84, "receiving_tds": 1,
                    "rushing_yards": 12, "rushing_tds": 0})
    assert got == pytest.approx(6 * 1.0 + 84 * 0.1 + 6.0 + 1.2)


def test_a_quarterback_line_scores_correctly():
    got = C.points({"passing_yards": 300, "passing_tds": 2, "interceptions": 1,
                    "rushing_yards": 20, "rushing_tds": 1})
    assert got == pytest.approx(12.0 + 8.0 - 2.0 + 2.0 + 6.0)


def test_missing_components_are_zero_not_an_error():
    assert C.points({"receptions": 3}) == pytest.approx(3.0)


def test_an_empty_line_scores_nothing():
    assert C.points({}) == 0.0


# --- touchdown regression --------------------------------------------------

def test_touchdowns_are_replaced_by_the_positional_rate_on_his_own_yards():
    """The measured optimum is a full swap, not a partial shrink."""
    lucky = {"receiving_yards": 800.0, "receiving_tds": 12.0}
    got = C.regress_touchdowns(lucky, "WR")
    assert got["receiving_tds"] == pytest.approx(800.0 * C.td_rate("WR", "rec"))
    assert got["receiving_tds"] < 12.0


def test_yardage_is_left_alone():
    """Volume persists; only the scoring rate regresses. Touching the yards would throw
    away the half of the signal that does carry forward."""
    got = C.regress_touchdowns({"receiving_yards": 800.0, "receptions": 70.0,
                                "receiving_tds": 12.0}, "WR")
    assert got["receiving_yards"] == 800.0 and got["receptions"] == 70.0


def test_an_unlucky_player_is_regressed_upward_too():
    got = C.regress_touchdowns({"receiving_yards": 900.0, "receiving_tds": 1.0}, "WR")
    assert got["receiving_tds"] > 1.0


def test_a_position_with_negligible_volume_in_a_phase_uses_the_fallback_rate():
    """The trap this guards. Quarterbacks catch a handful of passes a decade, almost all
    of them trick plays that score, so the raw QB receiving rate is 0.125 touchdowns per
    yard -- twenty times any real rate. Applying it to a quarterback who happened to catch
    a pass would invent points out of a sample of nothing."""
    assert C.td_rate("QB", "rec") == pytest.approx(C.FALLBACK_TD_RATE["rec"])
    assert C.td_rate("TE", "rush") == pytest.approx(C.FALLBACK_TD_RATE["rush"])
    assert C.td_rate("WR", "rec") != pytest.approx(C.FALLBACK_TD_RATE["rec"])


# --- the sampler ----------------------------------------------------------

def _wr(ppg_target=12.0):
    """A receiver's per-game component line, scaled to roughly a target points-per-game."""
    base = {"receptions": 4.5, "receiving_yards": 58.0, "receiving_tds": 0.35}
    s = ppg_target / C.points(base)
    return {k: v * s for k, v in base.items()}


def test_sampled_weeks_average_to_the_component_projection():
    """The sampler must not move the mean. If it does, every projection is biased by
    whatever the sampling does."""
    got = C.sample_weeks(_wr(12.0), "WR", n=40000, rng=np.random.default_rng(0))
    assert got.mean() == pytest.approx(12.0, rel=0.03)


def test_sampled_spread_follows_the_square_root_law():
    """Derived, not imposed. `docs/weekly-spread.md` measured sd = k*sqrt(mu) with k near
    2.0 by fitting it; here it falls out of sampling the counts, which is the point."""
    rng = np.random.default_rng(1)
    for mu in (6.0, 12.0, 20.0):
        sd = C.sample_weeks(_wr(mu), "WR", n=40000, rng=rng).std()
        assert sd / np.sqrt(mu) == pytest.approx(2.05, abs=0.45)


def test_quadrupling_the_projection_doubles_the_spread():
    """The law stated as the thing a user would notice, and the thing sd = 0.55*mu got
    wrong: spread grows with the root of the mean, so relative volatility falls."""
    rng = np.random.default_rng(2)
    lo = C.sample_weeks(_wr(5.0), "WR", n=40000, rng=rng)
    hi = C.sample_weeks(_wr(20.0), "WR", n=40000, rng=rng)
    assert hi.std() / lo.std() == pytest.approx(2.0, rel=0.2)
    assert (lo.std() / lo.mean()) > 1.6 * (hi.std() / hi.mean())


def test_the_sampled_week_is_right_skewed():
    """Measured at 1.04 pooled across 12,673 player-weeks. A normal is 0.00, which is what
    the model drew before this existed."""
    got = C.sample_weeks(_wr(12.0), "WR", n=60000, rng=np.random.default_rng(3))
    skew = float(((got - got.mean()) ** 3).mean() / got.std() ** 3)
    assert 0.6 < skew < 1.7


def test_the_typical_week_is_below_the_mean():
    """The consequence that matters for setting a lineup: the median week is about 0.90 of
    the mean, because the mean is carried by touchdown spikes. A normal model says the
    typical week is the projection, and it is not."""
    got = C.sample_weeks(_wr(12.0), "WR", n=60000, rng=np.random.default_rng(4))
    assert 0.83 < float(np.median(got)) / got.mean() < 0.97


def test_a_quarterback_is_less_skewed_than_a_receiver():
    """Empirically 0.29 against 1.07: passing yardage is high-volume and steady, so the
    lumpy touchdown term is a smaller share of the total."""
    rng = np.random.default_rng(5)
    qb = C.sample_weeks({"passing_yards": 250.0, "passing_tds": 1.6,
                         "interceptions": 0.7, "rushing_yards": 15.0,
                         "rushing_tds": 0.15}, "QB", n=60000, rng=rng)
    wr = C.sample_weeks(_wr(12.0), "WR", n=60000, rng=rng)
    sk = lambda x: float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    assert sk(qb) < sk(wr)


def test_a_player_projected_at_nothing_never_scores():
    got = C.sample_weeks({}, "WR", n=1000, rng=np.random.default_rng(6))
    assert got.max() == 0.0


def test_sampling_is_deterministic_given_a_seed():
    a = C.sample_weeks(_wr(), "WR", n=500, rng=np.random.default_rng(7))
    b = C.sample_weeks(_wr(), "WR", n=500, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_moments_summarise_without_the_caller_sampling():
    got = C.moments(_wr(12.0), "WR", n=20000, rng=np.random.default_rng(8))
    assert got["mu"] == pytest.approx(12.0, rel=0.04)
    assert got["sd"] > 0 and got["skew"] > 0
    assert got["p10"] < got["p50"] < got["p90"]
    assert got["p50"] < got["mu"], "right skew puts the median below the mean"


# --- projecting a season forward ------------------------------------------

def test_projection_regresses_touchdowns_and_keeps_volume():
    prior = pl.DataFrame({"player_id": ["a"], "position": ["WR"], "g": [16],
                          "receptions": [80.0], "receiving_yards": [1000.0],
                          "receiving_tds": [12.0], "rushing_yards": [0.0],
                          "rushing_tds": [0.0], "passing_yards": [0.0],
                          "passing_tds": [0.0], "interceptions": [0.0]})
    got = C.project(prior)
    assert got["receiving_yards"][0] == pytest.approx(1000.0 / 16)
    assert got["receiving_tds"][0] < 12.0 / 16
    assert got["proj_ppg"][0] > 0
