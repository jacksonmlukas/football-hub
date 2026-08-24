"""Volume projected from the market's own draft pick.

`docs/component-projection.md` screened a volume model that shrank toward a positional mean
and got a null: RMSE 3.311 against 3.303 for regressing touchdowns alone. The diagnosis
there was that shrinking a WR1 and a WR5 toward the same place over-shrinks the studs, and
that an ADP-implied prior was "a different and more promising thing". This is that thing.

It is promising, and the honest result is mixed -- see `docs/volume-model.md`. Anchoring on
the pick beats carrying volume forward decisively (99.9% on a paired bootstrap). It does not
beat simply reading the pick, at 87%, which is below anything this repo calls a result.

So what ships is a *decomposition*, not a replacement projection: given the market's points
projection, hand back the component line that produces it. That is what the sampler needs
and what a points projection cannot give, and it does not claim an edge nobody demonstrated.
"""
import pytest

from hub.models import components as C
from hub.models import volume as V


# --- the pick-implied prior -----------------------------------------------

def test_an_earlier_pick_implies_more_volume():
    early = V.pick_prior(6, "WR")
    late = V.pick_prior(150, "WR")
    assert early["targets"] > late["targets"]


def test_each_position_gets_the_volume_it_actually_has():
    """A quarterback has attempts and no targets; a receiver the reverse. A curve fitted per
    position without this check happily gives a quarterback four targets a game."""
    qb, wr, rb = V.pick_prior(40, "QB"), V.pick_prior(40, "WR"), V.pick_prior(40, "RB")
    assert qb["attempts"] > 25 and qb["targets"] < 1
    assert wr["targets"] > 4 and wr["attempts"] < 1
    assert rb["carries"] > wr["carries"]


def test_the_prior_produces_a_plausible_points_line():
    """End to end: an early receiver should come out somewhere near a real WR1's scoring."""
    ppg = C.points(V.pick_prior(8, "WR"))
    assert 11.0 < ppg < 22.0


def test_extrapolation_past_the_observed_range_is_clamped():
    """This league never drafted a tight end before pick 11 or a quarterback before 24, so
    the curve has nothing to say about pick 3. Unclamped it claims a tight end sees 10.8
    targets a game, more than any real one."""
    assert V.pick_prior(1, "TE") == V.pick_prior(V.PICK_RANGE["TE"][0], "TE")
    assert V.pick_prior(400, "QB") == V.pick_prior(V.PICK_RANGE["QB"][1], "QB")


def test_an_unknown_position_falls_back_rather_than_raising():
    assert V.pick_prior(50, "K") == {}


# --- blending his own season with the market's opinion --------------------

def test_projection_sits_between_his_own_volume_and_the_market_prior():
    own = {"targets": 9.0, "receptions": 6.0, "receiving_yards": 110.0,
           "carries": 0.0, "attempts": 0.0}
    got = V.project(own, pick=90, position="WR")
    prior = V.pick_prior(90, "WR")
    assert min(own["targets"], prior["targets"]) <= got["targets"] <= max(
        own["targets"], prior["targets"])


def test_volume_is_trusted_less_than_efficiency():
    """Fitted on held-out data: keep half his own volume, seven tenths of his own
    efficiency. Volume is the thing a changed situation moves most, and the pick is the only
    thing that knows the situation changed."""
    assert V.KEEP_VOLUME < V.KEEP_EFFICIENCY


def test_a_player_with_no_prior_season_falls_back_to_the_pick():
    """Rookies. There is no prior season to blend, and the market's pick is the whole of
    what is known about them."""
    assert V.project({}, pick=30, position="RB") == V.pick_prior(30, "RB")


# --- the decomposition, which is what ships -------------------------------

def test_decompose_hits_the_points_projection_it_was_given():
    """The contract. The market's mean is not second-guessed -- it is reproduced exactly,
    and only the shape comes from here."""
    got = V.decompose(pick=12, position="WR", target_ppg=15.5)
    assert C.points(got) == pytest.approx(15.5, rel=1e-6)


def test_decompose_keeps_the_shape_of_the_pick_implied_line():
    """Scaling, not reshaping: a receiver's yards-per-target should survive it."""
    prior = V.pick_prior(12, "WR")
    got = V.decompose(pick=12, position="WR", target_ppg=15.5)
    assert (got["receiving_yards"] / got["targets"]
            == pytest.approx(prior["receiving_yards"] / prior["targets"], rel=1e-6))


def test_decompose_of_a_zero_projection_is_empty_not_a_division_by_zero():
    assert C.points(V.decompose(pick=200, position="WR", target_ppg=0.0)) == 0.0


def test_a_decomposed_line_can_be_sampled():
    """The reason this exists: the sampler needs components, and the board only has points.
    This is the bridge."""
    import numpy as np
    got = V.decompose(pick=12, position="WR", target_ppg=15.5)
    draws = C.sample_weeks(got, "WR", n=20000, rng=np.random.default_rng(0))
    assert draws.mean() == pytest.approx(15.5, rel=0.05)
    assert draws.std() > 0
