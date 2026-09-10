"""The reliability diagram, the generalisation that bins on an input, and the continuous rule.

`reliability` bins a probability against its own value, which answers "when the model said
70%, did it happen 70% of the time". `reliability_by` bins on something else -- the spread a
survivor pick was chosen by -- because a miss concentrated in one range of the *input* is
invisible in a diagram whose bins are probabilities.

The two share a binning loop rather than holding two copies of it, so the first thing tested
here is that the shared version did not change what the published curve looks like.
"""
import math

import numpy as np
import polars as pl
import pytest

from hub.models import predict, scoring_rules


def _frame(probs, won):
    return pl.DataFrame({"home_win_prob": [float(p) for p in probs],
                         "home_won": [int(w) for w in won]})


# --- what the published curve already promised ---------------------------

def test_the_bins_and_their_labels_are_unchanged():
    got = scoring_rules.reliability(_frame([0.05, 0.15, 0.95], [1, 0, 1]), n_bins=10)
    assert [b["bin"] for b in got] == [
        "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
        "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    assert [b["n"] for b in got] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]


def test_an_empty_bin_is_kept_and_says_nothing():
    got = scoring_rules.reliability(_frame([0.05], [1]), n_bins=2)
    assert got[1] == {"bin": "0.5-1.0", "n": 0, "predicted": None, "actual": None,
                      "gap": None}


def test_a_probability_of_exactly_one_lands_in_the_top_bin():
    """The last bin used to admit its upper edge by comparing against 1.01. It now includes
    the edge, which for a probability validated into [0, 1] is the same set of rows -- and
    a certainty falling out of the diagram entirely is the failure either spelling avoids."""
    got = scoring_rules.reliability(_frame([1.0], [1]), n_bins=10)
    assert got[-1]["n"] == 1


def test_an_empty_frame_is_no_diagram_rather_than_ten_empty_bins():
    assert scoring_rules.reliability(_frame([], [])) == []


def test_predicted_and_actual_are_the_two_things_the_page_compares():
    got = scoring_rules.reliability(_frame([0.72, 0.78], [1, 0]), n_bins=10)
    hit = next(b for b in got if b["n"])
    assert hit["predicted"] == pytest.approx(0.75)
    assert hit["actual"] == pytest.approx(0.5)
    assert hit["gap"] == pytest.approx(-0.25), "gap is actual minus predicted"


# --- binning on the input ------------------------------------------------

def test_it_bins_on_the_named_column_not_on_the_probability():
    """Two games with the same probability and different spreads must land in different
    bins, which is exactly what a probability-binned diagram cannot do."""
    df = pl.DataFrame({"spread": [1.0, 12.0], "p": [0.6, 0.6], "won": [0, 1]})
    got = scoring_rules.reliability_by(df, [0.0, 6.0, 20.0], on="spread", prob="p",
                                       outcome="won")
    assert [b["n"] for b in got] == [1, 1]
    assert [b["actual"] for b in got] == [0.0, 1.0]


def test_the_top_bucket_includes_its_upper_edge():
    df = pl.DataFrame({"spread": [20.0], "p": [0.9], "won": [1]})
    got = scoring_rules.reliability_by(df, [0.0, 6.0, 20.0], on="spread", prob="p",
                                       outcome="won")
    assert got[-1]["n"] == 1, "a game at the top of the range must not fall out"


def test_a_positive_gap_is_a_model_that_was_under_confident():
    df = pl.DataFrame({"spread": [10.0] * 10, "p": [0.7] * 10, "won": [1] * 9 + [0]})
    got = scoring_rules.reliability_by(df, [0.0, 20.0], on="spread", prob="p",
                                       outcome="won")
    assert got[0]["gap"] == pytest.approx(0.2)


def test_rows_outside_the_edges_are_not_counted():
    df = pl.DataFrame({"spread": [-5.0, 3.0, 99.0], "p": [0.4, 0.6, 0.99],
                       "won": [0, 1, 1]})
    got = scoring_rules.reliability_by(df, [0.0, 6.0], on="spread", prob="p",
                                       outcome="won")
    assert sum(b["n"] for b in got) == 1


def test_the_bin_label_carries_the_places_it_is_asked_for():
    df = pl.DataFrame({"spread": [3.5], "p": [0.6], "won": [1]})
    got = scoring_rules.reliability_by(df, [0.0, 7.0], on="spread", prob="p",
                                       outcome="won", places=0)
    assert got[0]["bin"] == "0-7"


# --- the continuous rule (#177) ---------------------------------------------
#
# Everything above grades a probability against an outcome that happened or did not. The
# Weekly projection emits neither: it emits a mean, a spread and a skew, and MAE -- what
# decides today -- is minimised by the median and reads only the first.
#
# **These tests are written against the mathematics and not against the implementation.**
# `crps_normal` is a closed form, so testing it against another closed form written the same
# way would prove that two copies of one expression agree. Instead the anchor is the
# *definition*, `INTEGRAL (F(x) - 1{x >= y})^2 dx`, integrated numerically here, plus the
# value at `y = mu` worked out by hand. Both are independent of how the module computes it.

def _normal_cdf(x):
    """Phi, written out here rather than imported, so the anchor owes the module nothing."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _crps_by_definition(mu, sd, y, width=40.0, n=200_001):
    """`INTEGRAL (F(x) - 1{x >= y})^2 dx`, straight off the definition.

    Split at `y`, because the integrand has a step there and quadrature across a step is
    what makes a numerical check disagree with an exact one for reasons that have nothing to
    do with either.
    """
    total = 0.0
    for lo, hi, indicator in ((mu - width * sd, y, 0.0), (y, mu + width * sd, 1.0)):
        if hi <= lo:
            continue
        x = np.linspace(lo, hi, n)
        total += float(np.trapezoid((_normal_cdf((x - mu) / sd) - indicator) ** 2, x))
    return total


@pytest.mark.parametrize(("mu", "sd", "y"), [
    (0.0, 1.0, 0.0), (0.0, 1.0, 1.5), (10.0, 3.0, 4.0), (12.0, 5.0, 30.0), (8.0, 2.0, 0.0)])
def test_the_closed_form_is_the_defining_integral(mu, sd, y):
    """#177's first criterion: exact on a case with a known closed form.

    The known form is Gneiting & Raftery's for a normal predictive distribution; what it is
    checked against is the integral the score is *defined* as, computed here. A test that
    compared the module's closed form against the same closed form retyped would pass over
    any shared misreading of it.
    """
    got = float(scoring_rules.crps_normal([mu], [sd], [y])[0])
    assert got == pytest.approx(_crps_by_definition(mu, sd, y), abs=2e-3, rel=1e-4)


def test_at_its_own_centre_it_is_the_spread_times_a_hand_computed_constant():
    """`z = 0` collapses the closed form to `sd * (sqrt(2) - 1)/sqrt(pi)`, which is arithmetic
    rather than a second implementation -- so this is exact to the last bit rather than to a
    quadrature tolerance."""
    constant = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    assert constant == pytest.approx(0.2336949772551, abs=1e-12)
    for sd in (1.0, 3.0, 7.5):
        got = float(scoring_rules.crps_normal([11.0], [sd], [11.0])[0])
        assert got == pytest.approx(sd * constant, rel=1e-12)


def test_a_point_mass_is_scored_as_the_absolute_error():
    """The identity that lets CRPS be printed beside MAE and subtracted from it.

    A projection published without a distribution is a point mass; its CDF is the same step
    function the outcome is, and the defining integral collapses to `|y - mu|`. So MAE is
    not a different scale to be reconciled -- it is this rule's score for the same number
    published with nothing around it.
    """
    got = scoring_rules.crps_normal([10.0, 10.0, 0.0], [0.0, 0.0, 0.0], [4.0, 10.0, -3.0])
    assert list(got) == [6.0, 0.0, 3.0]


def test_a_calibrated_distribution_scores_lower_than_the_point_mass_at_its_centre():
    """#177's second criterion, with both sides in closed form and no random numbers.

    Over outcomes drawn from the forecast itself the expected CRPS is `sd/sqrt(pi)` and the
    expected absolute error is `sd*sqrt(2/pi)`, so a calibrated forecast scores exactly
    `1/sqrt(2)` -- about 71% -- of what publishing only its centre scores. The outcomes are
    the forecast's own midpoint quantiles rather than a sample, so this is a quadrature of
    that expectation and not a draw that could have gone either way.
    """
    mu, sd = 12.0, 5.0
    outcomes = mu + sd * scoring_rules.normal_quantile(
        scoring_rules.quantile_levels(20_000))
    calibrated = scoring_rules.crps_normal(np.full(outcomes.size, mu),
                                           np.full(outcomes.size, sd), outcomes).mean()
    point_mass = scoring_rules.crps_normal(np.full(outcomes.size, mu),
                                           np.zeros(outcomes.size), outcomes).mean()
    assert calibrated < point_mass
    assert calibrated == pytest.approx(sd / math.sqrt(math.pi), rel=1e-3)
    assert point_mass == pytest.approx(sd * math.sqrt(2.0 / math.pi), rel=1e-3)
    assert calibrated / point_mass == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-3)


def test_a_forecast_cannot_score_better_by_misstating_its_spread():
    """Propriety, in the direction MAE is blind to -- and the reason for the whole ticket.

    Sweeping the published spread while the outcomes stay drawn from one distribution: CRPS
    is minimised at the truth and rises on both sides of it, so a model cannot buy a better
    score by claiming to be more or less certain than it is. Mean absolute error is
    *identical* at every one of those spreads, which is the same statement from the other
    side: the quantity the model works hardest to produce does not enter it at all.
    """
    mu, sigma = 12.0, 5.0
    outcomes = mu + sigma * scoring_rules.normal_quantile(
        scoring_rules.quantile_levels(20_000))
    scores = {s: float(scoring_rules.crps_normal(np.full(outcomes.size, mu),
                                                 np.full(outcomes.size, s), outcomes).mean())
              for s in (2.0, 3.5, 5.0, 7.0, 10.0)}
    assert min(scores, key=lambda s: scores[s]) == sigma, scores
    assert scores[3.5] > scores[5.0] < scores[7.0]
    maes = {s: float(np.abs(outcomes - mu).mean()) for s in scores}
    assert len(set(maes.values())) == 1, (
        "mean absolute error moved when only the published spread did, which it cannot -- "
        "the point of this test is that it does not")


@pytest.mark.parametrize("m", [100, 400, 2000])
def test_the_quantile_form_converges_on_the_closed_form(m):
    """The general rule, for the distribution this repo actually publishes.

    `crps_from_quantiles` grades a forecast given by its quantiles, which is what a
    Cornish-Fisher skew clipped at zero leaves. Checked where a closed form exists -- on a
    normal, where the two must agree -- because that is the only place the general one can
    be checked against something other than itself. The error is in the tails, where the
    grid's outermost level stops short, so it is quoted as an absolute tolerance on a
    four-sigma outcome rather than as a relative one.
    """
    mu, sd = 10.0, 3.0
    outcomes = np.array([4.0, 10.0, 22.0])
    levels = scoring_rules.quantile_levels(m)
    quantiles = mu + sd * scoring_rules.normal_quantile(levels)[None, :]
    got = scoring_rules.crps_from_quantiles(np.repeat(quantiles, outcomes.size, axis=0),
                                            outcomes)
    exact = scoring_rules.crps_normal(np.full(3, mu), np.full(3, sd), outcomes)
    assert got == pytest.approx(exact, abs=4.0 / m)


def test_the_quantile_form_reads_a_skew_no_two_moment_rule_can_see():
    """The half of #177 that MAE and a normal CRPS both miss.

    Two forecasts with **the same mean and the same spread**, one skewed the way weekly
    fantasy scoring actually is and one not. Graded over outcomes drawn from the skewed
    truth, the skewed forecast wins -- which is propriety in the third moment, and it is the
    reason the weekly report grades `predict.skewed`'s quantiles rather than a normal fitted
    to the same two moments.
    """
    mu, sd = 12.0, 5.0
    z = scoring_rules.normal_quantile(scoring_rules.quantile_levels(4_000))
    truth = predict.skewed(mu, sd, 0.66, z)                  # the outcomes, and the forecast
    skewed = predict.skewed(mu, sd, 0.66, z)[None, :]
    symmetric = predict.skewed(mu, sd, 0.0, z)[None, :]
    honest = scoring_rules.crps_from_quantiles(np.repeat(skewed, truth.size, axis=0),
                                               truth).mean()
    two_moment = scoring_rules.crps_from_quantiles(np.repeat(symmetric, truth.size, axis=0),
                                                   truth).mean()
    assert honest < two_moment, (honest, two_moment)
    assert np.abs(truth.mean() - mu) < 0.05 and np.abs(truth.std() - sd) < 0.05, (
        "the two forecasts no longer share their first two moments, so this compares "
        "something other than the skew")
