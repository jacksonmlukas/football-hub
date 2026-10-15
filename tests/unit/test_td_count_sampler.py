"""#307: the touchdown count has no support ceiling, holds its dispersion across the mean range,
and is drawn given the yards.

The sampler this replaced drew `Binomial(round(mean / (1 - phi)), ...)`. At a WR1's receiving
rate that is two trials, so three touchdowns in a week had probability exactly zero where the
Poisson gives 0.6%, and the rounding moved the realised dispersion off the constant the family
was chosen to hold. `_shipped_binomial` below is that sampler, kept here as the **control**
(`docs/method.md` rule 18): every test that says the ceiling is gone or the dispersion holds is
run against it first, and each of those assertions fails on it.
"""
from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from hub.config import fitted_constants, fitted_digest
from hub.models import components as C
from hub.models import props

WR1_RATE = 0.36   # receiving touchdowns a game for a WR1; the Poisson's P(3+) is 0.59%


def _shipped_binomial(rng, mean: float, phi: float, n: int) -> np.ndarray:
    """The branch #307 replaced, verbatim."""
    p = 1.0 - phi
    trials = round(mean / p)
    return rng.binomial(max(trials, 1), min(mean / max(trials, 1), 1.0), n)


def _poisson_at_least_three(mean: float) -> float:
    return 1.0 - math.exp(-mean) * (1.0 + mean + mean ** 2 / 2.0)


def _exact_dispersion(pmf: np.ndarray) -> float:
    k = np.arange(pmf.size)
    mean = float(pmf @ k)
    return (float(pmf @ k ** 2) - mean ** 2) / mean


# --- no support ceiling -----------------------------------------------------------------

def test_the_shipped_binomial_could_not_draw_three_touchdowns_at_a_wr1_rate():
    """The control. If this stopped holding, the tests below would be asserting something the
    old sampler also did."""
    d = _shipped_binomial(np.random.default_rng(0), WR1_RATE, 0.83, 2_000_000)
    assert d.max() == 2
    assert (d >= 3).mean() == 0.0


def test_three_touchdowns_are_possible_at_a_wr1_rate_and_close_to_what_the_dispersion_implies():
    """Not zero, under the Poisson's (the data are under-dispersed: fewer big weeks than
    Poisson), and the number the fitted dispersion's own distribution gives."""
    n = 2_000_000
    d = C.draw_counts(np.random.default_rng(0), WR1_RATE, 0.83, n)
    got = float((d >= 3).mean())
    implied = float(C.cmp_pmf(WR1_RATE, C.cmp_nu(WR1_RATE, 0.83))[3:].sum())
    assert 0.0 < got < _poisson_at_least_three(WR1_RATE)
    assert got == pytest.approx(implied, rel=0.1)
    assert d.max() >= 3


@pytest.mark.parametrize("mean", [0.18, 0.2, 0.22])
def test_a_small_mean_is_not_capped_at_one_touchdown(mean):
    """The shipped sampler allowed one touchdown at most at a mean this small (one trial).
    Here two happen, as they do for any count whose dispersion is reachable at its mean --
    0.85 is, from a mean of 0.15 up."""
    old = _shipped_binomial(np.random.default_rng(1), mean, 0.85, 1_000_000)
    new = C.draw_counts(np.random.default_rng(1), mean, 0.85, 1_000_000)
    assert old.max() == 1
    assert new.max() >= 2


class _Uniform:
    """A generator whose `random` is a planted constant, for the draw that falls off the end."""

    def __init__(self, value: float):
        self.value = value

    def random(self, n):
        return np.full(n, self.value)


def _planted(value: float) -> np.random.Generator:
    return cast(np.random.Generator, _Uniform(value))


def test_a_uniform_past_the_end_of_the_table_is_capped_at_the_last_count():
    """The inverse-CDF search returns one past the table when the uniform is not below its last
    entry (rounding leaves it a hair under one). That index is a count outside the support the
    table was built on; it is capped at the last count, not returned."""
    top = C._cmp_support(WR1_RATE)
    got = C.draw_counts(_planted(2.0), WR1_RATE, 0.83, 5)
    assert got.tolist() == [top] * 5
    assert C.draw_counts(_planted(0.0), WR1_RATE, 0.83, 3).tolist() == [0, 0, 0]


def test_no_draw_exceeds_the_table_the_support_was_built_on():
    top = C._cmp_support(3.0)
    d = C.draw_counts(np.random.default_rng(2), 3.0, 0.83, 500_000)
    assert d.max() <= top


# --- the realised dispersion matches the constant, across the mean range ----------------

MEANS = [0.2, 0.36, 0.7, 1.0, 1.6, 3.0, 6.0]


@pytest.mark.parametrize("phi", [0.83, 0.86])
@pytest.mark.parametrize("mean", MEANS)
def test_the_realised_dispersion_is_the_constant_at_every_mean(mean, phi):
    pmf = C.cmp_pmf(mean, C.cmp_nu(mean, phi))
    assert _exact_dispersion(pmf) == pytest.approx(phi, abs=1e-3)
    d = C.draw_counts(np.random.default_rng(3), mean, phi, 600_000)
    assert d.mean() == pytest.approx(mean, rel=0.01)
    assert d.var() / d.mean() == pytest.approx(phi, rel=0.02)


def test_the_shipped_binomial_drifted_off_its_dispersion_where_the_trials_round():
    """The control for the test above: at a mean of 0.26 the trials round from 1.5 to 2 and the
    binomial's variance over mean is 0.87 against the 0.83 it was fitted to hold. The test
    above could fail, and this is the place it would have."""
    old = _shipped_binomial(np.random.default_rng(4), 0.26, 0.83, 1_000_000)
    assert old.var() / old.mean() == pytest.approx(0.87, abs=0.01)
    assert abs(old.var() / old.mean() - 0.83) > 0.03
    new = C.draw_counts(np.random.default_rng(4), 0.26, 0.83, 1_000_000)
    assert abs(new.var() / new.mean() - 0.83) < 0.015


def test_a_dispersion_no_count_of_that_mean_can_reach_runs_to_the_narrowest_law():
    """A count of mean m < 1 has variance at least m(1-m), so 0.83 is out of reach under a mean
    of 0.17. `nu` is capped at `CMP_NU_MAX` and the realised dispersion is the floor, 1 - m."""
    assert C.cmp_nu(0.05, 0.83) == C.CMP_NU_MAX
    d = C.draw_counts(np.random.default_rng(5), 0.05, 0.83, 1_000_000)
    assert d.var() / d.mean() == pytest.approx(0.95, abs=0.01)
    assert C.cmp_nu(0.5, 0.83) < C.CMP_NU_MAX


def test_nu_is_a_function_of_the_dispersion_asked_for():
    assert C.cmp_nu(1.6, 0.7) > C.cmp_nu(1.6, 0.9) > 1.0
    for phi in (0.0, 1.0, 1.4):
        with pytest.raises(ValueError, match="not in"):
            C.cmp_nu(1.0, phi)
    with pytest.raises(ValueError, match="no count"):
        C.cmp_nu(0.0, 0.8)


def test_a_mean_far_above_a_touchdown_count_still_draws_it():
    d = C.draw_counts(np.random.default_rng(6), 40.0, 0.9, 200_000)
    assert d.mean() == pytest.approx(40.0, rel=0.01)
    assert d.var() / d.mean() == pytest.approx(0.9, rel=0.03)


def test_no_mean_no_count_and_the_other_families_are_unmoved():
    assert C.draw_counts(np.random.default_rng(0), 0.0, 0.83, 7).tolist() == [0] * 7
    over = C.draw_counts(np.random.default_rng(0), 4.0, 1.5, 200_000)
    assert over.var() / over.mean() == pytest.approx(1.5, rel=0.05)
    assert np.array_equal(C.draw_counts(np.random.default_rng(8), 4.0, 1.5, 50),
                          np.random.default_rng(8).negative_binomial(8.0, 1 / 1.5, 50))


# --- drawn given the yards --------------------------------------------------------------

def _yards(n: int = 200_000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).gamma(2.0, 30.0, n)


def _corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_independent_touchdowns_are_uncorrelated_with_the_yards_the_planted_condition():
    """The structure the issue found, planted: `rho = 0` is the old draw. The correlation test
    below has to see the difference."""
    y = _yards()
    td = C.draw_touchdowns(np.random.default_rng(1), 0.45, 0.83, y, 0.0)
    assert abs(_corr(y, td)) < 0.01


def test_touchdowns_drawn_given_the_yards_follow_them():
    y = _yards()
    td = C.draw_touchdowns(np.random.default_rng(1), 0.45, 0.83, y, C.TD_YARDS_RHO["rec"])
    assert 0.25 < _corr(y, td) < 0.5
    top, rest = td[y > np.quantile(y, 0.9)], td[y <= np.quantile(y, 0.9)]
    assert top.mean() > 2.0 * rest.mean()
    stronger = C.draw_touchdowns(np.random.default_rng(1), 0.45, 0.83, y, 0.9)
    assert _corr(y, stronger) > _corr(y, td)


def test_the_coupling_reorders_the_counts_and_adds_no_new_ones():
    """The marginal law is the one drawn: the same seed gives the same multiset of counts with
    and without the yards, so the measured mean and dispersion are not touched."""
    y = _yards(50_000)
    marginal = C.draw_counts(np.random.default_rng(9), 0.7, 0.83, y.size)
    coupled = C.draw_touchdowns(np.random.default_rng(9), 0.7, 0.83, y, 0.5)
    assert np.array_equal(np.sort(coupled), np.sort(marginal))
    assert not np.array_equal(coupled, marginal)


def test_no_correlation_or_no_yards_or_no_touchdowns_is_the_marginal_draw():
    y = _yards(5_000)
    marginal = C.draw_counts(np.random.default_rng(9), 0.7, 0.83, y.size)
    assert np.array_equal(C.draw_touchdowns(np.random.default_rng(9), 0.7, 0.83, y, 0.0), marginal)
    assert np.array_equal(
        C.draw_touchdowns(np.random.default_rng(9), 0.7, 0.83, np.zeros(y.size), 0.5), marginal)
    spent = np.random.default_rng(9)
    assert C.draw_touchdowns(spent, 0.0, 0.83, y, 0.5).tolist() == [0] * y.size
    # nothing to couple, so nothing was drawn: the stream is where a fresh one is
    assert spent.random() == np.random.default_rng(9).random()


def test_a_phase_draws_its_units_and_yards_before_its_touchdowns():
    """Only the touchdowns follow the yards: a seed's units and yards are the draws they were,
    so a yards prop is not re-priced by this change."""
    line = {"receptions": 5.0, "receiving_yards": 62.0, "receiving_tds": 0.45}
    units, yards, tds = C.phase_draws(np.random.default_rng(11), line, "rec", 2_000)
    rng = np.random.default_rng(11)
    want_units = C.draw_counts(rng, 5.0, C.COUNT_DISPERSION["rec"], 2_000)
    want_yards = np.zeros(2_000)
    nz = want_units > 0
    cv = C.PER_UNIT_CV["rec"]
    want_yards[nz] = rng.gamma(want_units[nz] / cv ** 2, (62.0 / 5.0) * cv ** 2)
    assert np.array_equal(units, want_units)
    assert np.array_equal(yards, want_yards)
    assert tds.shape == (2_000,)


@pytest.mark.parametrize("phase", ["rec", "rush", "pass"])
def test_a_phase_keeps_its_touchdown_mean_and_dispersion_while_following_its_yards(phase):
    lines = {"rec": {"receptions": 5.0, "receiving_yards": 62.0, "receiving_tds": 0.45},
             "rush": {"carries": 16.0, "rushing_yards": 68.0, "rushing_tds": 0.55},
             "pass": {"attempts": 34.0, "passing_yards": 250.0, "passing_tds": 1.7}}
    tds_key = C._PHASE_TDS[phase]
    _, yards, tds = C.phase_draws(np.random.default_rng(12), lines[phase], phase, 300_000)
    assert tds.mean() == pytest.approx(lines[phase][tds_key], rel=0.01)
    assert tds.var() / tds.mean() == pytest.approx(C.TD_DISPERSION[phase], rel=0.03)
    assert _corr(yards, tds) > 0.25


def test_touchdowns_without_yards_are_drawn_marginally():
    _, yards, tds = C.phase_draws(np.random.default_rng(13), {"rushing_tds": 0.5}, "rush", 20_000)
    assert not yards.any()
    assert tds.mean() == pytest.approx(0.5, rel=0.05)


def test_a_big_week_is_big_because_the_yards_and_the_touchdowns_happened_together(monkeypatch):
    """The reason for the change, at the level a lineup decision reads: the upper tail of a
    week's points. The same line is sampled with the touchdowns tied to the yards and, planted,
    drawn apart; the tied tail is heavier at the same mean."""
    line = {"receptions": 5.0, "receiving_yards": 62.0, "receiving_tds": 0.45}
    tied = C.sample_weeks(line, "WR", 300_000, np.random.default_rng(3))
    monkeypatch.setitem(C.TD_YARDS_RHO, "rec", 0.0)
    apart = C.sample_weeks(line, "WR", 300_000, np.random.default_rng(3))
    assert tied.mean() == pytest.approx(apart.mean(), rel=0.01)
    assert (tied >= 30).mean() > 1.5 * (apart >= 30).mean()
    assert (tied >= 35).mean() > 1.8 * (apart >= 35).mean()


# --- one statement of the draw, and the version that carries it -------------------------

def test_a_prop_is_priced_from_the_draw_the_points_use():
    """`props` held a second copy of the count sampler and of `phase_draws`; it takes both from
    `hub.models.components` now, so a prop and a projection cannot disagree about the family."""
    line = {"attempts": 34.0, "passing_yards": 250.0, "passing_tds": 1.7,
            "carries": 3.0, "rushing_yards": 12.0, "rushing_tds": 0.1}
    got = props.stat_draws(line, "passing_tds", n=5_000)
    want = C.phase_draws(np.random.default_rng(props.SEED), line, "pass", 5_000)[2]
    assert np.array_equal(got, want)
    rng = np.random.default_rng(props.SEED)
    rush = C.phase_draws(rng, line, "rush", 5_000)[2]
    rec = C.phase_draws(rng, {"receptions": 1.0, **line}, "rec", 5_000)[2]
    both = props.stat_draws({**line, "receptions": 1.0}, "anytime_td", n=5_000)
    assert np.array_equal(both, (rush + rec).astype(float))


def test_the_new_constants_are_in_the_model_version():
    """A prop price carries `fitted_digest`; a touchdown draw that changed without moving it
    would be two prices under one version (ADR-0006)."""
    have = fitted_constants()
    assert "components.TD_YARDS_RHO" in have and "components.CMP_NU_MAX" in have
    base = fitted_digest()
    moved = {**have, "components.TD_YARDS_RHO": {**C.TD_YARDS_RHO, "rec": 0.0}}
    assert fitted_digest(moved) != base
    moved = {**have, "components.CMP_NU_MAX": 8.0}
    assert fitted_digest(moved) != base
