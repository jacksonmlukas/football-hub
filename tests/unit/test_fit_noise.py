"""Fitting the availability model to league history.

fit_espn_weight used to return ~1.0 for any input -- `abs(i - i) <= abs(ecr - i)` is
true whenever the right side is non-negative, so every pick counted as a win for ESPN.
That is not a small error: w=1.0 asserts the entire room drafts off ESPN's board, the
opposite of this league's premise, and w feeds every availability probability.

It cannot be fixed, only retired: ESPN does not retain historical ADP. What can be
fitted is sigma, which is what these tests cover.
"""
import numpy as np
import polars as pl
import pytest
from hub.draft import availability as av


def test_espn_weight_returns_the_prior_and_says_why(capsys):
    w = av.fit_espn_weight(123, [2024, 2025])
    assert w == av.DEFAULT_ESPN_WEIGHT
    assert "unavailable" in capsys.readouterr().out


def test_espn_weight_never_claims_a_fully_espn_room():
    """The specific old failure: a confident 1.0."""
    assert av.fit_espn_weight(123, [2024, 2025]) < 1.0


def _history(n, a, b, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.linspace(1, 180, n)
    resid = rng.normal(0.0, a + b * mu)
    return pl.DataFrame({"year": [2025] * n, "pick": mu + resid, "ecr": mu})


def test_fitted_sigma_recovers_a_known_slope(monkeypatch):
    monkeypatch.setattr(av, "historical_picks", lambda *a, **k: _history(4000, 3.0, 0.20))
    a, b = av.fit_pick_noise(123, [2025])
    assert a == pytest.approx(3.0, abs=1.5)
    assert b == pytest.approx(0.20, abs=0.05)


def test_thin_history_falls_back_to_the_prior(monkeypatch, capsys):
    monkeypatch.setattr(av, "historical_picks", lambda *a, **k: _history(10, 3.0, 0.2))
    assert av.fit_pick_noise(123, [2025]) == (2.0, 0.18)
    assert "keeping" in capsys.readouterr().out


def test_explosive_fit_is_rejected_rather_than_used(monkeypatch, capsys):
    """A history where picks bear no relation to ECR must not produce sigma > 2*mu."""
    ecr = [float(i + 1) for i in range(100)]
    bad = pl.DataFrame({"year": [2025] * 100, "pick": [e * 10 for e in ecr], "ecr": ecr})
    monkeypatch.setattr(av, "historical_picks", lambda *a, **k: bad)
    assert av.fit_pick_noise(123, [2025]) == (2.0, 0.18)
    assert "out of range" in capsys.readouterr().out


def test_negative_intercept_is_pinned_not_discarded(monkeypatch):
    """Real history wants a negative intercept; the slope is still worth keeping.

    Early picks are near-deterministic, so an unconstrained line dips below zero at the
    top of the board. Pin the intercept at the floor and refit the slope through it.
    """
    ecr = np.linspace(1, 180, 500)
    pick = ecr + np.sign(np.arange(500) % 2 - 0.5) * (0.30 * ecr - 4.0).clip(0)
    monkeypatch.setattr(av, "historical_picks", lambda *a, **k: pl.DataFrame(
        {"year": [2025] * 500, "pick": pick, "ecr": ecr}))
    a, b = av.fit_pick_noise(123, [2025])
    assert a == av.MIN_SIGMA
    assert 0.0 < b < 2.0


def test_sigma_grows_with_pick_number():
    """The back of the board is genuinely less predictable than the front."""
    df = pl.DataFrame({"mu_pick": [5.0, 150.0]})
    s = av._sigma(df)
    assert s[1] > s[0]
