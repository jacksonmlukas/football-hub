"""Fitting the margin dispersion around the closing spread.

`MARGIN_SD = 13.5` turns every closing spread into a win probability and was asserted, not
measured — while sitting in `config.FITTED_MODULES`, hashed as though it had been fitted.

These tests are written before the walk-forward is run, so the gate cannot be tuned to its own
answer. Everything here is offline.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import margin
from hub.models.market import MARGIN_SD


def _sched(rows):
    """(season, spread_line, result) triples."""
    return pl.DataFrame({"season": [r[0] for r in rows],
                         "spread_line": [float(r[1]) for r in rows],
                         "result": [r[2] for r in rows]},
                        schema={"season": pl.Int32, "spread_line": pl.Float64,
                                "result": pl.Int32})


# --- the residual ---------------------------------------------------------

def test_the_residual_is_home_relative_on_both_sides():
    """`result` is home_score - away_score and `spread_line` is home-relative, so the residual
    needs no sign juggling. Getting this backwards would look like a systematic bias."""
    got = margin.residuals(_sched([(2023, 4.0, 10)]))
    assert got["resid"][0] == pytest.approx(6.0)


def test_a_game_without_a_closing_spread_is_dropped():
    got = margin.residuals(pl.DataFrame(
        {"season": [2023, 2023], "spread_line": [3.0, None], "result": [7, 7]},
        schema={"season": pl.Int32, "spread_line": pl.Float64, "result": pl.Int32}))
    assert got.height == 1


def test_an_unplayed_game_is_dropped():
    got = margin.residuals(pl.DataFrame(
        {"season": [2023, 2023], "spread_line": [3.0, 3.0], "result": [7, None]},
        schema={"season": pl.Int32, "spread_line": pl.Float64, "result": pl.Int32}))
    assert got.height == 1


def test_ties_are_dropped():
    """A tie is neither a home win nor an away win, and there is no probability to score it
    against. Inventing a convention would be worse than excluding a handful of games."""
    got = margin.residuals(_sched([(2023, 3.0, 0), (2023, 3.0, 7)]))
    assert got.height == 1


def test_missing_columns_raise_rather_than_return_empty():
    """An empty frame here would fit a dispersion of nan and report it as a result."""
    with pytest.raises(ValueError, match="spread_line"):
        margin.residuals(pl.DataFrame({"season": [2023], "result": [7]}))


# --- the outcome convention, read from one place (issue #64) ---------------
#
# The repo scored a tied game two ways: this module and `hub.models.eval` dropped it, and
# `hub.publish._scored` -- the one that feeds the public record -- derived the outcome as
# `result > 0` and took log-loss credit for a game nobody won. `home_won` is now the single
# place that turns a realised margin into the binary outcome a proper scoring rule reads,
# so the three cannot disagree again.


def _games(rows):
    """(game_id, result) -- nflverse's realised margin, home minus away."""
    return pl.DataFrame({"game_id": [r[0] for r in rows], "result": [r[1] for r in rows]},
                        schema={"game_id": pl.Utf8, "result": pl.Float64})


def test_home_won_is_home_relative():
    got = margin.home_won(_games([("g1", 7.0), ("g2", -3.0)]))
    assert got["home_won"].to_list() == [1, 0]


def test_home_won_drops_an_unplayed_game():
    """An unplayed game arriving as a home loss is scored by log loss exactly as
    confidently as a real one, and nothing downstream can tell the two apart."""
    assert margin.home_won(_games([("g1", 7.0), ("g2", None)]))["game_id"].to_list() == ["g1"]


def test_home_won_drops_a_tie():
    got = margin.home_won(_games([("g1", 0.0), ("g2", 7.0)]))
    assert got["game_id"].to_list() == ["g2"]


def test_flipping_drop_ties_moves_home_won(monkeypatch):
    """The citation defect this replaced: `DROP_TIES` was quoted in `hub.models.eval`'s
    docstring and read by nothing outside `residuals`, so flipping it to False changed no
    behaviour and broke no test. Every caller now goes through here, so it does both."""
    monkeypatch.setattr(margin, "DROP_TIES", False)
    got = margin.home_won(_games([("g1", 0.0)]))
    assert got["home_won"].to_list() == [0]


def test_residuals_reads_the_constant_rather_than_restating_it(monkeypatch):
    monkeypatch.setattr(margin, "DROP_TIES", False)
    assert margin.residuals(_sched([(2023, 3.0, 0), (2023, 3.0, 7)])).height == 2


# --- the fit --------------------------------------------------------------

def test_the_fit_recovers_a_known_dispersion():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.0, 11.0, 6000)
    got = margin.fit(pl.DataFrame({"resid": draws}))
    assert got["sd"] == pytest.approx(11.0, rel=0.03)
    assert abs(got["mean"]) < 0.5


def test_the_standard_error_shrinks_with_n():
    """Reported because a point estimate of a dispersion invites being compared to another
    point estimate, and the entire question is whether 13.5 is far enough away to matter."""
    rng = np.random.default_rng(1)
    small = margin.fit(pl.DataFrame({"resid": rng.normal(0, 13, 200)}))
    large = margin.fit(pl.DataFrame({"resid": rng.normal(0, 13, 5000)}))
    assert large["se"] < small["se"]
    assert small["se"] == pytest.approx(small["sd"] / np.sqrt(2 * 199), rel=1e-6)


def test_too_few_games_gives_nan_not_a_crash():
    got = margin.fit(pl.DataFrame({"resid": [1.0]}))
    assert np.isnan(got["sd"])


# --- the probability model ------------------------------------------------

def test_a_pick_em_is_a_coin_flip():
    assert margin.home_win_prob(np.array([0.0]), 13.5)[0] == pytest.approx(0.5)


def test_a_smaller_dispersion_makes_a_favourite_more_confident():
    """The whole consequence of the number: a wider sd drags every game toward 50%."""
    wide = margin.home_win_prob(np.array([7.0]), 13.5)[0]
    tight = margin.home_win_prob(np.array([7.0]), 12.5)[0]
    assert tight > wide > 0.5


def test_the_sign_of_the_spread_picks_the_favourite():
    assert margin.home_win_prob(np.array([-7.0]), 13.5)[0] < 0.5


def test_log_loss_rewards_being_right_confidently():
    confident_right = margin.log_loss(np.array([0.9]), np.array([1.0]))
    hedged = margin.log_loss(np.array([0.5]), np.array([1.0]))
    confident_wrong = margin.log_loss(np.array([0.9]), np.array([0.0]))
    assert confident_right < hedged < confident_wrong


# --- the walk-forward never peeks ----------------------------------------

def _synthetic(seasons=range(2000, 2011), per=100, sd=11.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for yr in seasons:
        spreads = rng.uniform(-10, 10, per)
        margins = spreads + rng.normal(0, sd, per)
        rows += [(yr, float(s), round(m) or 1) for s, m in zip(spreads, margins, strict=True)]
    return _sched(rows)


def test_the_walk_forward_scores_every_season_but_the_first():
    """The earliest season has no history, so it can only be fitted on."""
    resid = margin.residuals(_synthetic())
    wf = margin.walk_forward(resid)
    assert wf["season"].min() == 2001
    assert wf.height == 10


def test_each_season_is_fitted_only_on_earlier_ones():
    """The leak that would make any fitted candidate look good. If the 2005 fit saw 2005, a
    dispersion fitted to that season would score itself."""
    resid = margin.residuals(_synthetic())
    wf = margin.walk_forward(resid)
    row = wf.filter(pl.col("season") == 2005).row(0, named=True)
    upto = margin.fit(resid.filter(pl.col("season") < 2005))["sd"]
    assert row["sd_all"] == pytest.approx(upto)


def test_the_trailing_window_ignores_seasons_beyond_it():
    resid = margin.residuals(_synthetic())
    wf = margin.walk_forward(resid, trailing=3)
    row = wf.filter(pl.col("season") == 2008).row(0, named=True)
    window = margin.fit(resid.filter(pl.col("season").is_between(2005, 2007)))["sd"]
    assert row["sd_trailing10"] == pytest.approx(window)


def test_a_fitted_candidate_wins_when_the_incumbent_is_plainly_wrong():
    """Sanity on the machinery: with a true dispersion of 11 and an incumbent of 13.5, a
    candidate fitted on history must score better out of sample."""
    resid = margin.residuals(_synthetic(sd=11.0, per=250))
    wf = margin.walk_forward(resid)
    assert margin._mean(wf, "ll_all") < margin._mean(wf, "ll_incumbent")


# --- the pre-registered rule ---------------------------------------------

def _wf(inc, all_, tr):
    return pl.DataFrame({"season": [2020, 2021], "n": [100, 100],
                         "ll_incumbent": inc, "ll_all": all_, "ll_trailing10": tr})


def test_a_better_challenger_is_adopted():
    winner, text = margin.verdict(_wf([0.60, 0.60], [0.55, 0.55], [0.58, 0.58]))
    assert winner == "all" and text.startswith("ADOPT")


def test_the_incumbent_wins_a_tie():
    """Replacing a constant hashed into every model version, for no measured gain, is churn."""
    winner, text = margin.verdict(_wf([0.60, 0.60], [0.60, 0.60], [0.60, 0.60]))
    assert winner == "incumbent" and text.startswith("KEEP")


def test_a_worse_challenger_leaves_the_incumbent_standing():
    winner, text = margin.verdict(_wf([0.60, 0.60], [0.70, 0.70], [0.65, 0.65]))
    assert winner == "incumbent"
    assert "no longer asserted" in text


def test_no_held_out_seasons_defaults_to_the_incumbent():
    winner, _ = margin.verdict(pl.DataFrame(schema={"season": pl.Int32}))
    assert winner == "incumbent"


def test_the_rule_reads_the_incumbent_from_market_not_a_copy():
    """If the two drifted, this module would gate against a number nothing uses."""
    import inspect
    assert "from hub.models.market import MARGIN_SD" in inspect.getsource(margin)


def test_the_live_constant_sits_inside_its_fitted_interval():
    """Guards against a silent revert to the guessed 13.5, which sat 2.6 se above the
    full-sample fit. Same pattern `calibrate.FITTED_CI95` uses for TALENT_CV."""
    lo = margin.FITTED_SD - 2 * margin.FITTED_SE
    hi = margin.FITTED_SD + 2 * margin.FITTED_SE
    assert lo <= MARGIN_SD <= hi, f"{MARGIN_SD} outside [{lo:.3f}, {hi:.3f}]"


def test_the_old_asserted_value_would_now_fail_that_guard():
    """13.5 is outside the interval, which is the point -- it was never measured."""
    lo = margin.FITTED_SD - 2 * margin.FITTED_SE
    hi = margin.FITTED_SD + 2 * margin.FITTED_SE
    assert not (lo <= 13.5 <= hi)


# --- the CLI, offline -----------------------------------------------------

def test_help_path_needs_no_network():
    assert margin.main([]) == 0


def test_the_fit_path_reports_and_gates(monkeypatch, capsys, tmp_path):
    """The whole reporting path, on synthetic seasons, with the fetch patched out. A gate
    whose CLI is only exercisable against the live API is one nobody re-runs."""
    import nflreadpy as nfl
    sched = _synthetic(seasons=range(2010, 2021), per=200, sd=11.0)
    monkeypatch.setattr(nfl, "load_schedules", lambda *a, **k: sched)

    out = tmp_path / "wf.parquet"
    assert margin.main(["--fit", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "full sample" in text
    assert "Walk-forward" in text
    # true dispersion 11 against an incumbent of 13.5: a fitted candidate must win
    assert "ADOPT" in text
    assert "Value to adopt" in text
    assert out.exists()


def test_the_fit_path_keeps_the_incumbent_when_it_is_right(monkeypatch, capsys):
    """The branch that matters more: an asserted number that survives a fit is no longer
    asserted, and the CLI has to be able to say so."""
    import nflreadpy as nfl

    from hub.models.market import MARGIN_SD as live
    sched = _synthetic(seasons=range(2010, 2021), per=200, sd=live)
    monkeypatch.setattr(nfl, "load_schedules", lambda *a, **k: sched)

    assert margin.main(["--fit"]) == 0
    assert "KEEP" in capsys.readouterr().out
