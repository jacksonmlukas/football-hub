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


def test_a_challenger_better_on_average_but_not_every_season_is_not_adopted():
    """The width gate is the same rule (#285): `docs/margin-sd.md` records that its first
    verdict fired on a mean at 15/26 seasons and said a future gate should ask for more."""
    wf = pl.DataFrame({"season": [2020, 2021, 2022], "n": [100] * 3,
                       "ll_incumbent": [0.60, 0.60, 0.60], "ll_all": [0.50, 0.50, 0.61],
                       "ll_trailing10": [0.59, 0.59, 0.59]})
    winner, text = margin.verdict(wf)
    assert winner == "trailing10" and "2/3" in text and "3/3" in text


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
    sched = _synthetic(seasons=range(2010, 2021), per=400, sd=8.0)
    monkeypatch.setattr(nfl, "load_schedules", lambda *a, **k: sched)

    out = tmp_path / "wf.parquet"
    assert margin.main(["--fit", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "full sample" in text
    assert "Walk-forward" in text
    # true dispersion 8 against an incumbent of 12.741, 400 games a season: a fitted
    # candidate must win in every held-out season, which is what the house rule asks (#285).
    # At 11 against 12.741 and 200 games the gain is real on average and lost in three
    # seasons of ten -- the case the old rule adopted and this one must not.
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


# --- the shape: mass on the key numbers (#185) -----------------------------
#
# Written before the walk-forward was run, as the tests above were. The recorded outcome
# (`FITTED_KEY_EXCESS`, `FITTED_SHAPE_GAIN`) was filled in afterwards and the last two tests
# guard the live shape against it.

def _lumpy_synthetic(seasons=range(2000, 2011), per=200, sd=11.0, seed=1, at=3, share=0.15,
                     symmetric=True):
    """Gaussian margins with `share` of games moved onto a margin of exactly +/-`at`.

    `symmetric=True` gives the sign a coin flip -- the lump sits on both sides of the spread,
    which is the shape the bump model is built for. `symmetric=False` puts it on the
    favourite's side only, which is a lump the bump model cannot see the sign of.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for yr in seasons:
        spreads = rng.uniform(-10, 10, per)
        margins = np.rint(spreads + rng.normal(0, sd, per))
        lump = rng.uniform(size=per) < share
        sign = rng.choice([-1.0, 1.0], size=per) if symmetric else np.sign(spreads)
        margins[lump] = at * sign[lump]
        margins[margins == 0] = 1
        rows += [(yr, float(s), int(m)) for s, m in zip(spreads, margins, strict=True)]
    return _sched(rows)


def test_key_number_mass_reads_the_histogram_off_the_sample():
    """Every game ending on exactly 3 puts all the empirical mass there and none at 7."""
    resid = margin.residuals(_sched([(2023, 2.0, 3), (2023, -4.0, -3), (2023, 6.0, 3)]))
    t = margin.key_number_mass(resid)
    at = dict(zip(t["key"].to_list(), t["empirical"].to_list(), strict=True))
    assert at[3] == pytest.approx(1.0)
    assert at[7] == pytest.approx(0.0)
    ex = dict(zip(t["key"].to_list(), t["excess"].to_list(), strict=True))
    assert ex[3] > 0 and ex[7] == pytest.approx(-1.0)


def test_the_spine_share_is_the_gaussian_cell_around_the_key_number():
    """A pick'em with sd 12.741 puts about 3.1% of its mass in (2.5, 3.5], and the same in
    (-3.5, -2.5]; the pooled spine share is their sum. At a 10-point spread the two cells
    differ, and the share is exactly the Gaussian's mass in (k - 1/2, k + 1/2] at +k and -k
    -- a pooled pick'em alone would not notice a cell shifted by a whole point."""
    from math import erf, sqrt

    def cdf(x):
        return 0.5 * (1.0 + erf(x / (MARGIN_SD * sqrt(2.0))))

    pick = margin.key_number_mass(margin.residuals(_sched([(2023, 0.0, 10)])), sd=MARGIN_SD)
    spine = dict(zip(pick["key"].to_list(), pick["spine"].to_list(), strict=True))
    assert spine[3] == pytest.approx(2 * 0.0305, abs=5e-4)

    ten = margin.key_number_mass(margin.residuals(_sched([(2023, 10.0, 10)])), sd=MARGIN_SD)
    spine = dict(zip(ten["key"].to_list(), ten["spine"].to_list(), strict=True))
    for k in (3, 7, 14):
        expect = (cdf(k + 0.5 - 10) - cdf(k - 0.5 - 10)) + (cdf(-k + 0.5 - 10) - cdf(-k - 0.5 - 10))
        assert spine[k] == pytest.approx(expect, rel=1e-9)


def test_no_bumps_is_the_plain_gaussian():
    """The spine is unchanged: with every excess at zero the lumpy price is `home_win_prob`."""
    s = np.array([-7.0, -3.0, 0.0, 3.0, 7.0, 14.0])
    got = margin.lumpy_home_win_prob(s, MARGIN_SD, dict.fromkeys(margin.KEY_NUMBERS, 0.0))
    assert got == pytest.approx(margin.home_win_prob(s, MARGIN_SD))


def test_the_lumpy_price_is_a_probability_and_sign_symmetric():
    s = np.linspace(-20, 20, 81)
    p = margin.lumpy_home_win_prob(s, MARGIN_SD, margin.FITTED_KEY_EXCESS)
    assert np.all((p > 0) & (p < 1))
    assert p + p[::-1] == pytest.approx(np.ones_like(p))
    assert margin.lumpy_home_win_prob(np.array([0.0]), MARGIN_SD, margin.FITTED_KEY_EXCESS)[0] \
        == pytest.approx(0.5)


def test_a_symmetric_bump_pulls_a_favourite_toward_one_half():
    """The mechanism behind the recorded null, stated as a test so it cannot be forgotten: a
    bump at +/-3 adds mass on both sides of zero, and for a 7-point favourite the losing side
    gains proportionally more than the winning side already holds."""
    s = np.array([7.0])
    plain = margin.home_win_prob(s, MARGIN_SD)[0]
    bumped = margin.lumpy_home_win_prob(s, MARGIN_SD, {3: 2.0})[0]
    assert 0.5 < bumped < plain


def test_the_bump_s_mass_is_credited_to_the_side_it_sits_on():
    """The one case a symmetric bump *helps* a favourite: a key number far enough out that
    its mirror cell is empty. A 14-point favourite with a bump at 14 gains mass at +14 and
    almost none at -14, so it wins more often -- and crediting the bump to the wrong side
    would price it well below the plain spine instead."""
    s = np.array([14.0])
    plain = margin.home_win_prob(s, MARGIN_SD)[0]
    bumped = margin.lumpy_home_win_prob(s, MARGIN_SD, {14: 5.0})[0]
    assert bumped > plain


def test_the_ceiling_is_never_negative_and_is_zero_for_a_perfect_gaussian_sample():
    """An in-sample oracle by bucket cannot score worse than the model it bounds, and a
    sample the Gaussian prices exactly leaves it nothing to gain."""
    resid = margin.residuals(_synthetic(sd=MARGIN_SD, per=400))
    top = margin.ceiling(resid)
    assert top["gain"] >= 0
    # three games at one spread, all won: the oracle says 1.0 there and the Gaussian cannot
    sure = margin.residuals(_sched([(2023, 3.0, 7), (2023, 3.0, 10), (2023, 3.0, 1)]))
    assert margin.ceiling(sure)["gain"] > 0.3


def test_the_shape_walk_forward_fits_bumps_only_on_earlier_seasons():
    """The leak that would make any fitted shape look good, in the same form as the width
    test above: the 2005 lumpy price must not have seen 2005."""
    resid = margin.residuals(_lumpy_synthetic())
    wf = margin.walk_forward_shape(resid)
    assert wf["season"].min() == 2001 and wf.height == 10
    row = wf.filter(pl.col("season") == 2005).row(0, named=True)
    past = resid.filter((pl.col("season") < 2005) & (pl.col("season") >= 2005 - margin.TRAILING))
    now = resid.filter(pl.col("season") == 2005)
    bumps = margin.key_excess(past)
    expect = margin.log_loss(
        margin.lumpy_home_win_prob(now["spread_line"].to_numpy(), MARGIN_SD, bumps),
        now["home_won"].to_numpy().astype(float))
    assert row["ll_lumpy"] == pytest.approx(expect)


def test_a_lump_symmetric_about_the_spread_is_adopted():
    """When the extra mass at 3 sits on both sides of the spread, the bump model reproduces
    the flattening it causes and prices P(win) better than the spine: the rule must say
    ADOPT -- the branch the real data did not take, held so the rule is known to fire. Half
    the games on the lump, because the house rule (#285) asks for every held-out season and
    at 200 games a season a lump of 0.4 loses one of ten to noise."""
    resid = margin.residuals(_lumpy_synthetic(share=0.5, symmetric=True))
    shape, sentence = margin.shape_verdict(margin.walk_forward_shape(resid))
    assert shape == "lumpy" and "ADOPT" in sentence


def test_a_lump_on_the_favourite_s_side_keeps_the_gaussian():
    """The real data's mechanism, as a synthetic: the excess at 3 is pooled over both signs,
    so a lump that lives on the favourite's side is fitted as a symmetric one and pulls the
    favourite the wrong way. The rule keeps the Gaussian and says so."""
    resid = margin.residuals(_lumpy_synthetic(share=0.4, symmetric=False))
    shape, sentence = margin.shape_verdict(margin.walk_forward_shape(resid))
    assert shape == "gaussian" and "KEEP" in sentence


def _shape_wf(gains):
    seasons = list(range(2001, 2001 + len(gains)))
    return pl.DataFrame({"season": seasons, "n": [100] * len(gains),
                         "ll_gaussian": [0.6] * len(gains),
                         "ll_lumpy": [0.6 - g for g in gains], "gain": gains})


def test_the_verdict_needs_every_season_and_not_just_the_mean():
    """The rule #285 replaced adopted on `mean > 0`: one big season could carry two losing
    ones. The house rule (ADR-0019) asks for the sign in every held-out season and an
    interval excluding zero, and the sentence still reports the count either way."""
    shape, sentence = margin.shape_verdict(_shape_wf([-0.01, -0.01, 0.1]))
    assert shape == "gaussian" and "1/3 seasons" in sentence
    assert margin.shape_verdict(pl.DataFrame({"gain": []}))[0] == "gaussian"


def test_the_shape_verdict_is_the_house_rule():
    """Same inputs, same answer as `experiment.gate` over a season-clustered `summarise`:
    ADOPT is the only verdict that changes the shape, and both halves have to hold."""
    from hub.models import experiment
    for gains in ([0.02, 0.01, 0.03], [-0.01, -0.01, 0.1], [-0.02, -0.01, -0.03], [0.0, 0.01]):
        wf = _shape_wf(gains)
        paired = wf.select("season", pl.col("gain").alias("diff"))
        house, _ = experiment.gate(
            experiment.summarise(paired, cluster=experiment.SEASON_CLUSTER),
            experiment.per_season(paired), margin.SHAPE_ACTIONS)
        shape, sentence = margin.shape_verdict(wf)
        assert (shape == "lumpy") == (house == "ADOPT"), gains
        assert ("ADOPT" in sentence) == (house == "ADOPT"), gains


def test_the_recorded_shape_verdict_is_unchanged_under_the_house_rule():
    """Re-derived from the record: a walk-forward with the recorded mean, the recorded
    seasons-better count and the recorded season count keeps the Gaussian under the house
    rule as it did under the mean -- 7 of 27 fails the every-season half on its own."""
    won, total = margin.FITTED_SHAPE_SEASONS_BETTER, margin.FITTED_SHAPE_SEASONS
    assert won < total and margin.FITTED_SHAPE_GAIN < 0
    lose = (margin.FITTED_SHAPE_GAIN * total - 0.001 * won) / (total - won)
    shape, sentence = margin.shape_verdict(_shape_wf([0.001] * won + [lose] * (total - won)))
    assert shape == "gaussian" and f"{won}/{total} seasons" in sentence


def test_the_calibration_table_grades_the_favourite_side_of_every_held_out_game():
    """Both sides go in and the buckets span [0, 30), so they partition the favourite sides:
    one per held-out game. The favourites row is a subset of them, not a further bucket."""
    resid = margin.residuals(_lumpy_synthetic())
    cal = margin.calibration_by_spread(resid)
    assert cal["bucket"].to_list()[-1] == "favourites"
    buckets = cal.filter(pl.col("bucket") != "favourites")
    assert buckets["n"].sum() == resid.filter(pl.col("season") > 2000).height
    fav = cal.filter(pl.col("bucket") == "favourites")["n"][0]
    assert 0 < fav < buckets["n"].sum()
    for col in ("gaussian", "lumpy", "actual"):
        assert cal.filter(pl.col("n") > 0)[col].is_between(0.0, 1.0).all()


def test_survival_is_the_favourites_price_to_the_power_of_the_season():
    cal = pl.DataFrame({"bucket": ["[0, 3)", "favourites"], "n": [10, 10],
                        "gaussian": [0.55, 0.8], "lumpy": [0.55, 0.75], "actual": [0.5, 0.9]})
    got = margin.survival_beside(cal, picks=3)
    assert got["gaussian"] == pytest.approx(0.8 ** 3)
    assert got["lumpy"] == pytest.approx(0.75 ** 3)
    assert got["actual"] == pytest.approx(0.9 ** 3)
    assert margin.survival_beside(cal)["picks"] == 18


def test_the_recorded_excess_is_largest_at_three_and_prices_a_favourite_below_the_spine():
    """The record of 2026-09-11, guarded. A margin of 3 carries the largest excess, and the
    recorded shape moves a survivor-range favourite *down* -- the direction that made the
    Gaussian win. A refit that flips either updates these numbers and this test together."""
    assert max(margin.FITTED_KEY_EXCESS.items(), key=lambda kv: kv[1])[0] == 3
    assert margin.FITTED_KEY_EXCESS[3] > 1.5
    s = np.array([7.0, 10.0])
    lumpy = margin.lumpy_home_win_prob(s, MARGIN_SD, margin.FITTED_KEY_EXCESS)
    assert np.all(lumpy < margin.home_win_prob(s, MARGIN_SD))


def test_the_live_shape_is_the_one_the_record_supports():
    """The recorded held-out gain is negative beyond two standard errors, so the number the
    repo prices with must still be the plain Gaussian: `survivor` and `MarketBaseline` read
    `normal_cdf(spread / MARGIN_SD)` and nothing reads `lumpy_home_win_prob`."""
    import inspect

    from hub.models import market
    from hub.season import survivor
    assert margin.FITTED_SHAPE_GAIN + 2 * margin.FITTED_SHAPE_SE < 0
    assert margin.FITTED_SHAPE_CEILING > 0
    for mod in (market, survivor):
        assert "lumpy" not in inspect.getsource(mod)


def test_the_shape_path_reports_the_ceiling_first_and_keeps_the_gaussian(monkeypatch, capsys):
    """Rule 8 in the printed order: the ceiling before the histogram before the verdict. On
    the favourite-side lump, the verdict is the one the real data gave."""
    import nflreadpy as nfl
    sched = _lumpy_synthetic(seasons=range(2010, 2021), share=0.4, symmetric=False)
    monkeypatch.setattr(nfl, "load_schedules", lambda *a, **k: sched)
    assert margin.main(["--shape"]) == 0
    text = capsys.readouterr().out
    assert text.index("Ceiling") < text.index("Mass on the key numbers") < text.index("KEEP")
    assert "Survival over 18" in text


# --- empty inputs answer with NaN, not an exception (#185 coverage) ---------

def test_the_ceiling_on_no_games_is_nan_and_says_so():
    """A ceiling over zero games is not zero -- it is unmeasured. NaN with the count."""
    empty = margin.residuals(_synthetic()).head(0)
    got = margin.ceiling(empty)
    assert got["n"] == 0.0
    assert got["ll_gaussian"] != got["ll_gaussian"], "an empty ceiling reported a number"


def test_calibration_on_no_games_is_an_empty_typed_frame():
    """The bucket table with nothing in it keeps its schema, so a consumer joining on it
    gets zero rows rather than a column-not-found."""
    empty = margin.residuals(_synthetic()).head(0)
    got = margin.calibration_by_spread(empty)
    assert got.height == 0
    assert {"bucket", "n", "gaussian"} <= set(got.columns)


def test_survival_beside_is_labelled_as_the_independence_bound_it_is(monkeypatch, capsys):
    """#286. An eighteenth power of one bucket's rate asserts one price every week and
    independence across weeks, and it was printed in the register of the measured numbers
    beside it with no interval, so it read as one. The figure now carries the games its rate
    rests on, and the line names it as the bound it is -- on the report path, not only on
    the function."""
    cal = pl.DataFrame({"bucket": ["[0, 3)", "favourites"], "n": [40, 10],
                        "gaussian": [0.55, 0.8], "lumpy": [0.55, 0.75], "actual": [0.5, 0.9]})
    got = margin.survival_beside(cal, picks=3)
    assert got["n"] == 10.0
    line = margin.survival_line(got)
    assert "Survival over 3 such favourites" in line
    assert "independence bound" in line and "10 games" in line
    assert "not a measurement" in line and f"{0.9 ** 3:.4f}" in line

    import nflreadpy as nfl
    sched = _lumpy_synthetic(seasons=range(2010, 2021), share=0.4, symmetric=False)
    monkeypatch.setattr(nfl, "load_schedules", lambda *a, **k: sched)
    assert margin.main(["--shape"]) == 0
    assert "independence bound" in capsys.readouterr().out


def test_survival_beside_an_empty_calibration_is_nan():
    """A season-long product over no buckets is undefined, and the picks count is still
    reported so a reader can see what was asked for."""
    empty = margin.calibration_by_spread(margin.residuals(_synthetic()).head(0))
    got = margin.survival_beside(empty, picks=18)
    assert got["picks"] == 18.0
    assert got["gaussian"] != got["gaussian"] and got["lumpy"] != got["lumpy"]


def test_since_skips_the_seasons_before_it_in_the_walk_forward():
    """`since` is how a caller reads the verdict from one season on without refitting the
    bumps -- earlier seasons are skipped, not included with zero weight."""
    resid = margin.residuals(_synthetic(per=120))
    seasons = sorted(resid["season"].unique().to_list())
    late = margin.calibration_by_spread(resid, since=seasons[-1])
    full = margin.calibration_by_spread(resid)
    assert late["n"].sum() < full["n"].sum(), "since= did not skip anything"


def test_main_degrades_when_the_schedule_pull_fails(monkeypatch, capsys):
    """CLAUDE.md's rule: a failing source is reported, not raised. `main` returns
    `unavailable`'s code and says which source, rather than a stack trace."""
    import types

    fake = types.SimpleNamespace(load_schedules=lambda: (_ for _ in ()).throw(
        RuntimeError("nflverse down")))
    monkeypatch.setitem(__import__("sys").modules, "nflreadpy", fake)
    code = margin.main(["--shape"])
    captured = capsys.readouterr()
    said = captured.out + captured.err
    assert code != 0
    assert "nflverse schedules" in said or "unavailable" in said.lower()
