"""Season simulation to a champion.

The properties that matter are not "the numbers look plausible" but that the thing
responds correctly to roster construction -- because that is the entire reason to
simulate instead of ranking by VOR.
"""
import numpy as np
import polars as pl
import pytest
from hub.draft.season import (REG_SEASON_WEEKS, _lineup_points, _round_robin,
                              champion_probability,
                              simulate_weeks)


def _roster(spec):
    """spec: list of (position, mu). Returns (idx, mu, sd, pos) for a one-team league."""
    pos = np.array([p for p, _ in spec])
    mu = np.array([m for _, m in spec], dtype=float)
    return np.arange(len(spec)), mu, np.full(len(spec), 1e-9), pos


# --- lineup ---------------------------------------------------------------

def test_lineup_starts_the_legal_maximum():
    """QB1 RB2 WR3 TE1 + FLEX1 = 8 starters, best available at each slot."""
    pos = np.array(["QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE"])
    sc = np.array([[[10, 9, 8, 7, 6, 5, 4, 3, 2]]], dtype=float)
    # QB 10 | RB 9+8 | WR 6+5+4 | TE 2 | FLEX best leftover = RB 7
    assert _lineup_points(sc, pos)[0, 0] == pytest.approx(10 + 9 + 8 + 6 + 5 + 4 + 2 + 7)


def test_flex_takes_the_best_leftover_across_positions():
    pos = np.array(["QB", "RB", "RB", "WR", "WR", "WR", "WR", "TE"])
    sc = np.array([[[1, 1, 1, 1, 1, 1, 9, 1]]], dtype=float)
    # the spare WR (9) must flex ahead of any other bench option
    assert _lineup_points(sc, pos)[0, 0] == pytest.approx(1 * 7 + 9)


def test_a_surplus_player_is_worth_nothing():
    """The core reason to simulate: VOR prices talent, lineups price slots.

    Once QB/RB2/WR3/TE/FLEX are filled, another WR of identical talent contributes
    exactly zero. VOR would happily rank him above a starter-quality TE.
    """
    full = ["QB", "RB", "RB", "WR", "WR", "WR", "WR", "WR"]   # TE slot empty, flex used
    sc8 = np.array([[[10.0] * 8]])
    baseline = _lineup_points(sc8, np.array(full))[0, 0]

    sc9 = np.array([[[10.0] * 9]])
    surplus = _lineup_points(sc9, np.array(full + ["WR"]))[0, 0]
    fills_hole = _lineup_points(sc9, np.array(full + ["TE"]))[0, 0]

    assert surplus == pytest.approx(baseline), "a sixth WR must add nothing"
    assert fills_hole > baseline, "a TE fills the empty TE slot and must add its points"


def test_missing_position_does_not_crash():
    pos = np.array(["RB", "RB"])
    assert _lineup_points(np.array([[[5.0, 5.0]]]), pos)[0, 0] == pytest.approx(10.0)


# --- schedule -------------------------------------------------------------

def test_every_team_plays_once_a_week():
    for week in _round_robin(12, 14):
        assert sorted(t for pair in week for t in pair) == list(range(12))


def test_schedule_covers_the_full_season():
    assert len(_round_robin(12, 14)) == 14


# --- end to end -----------------------------------------------------------

def _league(strengths):
    """One roster per strength value; identical shape, scaled talent."""
    shape = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB"]
    mu, pos, rosters = [], [], []
    for s in strengths:
        idx = np.arange(len(mu), len(mu) + len(shape))
        mu += [s] * len(shape)
        pos += shape
        rosters.append(idx)
    return rosters, np.array(mu, float), np.full(len(mu), 1.0), np.array(pos)


def test_probabilities_form_a_distribution():
    r, mu, sd, pos = _league([10.0] * 12)
    p = champion_probability(r, mu, sd, pos, n_sims=200)
    assert p.shape == (12,)
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()


def test_a_stronger_roster_wins_more_often():
    r, mu, sd, pos = _league([20.0] + [10.0] * 11)
    p = champion_probability(r, mu, sd, pos, n_sims=300)
    assert p[0] > p[1:].max()


def test_equal_rosters_are_roughly_equally_likely():
    r, mu, sd, pos = _league([10.0] * 12)
    p = champion_probability(r, mu, sd, pos, n_sims=600)
    assert p.max() < 0.30, "no team should dominate a league of clones"


def test_variance_matters_not_just_means():
    """Two rosters, same expected points, different volatility -- P(win) must differ."""
    r, mu, _, pos = _league([10.0] * 12)
    lo = np.full(mu.size, 1.0)
    hi = lo.copy()
    hi[r[0]] = 12.0
    a = champion_probability(r, mu, lo, pos, n_sims=500, rng=np.random.default_rng(1))
    b = champion_probability(r, mu, hi, pos, n_sims=500, rng=np.random.default_rng(1))
    assert a[0] != b[0]


def test_simulate_weeks_shape():
    r, mu, sd, pos = _league([10.0] * 12)
    assert simulate_weeks(r, mu, sd, pos, n_sims=5).shape == (5, 14, 12)


# --- projection uncertainty -----------------------------------------------

def test_talent_uncertainty_flattens_the_field():
    """With no projection error the best-projected roster wins nearly always; with
    realistic error the league becomes competitive. This is the difference between
    measuring an edge and measuring a leak."""
    r, mu, sd, pos = _league([16.0] + [10.0] * 11)
    certain = champion_probability(r, mu, sd, pos, n_sims=400, talent_cv=0.0,
                                   rng=np.random.default_rng(3))
    noisy = champion_probability(r, mu, sd, pos, n_sims=400, talent_cv=0.35,
                                 rng=np.random.default_rng(3))
    assert certain[0] > noisy[0], "projection error must reduce the favourite's edge"


def test_a_real_edge_survives_the_noise():
    """Flattening must not go so far that talent stops mattering at all."""
    r, mu, sd, pos = _league([16.0] + [10.0] * 11)
    p = champion_probability(r, mu, sd, pos, n_sims=600, talent_cv=0.35)
    assert p[0] > 1.0 / 12.0, "a clearly better roster should still beat baseline"


def test_zero_cv_reproduces_deterministic_talent():
    r, mu, sd, pos = _league([10.0] * 12)
    p = champion_probability(r, mu, sd, pos, n_sims=200, talent_cv=0.0)
    assert p.sum() == pytest.approx(1.0)


# --- weekly spread has to follow realised talent, not the projection -------

def test_a_player_whose_talent_collapses_scores_close_to_nothing():
    """The model could not produce a bust.

    Weekly points were drawn as N(realised talent, 0.55 * *projection*) and clipped at zero.
    For a player projected at 15 whose talent went to zero, that is N(0, 8.25) clipped --
    a half-normal averaging 3.3 points a game, 22% of his preseason projection, out of
    nothing but the clip. Every drafted bust in the simulation was quietly a useful bench
    player, and TALENT_CV was being asked to absorb the difference.

    Weekly spread scales with realised talent instead, so a player who loses his job loses
    his variance with it.
    """
    mu, sd = np.array([15.0]), np.array([15.0 * 0.55])
    got = simulate_weeks([np.array([0])], mu, sd, np.array(["RB"]),
                         n_sims=20000, talent_cv=0.45)
    season = got[:, :, 0].mean(axis=1)
    # At cv=0.45, P(talent below 20% of projection) = Phi(-0.8/0.45) = 3.8%, so seasons
    # that bad have to be reachable. Under the old formulation the floor was 22% of the
    # projection and this fraction was zero by construction.
    assert (season < 0.2 * 15.0).mean() > 0.02


def test_weekly_spread_still_scales_with_the_player():
    """The other half: a good player must stay volatile in absolute terms, or the model
    loses the week-to-week noise that head-to-head is made of."""
    r = [np.array([0])]
    big = simulate_weeks(r, np.array([20.0]), np.array([11.0]), np.array(["RB"]),
                         n_sims=2000, talent_cv=0.0)
    small = simulate_weeks(r, np.array([5.0]), np.array([2.75]), np.array(["RB"]),
                           n_sims=2000, talent_cv=0.0)
    assert big.std() > 2.5 * small.std()


def test_an_average_player_is_unaffected_by_the_change():
    """Regression guard: at realised talent equal to projection the two formulations are
    identical, so ordinary players must be untouched."""
    got = simulate_weeks([np.array([0])], np.array([12.0]), np.array([6.6]),
                         np.array(["RB"]), n_sims=4000, talent_cv=0.0)
    assert got.mean() == pytest.approx(12.0, rel=0.06)


# --- TALENT_CV varies by position -----------------------------------------

def test_each_position_gets_its_own_fitted_dispersion():
    """Fitted in `hub.draft.calibrate`, written up in `docs/talent-cv.md`. Only RB and TE
    are far enough from the pool to differ: RB at +2.6 se and TE at -3.8 se, while QB and
    WR sit within one standard error and shrink back onto the pooled value."""
    from hub.draft.season import TALENT_CV, TALENT_CV_BY_POS, talent_cv_for
    assert TALENT_CV_BY_POS["RB"] > TALENT_CV > TALENT_CV_BY_POS["TE"]
    got = talent_cv_for(np.array(["RB", "TE", "QB"]))
    assert got.tolist() == [TALENT_CV_BY_POS["RB"], TALENT_CV_BY_POS["TE"],
                            TALENT_CV_BY_POS["QB"]]


def test_an_unknown_position_falls_back_to_the_pooled_value():
    """K and DST are not drafted here and were never fitted. Falling back beats a KeyError
    in the middle of a draft."""
    from hub.draft.season import TALENT_CV, talent_cv_for
    assert talent_cv_for(np.array(["K", "DST"])).tolist() == [TALENT_CV, TALENT_CV]


def test_a_running_back_roster_is_more_of_a_lottery_than_a_tight_end_roster():
    """The behavioural consequence, and the whole point of doing this per position: at the
    same projection, a roster of RBs has a wider spread of seasons than a roster of TEs."""
    mu, sd = np.full(6, 12.0), np.full(6, 6.6)
    r = [np.arange(6)]
    rb = simulate_weeks(r, mu, sd, np.array(["RB"] * 6), n_sims=6000)
    te = simulate_weeks(r, mu, sd, np.array(["TE"] * 6), n_sims=6000)
    assert rb.mean(axis=1).std() > 1.15 * te.mean(axis=1).std()


def test_passing_a_single_number_still_works():
    """Back-compat, and the escape hatch every sweep in `hub.draft.leverage` uses."""
    r = [np.arange(2)]
    mu, sd, pos = np.full(2, 12.0), np.full(2, 6.6), np.array(["RB", "TE"])
    a = simulate_weeks(r, mu, sd, pos, n_sims=500, talent_cv=0.0)
    b = simulate_weeks(r, mu, sd, pos, n_sims=500, talent_cv=0.0)
    assert np.allclose(a, b)


def test_the_fitted_constants_are_inside_their_fitted_intervals():
    """Guard against a silent revert to a guessed value, same as the pooled one."""
    from hub.draft.season import TALENT_CV, TALENT_CV_BY_POS
    from hub.draft.calibrate import FITTED_CI95, FITTED_BY_POS
    assert FITTED_CI95[0] <= TALENT_CV <= FITTED_CI95[1]
    for pos, v in TALENT_CV_BY_POS.items():
        assert v == pytest.approx(FITTED_BY_POS[pos], abs=0.01)


# --- weekly spread follows a square-root law, not a proportional one -------

def test_weekly_spread_scales_with_the_square_root_of_the_mean():
    """`sd = 0.55 * mu` assumed spread is proportional to the mean. Fitted against 1,174
    player-seasons of nflverse weekly scoring, the exponent is 0.498 +/- 0.012 -- flat on
    the Poisson value and about 42 standard errors from 1.

    That is not a curve-fitting accident, it is what aggregating component stats produces:
    weekly points are a sum of count-driven pieces (receptions, carries, touchdowns) whose
    variance grows with their mean, so the spread of the total grows with its square root.
    See docs/weekly-spread.md.
    """
    from hub.draft.season import weekly_moments
    xp = pl.DataFrame({"proj_ppg": [4.0, 16.0], "position": ["WR", "WR"]})
    got = weekly_moments(xp)
    # four times the mean is twice the spread, not four times
    assert got["sd"][1] / got["sd"][0] == pytest.approx(2.0, rel=0.01)


def test_a_boom_bust_profile_is_a_property_of_the_level_not_a_setting():
    """The consequence that matters for the draft board: relative volatility falls as
    projection rises. A 5-point-a-game flier really is nearly twice the lottery, per point,
    that a 20-point-a-game starter is -- and the old constant said they were identical."""
    from hub.draft.season import weekly_moments
    xp = pl.DataFrame({"proj_ppg": [5.0, 20.0], "position": ["RB", "RB"]})
    got = weekly_moments(xp)
    cv = (got["sd"] / got["mu"]).to_list()
    assert cv[0] > 1.8 * cv[1]


def test_positions_keep_their_own_coefficient():
    from hub.draft.season import WEEKLY_K, weekly_moments
    xp = pl.DataFrame({"proj_ppg": [10.0, 10.0], "position": ["QB", "WR"]})
    got = weekly_moments(xp)
    assert got["sd"][0] < got["sd"][1]
    assert WEEKLY_K["WR"] > WEEKLY_K["QB"]


def test_a_missing_position_falls_back_to_the_pooled_coefficient():
    from hub.draft.season import WEEKLY_K_POOLED, weekly_moments
    xp = pl.DataFrame({"proj_ppg": [9.0]},
                      schema={"proj_ppg": pl.Float64}).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("position"))
    got = weekly_moments(xp)
    assert got["sd"][0] == pytest.approx(WEEKLY_K_POOLED * 3.0, rel=1e-6)


def test_a_player_projected_at_nothing_has_no_spread():
    """Under the old floor a zero-projection player still carried sd = 2.0, which the
    best-lineup rule turned into free points. sqrt(0) is 0."""
    from hub.draft.season import weekly_moments
    got = weekly_moments(pl.DataFrame({"proj_ppg": [0.0], "position": ["WR"]}))
    assert got["sd"][0] == 0.0


def test_simulate_weeks_scales_spread_by_the_root_of_realised_talent():
    """Consequence of the same law inside the simulation. Spread follows realised talent,
    and under a square-root law that means sqrt(realised/projected), not the ratio itself.
    A player who realises a quarter of his projection keeps half his spread."""
    r = [np.array([0])]
    mu, sd, pos = np.array([16.0]), np.array([8.0]), np.array(["WR"])
    full = simulate_weeks(r, mu, sd, pos, n_sims=20000, talent_cv=0.0)
    assert full.std() == pytest.approx(8.0, rel=0.08)


# --- the simulator draws a skewed week, not a normal one ------------------

def test_the_typical_simulated_week_is_below_the_projection():
    """Real weekly scoring is right-skewed: the median week is about 0.90 of the mean,
    because the mean is carried by touchdown spikes. Drawing normals made the simulator
    believe the typical week *was* the projection, which flatters every floor-based
    decision. Measured per position in docs/component-projection.md."""
    got = simulate_weeks([np.array([0])], np.array([12.0]), np.array([7.0]),
                         np.array(["WR"]), n_sims=40000, talent_cv=0.0)
    w = got[:, :, 0].ravel()
    assert 0.85 < float(np.median(w)) / w.mean() < 0.97


def test_the_simulated_week_carries_the_measured_skew():
    got = simulate_weeks([np.array([0])], np.array([12.0]), np.array([7.0]),
                         np.array(["WR"]), n_sims=40000, talent_cv=0.0)
    w = got[:, :, 0].ravel()
    skew = float(((w - w.mean()) ** 3).mean() / w.std() ** 3)
    assert 0.3 < skew < 1.1


def test_a_quarterback_week_is_nearly_symmetric():
    """QB 0.15 against WR 0.66 empirically -- passing yardage is high-volume and steady, so
    the lumpy touchdown term is a smaller share of the total."""
    qb = simulate_weeks([np.array([0])], np.array([18.0]), np.array([8.0]),
                        np.array(["QB"]), n_sims=40000, talent_cv=0.0)[:, :, 0].ravel()
    wr = simulate_weeks([np.array([0])], np.array([18.0]), np.array([8.0]),
                        np.array(["WR"]), n_sims=40000, talent_cv=0.0)[:, :, 0].ravel()
    sk = lambda x: float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    assert sk(qb) < sk(wr)


def test_skewing_the_draw_leaves_the_mean_and_spread_alone():
    """The distribution changes shape, not location or scale. If this drifts, every
    projection and every variance result in the repo moves with it."""
    got = simulate_weeks([np.array([0])], np.array([12.0]), np.array([7.0]),
                         np.array(["WR"]), n_sims=60000, talent_cv=0.0)[:, :, 0].ravel()
    assert got.mean() == pytest.approx(12.0, rel=0.03)
    assert got.std() == pytest.approx(7.0, rel=0.06)


def test_scores_are_never_negative():
    """A skewed draw with a long left shift could go below zero where a normal did not."""
    got = simulate_weeks([np.array([0])], np.array([3.0]), np.array([4.0]),
                         np.array(["RB"]), n_sims=20000, talent_cv=0.0)
    assert got.min() >= 0.0


# --- the simulator prices teammates -------------------------------------

def test_a_stacked_roster_is_more_volatile_in_the_simulator():
    """docs/correlation.md measured QB-WR at +0.232 and showed independence gives an 80%
    interval that covers 72.9%. `lineup.py` prices it; until now the simulator did not, so
    the draft optimizer understated the variance of a stacked roster in exactly the weeks a
    stack is for."""
    mu = np.array([18.0, 13.0])
    sd = np.array([8.0, 7.0])
    pos = np.array(["QB", "WR"])
    r = [np.array([0, 1])]
    same = simulate_weeks(r, mu, sd, pos, n_sims=20000, talent_cv=0.0,
                          nfl_team=np.array(["KC", "KC"]))
    apart = simulate_weeks(r, mu, sd, pos, n_sims=20000, talent_cv=0.0,
                           nfl_team=np.array(["KC", "DEN"]))
    assert same[:, :, 0].std() > apart[:, :, 0].std() * 1.03


def test_correlation_does_not_move_the_mean():
    mu, sd, pos = np.array([18.0, 13.0]), np.array([8.0, 7.0]), np.array(["QB", "WR"])
    r = [np.array([0, 1])]
    same = simulate_weeks(r, mu, sd, pos, n_sims=20000, talent_cv=0.0,
                          nfl_team=np.array(["KC", "KC"]))
    apart = simulate_weeks(r, mu, sd, pos, n_sims=20000, talent_cv=0.0,
                           nfl_team=np.array(["KC", "DEN"]))
    assert same.mean() == pytest.approx(apart.mean(), rel=0.02)


def test_two_receivers_on_a_team_are_not_correlated_by_the_simulator():
    """Measured at +0.014, so it must not pick up a spurious pairing."""
    mu, sd, pos = np.array([13.0, 11.0]), np.array([7.0, 6.0]), np.array(["WR", "WR"])
    r = [np.array([0, 1])]
    same = simulate_weeks(r, mu, sd, pos, n_sims=20000, talent_cv=0.0,
                          nfl_team=np.array(["KC", "KC"]))
    apart = simulate_weeks(r, mu, sd, pos, n_sims=20000, talent_cv=0.0,
                           nfl_team=np.array(["KC", "DEN"]))
    assert same[:, :, 0].std() == pytest.approx(apart[:, :, 0].std(), rel=0.05)


def test_omitting_team_information_behaves_as_before():
    mu, sd, pos = np.array([18.0, 13.0]), np.array([8.0, 7.0]), np.array(["QB", "WR"])
    got = simulate_weeks([np.array([0, 1])], mu, sd, pos, n_sims=2000, talent_cv=0.0)
    assert got.shape == (2000, REG_SEASON_WEEKS, 1)


def test_the_marginals_survive_correlation():
    """Correlation changes the joint, never a player's own distribution. If a player's mean,
    spread or skew moves when a teammate is added, every projection moves with it."""
    mu, sd, pos = np.array([18.0, 13.0]), np.array([8.0, 7.0]), np.array(["QB", "WR"])
    r = [np.array([0]), np.array([1])]
    same = simulate_weeks(r, mu, sd, pos, n_sims=40000, talent_cv=0.0,
                          nfl_team=np.array(["KC", "KC"]))
    qb = same[:, :, 0].ravel()
    assert qb.mean() == pytest.approx(18.0, rel=0.03)
    assert qb.std() == pytest.approx(8.0, rel=0.07)
