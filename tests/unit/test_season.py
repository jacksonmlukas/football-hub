"""Season simulation to a champion.

The properties that matter are not "the numbers look plausible" but that the thing
responds correctly to roster construction -- because that is the entire reason to
simulate instead of ranking by VOR.
"""
import numpy as np
import pytest

from hub.draft import season
from hub.draft.season import (
    REG_SEASON_WEEKS,
    _round_robin,
    lineup_points,
    simulate_weeks,
)

# `champion_probability` is the simulator's top -- `simulate_weeks`, then `seed_table`, then
# `champion`, read off as P(each team wins) -- and the end-to-end tests below exercise the
# simulator through it. It lives in the exhibit since #198 because its only caller is the
# removed arm; the tests stay here because what they assert is the season, not the arm.
from hub.exhibits.championship_equity import champion_probability


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
    assert lineup_points(sc, pos)[0, 0] == pytest.approx(10 + 9 + 8 + 6 + 5 + 4 + 2 + 7)


def test_flex_takes_the_best_leftover_across_positions():
    pos = np.array(["QB", "RB", "RB", "WR", "WR", "WR", "WR", "TE"])
    sc = np.array([[[1, 1, 1, 1, 1, 1, 9, 1]]], dtype=float)
    # the spare WR (9) must flex ahead of any other bench option
    assert lineup_points(sc, pos)[0, 0] == pytest.approx(1 * 7 + 9)


def test_a_surplus_player_is_worth_nothing():
    """The core reason to simulate: VOR prices talent, lineups price slots.

    Once QB/RB2/WR3/TE/FLEX are filled, another WR of identical talent contributes
    exactly zero. VOR would happily rank him above a starter-quality TE.
    """
    full = ["QB", "RB", "RB", "WR", "WR", "WR", "WR", "WR"]   # TE slot empty, flex used
    sc8 = np.array([[[10.0] * 8]])
    baseline = lineup_points(sc8, np.array(full))[0, 0]

    sc9 = np.array([[[10.0] * 9]])
    surplus = lineup_points(sc9, np.array([*full, "WR"]))[0, 0]
    fills_hole = lineup_points(sc9, np.array([*full, "TE"]))[0, 0]

    assert surplus == pytest.approx(baseline), "a sixth WR must add nothing"
    assert fills_hole > baseline, "a TE fills the empty TE slot and must add its points"


def test_missing_position_does_not_crash():
    pos = np.array(["RB", "RB"])
    assert lineup_points(np.array([[[5.0, 5.0]]]), pos)[0, 0] == pytest.approx(10.0)


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
    from hub.draft.calibrate import FITTED_BY_POS, FITTED_NOMINAL_CI95
    from hub.draft.season import TALENT_CV, TALENT_CV_BY_POS
    assert FITTED_NOMINAL_CI95[0] <= TALENT_CV <= FITTED_NOMINAL_CI95[1]
    for pos, v in TALENT_CV_BY_POS.items():
        assert v == pytest.approx(FITTED_BY_POS[pos], abs=0.01)


# The square-root spread law itself is `hub.models.predict`'s, and its tests moved to
# `tests/unit/test_predict.py` with it. What stays here is the law's consequence *inside a
# simulated season*, which is this module's subject.

def test_the_caller_can_supply_the_skew_it_already_computed():
    """`predict.moments` returns a skew column, and the simulator used to throw it away and
    recompute skew from position. The two agreed, which is exactly why nobody noticed: they
    would go on agreeing until skew stopped being a function of position alone."""
    from hub.draft.season import simulate_weeks
    r = [np.array([0])]
    mu, sd, pos = np.array([14.0]), np.array([7.0]), np.array(["WR"])
    flat = simulate_weeks(r, mu, sd, pos, n_sims=400, weeks=14,
                          rng=np.random.default_rng(3), skew=np.array([0.05]))
    lumpy = simulate_weeks(r, mu, sd, pos, n_sims=400, weeks=14,
                           rng=np.random.default_rng(3), skew=np.array([1.20]))
    assert scipy_skew(flat.ravel()) < scipy_skew(lumpy.ravel())


def test_omitting_the_skew_falls_back_to_the_positional_table():
    """Every caller that predates the parameter must keep its numbers exactly."""
    from hub.draft.season import simulate_weeks, weekly_skew_for
    r = [np.array([0, 1])]
    mu, sd = np.array([14.0, 9.0]), np.array([7.0, 5.0])
    pos = np.array(["WR", "QB"])
    kw = {"n_sims": 50, "weeks": 14}
    implicit = simulate_weeks(r, mu, sd, pos, rng=np.random.default_rng(7), **kw)
    explicit = simulate_weeks(r, mu, sd, pos, rng=np.random.default_rng(7),
                              skew=weekly_skew_for(pos), **kw)
    assert np.array_equal(implicit, explicit)


def scipy_skew(x: np.ndarray) -> float:
    """Sample skewness, without pulling scipy in for one moment."""
    d = x - x.mean()
    return float((d ** 3).mean() / x.std() ** 3)


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
    def sk(x):
        return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
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


# --- the bracket gets its own weeks ----------------------------------------
#
# `champion_probability` exposed eight knobs and not `weeks`, and `_week_winner` resolved a
# playoff week as `week % pts.shape[1]` -- so a 14-week simulation replayed weeks 1-3 as the
# three playoff rounds. None of the assertions below could be written against that interface,
# which is why `leverage.py` forked the whole bracket instead.

def _pts(n_sims=1, weeks=None, teams=12, seed=0):
    weeks = weeks or (season.REG_SEASON_WEEKS + season.PLAYOFF_ROUNDS)
    rng = np.random.default_rng(seed)
    return rng.normal(110.0, 20.0, size=(n_sims, weeks, teams))


def test_the_champion_ignores_the_regular_season_once_seeded():
    """The property the old code violated. Given the seeds, the title is decided by the
    playoff weeks alone -- so overwriting every regular-season week must change nothing.
    Under `week % pts.shape[1]` it changed the champion, because the bracket was reading
    weeks 0-2."""
    pts = _pts()
    _, seeds = season.seed_table(pts)
    before = season.champion(pts, seeds, 0)
    scrambled = pts.copy()
    scrambled[:, :season.REG_SEASON_WEEKS, :] = 0.0
    assert season.champion(scrambled, seeds, 0) == before


def test_changing_a_playoff_week_can_change_the_champion():
    """The other direction, so the test above cannot pass by the bracket ignoring `pts`."""
    pts = _pts(seed=3)
    _, seeds = season.seed_table(pts)
    winner = season.champion(pts, seeds, 0)
    rigged = pts.copy()
    loser = next(t for t in seeds[0] if t != winner)
    rigged[0, season.REG_SEASON_WEEKS:, loser] = 10_000.0
    assert season.champion(rigged, seeds, 0) == loser


def test_seeding_ignores_the_playoff_weeks():
    """The tiebreak is total points, and `champion_probability` summed the whole array. Once
    the array carries playoff weeks, that would let a semi-final decide the seed a team
    entered the playoffs with."""
    pts = _pts(n_sims=4)
    _, seeds = season.seed_table(pts)
    loud = pts.copy()
    loud[:, season.REG_SEASON_WEEKS:, :] = 9_999.0
    _, seeds_loud = season.seed_table(loud)
    assert (seeds == seeds_loud).all()


def test_champion_probability_simulates_the_playoff_weeks():
    rosters = [np.arange(t * 3, (t + 1) * 3) for t in range(12)]
    mu = np.full(36, 12.0)
    sd = np.full(36, 5.0)
    pos = np.array(["RB"] * 36)
    p = champion_probability(rosters, mu, sd, pos, n_sims=200, rng=np.random.default_rng(0))
    assert p.shape == (12,)
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).sum() >= 6, "twelve identical teams should spread the title around"


def test_leverage_and_season_share_one_bracket():
    """`leverage.py` re-implemented the seeding loop line for line and the bracket beside it.
    A copy that drifts is how this file's own docstring says the weekly model went stale."""
    import inspect

    from hub.exhibits import leverage
    src = inspect.getsource(leverage)
    assert "_champion" not in src, "leverage must not carry its own bracket"
    assert "seed_table" in src and "champion(" in src


# --- the starting-lineup rule, in the module that owns the shape it reads ----
#
# It was written twice, character for character, in `hub.season.lineup_gate` and
# `hub.season.weekly_gate`, kept in agreement by a docstring reading "the same rule as
# lineup_gate.projection_lineup_points". These tests lived against one of the two copies.


def _lineup_pos(n_qb=2, n_rb=4, n_wr=5, n_te=2):
    return ["QB"] * n_qb + ["RB"] * n_rb + ["WR"] * n_wr + ["TE"] * n_te


def test_required_slots_fill_before_the_flex():
    import numpy as np

    from hub.draft.season import FLEX_SLOTS, STARTERS, starting_lineup
    pos = _lineup_pos()
    got = [pos[i] for i in starting_lineup(pos, np.arange(len(pos), dtype=float))]
    for p, need in STARTERS.items():
        assert got.count(p) >= need, f"{p} short of its required {need}"
    assert len(got) == sum(STARTERS.values()) + FLEX_SLOTS


def test_the_flex_takes_the_best_leftover_not_the_first():
    from hub.draft.season import starting_lineup
    pos = ["QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE"]
    idx = starting_lineup(pos, [1, 9, 8, 7, 6, 5, 4, 3, 2])
    assert 3 in idx, "the third RB at 7 is the best flex-eligible leftover"


def test_a_quarterback_cannot_fill_the_flex():
    from hub.draft.season import starting_lineup
    pos = ["QB", "QB", "RB", "RB", "WR", "WR", "WR", "TE"]
    idx = starting_lineup(pos, [9, 8, 1, 1, 1, 1, 1, 1])
    assert [pos[i] for i in idx].count("QB") == 1, \
        "the second QB scores highest and still sits"


def test_the_rule_reads_the_shape_rather_than_restating_it():
    """`STARTERS`, `FLEX_FROM` and `FLEX_SLOTS` are three lines above it, derived from
    `RosterConfig`. A superflex league moves the config and the rule follows."""
    from dataclasses import replace

    from hub.config import RosterConfig, required_starters
    from hub.draft.season import STARTERS
    assert STARTERS == required_starters(RosterConfig())
    assert required_starters(replace(RosterConfig(), qb=2))["QB"] == 2


def test_both_gates_use_the_one_rule():
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "season"
    for rel in ("lineup_gate.py", "weekly_gate.py"):
        names = {a.name for n in ast.walk(ast.parse((root / rel).read_text()))
                 if isinstance(n, ast.ImportFrom) for a in n.names}
        assert "starting_lineup" in names, f"{rel} still carries its own copy"


# --- absence is absence, not a low-variance player (#183) ------------------
#
# The defect: talent was drawn per season, the mean clipped at zero, and weekly spread scaled
# by sqrt(realised/projected) -- so a player whose draw collapsed came out low-mean AND
# low-variance, which is a bust and is the opposite of an injury. Games played are now drawn
# from the player's own history through `durability.MISSED_YOY_R`, and the weeks he misses
# are zeros. Every assertion below is one of the ticket's two falsifiable directions or the
# guarantee that the `missed=None` path is untouched.

# A spread of prior-season histories, so `next_season_absence` has a population mean and
# spread to shrink toward -- the sample `games_missed` would hand it. The last two are the
# pair the ticket's second criterion is about: identical players, opposite histories.
ABSENCE_HIST = [float(i % 9) for i in range(22)] + [0.0, 16.0]
DURABLE, FRAGILE = -2, -1


def _one_team_per_player(hist):
    """A league of one-man teams, so a team's points ARE that player's points.

    `simulate_weeks` returns `lineup_points`, and over a shared roster that takes the best
    few of many identical players -- which hides the per-player effect these tests are about,
    and breaks mean-preservation at the team level even where it holds per player. One player
    per roster takes the lineup out of the question.
    """
    n = len(hist)
    return ([np.array([i]) for i in range(n)], np.full(n, 12.0), np.full(n, 3.0),
            np.array(["RB"] * n), np.asarray(hist, dtype=float))


def _season_totals(*, absence: bool, weeks=14, n_sims=4000, seed=5):
    """Season totals per player, shaped (sims, players)."""
    rosters, mu, sd, pos, hist = _one_team_per_player(ABSENCE_HIST)
    got = simulate_weeks(rosters, mu, sd, pos, n_sims=n_sims, weeks=weeks,
                         rng=np.random.default_rng(seed), talent_cv=0.0,
                         missed=(hist if absence else None))
    return got.sum(axis=1)


def test_a_player_who_misses_time_contributes_zeros_for_those_weeks():
    """The first acceptance criterion, and the one a shrunken mean cannot satisfy.

    A single player with a large mean and a tiny spread cannot score zero by drawing badly --
    the weekly distribution is nowhere near the clip -- so a zero week is absence or nothing.
    Without the draw there are none; with it there are.
    """
    hist = np.array([14.0])
    kw = {"n_sims": 400, "weeks": 14, "talent_cv": 0.0}
    args = ([np.array([0])], np.array([25.0]), np.array([0.5]), np.array(["RB"]))
    without = simulate_weeks(*args, rng=np.random.default_rng(3), missed=None, **kw)
    with_ = simulate_weeks(*args, rng=np.random.default_rng(3), missed=hist, **kw)
    assert not (without == 0.0).any(), (
        "the fixture can reach zero on its own, so a zero proves nothing about absence")
    assert (with_ == 0.0).any(), (
        "no week is a zero, so the player who misses the season is still being expressed as "
        "a shrunken mean rather than as games he did not play")


def test_injury_history_raises_season_variance():
    """The second acceptance criterion, and the pre-registered falsifiable direction: if the
    games-played draw left variance unchanged or lower, the implementation is wrong and
    nothing ships until that is explained."""
    with_, without = _season_totals(absence=True), _season_totals(absence=False)
    assert with_[:, FRAGILE].var() > without[:, FRAGILE].var() * 1.05, (
        "drawing absence did not raise season variance, which is the direction #183 "
        "pre-registered and the one that says whether absence is modelled at all")
    # Not only for the fragile player: absence is a population fact, so every player with any
    # expected absence is more variable than he was, which the old model could not express
    # without also making him worse.
    assert (with_.var(axis=0) > without.var(axis=0)).all()


def test_a_fragile_player_is_more_variable_than_an_identical_durable_one():
    """The same claim between two players rather than between two models, which is how the
    ticket words it: the two are identical in `mu`, `sd` and position, and `missed` is the
    only thing that differs."""
    got = _season_totals(absence=True)
    assert got[:, FRAGILE].var() > got[:, DURABLE].var() * 1.05


def test_absence_is_mean_preserving():
    """`mu` is points per *team* game and `durability.correct_projection` has already marked
    QB and WR down for expected absence, so a zero-one mask would price absence twice and
    re-level every downstream constant. The played weeks are scaled by the inverse of the
    expected played fraction, which is the definitional conversion to points per game
    *played* -- so a season total keeps its mean and only its spread moves.

    Across the pool rather than per player: the count of missed weeks is rounded to a whole
    week, so an individual is preserved only up to that rounding, and the claim being made is
    about the level of the board rather than about one man's projection.
    """
    with_, without = _season_totals(absence=True), _season_totals(absence=False)
    assert abs(with_.mean() - without.mean()) < 0.03 * without.mean()


def test_no_history_means_not_modelled_and_never_reaches_the_absence_path(monkeypatch):
    """Graceful degradation, and the reason no seeded result in this suite was re-baselined.

    `missed=None` is the honest answer for a caller with no durability history -- a board
    whose advisory durability stage was absorbed, or `leverage`'s synthetic pool. The
    assertion that matters is not "the two agree" (a default value agrees with itself); it is
    that the absence code is not *entered*, which is what leaves the generator where the
    pre-#183 simulator left it and every seeded expectation in this file intact.
    """
    rosters, mu, sd, pos, _ = _one_team_per_player(ABSENCE_HIST)
    kw = {"n_sims": 50, "weeks": 14, "talent_cv": 0.3}

    def refuse(*a, **k):
        raise AssertionError("the absence path ran for a caller that has no history, so it "
                             "has taken draws the old simulator did not and every seeded "
                             "result in this suite now means something else")

    monkeypatch.setattr(season, "_absence_factor", refuse)
    a = simulate_weeks(rosters, mu, sd, pos, rng=np.random.default_rng(11), **kw)
    b = simulate_weeks(rosters, mu, sd, pos, rng=np.random.default_rng(11), missed=None, **kw)
    assert np.array_equal(a, b)


def test_a_player_with_no_prior_role_draws_the_marginal_not_the_average():
    """A null `missed` is "nothing is known", which is not the same as "he is average". The
    conditional spread of a known player is narrowed by sqrt(1 - r^2); an unknown one keeps
    the full marginal, and that difference is the variance a rookie should carry."""
    from hub.draft.durability import next_season_absence
    hist = np.array([0.0, 1.0, 2.0, 6.0, 9.0, 12.0])
    _, sd_k = next_season_absence(hist)
    mu_u, sd_u = next_season_absence(np.append(hist, np.nan))
    assert sd_u[-1] > sd_k[0]
    assert abs(mu_u[-1] - hist.mean()) < 1e-9


def test_the_persistence_is_the_published_one_and_shrinks_toward_the_mean():
    """The whole fitted surface of #183 is one number that was already measured. If the
    shrinkage stopped being `mbar + r * (prior - mbar)`, absence would have acquired a
    parameter nothing in this repo derives -- ADR-0006's line, and the reason the disposition
    took the games-played draw over a mixture component."""
    from hub.draft.durability import MISSED_YOY_R, next_season_absence
    assert MISSED_YOY_R == 0.407
    hist = np.array([0.0, 4.0, 8.0, 12.0])
    mu, _ = next_season_absence(hist)
    mbar = hist.mean()
    assert np.allclose(mu, mbar + MISSED_YOY_R * (hist - mbar))


def test_absence_degrades_rather_than_raising_when_nothing_is_known():
    """A simulator that refuses to run because an advisory column is all-null is the
    operator-dependence CLAUDE.md warns about."""
    from hub.draft.durability import next_season_absence
    mu, sd = next_season_absence(np.array([np.nan, np.nan]))
    assert not mu.any() and not sd.any()
    got = simulate_weeks([np.array([0])], np.array([12.0]), np.array([3.0]),
                         np.array(["RB"]), n_sims=20, weeks=3,
                         rng=np.random.default_rng(0), missed=np.array([np.nan]))
    assert np.isfinite(got).all()


# --- byes (issue #226) ------------------------------------------------------

def test_a_player_scores_nothing_in_his_teams_bye_week_and_normally_elsewhere():
    """The second criterion. A bye is a scheduled zero, deterministic across sims -- not a
    draw -- so every sim has the zero in the same week and no other week is touched. The
    fixture cannot reach zero on its own (mean 25, spread 0.5), so a zero is the bye or
    nothing."""
    kw = {"n_sims": 200, "weeks": 14, "talent_cv": 0.0}
    args = ([np.array([0])], np.array([25.0]), np.array([0.5]), np.array(["RB"]))
    without = simulate_weeks(*args, rng=np.random.default_rng(3), **kw)
    with_ = simulate_weeks(*args, rng=np.random.default_rng(3), bye_week=np.array([7]), **kw)
    assert not (without == 0.0).any(), "the fixture reaches zero on its own"
    assert (with_[:, 6, 0] == 0.0).all(), "week 7 (index 6) was not zeroed in every sim"
    other = [w for w in range(14) if w != 6]
    assert not (with_[:, other, 0] == 0.0).any(), "a week other than the bye was zeroed"


def test_a_bye_is_not_mean_preserving_because_it_is_a_real_missed_game():
    """Unlike #183's injury absence, which rescales played weeks because `mu` is already per
    team game, a bye is a week the team does not play at all. Fourteen fantasy weeks with one
    bye is thirteen games, and the season total is thirteen times the mean, not fourteen."""
    kw = {"n_sims": 2000, "weeks": 14, "talent_cv": 0.0}
    args = ([np.array([0])], np.array([20.0]), np.array([1.0]), np.array(["RB"]))
    without = simulate_weeks(*args, rng=np.random.default_rng(4), **kw).sum(axis=1).mean()
    with_ = simulate_weeks(*args, rng=np.random.default_rng(4), bye_week=np.array([9]),
                           **kw).sum(axis=1).mean()
    assert with_ == pytest.approx(without * 13 / 14, rel=0.02)


def test_a_player_with_no_bye_is_untouched():
    """Free agents and unmatched teams carry 0, which is no week, so nothing is zeroed."""
    kw = {"n_sims": 100, "weeks": 14, "talent_cv": 0.0}
    # One player per roster, so the per-team output is the per-player one.
    args = ([np.array([0]), np.array([1])], np.array([25.0, 25.0]), np.array([0.5, 0.5]),
            np.array(["RB", "WR"]))
    got = simulate_weeks(*args, rng=np.random.default_rng(5), bye_week=np.array([0, 7]), **kw)
    assert not (got[:, :, 0] == 0.0).any(), "a player with no bye was zeroed somewhere"
    assert (got[:, 6, 1] == 0.0).all()
    assert not (got[:, [w for w in range(14) if w != 6], 1] == 0.0).any()


def test_a_bye_outside_the_fantasy_season_is_refused():
    """#226's fourth criterion: `REG_SEASON_WEEKS` and the bye placement agree, and a bye
    the fantasy season cannot see is a schedule shape nobody has modelled -- refused rather
    than silently a no-op, because a no-op here would read as 'no bye'."""
    kw = {"n_sims": 10, "weeks": 14, "talent_cv": 0.0}
    args = ([np.array([0])], np.array([25.0]), np.array([0.5]), np.array(["RB"]))
    with pytest.raises(ValueError, match="bye"):
        simulate_weeks(*args, rng=np.random.default_rng(6), bye_week=np.array([15]), **kw)


def test_bye_weeks_are_derived_from_the_schedule_one_per_team(monkeypatch):
    """The first criterion: a team-week schedule source through the validated loader, and a
    bye is the regular-season week a team has no game. One per team, or the shape changed."""
    import polars as pl

    from hub.draft import season as S

    # Five weeks, four teams. AAA-BBB sit in week 2, CCC-DDD in week 4; every other week
    # both pairs play. A postseason row is present and must be ignored.
    rows = []
    for w in range(1, 6):
        if w != 2:
            rows.append({"game_id": f"{w}_ab", "season": 2026, "week": w, "home_team": "AAA",
                         "away_team": "BBB", "game_type": "REG"})
        if w != 4:
            rows.append({"game_id": f"{w}_cd", "season": 2026, "week": w, "home_team": "CCC",
                         "away_team": "DDD", "game_type": "REG"})
    rows.append({"game_id": "wc", "season": 2026, "week": 6, "home_team": "AAA",
                 "away_team": "CCC", "game_type": "WC"})
    sched = pl.DataFrame(rows)
    monkeypatch.setattr(S, "_schedule_for", lambda season: sched)
    got = S.bye_weeks(2026)
    assert got == {"AAA": 2, "BBB": 2, "CCC": 4, "DDD": 4}, got


def test_bye_weeks_refuse_a_team_with_two_byes_or_none(monkeypatch):
    """A schedule where a team sits twice is not this league's shape, and the derivation
    must not quietly pick one of the two."""
    import polars as pl

    from hub.draft import season as S

    # Four regular-season weeks. AAA and BBB play each other in weeks 1 and 3 only, so both
    # sit in weeks 2 and 4 -- two byes each.
    sched = pl.DataFrame([{"game_id": "1", "season": 2026, "week": 1, "home_team": "AAA",
                           "away_team": "BBB", "game_type": "REG"},
                          {"game_id": "3", "season": 2026, "week": 3, "home_team": "BBB",
                           "away_team": "AAA", "game_type": "REG"},
                          {"game_id": "2c", "season": 2026, "week": 2, "home_team": "CCC",
                           "away_team": "DDD", "game_type": "REG"},
                          {"game_id": "4c", "season": 2026, "week": 4, "home_team": "DDD",
                           "away_team": "CCC", "game_type": "REG"}])
    monkeypatch.setattr(S, "_schedule_for", lambda season: sched)
    with pytest.raises(ValueError, match="AAA"):
        S.bye_weeks(2026)


def test_attach_bye_joins_by_canonical_team_and_leaves_the_unplaced_null():
    """The board column. A player whose team cannot be placed keeps a null, not a zero: 0 is
    what the simulator reads as 'no bye', and a placement failure is not that."""
    import polars as pl

    from hub.draft.season import attach_bye

    board = pl.DataFrame({"player": ["a", "b", "c", "d"],
                          "team": ["KC", "JAC", "FA", None]})
    got = attach_bye(board, {"KC": 10, "JAX": 12})
    assert got["bye_week"].to_list() == [10, 12, None, None]
    assert got["bye_week"].dtype == pl.Int64
    assert got.columns == ["player", "team", "bye_week"], "no helper column left behind"


def test_a_bye_array_of_the_wrong_length_is_refused():
    """One week per player or nothing: a shorter array would broadcast onto the wrong men."""
    kw = {"n_sims": 10, "weeks": 14, "talent_cv": 0.0}
    args = ([np.array([0]), np.array([1])], np.array([25.0, 25.0]), np.array([0.5, 0.5]),
            np.array(["RB", "WR"]))
    with pytest.raises(ValueError, match="shape"):
        simulate_weeks(*args, rng=np.random.default_rng(6), bye_week=np.array([7]), **kw)


def test_the_schedule_comes_through_the_validated_loader(monkeypatch):
    """The seam the derivation reads through, and the columns it asks for -- not a second
    reader of the schedule beside `playoff_sos`."""
    from hub.draft import season as S
    from hub.fetch import nflverse

    seen = {}

    def fake(name, **kw):
        seen.update(name=name, **kw)
        return "frame"
    monkeypatch.setattr(nflverse, "load", fake)
    assert S._schedule_for(2026) == "frame"
    assert seen["name"] == "schedules" and seen["seasons"] == [2026]
    assert {"week", "home_team", "away_team", "game_type"} <= set(seen["cols"])


def test_an_imputed_player_carries_a_wider_talent_spread_than_an_observed_one():
    """#87. Two players at the same rank, one with an observed prior season and one whose
    projection was interpolated from rank: the guess is less certain by the imputation's
    own measured error, added in quadrature; the observed one is unchanged."""
    from hub.draft.season import talent_cv_for
    from hub.models.predict import IMPUTE_CV_BY_POS, TALENT_CV_BY_POS
    pos = np.array(["RB", "RB", "WR", "K"])
    plain = talent_cv_for(pos)
    got = talent_cv_for(pos, imputed=np.array([False, True, True, True]))
    assert got[0] == plain[0] == TALENT_CV_BY_POS["RB"]
    assert got[1] > plain[1]
    assert got[1] == pytest.approx(np.hypot(TALENT_CV_BY_POS["RB"], IMPUTE_CV_BY_POS["RB"]))
    assert got[2] == pytest.approx(np.hypot(TALENT_CV_BY_POS["WR"], IMPUTE_CV_BY_POS["WR"]))
    assert got[3] > plain[3], "an unfitted position still widens, by the pooled error"
    assert talent_cv_for(pos, imputed=None).tolist() == plain.tolist()
