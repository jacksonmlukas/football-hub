"""Season simulation to a champion.

The properties that matter are not "the numbers look plausible" but that the thing
responds correctly to roster construction -- because that is the entire reason to
simulate instead of ranking by VOR.
"""
import numpy as np
import pytest
from hub.draft.season import (_lineup_points, _round_robin, champion_probability,
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
