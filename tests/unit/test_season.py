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
