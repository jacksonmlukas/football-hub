"""Does the lineup optimiser beat sorting a column?

`hub.season.lineup` is 97% covered by unit tests, which is a claim about its internals and not
about whether it is right. The draft board was also well covered while recommending a fourth
quarterback. So it gets a gate: a model is tested against the simplest thing that already
works, which here is "start your highest projections".

Everything below runs offline. The statistics have to be exercisable without nflverse, or the
gate is one nobody re-runs -- which is how P0 ended up unreproducible.
"""
import polars as pl
import pytest

from hub.season import lineup_gate as lg


def _realised(rows):
    from hub.draft.state import _norm
    return pl.DataFrame({"player": [_norm(r[0]) for r in rows],
                         "week": [r[1] for r in rows],
                         "points": [r[2] for r in rows]},
                        schema={"player": pl.Utf8, "week": pl.Int64, "points": pl.Float64})


# --- the pre-registered rule, as executable code --------------------------

def test_an_interval_above_zero_trusts_the_optimiser():
    assert lg.verdict({"lo": 0.5, "hi": 3.0}).startswith("TRUST")


def test_an_interval_containing_zero_says_start_your_projections():
    """The likely branch, and it has an action rather than being a disappointment."""
    assert lg.verdict({"lo": -1.0, "hi": 2.0}).startswith("START YOUR PROJECTIONS")


def test_an_interval_below_zero_removes_it():
    """Evidence demotes as well as promotes -- the asymmetry P0's rule originally lacked."""
    assert lg.verdict({"lo": -3.0, "hi": -0.5}).startswith("REMOVE")


def test_touching_zero_has_not_excluded_it():
    assert lg.verdict({"lo": 0.0, "hi": 3.0}).startswith("START YOUR PROJECTIONS")


# --- the two arms ---------------------------------------------------------
#
# Both see ONLY projections. An earlier version of this file scored the optimiser arm on
# realised weekly scores, which gave it information the baseline did not have: it measured
# the value of perfect foresight (+31 points a game) and could not fail. The optimiser's sole
# advantage over sorting is that it reads `sd` as well as `mu`.

def _roster(sd=None):
    """QB1 RB2 WR3 TE1 + flex candidates. (player, pos, mu, sd)."""
    base = [("QB1", "QB", 20.0), ("RB1", "RB", 18.0), ("RB2", "RB", 14.0),
            ("WR1", "WR", 16.0), ("WR2", "WR", 13.0), ("WR3", "WR", 11.0),
            ("TE1", "TE", 9.0), ("WR4", "WR", 8.0), ("RB3", "RB", 7.0)]
    sd = sd or {}
    return [(n, p, m, sd.get(n, 2.0)) for n, p, m in base]


def _flat(r, pts=10.0, extra=()):
    rows = [(n, w, pts) for n, _, _, _ in r for w in range(1, 15)]
    return _realised(rows + list(extra))


def test_the_baseline_starts_the_highest_projections():
    """Eight slots at ten points each. RB3 is projected lowest, so he sits."""
    r = _roster()
    grid = lg.weekly_grid([n for n, _, _, _ in r], _flat(r))
    got = lg.projection_lineup_points(grid, [p for _, p, _, _ in r], [m for _, _, m, _ in r])
    assert got == pytest.approx(80.0)


def test_the_baseline_lineup_does_not_change_week_to_week():
    """A projection does not change, so neither does the lineup it implies. Giving the
    baseline a weekly choice would be scoring it as an optimiser."""
    r = _roster()
    grid = lg.weekly_grid([n for n, _, _, _ in r], _flat(r, extra=[("RB3", 5, 500.0)]))
    got = lg.projection_lineup_points(grid, [p for _, p, _, _ in r], [m for _, _, m, _ in r])
    assert got == pytest.approx(80.0), "a benched player's spike must not be collected"


def test_the_optimiser_cannot_see_realised_scores():
    """The defect this file was rewritten to remove. Moving a huge week onto a player the
    optimiser did not start must not change what it starts -- it never saw it."""
    r = _roster()
    names, pos = [n for n, _, _, _ in r], [p for _, p, _, _ in r]
    mu, sd = [m for _, _, m, _ in r], [v for _, _, _, v in r]
    quiet = lg.weekly_grid(names, _flat(r))
    spike = lg.weekly_grid(names, _flat(r, extra=[("RB3", 5, 500.0)]))
    a = lg.optimiser_lineup_points(quiet, names, pos, mu, sd)
    b = lg.optimiser_lineup_points(spike, names, pos, mu, sd)
    # RB3 is the lowest projection and stays benched in both, so the spike is never collected.
    assert a == pytest.approx(b)


def test_the_optimiser_can_prefer_upside_over_projection():
    """The hypothesis under test, isolated. Against a strong opponent, a lower-mu player with
    much larger sd can carry a higher chance of winning -- and sorting on mu cannot see it."""
    from hub.season.lineup import optimize
    players = pl.DataFrame({
        "player": ["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "STEADY", "SWINGY"],
        "pos": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "WR"],
        "mu": [20.0, 18.0, 14.0, 16.0, 13.0, 11.0, 9.0, 9.0, 8.0],
        "sd": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 30.0],
    })
    # a hopeless matchup: you need variance, not expectation
    started = optimize(players, opp_mu=250.0, opp_sd=25.0)["starters"]["player"].to_list()
    assert "SWINGY" in started and "STEADY" not in started


def test_the_two_arms_agree_when_variance_is_uniform():
    """With identical sd there is nothing for variance-awareness to exploit, so the optimiser
    must not manufacture a difference out of tie-breaking."""
    r = _roster()
    names, pos = [n for n, _, _, _ in r], [p for _, p, _, _ in r]
    mu, sd = [m for _, _, m, _ in r], [v for _, _, _, v in r]
    grid = lg.weekly_grid(names, _flat(r))
    assert lg.optimiser_lineup_points(grid, names, pos, mu, sd) == pytest.approx(
        lg.projection_lineup_points(grid, pos, mu))


def test_a_roster_that_cannot_field_a_lineup_scores_zero():
    """Nine quarterbacks fills no RB slot. The gate must report, not raise."""
    r = [(f"QB{i}", "QB", 10.0, 2.0) for i in range(9)]
    names, pos = [n for n, _, _, _ in r], [p for _, p, _, _ in r]
    grid = lg.weekly_grid(names, _flat(r))
    assert lg.optimiser_lineup_points(grid, names, pos,
                                      [m for _, _, m, _ in r], [v for _, _, _, v in r]) == 0.0


def test_a_player_who_never_played_scores_zero_not_null():
    r = _roster()
    rows = [(n, w, 10.0) for n, _, _, _ in r[:-1] for w in range(1, 15)]
    grid = lg.weekly_grid([n for n, _, _, _ in r], _realised(rows))
    assert grid[-1].sum() == 0.0


def test_points_are_per_game():
    r = _roster()
    names, pos = [n for n, _, _, _ in r], [p for _, p, _, _ in r]
    mu, sd = [m for _, _, m, _ in r], [v for _, _, _, v in r]
    g7 = lg.weekly_grid(names, _flat(r), weeks=7)
    g14 = lg.weekly_grid(names, _flat(r), weeks=14)
    assert lg.optimiser_lineup_points(g7, names, pos, mu, sd) == pytest.approx(
        lg.optimiser_lineup_points(g14, names, pos, mu, sd))


# --- the paired comparison ------------------------------------------------

def test_compare_pairs_one_row_per_roster():
    r = _roster()
    real = _flat(r)
    got = lg.compare({2024: [r, r], 2025: [r]}, {2024: real, 2025: real})
    assert got.height == 3
    assert set(got["season"].to_list()) == {2024, 2025}
    assert (got["diff"].abs() < 1e-9).all()


def test_an_empty_roster_does_not_crash_the_gate():
    """A season where the draft produced nothing must report, not raise."""
    got = lg.compare({2024: [[]]}, {2024: _realised([])})
    assert got.height == 1
    assert got["projection"][0] == 0.0


# --- the gate reads its inputs the way the live tool does -----------------

def test_the_gate_uses_the_same_moments_object_the_simulator_does():
    """mu and sd come from `predict.moments`, not a private copy -- so the optimiser is fed
    exactly what it is fed live, and the gate cannot pass by being handed nicer numbers."""
    import inspect

    from hub.season import lineup_gate
    src = inspect.getsource(lineup_gate.main)
    assert "from hub.models.predict import moments" in src


def test_a_roster_of_one_position_still_reports():
    """Nine quarterbacks fills no RB slot, so the optimiser has no legal lineup. The paired
    frame must still have a row -- a gate that drops its failures overstates its arm."""
    r = [(f"QB{i}", "QB", 10.0, 2.0) for i in range(9)]
    got = lg.compare({2024: [r]}, {2024: _flat(r)})
    assert got.height == 1
    assert got["optimiser"][0] == 0.0
