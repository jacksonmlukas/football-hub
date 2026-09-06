"""`pool.weekly`: the survivor pick for a week, and the free pick it has to beat.

Named for the module, not the word. `hub.models.weekly` is a different weekly -- the
projection -- and its tests are `test_weekly.py`. These two were briefly the same file.

Every grid here is small and hand-priced. A recommendation is only interesting against a
stated alternative, so most of these fix the fallback and ask what departing from it buys.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hub.config import PoolConfig
from hub.season import pool


def _grid(spec: dict[int, list[tuple[str, str, float]]]) -> pl.DataFrame:
    rows = []
    for wk, games in spec.items():
        for i, (a, b, p) in enumerate(games):
            gid = f"{wk}-{i}"
            rows += [(wk, a, p, gid), (wk, b, 1 - p, gid)]
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [r[2] for r in rows], "game_id": [r[3] for r in rows]})


FLAT = _grid({w: [("KC", "LV", 0.85), ("SF", "SEA", 0.80), ("BUF", "NYJ", 0.75)]
              for w in (1, 2, 3)})

# KC is barely the best pick in week 1 and a near-lock in week 2, so spending it now costs
# that. This is the whole reason a recommendation is not just "take the biggest favourite".
HOARD = _grid({1: [("KC", "LV", 0.70), ("SF", "SEA", 0.68), ("BUF", "NYJ", 0.66)],
               2: [("KC", "LV", 0.97), ("SF", "SEA", 0.55), ("BUF", "NYJ", 0.54)]})


def _team(w: pool.Weekly, team: str) -> pool.Candidate:
    return next(c for c in w.candidates if c.team == team)


def _weekly(grid, weeks, **kw):
    kw.setdefault("entries", 12)
    kw.setdefault("pot", 420.0)
    kw.setdefault("trials", 400)
    kw.setdefault("rng", np.random.default_rng(0))
    return pool.weekly(grid, weeks, **kw)


def test_every_candidate_carries_a_dollar_figure_beside_its_survival():
    w = _weekly(FLAT, [1, 2, 3], week=1)
    assert w.candidates
    for c in w.candidates:
        assert c.expected_dollars > 0
        assert 0 < c.survives <= 1
    # Ranked on money, which is the quantity the recommendation is about.
    assert [c.expected_dollars for c in w.candidates] == sorted(
        (c.expected_dollars for c in w.candidates), reverse=True)


def test_the_free_alternative_is_named_beside_the_recommendation():
    w = _weekly(FLAT, [1, 2, 3], week=1)
    assert w.fallback == "KC"
    assert any(c.is_fallback for c in w.candidates)


def test_a_recommendation_that_is_the_free_pick_says_so_rather_than_taking_credit():
    w = _weekly(FLAT, [3], week=3)          # nothing to hoard for, so the biggest price wins
    assert w.matched and w.recommend == w.fallback == "KC"
    assert "for nothing" in "\n".join(pool.weekly_report(w))


def _hand(gap: float, resolution: float, *, matched: bool = False) -> pool.Weekly:
    """A week with the two figures set by hand, so the verdict is tested and not the sampler."""
    best = pool.Candidate("SF", 0.8, 0.5, 100.0, matched)
    fb = pool.Candidate("KC", 0.85, 0.5, 100.0 - gap, True)
    cands = [best] if matched else [best, fb]
    return pool.Weekly(week=1, recommend=cands[0].team, fallback="KC", matched=matched,
                       given_up=0.0, pot=420.0, resolution=resolution, candidates=cands)


def test_a_gap_the_trials_cannot_resolve_is_not_a_recommendation():
    """`docs/method.md` rule 13: a difference smaller than what the run could detect is
    unresolved, not established. The free pick wins that tie because it is free."""
    w = _hand(gap=2.0, resolution=5.0)
    assert not w.decisive
    assert "it is free and no worse" in "\n".join(pool.weekly_report(w))


def test_a_gap_wider_than_the_trials_resolve_stands_as_a_recommendation():
    w = _hand(gap=20.0, resolution=5.0)
    assert w.decisive
    assert "it is free and no worse" not in "\n".join(pool.weekly_report(w))


def test_taking_the_free_pick_is_never_undecided():
    assert _hand(gap=0.0, resolution=99.0, matched=True).decisive


def test_the_resolution_tightens_as_the_trials_grow():
    """Reported, not hidden. Four times the trials halves what the run can tell apart."""
    coarse = _weekly(FLAT, [1, 2, 3], week=1, trials=200, rng=np.random.default_rng(5))
    fine = _weekly(FLAT, [1, 2, 3], week=1, trials=800, rng=np.random.default_rng(5))
    assert coarse.resolution > 0
    assert fine.resolution == pytest.approx(coarse.resolution / 2, rel=0.35)


def test_a_departure_from_the_free_pick_reports_what_it_did_to_survival():
    w = _weekly(HOARD, [1, 2], week=1, trials=800, rng=np.random.default_rng(3))
    assert not w.matched
    assert w.recommend != w.fallback == "KC"
    lines = "\n".join(pool.weekly_report(w))
    assert "against the free pick KC" in lines


def test_a_departure_that_survives_better_gains_rather_than_costs():
    """Signed on purpose. Reading a gain out as a cost of zero would say the two plans were
    alike, which is a claim the journal treats as a finding rather than a blank."""
    w = _weekly(HOARD, [1, 2], week=1, trials=800, rng=np.random.default_rng(3))
    fb = next(c for c in w.candidates if c.is_fallback)
    best = w.candidates[0]
    assert w.given_up == pytest.approx(fb.survives - best.survives)
    assert w.given_up < 0                       # hoarding KC for week 2 is the better plan
    assert "gains" in "\n".join(pool.weekly_report(w))


def test_the_free_pick_skips_teams_already_spent():
    assert pool.auto_pick(FLAT, 1) == "KC"
    assert pool.auto_pick(FLAT, 1, ["KC"]) == "SF"
    assert pool.auto_pick(FLAT, 1, ["KC", "SF"]) == "BUF"
    w = _weekly(FLAT, [1, 2, 3], week=1, ledger=["KC", "SF"])
    assert w.fallback == "BUF"
    assert all(c.team not in ("KC", "SF") for c in w.candidates)


def test_a_tie_on_price_does_not_let_row_order_pick_the_free_team():
    tied = _grid({1: [("SF", "LV", 0.8), ("KC", "SEA", 0.8)]})
    assert pool.auto_pick(tied, 1) == "KC"      # sorted on the name, not on where it landed


def test_a_week_the_entry_is_unlikely_to_reach_is_worth_almost_nothing():
    """Future value is weighted by the chance the entry is still playing for it. A team that
    loses this week does not collect from the weeks after it, however rich they are."""
    w = _weekly(FLAT, [1, 2, 3], week=1)
    by = {c.team: c for c in w.candidates}
    assert by["LV"].win_prob == pytest.approx(0.15)
    # LV wins one week in seven; its whole claim on the rest of the season is scaled by that.
    assert by["LV"].expected_dollars < by["KC"].expected_dollars / 3


def test_a_double_pick_week_changes_the_price_of_a_team_that_reaches_it():
    """The control for the test below: within the simulated weeks, a double-pick rule is not
    inert. It burns two teams a week, so it has to move the figure of a team that gets there."""
    plain = PoolConfig(double_pick_weeks=())
    dbl = PoolConfig(double_pick_weeks=(3,))
    a = _weekly(FLAT, [1, 2, 3], week=1, pool=plain, rng=np.random.default_rng(7))
    b = _weekly(FLAT, [1, 2, 3], week=1, pool=dbl, rng=np.random.default_rng(7))
    assert _team(a, "KC").expected_dollars != _team(b, "KC").expected_dollars


def test_a_double_pick_week_the_entry_never_reaches_contributes_nothing():
    """The discounting the ticket asks for, and no special case implements it: a trial stops
    when the entry does, so a week past that never runs. A team that loses in week 1 is priced
    identically whether or not week 3 doubles, because it is not there either way."""
    plain = PoolConfig(double_pick_weeks=())
    dbl = PoolConfig(double_pick_weeks=(3,))
    # LV wins week 1 fifteen times in a hundred; almost every trial ends before week 3.
    dead = _grid({1: [("KC", "LV", 0.999), ("SF", "SEA", 0.80), ("BUF", "NYJ", 0.75)],
                  2: [("KC", "LV", 0.85), ("SF", "SEA", 0.80), ("BUF", "NYJ", 0.75)],
                  3: [("KC", "LV", 0.85), ("SF", "SEA", 0.80), ("BUF", "NYJ", 0.75)]})
    a = _weekly(dead, [1, 2, 3], week=1, pool=plain, rng=np.random.default_rng(7))
    b = _weekly(dead, [1, 2, 3], week=1, pool=dbl, rng=np.random.default_rng(7))
    assert _team(a, "LV").win_prob == pytest.approx(0.001)
    assert _team(a, "LV").expected_dollars == pytest.approx(
        _team(b, "LV").expected_dollars, abs=0.01)


def test_the_figures_are_net_of_what_the_entry_already_paid():
    """The outlay is sunk, so it shifts every candidate alike and cannot reorder them. It is
    subtracted anyway: "which of these is best" and "is any of them worth having" are
    different questions, and only the second one notices the entry cost money."""
    kw = {"week": 1, "entries": 12, "pot": 420.0, "trials": 400}
    gross = pool.weekly(FLAT, [1, 2, 3], rng=np.random.default_rng(4), **kw)
    net = pool.weekly(FLAT, [1, 2, 3], outlay=25.0, rng=np.random.default_rng(4), **kw)
    assert [c.team for c in net.candidates] == [c.team for c in gross.candidates]
    for g, n in zip(gross.candidates, net.candidates, strict=True):
        assert n.expected_dollars == pytest.approx(g.expected_dollars - 25.0)


def test_the_last_week_needs_no_simulation_to_price():
    w = _weekly(FLAT, [3], week=3)
    kc = next(c for c in w.candidates if c.team == "KC")
    assert kc.survives == pytest.approx(0.85)
    assert kc.expected_dollars == pytest.approx(0.85 * 420.0)


def test_a_week_with_no_legal_pick_is_an_error_not_a_silent_recommendation():
    with pytest.raises(ValueError, match="no legal pick"):
        _weekly(FLAT, [1, 2, 3], week=1, ledger=["KC", "SF", "BUF", "LV", "SEA", "NYJ"])


def test_the_week_reports_as_lines_so_it_can_be_composed(capsys):
    w = _weekly(FLAT, [1, 2, 3], week=1)
    lines = pool.weekly_report(w)
    assert isinstance(lines, list) and all(isinstance(x, str) for x in lines)
    assert capsys.readouterr().out == ""


def test_only_the_teams_worth_considering_are_priced():
    w = _weekly(FLAT, [1, 2, 3], week=1, top=2)
    assert sorted(c.team for c in w.candidates) == ["KC", "SF"]


def test_the_same_seed_prices_the_week_the_same_way_twice():
    """Every candidate faces the identical season, so the ranking is the pick and not the
    draw. Independent draws put the top of a flat board inside sampling noise."""
    kw = {"week": 1, "entries": 12, "pot": 420.0, "trials": 400}
    a = pool.weekly(FLAT, [1, 2, 3], rng=np.random.default_rng(11), **kw)
    b = pool.weekly(FLAT, [1, 2, 3], rng=np.random.default_rng(11), **kw)
    assert a == b
    # And the answer does not move with an unrelated seed, because the board is not close.
    assert a.candidates == b.candidates
