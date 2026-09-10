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

# The same shape with the hoard worth almost nothing: KC is a hair better this week and a
# hair better next week, so departing from it buys an edge inside what a few hundred trials
# can see. This is the board the *bar* is testable on -- on `HOARD` the gap is twenty-odd
# standard errors and every threshold agrees about it.
NEAR = _grid({1: [("KC", "LV", 0.70), ("SF", "SEA", 0.695), ("BUF", "NYJ", 0.66)],
              2: [("KC", "LV", 0.74), ("SF", "SEA", 0.72), ("BUF", "NYJ", 0.70)]})

# Week 2's only fixture is a game whose two teams are already spent, so an entry reaching it
# cannot field a legal pick and dies there whatever it took in week 1. Every candidate is
# then worth exactly nothing and no trial ever cashes -- the degenerate regime #159 asks to
# be reported rather than folded into a tie.
NEVER = _grid({1: [("SF", "SEA", 0.80), ("BUF", "NYJ", 0.75), ("DAL", "NYG", 0.70)],
               2: [("KC", "LV", 0.85)]})


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
    """`docs/method.md` rule 12: a gate that cannot run is an absence of evidence, so a
    difference smaller than the run could detect is unresolved rather than established. The free pick wins that tie because it is free."""
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
    """Reported, not hidden. Four times the trials halves what the run can tell apart.

    Asked on a board where the recommendation departs from the free pick, because since #159
    the resolution is the interval on *that* comparison and nothing else. Where the two are
    the same pick there is no difference to resolve: the arms are one arm, they differ by
    exactly zero in every trial, and the interval on that is zero rather than small.
    """
    coarse = _weekly(HOARD, [1, 2], week=1, trials=200, rng=np.random.default_rng(5))
    fine = _weekly(HOARD, [1, 2], week=1, trials=800, rng=np.random.default_rng(5))
    assert not coarse.matched and not fine.matched
    assert coarse.resolution > 0
    assert fine.resolution == pytest.approx(coarse.resolution / 2, rel=0.35)
    same = _weekly(FLAT, [1, 2, 3], week=1, trials=200, rng=np.random.default_rng(5))
    assert same.matched and same.resolution == 0.0


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


def test_an_absent_auto_pick_is_carried_through_rather_than_stringified(monkeypatch):
    """`fallback` was the string "none available" when there was nothing to fall back to.

    That is a value every reader had to know to compare against, and one of them is
    `hub.season.journal`, which would have taken it for a team and recorded a survival cost
    against a free pick that does not exist. `auto_pick` already answers `None`, and the week
    carries that answer through.

    Reached through `auto_pick` rather than by handing `weekly` a grid: the two apply the same
    filter, so today the empty-grid branch always raises first and the absence is a shape the
    type allows rather than a state this grid can reach. Pinning it at the seam is what says
    the answer is passed along and not converted on the way."""
    assert pool.auto_pick(FLAT, 1, ["KC", "SF", "BUF", "LV", "SEA", "NYJ"]) is None
    monkeypatch.setattr(pool, "auto_pick", lambda *a, **k: None)
    w = _weekly(FLAT, [1, 2, 3], week=1, top=2)
    assert w.fallback is None
    assert not w.matched
    assert not any(c.is_fallback for c in w.candidates)


def test_a_week_with_no_free_pick_says_so_rather_than_naming_one():
    """The other half: a report that has no fallback to name must not name one anyway. Under
    the sentinel it read "over auto-pick's none available", and under a bare `None` it would
    read "over auto-pick's None" -- a team by either spelling."""
    w = pool.Weekly(week=1, recommend="SF", fallback=None, matched=False, given_up=0.0,
                    pot=420.0, resolution=1.0,
                    candidates=[pool.Candidate("SF", 0.8, 0.5, 100.0, False)])
    lines = "\n".join(pool.weekly_report(w))
    assert "none available" not in lines and "None" not in lines
    assert "no team left to assign" in lines


# --- the trials are paired, and decisive is a two-sigma bar (#159) --------------------------
#
# Reseeding every candidate to one value is a shared season only while both consume the stream
# at the same rate, and they did not: the trial stopped when the last live entry died, ours
# included, so the first trial our entry outlasted the field in one candidate and not in the
# other offset everything after it. The pairing held up to the first week our fate differed --
# which is exactly the trials the comparison carries its signal in. `_play` now draws the whole
# season before anybody picks, so the count is fixed whatever becomes of us.
#
# Which comparison the interval is for: the recommended plan against the auto-pick plan, both
# replayed by our own entry against one shared season. Not our arm against the field's sampling
# rule, which cannot be paired at all -- see `test_pool.py`'s #151 section.


def _gap(w: pool.Weekly) -> float:
    """What the week's verdict is about: the recommendation less the free pick, in dollars."""
    fb = next(c for c in w.candidates if c.is_fallback)
    return abs(w.candidates[0].expected_dollars - fb.expected_dollars)


def test_the_bar_a_difference_is_called_real_at_is_two_standard_errors():
    """One standard error is a two-sided false positive rate of about one in three.

    The bar is a stated constant and the resolution is that constant times the paired standard
    error, so the two are wired together rather than a threshold that happens to sit near a
    spread. Asked on `NEAR`, because on `HOARD` the gap is twenty standard errors and every
    threshold agrees about it -- a bar can only be tested where it decides something.
    """
    assert pool.DECISIVE_SIGMA == 2.0
    w = _weekly(NEAR, [1, 2], week=1, trials=400, rng=np.random.default_rng(3))
    assert not w.matched and w.cashed > 0
    assert w.resolution / 2.0 < _gap(w) < w.resolution
    assert not w.decisive


def test_the_one_sigma_bar_this_replaces_would_have_called_that_week_a_departure(monkeypatch):
    """The same week at the old threshold, so the change is shown rather than asserted.

    Read from the module at call time, which is what makes this test possible and what makes
    the bar a number a reader can find. The ranking does not move: what moves is whether the
    week claims the gap is real, and the alternative when it does not is free.
    """
    strict = _weekly(NEAR, [1, 2], week=1, trials=400, rng=np.random.default_rng(3))
    monkeypatch.setattr(pool, "DECISIVE_SIGMA", 1.0)
    loose = _weekly(NEAR, [1, 2], week=1, trials=400, rng=np.random.default_rng(3))
    assert loose.resolution == pytest.approx(strict.resolution / 2.0)
    assert loose.decisive and not strict.decisive
    assert loose.recommend == strict.recommend


def test_a_trial_set_where_nothing_cashed_says_so_rather_than_reporting_a_tie():
    """The degenerate regime, distinguishable from a measured tie.

    Every candidate is worth exactly zero here because nothing survives week 2, so the
    difference between the recommendation and the free pick is zero -- and it is zero because
    nothing was measured, not because two plans were measured alike. Both send a reader to the
    free pick and only one of them is evidence. `docs/method.md` rule 12.
    """
    w = _weekly(NEVER, [1, 2], week=1, ledger=["KC", "LV"], trials=400,
                rng=np.random.default_rng(0))
    assert w.trials == 400 and w.cashed == 0
    assert w.unresolved and not w.decisive
    assert all(c.expected_dollars == 0.0 and c.survives == 0.0 for c in w.candidates)
    lines = "\n".join(pool.weekly_report(w))
    assert "unresolved rather than a measured tie" in lines
    assert "these trials can resolve" not in lines


def test_a_measured_tie_is_not_reported_as_an_unresolved_week():
    """The other side of it, on a set that did cash: a week where the two plans really do
    survive alike is a finding of no difference, not an absence of findings."""
    w = _weekly(FLAT, [1, 2, 3], week=1, trials=400, rng=np.random.default_rng(0))
    assert w.cashed > 0 and not w.unresolved
    assert w.matched and w.given_up == 0.0 and w.decisive
    assert "unresolved" not in "\n".join(pool.weekly_report(w))


def test_a_week_priced_in_closed_form_is_not_an_unresolved_one():
    """No trial ran, so there is no Monte Carlo error for anything to be inside of. A zero
    trial count is not zero cashes, and reading it as one would report the last week of a
    season -- the one week this simulator does not need -- as the week it could not answer."""
    w = _weekly(FLAT, [3], week=3)
    assert w.trials == 0 and not w.unresolved and w.decisive
    assert all(c.survives_se == 0.0 and c.dollars_se == 0.0 for c in w.candidates)


def test_no_figure_is_printed_finer_than_its_own_error():
    """Every printed figure, and each to its own error rather than to one `places` for the
    block. The candidates are not alike: a 30% dog's dollar figure is a fraction of the chalk
    team's and is resolved to a fraction of the precision. `$21.39` off 400 trials that
    resolve $2.53 is four digits of which two are the seed.
    """
    assert pool._places(2.53, cap=2) == 0
    assert pool._places(0.03, cap=2) == 1
    assert pool._places(0.004, cap=2) == 2
    assert pool._places(0.0, cap=2) == 2, "an exact figure is not an unknown one"
    w = _weekly(HOARD, [1, 2], week=1, trials=400, rng=np.random.default_rng(5))
    lines = pool.weekly_report(w)
    for c in w.candidates:
        row = next(ln for ln in lines if f" {c.team:<4} " in ln)
        shown = row.split()[0].lstrip("$")
        places = len(shown.split(".")[1]) if "." in shown else 0
        assert places == pool._places(c.dollars_se, cap=2)
        assert places == 0 or 10.0 ** -places >= c.dollars_se
    # And the survival cost is rounded to the error on the *difference*, which is the paired
    # one and the figure it is printed beside.
    assert w.given_up_se > 0
    said = next(ln for ln in lines if "points of survival" in ln).split()[1]
    shown = len(said.split(".")[1]) if "." in said else 0
    assert shown == pool._places(w.given_up_se * 100, cap=2)


def test_more_trials_buy_more_digits():
    """The visible half of the same rule: precision is a property of the run rather than of
    the format string, so a reader can see what a longer run bought."""
    kw = {"week": 1, "entries": 12, "pot": 420.0}
    coarse = pool.weekly(HOARD, [1, 2], trials=200, rng=np.random.default_rng(5), **kw)
    fine = pool.weekly(HOARD, [1, 2], trials=3200, rng=np.random.default_rng(5), **kw)
    dug = {c.team: pool._places(c.dollars_se, cap=2) for c in coarse.candidates}
    fig = {c.team: pool._places(c.dollars_se, cap=2) for c in fine.candidates}
    assert all(fig[t] >= dug[t] for t in dug)
    assert any(fig[t] > dug[t] for t in dug)
