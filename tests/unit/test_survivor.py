"""Survivor: one team a week, nobody twice, eighteen weeks.

The greedy pick -- take the biggest favourite each week -- is what most entrants do and it
is reliably wrong. Spending Kansas City in week 1 against a bad team costs you Kansas City
in week 12 when you need them more, and the schedule is known in advance, so the whole
season is a single assignment problem rather than eighteen independent choices.

Survival is multiplicative, so the objective is the sum of log win probabilities. That is
not a detail: maximising the *sum* of probabilities happily trades a 0.95 week for two
0.60s, which is a worse season and an easy mistake to make.

The pool-aware version -- deliberately going contrarian when the field is large -- is gated
on pool configuration nobody has yet, and `docs/decisions.md` lists that as the open
question. Maximising survival is the right default until then.
"""
import numpy as np
import polars as pl
import pytest

from hub.season import survivor


def _grid(rows):
    """rows: (week, team, win_prob)"""
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [float(r[2]) for r in rows]})


# --- the constraints ------------------------------------------------------

def test_one_pick_per_week():
    grid = _grid([(w, t, 0.5 + 0.01 * i) for w in (1, 2, 3)
                  for i, t in enumerate(["A", "B", "C"])])
    plan = survivor.solve(grid)
    assert sorted(plan["week"].to_list()) == [1, 2, 3]


def test_no_team_is_used_twice():
    grid = _grid([(w, t, 0.5 + 0.01 * i) for w in (1, 2, 3)
                  for i, t in enumerate(["A", "B", "C"])])
    plan = survivor.solve(grid)
    assert len(set(plan["team"].to_list())) == plan.height


def test_a_team_may_only_be_picked_in_a_week_it_plays():
    grid = _grid([(1, "A", 0.9), (2, "B", 0.6)])
    plan = survivor.solve(grid)
    assert plan.filter(pl.col("week") == 1)["team"][0] == "A"
    assert plan.filter(pl.col("week") == 2)["team"][0] == "B"


def test_an_infeasible_season_is_reported_not_silently_truncated():
    """Two weeks, one team available. There is no full-season plan and pretending
    otherwise would put a survivor entry into week 2 with nothing to pick."""
    grid = _grid([(1, "A", 0.9), (2, "A", 0.9)])
    with pytest.raises(survivor.Infeasible):
        survivor.solve(grid, weeks=[1, 2])


# --- the objective --------------------------------------------------------

def test_it_beats_the_greedy_pick():
    """The whole reason to solve rather than pick. A is the best team in both weeks, but
    spending it in week 1 leaves a 0.50 in week 2; saving it wins the season."""
    grid = _grid([(1, "A", 0.90), (1, "B", 0.85),
                  (2, "A", 0.90), (2, "B", 0.50)])
    plan = survivor.solve(grid, weeks=[1, 2])
    picks = dict(zip(plan["week"].to_list(), plan["team"].to_list(), strict=True))
    assert picks == {1: "B", 2: "A"}


def test_survival_is_multiplicative_not_additive():
    """A case where summing probabilities and multiplying them genuinely disagree.

    Two teams, two weeks, so the choices are coupled and only two plans exist:

        X then Y:  0.50 + 0.99 = 1.49 summed,  0.50 * 0.99 = 0.495 multiplied
        Y then X:  0.70 + 0.75 = 1.45 summed,  0.70 * 0.75 = 0.525 multiplied

    A linear objective takes the first and loses three points of survival. Surviving both
    weeks is a product, so the solver must take the second.
    """
    grid = _grid([(1, "X", 0.50), (1, "Y", 0.70),
                  (2, "X", 0.75), (2, "Y", 0.99)])
    plan = survivor.solve(grid, weeks=[1, 2])
    picks = dict(zip(plan["week"].to_list(), plan["team"].to_list(), strict=True))
    assert picks == {1: "Y", 2: "X"}
    assert survivor.survival(plan) == pytest.approx(0.525)


def test_the_plan_reports_its_survival_probability():
    grid = _grid([(1, "A", 0.9), (2, "B", 0.8)])
    plan = survivor.solve(grid, weeks=[1, 2])
    assert survivor.survival(plan) == pytest.approx(0.72)


def test_a_zero_probability_team_never_appears():
    """log(0) is negative infinity; the solver must handle it rather than crash."""
    grid = _grid([(1, "A", 0.0), (1, "B", 0.4), (2, "C", 0.5)])
    plan = survivor.solve(grid, weeks=[1, 2])
    assert "A" not in plan["team"].to_list()


# --- shape of the answer --------------------------------------------------

def test_a_full_season_is_assignable():
    """The done-when: eighteen weeks, a real-sized league, a feasible plan."""
    rng = np.random.default_rng(0)
    teams = [f"T{i:02d}" for i in range(32)]
    rows = []
    for w in range(1, 19):
        for t in rng.choice(teams, 26, replace=False):
            rows.append((w, str(t), float(rng.uniform(0.25, 0.85))))
    plan = survivor.solve(_grid(rows), weeks=list(range(1, 19)))
    assert plan.height == 18
    assert len(set(plan["team"].to_list())) == 18


def test_the_plan_is_ordered_by_week():
    rng = np.random.default_rng(1)
    rows = [(w, f"T{i}", float(rng.uniform(0.3, 0.8)))
            for w in range(1, 8) for i in range(10)]
    plan = survivor.solve(_grid(rows), weeks=list(range(1, 8)))
    assert plan["week"].to_list() == sorted(plan["week"].to_list())


def test_solving_is_deterministic():
    rng = np.random.default_rng(2)
    rows = [(w, f"T{i}", float(rng.uniform(0.3, 0.8)))
            for w in range(1, 6) for i in range(8)]
    grid = _grid(rows)
    assert (survivor.solve(grid, weeks=[1, 2, 3, 4, 5])["team"].to_list()
            == survivor.solve(grid, weeks=[1, 2, 3, 4, 5])["team"].to_list())


# --- what the market has not priced yet -----------------------------------

def test_uncovered_weeks_are_named_not_quietly_skipped():
    """The failure this catches, found on a real 2026 run: the board only had spreads for
    12 of 18 weeks in August, so the plan came back with 12 picks and announced it had
    survived "all 12 weeks". An entrant still has to pick in week 8. Planning around a week
    is fine; not saying so is how you show up on a Sunday with nothing."""
    grid = _grid([(1, "A", 0.7), (1, "B", 0.6), (3, "C", 0.8)])
    cov = survivor.coverage(grid, weeks=[1, 2, 3])
    assert cov["missing"] == [2]
    assert cov["covered"] == [1, 3]


def test_a_week_priced_by_a_single_game_is_flagged_as_thin():
    """Two teams in a week means one game on the board. The pick is whichever side of that
    game is favoured, not a choice among the league."""
    grid = _grid([(1, "A", 0.7), (1, "B", 0.3)]
                 + [(2, f"T{i}", 0.5) for i in range(20)])
    cov = survivor.coverage(grid, weeks=[1, 2])
    assert cov["thin"] == [1]


def test_full_coverage_reports_nothing_to_warn_about():
    grid = _grid([(w, f"T{i}", 0.5) for w in (1, 2) for i in range(20)])
    cov = survivor.coverage(grid, weeks=[1, 2])
    assert cov["missing"] == [] and cov["thin"] == []


def test_the_cli_says_so_when_the_season_is_not_fully_priced(capsys, monkeypatch):
    """docs/CLAUDE.md graceful degradation: serve the usable answer, but a partial plan
    must not read like a complete one."""
    monkeypatch.setattr(survivor, "grid_from_schedule",
                        lambda season, cache=None: _grid(
                            [(1, "A", 0.7), (1, "B", 0.6), (3, "C", 0.8), (3, "D", 0.5)]))
    assert survivor.main(["--season", "2026", "--weeks", "4"]) == 0
    out = capsys.readouterr().out
    assert "2" in out and "4" in out
    assert "not priced" in out.lower() or "no spread" in out.lower()
    assert "all 18 weeks" not in out


# --- the schedule -> probability boundary ---------------------------------

def test_the_home_favourite_is_the_one_with_the_higher_win_probability(tmp_path):
    """The sign convention, which this repo has already got wrong once at a different
    boundary (`docs/decisions.md`: The Odds API reports a handicap, nflverse a margin).
    nflverse `spread_line` is positive when the *home* team is favoured. Getting this
    backwards would produce a survivor plan that picks underdogs all season and still
    looks entirely plausible on the page."""
    import hub.fetch.nflverse as nflverse
    sched = pl.DataFrame({"game_id": ["2026_01_LV_KC"], "season": [2026], "week": [1],
                          "home_team": ["KC"], "away_team": ["LV"],
                          "spread_line": [9.5], "result": [None]})
    with pytest.MonkeyPatch.context() as m:
        m.setattr(nflverse, "load", lambda *a, **k: sched)
        grid = survivor.grid_from_schedule(2026, base=tmp_path)
    p = dict(zip(grid["team"].to_list(), grid["win_prob"].to_list(), strict=True))
    assert p["KC"] > 0.5 < 1.0 and p["LV"] < 0.5
    assert p["KC"] + p["LV"] == pytest.approx(1.0)


def test_games_without_a_posted_line_are_dropped_not_treated_as_coin_flips(tmp_path):
    """A null spread means the market has not priced it. Filling it in at 0.5 would put an
    unpriced game into the plan at a probability nobody quoted."""
    import hub.fetch.nflverse as nflverse
    sched = pl.DataFrame({"game_id": ["2026_01_LV_KC", "2026_02_SEA_SF"],
                          "season": [2026, 2026], "week": [1, 2],
                          "home_team": ["KC", "SF"], "away_team": ["LV", "SEA"],
                          "spread_line": [9.5, None], "result": [None, None]})
    with pytest.MonkeyPatch.context() as m:
        m.setattr(nflverse, "load", lambda *a, **k: sched)
        grid = survivor.grid_from_schedule(2026, base=tmp_path)
    assert grid["week"].to_list() == [1, 1]


# --- the weeks the snapshots reach and the moving field does not ---------------
#
# The ticket in one sentence: `spread_line` is a lookahead number upstream leaves empty for
# the late season, so survivor planned 12 of 18 weeks and called the other six "not priced
# yet" -- while the store held every game of the season, week 18 included. Solving twelve
# weeks now and the rest later, with the best teams already spent, is exactly the mistake
# this module exists to avoid.

import datetime as dt  # noqa: E402

from hub import store  # noqa: E402


def _late_season(tmp_path, snapshot_weeks=(), at=dt.datetime(2026, 9, 4)):
    """A schedule the moving field prices only in week 1, plus snapshots for named weeks."""
    rows = [("2026_01_LV_KC", 1, "KC", "LV", 9.5),
            ("2026_02_SEA_SF", 2, "SF", "SEA", None),
            ("2026_18_NYJ_BUF", 18, "BUF", "NYJ", None)]
    sched = pl.DataFrame({
        "game_id": [r[0] for r in rows], "season": [2026] * len(rows),
        "week": [r[1] for r in rows], "home_team": [r[2] for r in rows],
        "away_team": [r[3] for r in rows], "spread_line": [r[4] for r in rows],
        "result": [None] * len(rows)})
    for gid, wk, _, _, _ in rows:
        if wk in snapshot_weeks:
            store.write(
                pl.DataFrame({"game_id": [gid], "close_spread": [7.0],
                              "captured_at": [at]},
                             schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                     "captured_at": pl.Datetime}),
                "lines", "nfl", 2026, wk, base=tmp_path, name="snap-x")
    return sched


def test_a_week_only_the_snapshot_prices_still_enters_the_plan(tmp_path, monkeypatch):
    """Week 18 carries no `spread_line` and never will at the moment the plan is first
    wanted. The snapshot is what puts it in the season."""
    import hub.fetch.nflverse as nflverse
    sched = _late_season(tmp_path, snapshot_weeks=(18,))
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: sched)
    grid = survivor.grid_from_schedule(2026, at=dt.datetime(2026, 9, 5), base=tmp_path)
    cov = survivor.coverage(grid, [1, 2, 18])
    assert 18 in cov["covered"]
    assert 2 in cov["missing"], "week 2 has neither source and must still ask for a pick"
    assert 18 in survivor.solve(grid, weeks=cov["covered"])["week"].to_list()


def test_a_week_neither_source_prices_is_still_reported_as_needing_a_pick(tmp_path,
                                                                          monkeypatch):
    """The coverage report is what tells an entrant a week is theirs to fill. Reading the
    store must not turn a genuinely unpriced week into a silent absence."""
    import hub.fetch.nflverse as nflverse
    monkeypatch.setattr(nflverse, "load",
                        lambda *a, **k: _late_season(tmp_path, snapshot_weeks=(18,)))
    grid = survivor.grid_from_schedule(2026, at=dt.datetime(2026, 9, 5), base=tmp_path)
    assert survivor.coverage(grid, [1, 2, 18])["missing"] == [2]


def test_the_snapshot_does_not_change_a_week_the_moving_field_already_priced(tmp_path,
                                                                             monkeypatch):
    """Coverage, not repricing. Where both exist the two agree, and the plan for those weeks
    is the one it always was."""
    import hub.fetch.nflverse as nflverse
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: _late_season(tmp_path))
    before = survivor.grid_from_schedule(2026, at=dt.datetime(2026, 9, 5), base=tmp_path)
    monkeypatch.setattr(nflverse, "load",
                        lambda *a, **k: _late_season(tmp_path, snapshot_weeks=(18,)))
    after = survivor.grid_from_schedule(2026, at=dt.datetime(2026, 9, 5), base=tmp_path)
    wk1 = lambda g: g.filter(pl.col("week") == 1).sort("team")["win_prob"].to_list()  # noqa: E731
    assert wk1(before) == wk1(after)
    assert 18 not in before["week"].to_list() and 18 in after["week"].to_list()


def test_survivor_and_the_weekly_prediction_price_a_game_the_same_way(tmp_path, monkeypatch):
    """The claim `grid_from_schedule` makes in its own docstring, asserted rather than
    stated. It held until the weekly prediction moved onto the snapshots and this did not."""
    import hub.fetch.nflverse as nflverse
    from hub import schedule as sch
    from hub.models.market import MARGIN_SD, normal_cdf
    monkeypatch.setattr(nflverse, "load",
                        lambda *a, **k: _late_season(tmp_path, snapshot_weeks=(18,)))
    at = dt.datetime(2026, 9, 5)
    priced = sch.priced_games(2026, at=at, base=tmp_path)
    spread = priced.filter(pl.col("week") == 18)["close_spread"][0]
    grid = survivor.grid_from_schedule(2026, at=at, base=tmp_path)
    buf = grid.filter((pl.col("week") == 18) & (pl.col("team") == "BUF"))["win_prob"][0]
    assert buf == pytest.approx(normal_cdf(float(spread) / MARGIN_SD))


def test_a_week_both_sources_price_is_not_reported_as_snapshot_only(tmp_path, monkeypatch):
    """The measurement I got wrong first. Reading it off the *winning* source says
    "snapshot" for every game once the store covers the season, so every week looks like one
    the fallback could not reach -- which reported all eighteen and was caught only by the
    real data disagreeing. What decides it is whether the moving field carries the game at
    all, not which of the two was used."""
    import hub.fetch.nflverse as nflverse
    monkeypatch.setattr(nflverse, "load",
                        lambda *a, **k: _late_season(tmp_path, snapshot_weeks=(1, 18)))
    grid = survivor.grid_from_schedule(2026, at=dt.datetime(2026, 9, 5), base=tmp_path)
    # week 1 carries both and is priced from the snapshot; week 18 carries only the snapshot
    assert grid.filter(pl.col("week") == 1)["moving_field"].all()
    assert survivor.snapshot_only_weeks(grid, [1, 18]) == [18]


def test_snapshot_only_weeks_is_empty_when_the_moving_field_reaches_everything(tmp_path,
                                                                               monkeypatch):
    import hub.fetch.nflverse as nflverse
    sched = pl.DataFrame({"game_id": ["2026_01_LV_KC"], "season": [2026], "week": [1],
                          "home_team": ["KC"], "away_team": ["LV"],
                          "spread_line": [9.5], "result": [None]})
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: sched)
    grid = survivor.grid_from_schedule(2026, base=tmp_path)
    assert survivor.snapshot_only_weeks(grid, [1]) == []
