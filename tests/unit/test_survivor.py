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


@pytest.fixture(autouse=True)
def _no_real_pool_state(monkeypatch, tmp_path):
    """`prior_rows` reads the pool host's last-known state for the Ledger (#280), and a real
    one under `data/processed/` must not reach a test's plan through the CLI."""
    from hub.fetch import pool as fetch_pool
    monkeypatch.setattr(fetch_pool, "PROCESSED", tmp_path / "no-pool-state")


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


def test_the_survivor_plan_reports_its_survival_probability():
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


def test_the_survivor_plan_is_ordered_by_week():
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
    assert cov.missing == [2]
    assert cov.covered == [1, 3]


def test_a_week_priced_by_a_single_game_is_flagged_as_thin():
    """Two teams in a week means one game on the board. The pick is whichever side of that
    game is favoured, not a choice among the league."""
    grid = _grid([(1, "A", 0.7), (1, "B", 0.3)]
                 + [(2, f"T{i}", 0.5) for i in range(20)])
    cov = survivor.coverage(grid, weeks=[1, 2])
    assert cov.thin == [1]


def test_full_coverage_reports_nothing_to_warn_about():
    grid = _grid([(w, f"T{i}", 0.5) for w in (1, 2) for i in range(20)])
    cov = survivor.coverage(grid, weeks=[1, 2])
    assert cov.missing == [] and cov.thin == []


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


def test_both_sides_of_a_fixture_carry_the_same_game_id(tmp_path):
    """Two rows, one game. A week that takes two picks must be able to tell that these two
    teams are the same fixture, because taking both guarantees one of them loses. Under one
    pick a week the pairing was unreachable, so nothing on the row ever needed to say it."""
    import hub.fetch.nflverse as nflverse
    sched = pl.DataFrame({"game_id": ["2026_01_LV_KC", "2026_01_SEA_SF"],
                          "season": [2026, 2026], "week": [1, 1],
                          "home_team": ["KC", "SF"], "away_team": ["LV", "SEA"],
                          "spread_line": [9.5, 3.0], "result": [None, None]})
    with pytest.MonkeyPatch.context() as m:
        m.setattr(nflverse, "load", lambda *a, **k: sched)
        grid = survivor.grid_from_schedule(2026, base=tmp_path)
    by_team = dict(zip(grid["team"].to_list(), grid["game_id"].to_list(), strict=True))
    assert by_team["KC"] == by_team["LV"]
    assert by_team["SF"] == by_team["SEA"]
    assert by_team["KC"] != by_team["SF"]


def test_a_shared_kickoff_is_not_a_shared_game(tmp_path):
    """The reason the identifier has to ride along rather than being derived. A dozen games
    share a Sunday afternoon slot, so grouping on `kickoff` would forbid taking two teams
    from two entirely different fixtures -- a silently wrong plan of exactly the kind the
    same-game rule exists to prevent."""
    import hub.fetch.nflverse as nflverse
    sched = pl.DataFrame({"game_id": ["2026_01_LV_KC", "2026_01_SEA_SF"],
                          "season": [2026, 2026], "week": [1, 1],
                          "home_team": ["KC", "SF"], "away_team": ["LV", "SEA"],
                          "spread_line": [9.5, 3.0], "result": [None, None],
                          "gameday": ["2026-09-13", "2026-09-13"],
                          "gametime": ["13:00", "13:00"]})
    with pytest.MonkeyPatch.context() as m:
        m.setattr(nflverse, "load", lambda *a, **k: sched)
        grid = survivor.grid_from_schedule(2026, base=tmp_path)
    assert grid["kickoff"].n_unique() == 1, "the fixture is that they share a slot"
    by_team = dict(zip(grid["team"].to_list(), grid["game_id"].to_list(), strict=True))
    assert by_team["KC"] != by_team["SF"]


def test_the_game_id_does_not_change_any_plan():
    """It rides along and nothing reads it. The next change makes a week take two picks and
    uses it; until then a grid carrying the column must plan exactly as one without it, or
    this commit is not the no-op it claims to be."""
    rows = [(1, "KC", 0.9), (1, "SF", 0.8), (2, "KC", 0.7), (2, "SF", 0.6)]
    plain = _grid(rows)
    keyed = plain.with_columns(
        pl.Series("game_id", ["2026_01_LV_KC", "2026_01_SEA_ARI",
                              "2026_02_KC_DEN", "2026_02_SF_LAR"]))
    assert (survivor.solve(plain).drop("win_prob").to_dicts()
            == survivor.solve(keyed).drop("win_prob").to_dicts())


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


def test_a_week_only_the_snapshot_prices_still_enters_the_survivor_plan(tmp_path, monkeypatch):
    """Week 18 carries no `spread_line` and never will at the moment the plan is first
    wanted. The snapshot is what puts it in the season."""
    import hub.fetch.nflverse as nflverse
    sched = _late_season(tmp_path, snapshot_weeks=(18,))
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: sched)
    grid = survivor.grid_from_schedule(2026, at=dt.datetime(2026, 9, 5), base=tmp_path)
    cov = survivor.coverage(grid, [1, 2, 18])
    assert 18 in cov.covered
    assert 2 in cov.missing, "week 2 has neither source and must still ask for a pick"
    assert 18 in survivor.solve(grid, weeks=cov.covered)["week"].to_list()


def test_a_week_neither_source_prices_is_still_reported_as_needing_a_pick(tmp_path,
                                                                          monkeypatch):
    """The coverage report is what tells an entrant a week is theirs to fill. Reading the
    store must not turn a genuinely unpriced week into a silent absence."""
    import hub.fetch.nflverse as nflverse
    monkeypatch.setattr(nflverse, "load",
                        lambda *a, **k: _late_season(tmp_path, snapshot_weeks=(18,)))
    grid = survivor.grid_from_schedule(2026, at=dt.datetime(2026, 9, 5), base=tmp_path)
    assert survivor.coverage(grid, [1, 2, 18]).missing == [2]


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


# --- a plan is for the weeks that are left (issue #24) -----------------------
#
# Survivor is one assignment problem *because* spending a team in week 1 costs you that team
# in week 12. The grid priced every week of the season with no cutoff, so from week 2 onward
# the plan spent its strongest teams on weeks already over: every remaining pick was drawn
# from a pool degraded by picks that were never available, and the reported survival
# probability was the product over games already won or lost. Neither is visible from the
# output -- the plan looks like a plan.

def _dated_grid(rows):
    """rows: (week, team, win_prob, kickoff, result)"""
    return pl.DataFrame(
        {"week": [r[0] for r in rows], "team": [r[1] for r in rows],
         "win_prob": [float(r[2]) for r in rows], "kickoff": [r[3] for r in rows],
         "result": [r[4] for r in rows]},
        schema={"week": pl.Int64, "team": pl.Utf8, "win_prob": pl.Float64,
                "kickoff": pl.Datetime, "result": pl.Float64})


def test_a_team_already_spent_cannot_be_planned_again():
    """The constraint the solver already enforces within one plan, extended across the
    plans that came before it. Without it a September pick is silently available again in
    October and the plan is infeasible the moment it is entered."""
    grid = _grid([(2, "A", 0.90), (2, "B", 0.60), (2, "C", 0.55),
                  (3, "A", 0.95), (3, "B", 0.50), (3, "C", 0.45)])
    assert "A" in survivor.solve(grid)["team"].to_list(), (
        "the premise: A is worth picking, so leaving it out has to be `spent` doing it")
    plan = survivor.solve(grid, spent=["A"])
    assert "A" not in plan["team"].to_list()
    assert plan.height == 2


def test_spending_the_only_option_in_a_week_is_reported_not_silently_dropped():
    grid = _grid([(2, "A", 0.9), (3, "A", 0.9), (3, "B", 0.8)])
    with pytest.raises(survivor.Infeasible):
        survivor.solve(grid, weeks=[2, 3], spent=["A"])


def test_a_week_already_played_is_not_in_the_grid_the_remaining_plan_solves():
    """Week 1 has kicked off and has a result. It is not a choice any more, and the honest
    plan is over what is left."""
    import datetime as dt
    grid = _dated_grid([
        (1, "A", 0.9, dt.datetime(2026, 9, 10, 20, 15), 7.0),
        (1, "B", 0.1, dt.datetime(2026, 9, 10, 20, 15), 7.0),
        (2, "A", 0.8, dt.datetime(2026, 9, 17, 20, 15), None),
        (2, "B", 0.2, dt.datetime(2026, 9, 17, 20, 15), None),
    ])
    got = survivor.plan_remaining(grid, 2026, season_weeks=2,
                                  at=dt.datetime(2026, 9, 15))
    assert got.played == [1]
    assert got.coverage.covered == [2]
    assert got.picks["week"].to_list() == [2]


def test_a_week_in_progress_is_no_more_pickable_than_a_finished_one():
    """Both tests, because neither covers the other -- the same pair
    `schedule.forecastable` makes for a weekly prediction. A game under way has no result
    and is not a choice either."""
    import datetime as dt
    grid = _dated_grid([
        (1, "A", 0.9, dt.datetime(2026, 9, 10, 20, 15), None),
        (2, "B", 0.8, dt.datetime(2026, 9, 17, 20, 15), None),
    ])
    assert survivor.played(grid, at=dt.datetime(2026, 9, 10, 21, 0)) == [1]


def test_a_week_with_one_game_started_keeps_its_published_pick_and_is_not_re_solved():
    """#263. Week 2's Thursday game has kicked off and its Sunday games have not. The pick
    for week 2 was entered before that kickoff and is locked with the Pool, so the week is
    behind the remaining plan: the published pick -- SEA, which no re-solve would choose --
    is kept as spent, and the plan covers week 3 alone. A week fully ahead and a week fully
    behind read exactly as they did."""
    import datetime as dt
    thu, sun = dt.datetime(2026, 9, 17, 20, 15), dt.datetime(2026, 9, 20, 13, 0)
    grid = _dated_grid([
        (1, "KC", 0.9, dt.datetime(2026, 9, 10, 20, 15), 7.0),
        (1, "LV", 0.1, dt.datetime(2026, 9, 10, 20, 15), 7.0),
        (2, "BUF", 0.8, thu, None), (2, "NYJ", 0.2, thu, None),
        (2, "SF", 0.6, sun, None), (2, "SEA", 0.4, sun, None),
        (3, "SF", 0.7, dt.datetime(2026, 9, 27, 13, 0), None),
        (3, "SEA", 0.3, dt.datetime(2026, 9, 27, 13, 0), None),
        (3, "BUF", 0.65, dt.datetime(2026, 9, 27, 13, 0), None),
        (3, "NYJ", 0.35, dt.datetime(2026, 9, 27, 13, 0), None),
    ])
    at = dt.datetime(2026, 9, 18, 9, 0)      # Friday: Thursday played, Sunday ahead
    assert survivor.played(grid, at=at) == [1, 2]
    prior = [{"week": 1, "team": "KC"}, {"week": 2, "team": "SEA"}, {"week": 3, "team": "SF"}]
    got = survivor.plan_remaining(grid, 2026, prior=prior, season_weeks=3, at=at)
    assert got.played == [1, 2]
    assert got.spent == ["KC", "SEA"], "the locked pick is kept, not re-solved away"
    assert got.picks["week"].to_list() == [3]
    assert got.coverage.covered == [3] and got.coverage.missing == []
    # Unchanged either side of it: a week nothing in has started is ahead, and one with
    # every game over is behind, as before.
    assert survivor.played(grid, at=dt.datetime(2026, 9, 15)) == [1]
    assert survivor.played(grid, at=dt.datetime(2026, 9, 21)) == [1, 2]


def test_survival_is_the_product_over_the_weeks_still_to_come():
    """The reported number was the product over games already won or lost, which says
    nothing about whether the entry survives from here."""
    import datetime as dt
    grid = _dated_grid([
        (1, "A", 0.50, dt.datetime(2026, 9, 10, 20, 15), 7.0),
        (2, "B", 0.80, dt.datetime(2026, 9, 17, 20, 15), None),
        (3, "C", 0.90, dt.datetime(2026, 9, 24, 20, 15), None),
    ])
    got = survivor.plan_remaining(grid, 2026, season_weeks=3,
                                  at=dt.datetime(2026, 9, 15))
    assert survivor.survival(got.picks) == pytest.approx(0.8 * 0.9)


def test_a_grid_with_no_kickoffs_is_entirely_still_to_come():
    """Preseason, and the fixture shape every other test here uses. A schedule without
    times carries a null kickoff, and an invented one would silently decide this."""
    grid = _grid([(1, "A", 0.9), (2, "B", 0.8)])
    assert survivor.played(grid) == []
    assert survivor.plan_remaining(grid, 2026, season_weeks=2).coverage.covered == [1, 2]


# --- the remaining plan is one function, and the rule has one name -----------
#
# The six steps -- what is still ahead, which weeks are behind, what those weeks spent,
# which of the weeks left are priced, solve against the rest -- were written out twice,
# verbatim, in `hub.publish.survivor` and in this module's CLI. That sequence *is* the rule
# issue #24 was about, so two copies is one plan silently wrong with nothing in either
# output to say which. And `forthcoming` was a second name for `schedule.forecastable`
# under ten lines of docstring saying the rule was "unchanged and unrestated".

def test_the_remaining_plan_carries_the_scope_it_is_a_plan_over():
    """A plan is not readable without its scope: which weeks are behind it, which of the
    weeks left the market has not priced, and which teams it was not allowed to use."""
    import datetime as dt
    kicks = [dt.datetime(2026, 9, 3 + 7 * w, 20, 15) for w in range(4)]
    grid = _dated_grid([
        (1, "KC", 0.90, kicks[0], 7.0), (1, "LV", 0.10, kicks[0], 7.0),
        (2, "KC", 0.85, kicks[1], None), (2, "SF", 0.70, kicks[1], None),
        (3, "KC", 0.80, kicks[2], None), (3, "SEA", 0.60, kicks[2], None),
    ])
    got = survivor.plan_remaining(grid, 2026, prior=[{"week": 1, "team": "KC"}],
                                  season_weeks=4, at=dt.datetime(2026, 9, 5))
    assert got.played == [1], "week 1 is behind us and is neither planned nor unpriced"
    assert got.spent == ["KC"], "the published plan is the only record of what was used"
    assert "KC" not in got.picks["team"].to_list()
    assert got.coverage.covered == [2, 3] and got.coverage.missing == [4]
    assert got.coverage.thin == [2, 3], "two teams in a week is one game, not a choice"
    assert got.snapshot_only == []


def test_a_remaining_plan_with_nothing_priced_says_which_weeks_it_wanted():
    """The CLI's own message, raised rather than printed, so both callers get it. Solving
    zero weeks would otherwise arrive as "no pickable team in any week", which is true of
    the grid and not the answer to what was asked."""
    grid = _grid([(9, "A", 0.7)])
    with pytest.raises(survivor.Infeasible, match="1-4"):
        survivor.plan_remaining(grid, 2026, season_weeks=4)


def test_the_cli_asks_for_a_plan_rather_than_assembling_one(monkeypatch):
    """Half the drift guard, and it says which half: this asserts only that the CLI reaches
    the plan through `plan_remaining`, not that the two agree. That the panel and the CLI
    *print the same plan* is asserted in `tests/unit/test_publish.py`, by running both."""
    seen = []
    real = survivor.plan_remaining
    monkeypatch.setattr(survivor, "plan_remaining",
                        lambda *a, **k: (seen.append((a, k)), real(*a, **k))[1])
    monkeypatch.setattr(survivor, "grid_from_schedule",
                        lambda season, cache=None: _grid([(1, "A", 0.7), (1, "B", 0.6),
                                                          (2, "C", 0.8), (2, "D", 0.5)]))
    assert survivor.main(["--season", "2026", "--weeks", "2"]) == 0
    assert seen, "the CLI builds its own plan instead of asking for one"


def test_the_season_length_is_spelled_once():
    """`hub.publish` carried its own `NFL_WEEKS = 18` beside this CLI's `--weeks` default of
    18. One number, two spellings, in the two places that plan the same season."""
    import inspect

    from hub import publish
    assert survivor.NFL_WEEKS == 18
    assert (inspect.signature(survivor.plan_remaining).parameters["season_weeks"].default
            == survivor.NFL_WEEKS)
    assert not hasattr(publish, "NFL_WEEKS"), "the panel spells the season length again"


# --- a plan belongs to one season, like a published week does ----------------
#
# Found reviewing the change above. `spent_teams` matched on week alone, so a `survivor.json`
# left from a previous season contributes its weeks-1..N picks as "spent" in the new one.
# `site/data/survivor.json` is committed and `data/processed/` is not, so the case is a
# scheduled run that starts a season mid-way -- a dormant repo, or a fresh clone. Exactly
# the collision issue #23 fixed for the weekly artifact, one file over.

def test_a_plan_from_another_season_spends_nothing():
    prior = [{"season": 2025, "week": 1, "team": "KC"},
             {"season": 2025, "week": 2, "team": "SF"}]
    assert survivor.spent_teams(prior, [1, 2], season=2026) == []
    assert survivor.spent_teams(prior, [1, 2], season=2025) == ["KC", "SF"]


def test_a_plan_with_no_season_is_read_as_this_one():
    """Artifacts written before the plan carried a season. Refusing them would silently
    forget what a running entry had spent, which is the worse of the two errors."""
    prior = [{"week": 1, "team": "KC"}]
    assert survivor.spent_teams(prior, [1], season=2026) == ["KC"]


def test_a_row_with_no_week_is_skipped_rather_than_raising():
    """A partially written or hand-edited artifact. `int(None)` raised, and `publish`
    caught it in its broad except -- so the panel went stale citing an unavailable
    schedule, which is the wrong cause reported for the wrong reason."""
    prior = [{"week": None, "team": "KC"}, {"week": 1, "team": "SF"}]
    assert survivor.spent_teams(prior, [1], season=2026) == ["SF"]


def test_the_ledger_has_one_reading_the_pool_host_and_the_published_plan_both_feed(tmp_path,
                                                                                  capsys):
    """#280. The default Ledger came from the published plan, not the fetched pool state
    and not the journal: three sources of truth for one Ledger. `prior_rows` is the one
    reading now, in the shape `spent_teams` takes: the host's Ledger for our entry -- what
    was entered and settled -- as week-0 ledger rows beside the plan's own, which carry what
    is locked and not yet settled. A state from another season contributes nothing; no
    state, and the plan's rows stand alone; a drifted state is said and skipped."""
    import datetime as dt
    import json

    from hub.fetch import pool as fetch_pool
    art = tmp_path / "survivor.json"
    art.write_text(json.dumps({"name": "survivor", "season": 2026, "n": 1, "spent": ["DAL"],
                               "rows": [{"week": 2, "team": "SF", "win_prob": 0.7}]}))
    store = tmp_path / "processed"
    assert survivor.prior_rows(2026, path=art, store=store) == survivor.published_plan(art)

    state = fetch_pool.PoolState(season=2026, week=2, field_size=3, pot=60.0, entries=(
        fetch_pool.Entry(0, True, ("KC",)), fetch_pool.Entry(1, True, ("PHI",)),
        fetch_pool.Entry(2, False, ())))
    fetch_pool.write_state(state, store, when=dt.datetime(2026, 9, 16, 9, 0, tzinfo=dt.UTC))
    rows = survivor.prior_rows(2026, path=art, store=store)
    assert {"week": 0, "team": "KC", "ledger": True, "season": 2026} in rows
    assert not any(r["team"] == "PHI" for r in rows), "a rival's Ledger is not ours"
    assert survivor.spent_teams(rows, [2], season=2026) == ["DAL", "KC", "SF"]
    assert survivor.spent_teams(survivor.prior_rows(2025, path=art, store=store), [2],
                                season=2025) == []

    # A state listing no entry at our index -- `parse_payload` never writes one, but the
    # store is a file -- contributes nothing rather than somebody else's Ledger.
    fetch_pool.write_state(state._replace(entries=state.entries[1:]), store,
                           when=dt.datetime(2026, 9, 16, 10, 0, tzinfo=dt.UTC))
    assert survivor.spent_teams(survivor.prior_rows(2026, path=art, store=store), [2],
                                season=2026) == ["DAL", "SF"]

    doc = json.loads(fetch_pool.state_path(store).read_text())
    del doc["entries"][0]["used"]
    fetch_pool.state_path(store).write_text(json.dumps(doc))
    assert survivor.spent_teams(survivor.prior_rows(2026, path=art, store=store), [2],
                                season=2026) == ["DAL", "SF"]
    assert "pool state" in capsys.readouterr().err


def test_the_published_plan_stamps_each_row_with_the_seasons_it_came_from(tmp_path):
    """The season is on the envelope, not the rows, so it is carried down here -- otherwise
    `spent_teams` would need the artifact's shape as well as its rows."""
    import json
    art = tmp_path / "survivor.json"
    art.write_text(json.dumps({"name": "survivor", "season": 2025, "n": 1,
                               "rows": [{"week": 1, "team": "KC", "win_prob": 0.9}]}))
    assert survivor.published_plan(art) == [
        {"week": 1, "team": "KC", "win_prob": 0.9, "season": 2025}]


def test_an_absent_plan_is_no_history_rather_than_an_error(tmp_path):
    assert survivor.published_plan(tmp_path / "nothing.json") == []


# --- weeks that take two teams ------------------------------------------------
#
# Both have to win. The objective already said so -- surviving both is the product, which the
# sum of logs expresses -- and the no-repeat constraint already spanned the season. What is new
# is that the two picks must not be the two sides of one fixture, which under one pick a week
# was unreachable and under two is reachable, fatal, and attractive across a season.

from hub.config import PoolConfig  # noqa: E402


def _fx(rows):
    """rows: (week, team_a, team_b, p_a) -- a grid carrying the fixture key `solve` needs."""
    out = []
    for i, (w, a, b, p) in enumerate(rows):
        gid = f"{w}-{i}"
        out += [(w, a, float(p), gid), (w, b, 1.0 - float(p), gid)]
    return pl.DataFrame({"week": [r[0] for r in out], "team": [r[1] for r in out],
                         "win_prob": [r[2] for r in out], "game_id": [r[3] for r in out]})


_DOUBLE = PoolConfig(double_pick_weeks=(2,))


def test_a_double_week_takes_two_teams_and_a_single_week_takes_one():
    g = _fx([(1, "KC", "LV", 0.9), (1, "SF", "SEA", 0.8),
             (2, "BUF", "NYJ", 0.85), (2, "DAL", "NYG", 0.75)])
    plan = survivor.solve(g, [1, 2], pool=_DOUBLE)
    by_week = plan.group_by("week").len().sort("week")
    assert by_week["len"].to_list() == [1, 2]
    wk2 = plan.filter(pl.col("week") == 2)["team"].to_list()
    assert len(set(wk2)) == 2


def test_both_teams_have_to_win_so_survival_multiplies_them():
    """Not one of them: a double week is survived only if neither pick loses."""
    g = _fx([(1, "KC", "LV", 0.9), (2, "BUF", "NYJ", 0.8), (2, "DAL", "NYG", 0.75)])
    plan = survivor.solve(g, [1, 2], pool=_DOUBLE)
    assert survivor.survival(plan) == pytest.approx(0.9 * 0.8 * 0.75)


def test_a_double_pick_week_with_one_usable_fixture_is_refused_for_the_true_reason():
    """The pre-check counts fixtures, because two picks have to come from two games.

    Counted on rows these two teams look like enough for two picks; they are the two sides of
    one fixture, so one of them loses and the week cannot be covered. Counted on rows the
    check passes, the `one_side` constraint below makes the program infeasible, and what
    surfaces is "no full-season plan" for the whole season -- naming neither the week nor the
    reason, which is the one thing an entrant needs to know."""
    g = _fx([(2, "KC", "LV", 0.8)])
    with pytest.raises(survivor.Infeasible, match="both sides of one fixture") as e:
        survivor.solve(g, [2], pool=_DOUBLE)
    assert "week 2" in str(e.value)
    assert "no full-season plan" not in str(e.value)


def test_the_two_picks_are_never_the_two_sides_of_one_fixture():
    """The constraint that only exists once a week takes two. Both sides of a game are the
    two highest-probability rows available here, so an unconstrained solver would reach for
    them: one of them loses, and the week is lost with it."""
    g = _fx([(2, "KC", "LV", 0.55), (2, "SF", "SEA", 0.54), (2, "BUF", "NYJ", 0.53)])
    plan = survivor.solve(g, [2], pool=_DOUBLE)
    picked = plan["team"].to_list()
    gids = g.filter(pl.col("team").is_in(picked) & (pl.col("week") == 2))["game_id"].to_list()
    assert len(set(gids)) == 2, "the two picks came from one fixture"


def test_two_teams_sharing_a_slot_but_not_a_fixture_stay_jointly_pickable():
    """The exclusion keys on the fixture, not the kickoff -- a dozen games share a Sunday."""
    g = _fx([(2, "KC", "LV", 0.9), (2, "SF", "SEA", 0.88)]).with_columns(
        pl.lit(dt.datetime(2026, 9, 13, 13, 0)).alias("kickoff"))
    plan = survivor.solve(g, [2], pool=_DOUBLE)
    assert sorted(plan["team"].to_list()) == ["KC", "SF"]


def test_a_team_spent_earlier_is_unavailable_in_either_slot():
    g = _fx([(2, "KC", "LV", 0.9), (2, "SF", "SEA", 0.85), (2, "BUF", "NYJ", 0.8)])
    plan = survivor.solve(g, [2], spent=["KC"], pool=_DOUBLE)
    assert "KC" not in plan["team"].to_list()
    assert len(plan) == 2


def test_a_double_week_priced_by_one_fixture_needs_a_pick_rather_than_raising():
    """The failure this avoids is season-wide: called covered, the week hands the solver an
    unsatisfiable equality and `publish.survivor` keeps a stale plan for the weeks that were
    fine. Reported as needing a pick instead."""
    g = _fx([(1, "KC", "LV", 0.9), (2, "SF", "SEA", 0.8)])
    cov = survivor.coverage(g, [1, 2], _DOUBLE)
    assert cov.covered == [1] and cov.missing == [2]
    survivor.solve(g, cov.covered, pool=_DOUBLE)      # the weeks that were fine still plan


def test_with_no_double_weeks_configured_the_survivor_plan_is_todays():
    g = _fx([(1, "KC", "LV", 0.9), (1, "SF", "SEA", 0.8),
             (2, "BUF", "NYJ", 0.85), (2, "DAL", "NYG", 0.75)])
    plain = survivor.solve(g, [1, 2])
    empty = survivor.solve(g, [1, 2], pool=PoolConfig(double_pick_weeks=()))
    assert plain.to_dicts() == empty.to_dicts()
    assert len(plain) == 2


def test_a_double_week_solves_the_same_way_twice():
    g = _fx([(2, "KC", "LV", 0.9), (2, "SF", "SEA", 0.85), (2, "BUF", "NYJ", 0.8)])
    a = survivor.solve(g, [2], pool=_DOUBLE)
    b = survivor.solve(g, [2], pool=_DOUBLE)
    assert a.to_dicts() == b.to_dicts()


def test_a_double_week_without_a_fixture_key_is_refused():
    """Rather than silently taking both sides of a game nobody can identify."""
    g = _fx([(2, "KC", "LV", 0.9), (2, "SF", "SEA", 0.8)]).drop("game_id")
    with pytest.raises(ValueError, match="game_id"):
        survivor.solve(g, [2], pool=_DOUBLE)


def test_the_cli_counts_weeks_and_reports_picks_beside_them(capsys, monkeypatch):
    """The caption printed `picks.height` as a week count, which reads "24 planned weeks" for
    an eighteen-week season once a week can take two."""
    g = _fx([(1, "KC", "LV", 0.9), (2, "BUF", "NYJ", 0.8), (2, "DAL", "NYG", 0.75)])
    plan = survivor.solve(g, [1, 2], pool=_DOUBLE)
    monkeypatch.setattr(survivor, "grid_from_schedule", lambda *a, **k: g)
    monkeypatch.setattr(survivor, "plan_remaining", lambda *a, **k: survivor.RemainingPlan(
        picks=plan, coverage=survivor.Coverage([1, 2], [], []), played=[], spent=[],
        snapshot_only=[]))
    survivor.main(["--season", "2026"])
    out = capsys.readouterr().out
    assert "survives the 2 planned weeks" in out
    assert "(3 picks)" in out


def test_one_rule_says_what_a_usable_week_is_and_names_its_two_halves():
    """The finding of #204, pinned: *pickable* and *drawable* are different statements about
    a fixture, and until now each consumer decided one of them privately.

    Three fixtures in a week. One is priced on both sides, one on a single side, and one on
    both sides of which the underdog is below the floor. A pick may be taken from all three
    -- each offers a side worth having -- and only two of them can be played out, because a
    game whose opponent has no row has no result to draw. The floor is not a coverage test:
    the 0.9999 favourite's fixture is drawable and its opponent is simply never pickable."""
    g = pl.concat([
        _fx([(1, "KC", "LV", 0.8)]),
        pl.DataFrame({"week": [1], "team": ["SF"], "win_prob": [0.7], "game_id": ["solo"]}),
        pl.DataFrame({"week": [1, 1], "team": ["BUF", "NYJ"],
                      "win_prob": [0.9999, 1e-5], "game_id": ["lop", "lop"]})])
    f = survivor.week_fixtures(g, [1])[0]
    assert f.week == 1 and f.needs == 1
    assert f.pickable == ("1-0", "lop", "solo")
    assert f.drawable == ("1-0", "lop")
    assert f.half == ("solo",)


def test_a_double_week_needs_two_fixtures_by_the_same_rule_both_consumers_read():
    """`needs` rides on the answer rather than being recomputed by each caller. It was, and
    the two callers disagreed: this week is missing here and was simulated there."""
    g = _fx([(1, "KC", "LV", 0.9), (2, "SF", "SEA", 0.8)])
    assert [f.needs for f in survivor.week_fixtures(g, [1, 2], _DOUBLE)] == [1, 2]
    cov = survivor.coverage(g, [1, 2], _DOUBLE)
    assert cov.covered == [1] and cov.missing == [2]


def test_a_grid_with_no_fixture_key_prices_teams_and_draws_nothing():
    """The fallback `coverage` already had, said once. Without `game_id` a fixture cannot be
    identified, so every priced team is its own pickable option -- the old row count -- and
    nothing at all is drawable, which is why `pool.weeks_from_grid` refuses such a grid."""
    g = pl.DataFrame({"week": [1, 1, 1], "team": ["KC", "LV", "SF"],
                      "win_prob": [0.8, 0.2, 1e-9]})
    f = survivor.week_fixtures(g, [1])[0]
    assert f.pickable == ("KC", "LV") and f.drawable == () and f.half == ()
    assert survivor.coverage(g, [1]).covered == [1]


# --- the grid is rated where no live price exists (#218) ----------------------------------
#
# The rule is `hub.models.quarterback`'s and the seam is `hub.models.ratings.rated_games`;
# what is held here is that survivor reads the same seam, so a game the weekly prediction
# adjusts is the same game, adjusted the same way, in the survivor plan.

def _qb_state(cache):
    import json
    from pathlib import Path

    from hub.fetch import nfeloqb
    rows = json.loads((Path(__file__).resolve().parents[1] / "golden" / "fixtures"
                       / "nfeloqb_qb_elos.synthetic.json").read_text())
    cols = list(rows[0])
    text = "\n".join([",".join(cols)] + [",".join("" if r[c] is None else str(r[c]) for c in cols)
                                         for r in rows]) + "\n"
    (cache / "nfeloqb").mkdir(parents=True)
    (cache / "nfeloqb" / nfeloqb.FILE).write_text(text)


def test_a_week_priced_only_by_the_moving_field_is_rated_from_the_quarterback_state(tmp_path,
                                                                                    monkeypatch):
    """LV's starter in the fixture is a fresh backup; the game is priced from the moving
    field, which nothing polls. The away side is worse by 13.2 points less its own gap, and
    KC's win probability rises to say so. The row carries the adjustment and its source."""
    import hub.fetch.nflverse as nflverse
    from hub.models.market import MARGIN_SD, normal_cdf
    sched = pl.DataFrame({"game_id": ["2026_09_LV_KC"], "season": [2026], "week": [9],
                          "home_team": ["KC"], "away_team": ["LV"],
                          "spread_line": [3.0], "result": [None]})
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: sched)
    _qb_state(tmp_path / "cache")
    grid = survivor.grid_from_schedule(2026, cache=tmp_path / "cache",
                                       at=dt.datetime(2026, 9, 12), base=tmp_path)
    p = dict(zip(grid["team"].to_list(), grid["win_prob"].to_list(), strict=True))
    adj = grid.filter(pl.col("team") == "KC")["qb_adjustment"][0]
    assert adj > 0
    assert p["KC"] == pytest.approx(normal_cdf((3.0 + adj) / MARGIN_SD))
    assert grid["adjusted_by"].to_list() == ["nfeloqb", "nfeloqb"]


def test_survivor_and_the_weekly_prediction_adjust_a_game_the_same_way(tmp_path, monkeypatch):
    """The claim the seam exists for, asserted across the two readers."""
    import hub.fetch.nflverse as nflverse
    from hub.models import ratings
    sched = pl.DataFrame({"game_id": ["2026_09_LV_KC"], "season": [2026], "week": [9],
                          "home_team": ["KC"], "away_team": ["LV"],
                          "spread_line": [3.0], "result": [None]})
    monkeypatch.setattr(nflverse, "load", lambda *a, **k: sched)
    _qb_state(tmp_path / "cache")
    at = dt.datetime(2026, 9, 12)
    grid = survivor.grid_from_schedule(2026, cache=tmp_path / "cache", at=at, base=tmp_path)
    preds = ratings.fit(2026, 9, at=at, base=tmp_path, cache=tmp_path / "cache")
    home = grid.filter(pl.col("team") == "KC")
    assert home["win_prob"][0] == pytest.approx(preds["home_win_prob"][0])
    assert home["qb_adjustment"][0] == pytest.approx(preds["qb_adjustment"][0])


def test_the_survivor_cli_reports_the_change_once(capsys, monkeypatch):
    g = _grid([(1, "A", 0.7), (1, "B", 0.3)]).with_columns(
        pl.lit("g1").alias("game_id"), pl.lit(3.0).alias("close_spread"),
        pl.lit(2.5).alias("qb_adjustment"), pl.lit("nfeloqb").alias("adjusted_by"))
    monkeypatch.setattr(survivor, "grid_from_schedule", lambda season, cache=None: g)
    assert survivor.main(["--season", "2026", "--weeks", "1"]) == 0
    out = capsys.readouterr().out
    assert "quarterback adjustment: 1 of 1 priced games touched" in out
    assert "2.50 points" in out
