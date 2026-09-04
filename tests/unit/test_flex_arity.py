"""One roster shape, read by every lineup rule.

`config.py` says the shape used to live in five places and "nothing made them agree". The
unification reached `STARTERS`, `FLEX_FROM` and `FLEX_CAPACITY` and stopped one constant
short: `FLEX_SLOTS` was exported and read by nobody, while four separate implementations of
"best legal lineup" each hardcoded exactly one flex.

The consequence was not duplication, it was disagreement. `optimize._need_score` reads
`FLEX_CAPACITY`, which moves 7 -> 8 when a second flex is added, so the draft would have
stopped calling for flex-eligible players at eight while every lineup score in the repo kept
fielding seven.

None of the assertions below could be written before: the flex count was a literal.

All offline.
"""
import numpy as np
import polars as pl

from hub import league
from hub.draft import evaluate, season
from hub.season import lineup, lineup_gate

# QB 20 | RB 10 9 8 | WR 7 6 5 4 | TE 3 2   with STARTERS QB1 RB2 WR3 TE1
#   starters  = 20 + (10+9) + (7+6+5) + 3 = 60
#   leftovers = RB 8, WR 4, TE 2
POS = ["QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "TE"]
PTS = [20.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
BASE, BEST, SECOND = 60.0, 8.0, 4.0


def test_the_vectorised_scorer_fields_the_configured_flex(monkeypatch):
    scores = np.array(PTS, dtype=float).reshape(1, 1, len(PTS))
    pos = np.array(POS)
    monkeypatch.setattr(season, "FLEX_SLOTS", 1)
    assert season.lineup_points(scores, pos)[0, 0] == BASE + BEST
    monkeypatch.setattr(season, "FLEX_SLOTS", 2)
    assert season.lineup_points(scores, pos)[0, 0] == BASE + BEST + SECOND


def test_the_greedy_season_scorer_fields_the_configured_flex(monkeypatch):
    roster = pl.DataFrame({"pos": POS, "actual_points": PTS})
    monkeypatch.setattr(evaluate, "FLEX_SLOTS", 1)
    assert evaluate.starter_points(roster) == BASE + BEST
    monkeypatch.setattr(evaluate, "FLEX_SLOTS", 2)
    assert evaluate.starter_points(roster) == BASE + BEST + SECOND


def test_the_projection_baseline_fields_the_configured_flex(monkeypatch):
    """Patched on `league`, not on `lineup_gate`, because the flex count is read in exactly
    one place: `league.starting_lineup`, which both gates call. Before, each gate carried its
    own copy of the selection loop and so its own read of the constant.

    Patched on `league` rather than `draft.season` since 2026-09-04: the rule moved to a leaf
    so `hub.season` would stop importing a draft simulator to learn the roster shape. This
    test caught the move -- patching the re-exported name no longer reached the function,
    which is the correct failure and the reason the guard is worth having."""
    grid = np.array(PTS, dtype=float).reshape(len(PTS), 1)
    monkeypatch.setattr(league, "FLEX_SLOTS", 1)
    assert lineup_gate.projection_lineup_points(grid, POS, PTS) == BASE + BEST
    monkeypatch.setattr(league, "FLEX_SLOTS", 2)
    assert lineup_gate.projection_lineup_points(grid, POS, PTS) == BASE + BEST + SECOND


def test_the_exhaustive_enumerator_fills_every_flex_slot():
    slots = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    one = list(lineup._legal_lineups(POS, slots, ("RB", "WR", "TE"), 100_000, flex_slots=1))
    two = list(lineup._legal_lineups(POS, slots, ("RB", "WR", "TE"), 100_000, flex_slots=2))
    assert {len(t) for t in one} == {8}, "seven starters plus one flex"
    assert {len(t) for t in two} == {9}, "seven starters plus two flex"
    assert all(len(set(t)) == len(t) for t in two), "a player cannot fill two slots"


def test_two_flex_can_come_from_one_position():
    """The reason this is combinations and not one-candidate-per-position. With two flex
    slots the best pair may both be running backs, which the old shape could not express."""
    pos = ["QB", "RB", "RB", "RB", "RB", "WR", "WR", "WR", "TE"]
    slots = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    got = list(lineup._legal_lineups(pos, slots, ("RB", "WR", "TE"), 100_000, flex_slots=2))
    rb_idx = {1, 2, 3, 4}
    assert any(len(set(t[7:]) & rb_idx) == 2 for t in got)


def test_the_greedy_rule_answers_to_the_configured_flex(monkeypatch):
    """`FLEX_SLOTS` was exported by `season.py` and read by nobody, which is how four
    implementations came to hardcode the number instead.

    Asserted on **behaviour** rather than on whether a module mentions the constant. The
    source-text version of this guard failed the moment both gates stopped carrying their own
    selection loop and started calling `league.starting_lineup` -- which is the fix, not a
    regression. The enumerator has its own test above; this is the greedy rule.
    """
    for flex in (1, 2):
        monkeypatch.setattr(league, "FLEX_SLOTS", flex)
        got = league.starting_lineup(POS, PTS)
        assert len(got) == sum(league.STARTERS.values()) + flex


def test_the_flex_count_is_read_where_a_rule_is_implemented():
    """The list is shorter than it was, and that is the point: `lineup_gate` and `weekly_gate`
    are off it because they no longer implement the rule -- they call the one that does."""
    import inspect
    for mod in (season, evaluate, lineup):
        assert "FLEX_SLOTS" in inspect.getsource(mod), f"{mod.__name__} ignores it"
    for mod in (lineup_gate,):
        assert "starting_lineup" in inspect.getsource(mod), \
            f"{mod.__name__} must delegate rather than reimplement"


def test_the_draft_and_the_lineup_agree_about_capacity():
    """`FLEX_CAPACITY` already adapted and the scorers did not. That disagreement is the
    defect: the draft would stop calling for flex-eligible players at eight while every
    lineup score still fielded seven."""
    from hub.config import RosterConfig, flex_capacity, required_starters
    for flex in (1, 2):
        cfg = RosterConfig(flex=flex)
        assert flex_capacity(cfg) == sum(required_starters(cfg).values()) - cfg.qb + flex
