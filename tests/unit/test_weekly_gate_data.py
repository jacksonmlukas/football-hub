"""The gate-side assembly, run against the frozen archive rather than described.

`hub.season.weekly_gate_data.assemble_universe` turns a Panel, a board and a **Cohort** into
the nine aligned collections `hub.season.weekly_gate` scores. Until this file its only mention
anywhere under `tests/` was one entry in a naming lint -- the statistics it feeds were tested
thoroughly on hand-built matrices, and the step that *builds* those matrices was not tested at
all. That is the shape issue #108 was filed about: pure functions extracted for testability
while the defect lives in how they are called.

The specific thing that only running can show is the **join failure**. `VOID_FLOOR` voids a
run when too many roster-weeks are a player the consensus page did not list who scored anyway,
because an unmatched player is ranked last and that benches the incumbent's arm rather than
adding noise -- a directional error, biased toward us. `coverage` and `verdict` are tested on
hand-built inputs next door in `test_weekly_gate.py`; what was missing is that a real assembly
can produce such a cell at all. Two of the archive's sixteen players carry one, observed:
Michael Pittman in 2024 week 6 and Gus Edwards in 2024 week 10.

`tests/panelarchive.py` says what the archive holds and what it is a capture of.
"""
import numpy as np
import panelarchive as arc
import polars as pl
import pytest

from hub.league import REG_SEASON_WEEKS
from hub.names import player_key
from hub.season import weekly_gate as G
from hub.season import weekly_gate_data as wgd
from hub.season.weekly_gate import UNRANKED, VOID_FLOOR, coverage

# The two observed join failures, and what each player really scored that week. A cell is a
# join failure when the consensus page did not list him *and* he played, which is the pair
# `coverage` counts and the only thing `VOID_FLOOR` is measured against.
UNLISTED_BUT_SCORED = [("michael pittman", 6, 12.5), ("gus edwards", 10, 5.5)]


@pytest.fixture
def universe(monkeypatch, tmp_path):
    """One assembly, at the recipe the gate actually runs: twenty drafts, seed 0.

    Not a smaller number for speed. Twenty drafts a season is `hub.draft.cohort`'s recipe and
    what every published result was measured on, and the share of roster-weeks lost to a join
    failure is a property of that draw -- reading it off five drafts would be reporting a
    different quantity under the same name.
    """
    arc.install(monkeypatch, tmp_path, board=True)
    return wgd.assemble_universe(arc.SEASONS)


def _board_index(key: str) -> int:
    board = arc.frame("draft_board")
    return [player_key(p) for p in board["player"].to_list()].index(key)


def test_the_gate_inputs_are_assembled_end_to_end_with_no_network(universe):
    """The premise every assertion below rests on: it runs, offline, and returns real shapes.

    The board is 200 players and the fantasy regular season is fourteen weeks, so every arm
    is scored on the same (universe, weeks) grid -- which is what lets a roster be a list of
    indices that changes week to week rather than a fixed matrix.
    """
    assert sorted(universe.realised) == [2024], \
        "2023 is the past `expanding_seasons` fits on and is never itself scored"
    yr = 2024
    shape = (arc.frame("draft_board").height, REG_SEASON_WEEKS)
    for grid in (universe.realised[yr], universe.consensus[yr], universe.weekly[yr],
                 universe.se[yr], universe.addable[yr]):
        assert grid.shape == shape
    assert len(universe.rosters[yr]) == 20, "twenty drafts, the recipe both gates share"
    assert all(len(r) == 14 for r in universe.rosters[yr]), "fourteen rounds each"
    assert len(universe.pos[yr]) == shape[0], "a position per universe index, not per roster"


def test_the_weeks_the_page_covers_come_from_the_page_and_not_from_the_calendar(universe):
    """`covered_weeks` is what stops a week the incumbent never ranked being scored as a loss
    for it. The archive's consensus page starts at 2023 week 2 and 2024 week 4, so those two
    gaps are real and the set has to show them."""
    assert (2024, 4) in universe.covered and (2024, 3) not in universe.covered
    assert (2023, 2) in universe.covered and (2023, 1) not in universe.covered
    assert len(universe.covered) == 24


@pytest.mark.parametrize(("key", "week", "points"), UNLISTED_BUT_SCORED)
def test_a_player_the_page_did_not_list_is_ranked_last_and_still_scores(universe, key, week,
                                                                       points):
    """The join failure, as the assembly actually produces it -- not as a matrix written by
    hand to look like one.

    Both cells are observed: FantasyPros' cross-position weekly page carried neither player
    that week, and both played. The incumbent's arm therefore sees `UNRANKED`, which sorts
    behind every listed player, while the realised grid scores the points he really put up.
    """
    i = _board_index(key)
    assert universe.consensus[2024][i, week - 1] == UNRANKED, \
        "the page did not list him, so the incumbent ranks him behind everyone"
    assert universe.realised[2024][i, week - 1] == pytest.approx(points), \
        "and he scored anyway, which is what makes it a failure rather than an omission"
    assert not universe.addable[2024][i, week - 1], \
        "a cell only one arm can score is out of the waiver pool, not silently offered to us"


def test_the_join_failure_the_floor_guards_against_is_reachable_from_the_assembly(universe):
    """What `VOID_FLOOR` is measured against, reached by running the assembly.

    Three roster-weeks of 3,080 -- 0.097% -- against a pre-registered floor of 2%. The number
    is a property of this capture and its draw, not of production; what it establishes is that
    the quantity the floor reads is one a real assembly can make non-zero. A version of this
    that could only ever report zero would leave the floor guarding nothing.

    `unranked` is deliberately not asserted as a rate. The board carries 200 players and the
    archive's consensus page carries 16 of them, so 88% of cells are unranked here against
    roughly 64% in production -- an artefact of the trim, recorded in the fixtures README. The
    two are separate numbers in `coverage` precisely because conflating them would either void
    every run or none.
    """
    cover = coverage(universe)
    assert cover["cells"] == 3080.0
    assert cover["join_failure"] * cover["cells"] == pytest.approx(3.0), \
        "three roster-weeks where the page had no row and the player played"
    assert 0 < cover["join_failure"] < VOID_FLOOR, \
        "non-zero, so the floor has something to read; under it, so this capture is not void"


def test_the_arm_under_test_falls_back_to_the_incumbent_where_it_has_no_projection(universe):
    """A player the Weekly projection cannot price is not scored as a zero -- he is scored at
    the incumbent's own number, so neither arm is handed information the other lacks. That is
    the defect that made the first `lineup_gate` unable to fail, and it lives in one `np.where`
    inside this assembly rather than anywhere in the gate.

    Which cells those are is re-derived from the Panel the assembly itself built, rather than
    read off `addable` -- that mask is the *other* half of the same `np.isnan`, so reading it
    here would compare the fallback against itself.
    """
    from hub.models.panel import PanelSpec, build_panel
    panel = build_panel(arc.SEASONS, PanelSpec(consensus=False))
    priced = {(r["key"], r["week"]) for r in
              panel.filter(pl.col("season") == 2024).select("key", "week").iter_rows(named=True)}
    board = arc.frame("draft_board")
    keys = [player_key(p) for p in board["player"].to_list()]
    fallback = np.array([[(k, w) not in priced for w in range(1, REG_SEASON_WEEKS + 1)]
                         for k in keys])

    weekly, cons, se = universe.weekly[2024], universe.consensus[2024], universe.se[2024]
    assert fallback.any() and (~fallback).any(), "both halves have to exist to compare them"

    # The fallback is still a fallback -- a player the model cannot price is scored, not
    # benched, or the arm under test would be handed a smaller universe than the incumbent
    # and the gate could not fail. What changed with #44 is its *units*: it used to be the
    # incumbent's own number, negated ECR, sitting in a column of fantasy points, so every
    # projected player outranked every unprojected one whatever either was worth.
    ranked_fallback = fallback & (cons > G.UNRANKED)
    assert ranked_fallback.any(), "no ranked-but-unprojected cell, so nothing to check"
    assert (weekly[ranked_fallback] > G.UNRANKED).all(), (
        "a player the model cannot price was benched rather than scored, so the two arms no "
        "longer see the same universe")
    # One scale: the fallback lands inside the range of the projections it sits beside.
    priced = weekly[~fallback]
    assert weekly[ranked_fallback].min() >= priced.min() - 1e-9
    assert weekly[ranked_fallback].max() <= priced.max() + 1e-9
    assert not np.isnan(weekly).any(), "and never a NaN, which no lineup search can order"
    assert se[fallback].max() == 0.0, \
        "no projection means no standard error, so a lower bound moves nothing"
    assert not np.array_equal(weekly[~fallback], cons[~fallback]), (
        "where it does have a projection the two arms are different numbers, or there is no "
        "comparison for the gate to make")


def test_the_panel_behind_the_gate_keeps_the_players_consensus_never_ranked(monkeypatch,
                                                                           tmp_path):
    """`assemble_universe` asks for `consensus=False`, and that is not a detail of taste.

    Being unranked is the incumbent's *answer*, so the gate needs a projection for a rostered
    player the page leaves out -- and the two cells above are exactly such players. Built with
    the screen's spec instead, they would not be on the Panel at all and the failure this file
    is about would be invisible rather than measured.
    """
    arc.install(monkeypatch, tmp_path)
    from hub.models.panel import PanelSpec, build_panel
    gate_side = build_panel(arc.SEASONS, PanelSpec(consensus=False))
    for key, week, _pts in UNLISTED_BUT_SCORED:
        rows = gate_side.filter((pl.col("key") == key) & (pl.col("season") == 2024)
                                & (pl.col("week") == week))
        assert rows.height == 1, f"{key} week {week} has to reach the gate's Panel"
    screen_side = build_panel(arc.SEASONS)
    assert screen_side.filter((pl.col("key") == UNLISTED_BUT_SCORED[0][0])
                              & (pl.col("season") == 2024)
                              & (pl.col("week") == UNLISTED_BUT_SCORED[0][1])).is_empty()


# --- the treatment arm scores on one scale (issue #44) ---------------------

def test_the_fallback_is_neither_above_nor_below_every_projected_player():
    """Both directions of the same defect.

    The old column put negated ECR beside fantasy points, so every projected player outranked
    every unprojected one -- a projected two-point scrub above a rank-1 star. The obvious
    naive fix, using ECR unnegated, inverts it. Neither is a scale; both are an ordering by
    which column a player arrived in.
    """
    cons = np.array([[-1.0], [-5.0], [-50.0], [-200.0]])       # negated ECR, best first
    mu = np.array([[np.nan], [20.0], [8.0], [np.nan]])          # two projected, two not
    got = wgd._one_scale(cons, mu)[:, 0]

    priced = got[[1, 2]]
    imputed = got[[0, 3]]
    assert not (imputed > priced.max()).all(), "the fallback sorts above every projection"
    assert not (imputed < priced.min()).all(), "the fallback sorts below every projection"
    # And it is monotone in rank: the rank-1 player is scored at least as well as rank-200.
    assert got[0] >= got[3]


def test_the_fallback_is_scored_at_the_points_its_neighbours_carry():
    """The calibration, stated as a number. A player the model cannot price, ranked exactly
    between two it can, is scored between their projections -- reading off nothing but the
    week's own paired observations."""
    cons = np.array([[-10.0], [-20.0], [-30.0]])
    mu = np.array([[18.0], [np.nan], [6.0]])
    got = wgd._one_scale(cons, mu)[:, 0]
    assert got[0] == 18.0 and got[2] == 6.0, "a projected player keeps his own projection"
    assert 6.0 < got[1] < 18.0
    assert got[1] == pytest.approx(12.0), "halfway in rank is halfway in points here"


def test_a_week_with_nothing_to_calibrate_against_leaves_the_player_unscoreable():
    """Guessing from no paired observation would be inventing a number and calling it a
    projection. Unscoreable is the honest answer, and it is what consensus already says about
    a player its own page does not list."""
    cons = np.array([[-10.0], [-20.0]])
    mu = np.array([[np.nan], [np.nan]])
    got = wgd._one_scale(cons, mu)[:, 0]
    assert (got == G.UNRANKED).all()
