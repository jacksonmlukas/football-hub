"""The gate-side assembly, run against the frozen archive rather than described.

`hub.season.weekly_gate_data.assemble_universe` turns a Panel, a board and a **Cohort** into
the nine aligned collections `hub.season.weekly_gate` scores. Until this file its only mention
anywhere under `tests/` was one entry in a naming lint -- the statistics it feeds were tested
thoroughly on hand-built matrices, and the step that *builds* those matrices was not tested at
all. That is the shape issue #108 was filed about: pure functions extracted for testability
while the defect lives in how they are called.

The specific thing that only running can show is the **join failure**. `VOID_FLOOR` voids a
run when too many roster-weeks are a player the consensus page did not list who scored anyway.
It was written when that error was *directional* -- an unmatched player was ranked last, which
benched the incumbent's arm rather than adding noise, biased toward us. Since #206 neither arm
scores a player consensus cannot price, so the same cell is a hole in the covered share
instead: a weaker fault, the same floor, and the reason the floor is still worth having is
that a run answering for one slate while naming another is the failure either way. `coverage`
and `verdict` are tested on hand-built inputs next door in `test_weekly_gate.py`; what was
missing is that a real assembly can produce such a cell at all. Two of the archive's sixteen
players carry one, observed: Michael Pittman in 2024 week 6 and Gus Edwards in 2024 week 10.

`tests/panelarchive.py` says what the archive holds and what it is a capture of.
"""
import numpy as np
import panelarchive as arc
import polars as pl
import pytest

from hub.league import REG_SEASON_WEEKS
from hub.models import experiment
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

    Four roster-weeks of 3,080 -- 0.130% -- against a pre-registered floor of 2%. The number
    is a property of this capture and its draw, not of production; what it establishes is that
    the quantity the floor reads is one a real assembly can make non-zero. A version of this
    that could only ever report zero would leave the floor guarding nothing.

    **Restated 2026-09-07 (#150): three roster-weeks, 0.097%, until the pick-noise refit.**
    The draw is the thing that moved. `assemble_universe` simulates twenty drafts through
    `availability.pick_noise`, and refitting it -- `1.00 + 0.253*pick` to `1.31 + 0.169*pick`,
    a narrower room -- put Gus Edwards on three of the twenty rosters where he had been on two.
    Michael Pittman Jr. is still on one. Same two players, one more roster-week, and the two
    the parametrized test above names are unchanged. Nothing about the floor moves: 0.130% is
    still non-zero and still an order of magnitude under 2%.

    `unranked` is deliberately not asserted as a rate. The board carries 200 players and the
    archive's consensus page carries 16 of them, so 88% of cells are unranked here against
    roughly 64% in production -- an artefact of the trim, recorded in the fixtures README. The
    two are separate numbers in `coverage` precisely because conflating them would either void
    every run or none.
    """
    cover = coverage(universe)
    assert cover["cells"] == 3080.0
    assert cover["join_failure"] * cover["cells"] == pytest.approx(4.0), \
        "four roster-weeks where the page had no row and the player played"
    assert 0 < cover["join_failure"] < VOID_FLOOR, \
        "non-zero, so the floor has something to read; under it, so this capture is not void"


def test_the_arm_under_test_falls_back_to_the_incumbent_where_it_has_no_projection(universe):
    """A player the Weekly projection cannot price is scored at the points its own week's
    paired observations carry at his rank, and never as a zero, a NaN or a negated rank.

    **What this column is for changed with #206 and what it contains did not.** It was the
    arm under test, and the argument for filling those cells at all was that benching them
    would hand the incumbent information the arm lacks. Both arms now decline to score them
    together -- `weekly_gate.priced_by_both` -- so nothing below is on a path the verdict is
    read off. It is kept, and kept under test, because two of the three treatments the gate
    still scores as a check are rebuilt from this column, and because the published -1.004
    was measured on it: a column that had already collapsed the fallback to `UNRANKED` would
    leave that figure re-derivable from nothing but the page that carries it.

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

    # The fallback is still a fallback in this column -- a player the model cannot price
    # carries a number rather than a bench. It stopped being a lineup rule with #206, where
    # both arms decline those cells together; what it still is, is the thing the other two
    # treatments are rebuilt from. What changed with #44 is its *units*: it used to be the
    # incumbent's own number, negated ECR, sitting in a column of fantasy points, so every
    # projected player outranked every unprojected one whatever either was worth.
    ranked_fallback = fallback & (cons > G.UNRANKED)
    assert ranked_fallback.any(), "no ranked-but-unprojected cell, so nothing to check"
    assert (weekly[ranked_fallback] > G.UNRANKED).all(), (
        "the interpolation is gone from the column, and with it the only reconstruction of "
        "the treatment the published -1.004 was measured under")
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


# --- the Board this assembly is handed (issue #131) ------------------------
#
# Both entry points here reached the Board with `board_as_of(yr)[0]`, so a season built while
# a Correction stage was absorbed arrived indistinguishable from a whole one. The rule is
# `hub.models.experiment.require_corrections` and the reason to refuse rather than record is
# written there; these two hold that Gate B actually asks.


def _short_board(monkeypatch, missing_flag: str) -> None:
    """Serve the frozen board with a report saying one correction stage did not run."""
    import hub.draft.board as brd
    report = brd.BuildReport(adp=True, td_luck=True, durability=True)
    setattr(report, missing_flag, False)
    monkeypatch.setattr(brd, "board_as_of",
                        lambda season: (arc.frame("draft_board"), report))


def test_the_assembly_refuses_a_board_whose_corrected_ranking_lost_a_term(monkeypatch,
                                                                         tmp_path):
    """The **Cohort** every season-level gate is scored on is drafted from this board, so a
    board short a term is a season drafted from a different ranking -- not a thinner one."""
    arc.install(monkeypatch, tmp_path, board=True)
    _short_board(monkeypatch, "durability")
    with pytest.raises(experiment.CorrectionMissing, match="durability"):
        wgd.assemble_universe(arc.SEASONS)


def test_the_preseason_ranks_refuse_the_same_board(monkeypatch, tmp_path):
    """This one reads `ecr` alone, which no Correction moves, and it refuses anyway.

    The shrinkage target it builds is scored in the same run that drafts a Cohort from the
    same board, so the question is not whether this column is affected but whether the run
    should start. Failing at the first board that is short a term costs an operator one build
    rather than four and a discarded interval.

    Written against `td_luck` until #48 emptied its coefficients. It asks `durability` now,
    which is what a Correction still means -- see the test below for the other half.
    """
    arc.install(monkeypatch, tmp_path, board=True)
    _short_board(monkeypatch, "durability")
    with pytest.raises(experiment.CorrectionMissing, match="durability"):
        wgd.preseason_ranks([2024])


def test_a_board_without_touchdown_luck_is_no_longer_refused(monkeypatch, tmp_path):
    """The other half of #48, and the reason the test above had to move.

    `require_corrections` refuses a Board whose *ranking* was computed from a subset of the
    Corrections, because a season drafted from a different ranking is a different arm rather
    than a thinner one. #186 found the touchdown-luck price loses to charging nothing held
    out, and #48 emptied `TD_LUCK_BETA` -- so the term multiplies nothing, its absence moves
    no ranking, and refusing on it would stop a run over a column that cannot change the
    answer.

    The signal itself is not withdrawn: `td_luck` stays on the board and in the report, shown
    and not priced, which is ADR-0013's shape. This asserts the refusal followed the price
    rather than the column.
    """
    arc.install(monkeypatch, tmp_path, board=True)
    _short_board(monkeypatch, "td_luck")
    got = wgd.preseason_ranks([2024])
    assert got.height > 0, "a board short only touchdown luck produced no preseason ranks"


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


# --- and the assembly says which cells those were (issue #207) --------------

def test_the_assembly_carries_the_mask_that_says_which_cells_are_the_models_own(universe):
    """`weekly` is a mixture and the arithmetic that made it is thrown away at the return.

    Until #207 nothing left this module saying which cells were the model's own number and
    which were the fallback, so the share of each -- and any re-scoring under a different
    treatment of the fallback -- had to be reconstructed by probing the column from outside.
    The mask is re-derived here from the Panel the assembly built, the same way the fallback
    test above does it, rather than read off `addable`: that mask is `projected` *and ranked*,
    so checking one against the other would compare the fallback with itself.
    """
    from hub.models.panel import PanelSpec, build_panel
    panel = build_panel(arc.SEASONS, PanelSpec(consensus=False))
    priced = {(r["key"], r["week"]) for r in
              panel.filter(pl.col("season") == 2024).select("key", "week").iter_rows(named=True)}
    board = arc.frame("draft_board")
    keys = [player_key(p) for p in board["player"].to_list()]
    want = np.array([[(k, w) in priced for w in range(1, REG_SEASON_WEEKS + 1)]
                     for k in keys])

    got = universe.projected[2024]
    assert got.dtype == np.dtype(bool)
    assert got.shape == universe.weekly[2024].shape
    assert want.any() and not want.all(), "both halves have to exist to compare them"
    assert np.array_equal(got, want)
    # The mask and the column agree about what the model priced, which is the join the
    # mixture and every treatment of the fallback are counted across.
    assert (universe.weekly[2024][got] > G.UNRANKED).all()
    assert (universe.addable[2024] & ~got).sum() == 0, \
        "addable is the other half of the same isnan and stays that way"


def test_the_three_treatments_are_reachable_from_a_real_assembly(universe):
    """The end of the path #207 opens, driven offline: a real `GateInputs` splits into three
    shares that sum to one and re-scores under all three treatments of its fallback.

    The published figures -- 54.2 / 16.7 / 29.1 and a 1.5-point spread -- are properties of the
    production run, not of this capture: the archive trims the board to 200 players and the
    consensus page to 16 of them, which is the same artefact the join-failure test above
    records for `unranked`. What is asserted here is that the quantities exist on a real
    assembly and are the three the gate names, so the shares the run prints are counted off
    the same column the verdict is read from.
    """
    mix = G.mixture(universe)
    assert mix["cells"] == coverage(universe)["cells"], "one universe, both blocks"
    assert (mix["projection"] + mix["fallback"] + mix["unscoreable"]) == pytest.approx(1.0)
    assert all(mix[k] > 0 for k in ("projection", "fallback", "unscoreable")), \
        "all three groups are non-empty here, so the treatments have something to disagree on"

    weekly = universe.weekly[2024]
    stripped = G.under_treatment(universe, "unscoreable").weekly[2024]
    mixed = G.under_treatment(universe, "mixed scale").weekly[2024]
    fallback = ~universe.projected[2024]
    assert (stripped[fallback] == UNRANKED).all()
    assert (mixed[fallback] == universe.consensus[2024][fallback]).all()
    assert np.array_equal(stripped[~fallback], weekly[~fallback])
    assert np.array_equal(mixed[~fallback], weekly[~fallback])


def test_the_fallback_reaches_no_result_on_a_real_assembly(universe):
    """#206 driven end to end offline, and **this capture is the case the guard is for.**

    The unit tests next door prove the restriction on fixtures whose arithmetic is on paper.
    What running it here found is the failure mode restricting the rows creates, arriving
    unprompted on real inputs: the archive's consensus page carries 16 of the board's 200
    players, so only **9.4%** of roster-week cells are priced by both arms, every roster's
    fieldable side fits inside the starting slots, and the two arms field the identical team
    in **every** scored roster-week. The effect is +0.000 -- not measured, forced.

    The share was 10.1% when this was written on 2026-09-10 and is 9.4% since #155 restated
    the pick-noise law on 2026-09-11. It moves because the Cohort this gate scores is drafted
    with pick noise, so a different sigma drafts different rosters and a different fraction of
    them is priceable -- the figure is a property of the archive *and* the draft simulator,
    not of the archive alone. Pinned to the current value so the next simulator change is
    noticed here rather than absorbed; the claim the test holds is the one below it, that the
    arms never disagree, and that has not moved.

    That is `lineup_gate`'s structural zero (ADR-0012) arriving down a different road, and it
    is exactly what a run must not report as a result. So the shares are asserted, and so is
    the loud line: a capture this thin is the right place to require the gate to say it could
    not have failed, because a production run would not fire it and nothing else would.

    The unrestricted arm still separates here (+0.075), which is what makes the zero above a
    property of the restriction rather than of a capture where nothing separates at all.
    Every figure is a property of this trim and none is comparable with production -- the same
    artefact the join-failure test records.
    """
    pop = G.priced_share(universe)
    assert pop["cells"] == coverage(universe)["cells"], "one universe, every block"
    assert pop["share"] == pytest.approx(0.0945, abs=5e-4), "9.4% of cells, both arms"
    assert pop["no_projection"] > 0.0 and pop["no_rank"] > 0.0

    means = [e.summary["mean"] for e in G.treatment_effects(universe, seed=0)]
    assert len(means) == 3 and not any(np.isnan(m) for m in means)
    assert max(means) - min(means) == 0.0, (
        "a treatment reached a scored cell: the restriction missed an arm, a week or the "
        "waiver rule, and the effect is once again partly the fallback")

    assert pop["forced"] == 1.0, "no scored roster-week here leaves either arm a choice"
    said = " ".join(" ".join(G.priced_report(pop)).split())
    assert "THE ARMS NEVER DISAGREE" in said
    assert "zero by construction and this run could not have failed" in said
    assert means == [pytest.approx(0.0)] * 3, "which is the number it is warning about"

    loose = [e.summary["mean"] for e in G.treatment_effects(universe, seed=0, restrict=False)]
    assert loose == [pytest.approx(0.075)] * 3, (
        "the pre-#206 universe separates the arms on this capture, so the forced zero above "
        "is the restriction and not a trim where nothing could ever separate")
