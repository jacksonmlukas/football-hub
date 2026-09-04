"""The Cohort: the rosters a season-level gate is scored on.

Both season-level gates draft twenty rosters a season, in seat 3, over fourteen rounds, with
the market strategy, seeded the same way -- and each built its own. The seed formula was
hand-copied into both, so if either drifted the two gates would silently score different
cohorts while describing themselves as identical harnesses.

The cause was a narrow interface, not carelessness. `backtest.play` returns names and
positions and discards the board indices and the undrafted remainder; the weekly gate's waiver
arm needs both, because a roster that changes week to week cannot be a fixed list of names. So
it could not use `play` and wrote the recipe again -- inside a `main()`, where no test reached
it.

What is asserted here is the recipe, not a golden roster: same ingredients in, same cohort out.
A frozen list of player indices would pass while the seed drifted, as long as it drifted in the
fixture too.
"""
import numpy as np
import polars as pl
import pytest

from hub.draft import cohort as C


def _board(n=200):
    """A board wide enough to fill twelve teams for fourteen rounds -- 168 picks, so a
    narrower one silently produces short rosters rather than failing."""
    pos = [("QB", "RB", "WR", "TE")[i % 4] for i in range(n)]
    return pl.DataFrame({
        "player": [f"P{i:03d}" for i in range(n)],
        "pos": pos,
        "ecr": [float(i + 1) for i in range(n)],
        "adp": [float(i + 1) for i in range(n)],
        "vor": [float(n - i) for i in range(n)],
        "proj_blend": [float(n - i) / 4 for i in range(n)],
    })


def test_a_cohort_is_the_drafts_asked_for():
    got = C.cohort(_board(), 2024, drafts=5)
    assert len(got.rosters) == 5 and len(got.pool) == 5


def test_every_roster_is_the_rounds_asked_for():
    got = C.cohort(_board(), 2024, drafts=3, rounds=14)
    assert {len(r) for r in got.rosters} == {14}


def test_the_same_season_and_seed_give_the_same_cohort():
    """The property the hand-copied formula put at risk."""
    a = C.cohort(_board(), 2024, drafts=4, seed=0)
    b = C.cohort(_board(), 2024, drafts=4, seed=0)
    assert a.rosters == b.rosters and a.pool == b.pool


def test_a_different_seed_gives_a_different_cohort():
    """Otherwise the seed is decoration and the two gates score one draft twenty times."""
    a = C.cohort(_board(), 2024, drafts=4, seed=0)
    b = C.cohort(_board(), 2024, drafts=4, seed=7)
    assert a.rosters != b.rosters


def test_a_different_season_gives_a_different_cohort():
    """The season is in the seed, so a four-season gate is not one cohort scored four times."""
    assert C.cohort(_board(), 2024, drafts=4).rosters != \
        C.cohort(_board(), 2025, drafts=4).rosters


def test_the_recipe_is_its_ingredients_not_a_frozen_answer():
    """The assertion that catches a drifted seed. Run the documented ingredients directly and
    require the cohort to equal them -- a golden list of indices would pass while the formula
    moved, as long as it moved in the fixture too."""
    from hub.draft.backtest import market_strategy
    from hub.draft.optimize import simulate_remaining_draft
    from hub.draft.state import DraftState

    board = _board()
    got = C.cohort(board, 2024, drafts=2, seed=0)
    for k in range(2):
        room = simulate_remaining_draft(
            board, DraftState(taken=[]), my_slot=C.SLOT, teams=C.TEAMS, rounds=C.ROUNDS,
            rng=np.random.default_rng(C.seed_for(0, 2024, k)), my_pick=market_strategy())
        assert got.rosters[k] == [int(i) for i in room[C.SLOT - 1]]


def test_the_pool_is_everyone_nobody_drafted():
    """The waiver arm adds from it, so a drafted player leaking in would let a roster pick up
    someone another team already holds."""
    got = C.cohort(_board(), 2024, drafts=3)
    for k, pool in enumerate(got.pool):
        assert not set(pool) & set(got.rosters[k])
        assert len(set(pool)) == len(pool)


def test_the_pool_excludes_every_seat_not_only_mine():
    """Twelve teams draft; only one roster is scored. A pool that removed just my players
    would offer eleven other rosters' picks as free agents."""
    board = _board()
    got = C.cohort(board, 2024, drafts=1, rounds=14)
    drafted_by_the_room = board.height - len(got.pool[0])
    assert drafted_by_the_room == C.TEAMS * 14


def test_positions_are_carried_so_a_caller_need_not_re_read_the_board():
    got = C.cohort(_board(), 2024, drafts=1)
    assert len(got.pos) == 200
    assert all(p in ("QB", "RB", "WR", "TE") for p in got.pos)


def test_a_null_position_becomes_a_stated_placeholder_not_none():
    """Both gates index positions into a lineup rule that expects strings."""
    board = _board().with_columns(
        pl.when(pl.col("player") == "P000").then(None).otherwise(pl.col("pos")).alias("pos"))
    assert C.cohort(board, 2024, drafts=1).pos[0] == "NA"


@pytest.mark.parametrize("drafts", [0, 1])
def test_a_degenerate_cohort_is_a_shape_not_a_crash(drafts):
    got = C.cohort(_board(), 2024, drafts=drafts)
    assert len(got.rosters) == drafts


def test_the_cohort_is_what_the_gates_used_to_build_for_themselves():
    """The criterion that matters. Both published gate results were measured on the old
    recipe, so a cohort differing from it -- by a round, a seed, a strategy -- would
    invalidate a recorded number without saying so.

    `backtest.play` is what the lineup gate called; it is still here, so the equality can be
    asserted directly rather than against a frozen list that would drift with the fixture."""
    from hub.draft.backtest import market_strategy, play

    board = _board()
    got = C.cohort(board, 2024, drafts=3, seed=0)
    names_at = board["player"].to_list()
    for k in range(3):
        was, was_pos = play(board, market_strategy(), my_slot=C.SLOT, teams=C.TEAMS,
                            rounds=14, rng=np.random.default_rng(0 + 1000 * 2024 + k))
        assert [names_at[i] for i in got.rosters[k]] == was
        assert [got.pos[i] for i in got.rosters[k]] == was_pos
