"""Championship-equity ranking.

The properties worth pinning are the ones that make this different from VOR: it must
respond to roster construction and to who is already gone, and it must return a real
probability rather than a score.
"""
import numpy as np
import polars as pl
import pytest
from hub.draft import optimize
from hub.draft.optimize import (market_pick, rank_tiers,
                                simulate_remaining_draft,
                                tag_for,
                                win_probability, _need_score)
from hub.draft.season import STARTERS
from hub.draft.state import DraftState, take


def _board(n=180):
    pos_cycle = ["RB", "WR", "WR", "TE", "QB"]
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": [pos_cycle[i % len(pos_cycle)] for i in range(n)],
        "ecr": [float(i + 1) for i in range(n)],
        "sd": [3.0] * n,
        "adp": [float(i + 1) for i in range(n)],
        "vor": [float(n - i) for i in range(n)],
        "xfp_per_game": [max(0.0, 20.0 - i * 0.1) for i in range(n)],
    })


# --- need scoring ---------------------------------------------------------

def test_empty_starting_slot_outranks_surplus():
    empty = {"QB": 1, "RB": 2, "WR": 3}          # no TE yet
    assert _need_score(empty, "TE") > _need_score(empty, "WR")


def test_surplus_beyond_bench_depth_scores_zero():
    saturated = {"RB": 4, "WR": 5, "TE": 2}
    assert _need_score(saturated, "WR") == 0


# --- draft simulation -----------------------------------------------------

def test_every_team_is_filled_and_nobody_is_drafted_twice():
    rosters = simulate_remaining_draft(_board(), DraftState(), my_slot=3, teams=12, rounds=8)
    assert len(rosters) == 12
    allp = np.concatenate(rosters)
    assert allp.size == len(set(allp.tolist())), "a player was drafted twice"


def test_forced_candidate_lands_on_my_roster():
    board = _board()
    rosters = simulate_remaining_draft(board, DraftState(), my_slot=3, teams=12,
                                       rounds=6, forced="P42")
    mine = [board["player"][int(i)] for i in rosters[2]]
    assert "P42" in mine


def test_already_drafted_players_are_never_drafted_again():
    """They now appear once, on the seat that took them -- see the seeding test below --
    but nobody may take them a second time."""
    board = _board()
    st = take(DraftState(), *[f"P{i}" for i in range(20)])
    rosters = simulate_remaining_draft(board, st, my_slot=3, teams=12, rounds=6)
    picked = [board["player"][int(i)] for r in rosters for i in r]
    assert len(picked) == len(set(picked)), "a player was drafted twice"


def test_every_seat_starts_from_the_roster_it_already_holds():
    """The defect this shape fixes. Rosters used to contain only picks made *during* the
    simulation, so championship equity could not see what you already owned -- holding a
    quarterback, it ranked a second one above a startable back."""
    from hub.draft.state import roster_for
    board = _board()
    st = take(DraftState(), *[f"P{i}" for i in range(20)])
    rosters = simulate_remaining_draft(board, st, my_slot=3, teams=12, rounds=6)
    for seat in range(1, 13):
        held = set(roster_for(st, seat, 12, 6))
        got = {board["player"][int(i)] for i in rosters[seat - 1]}
        assert held <= got, f"seat {seat} lost the players it already held"


def test_seeding_covers_opponents_not_just_me():
    """Seeding only my seat would leave eleven opponents fielding future picks alone,
    making them weaker than they are and inflating my p_win."""
    from hub.draft.state import roster_for
    board = _board()
    st = take(DraftState(), *[f"P{i}" for i in range(20)])
    rosters = simulate_remaining_draft(board, st, my_slot=3, teams=12, rounds=6)
    other = [s for s in range(1, 13) if s != 3 and roster_for(st, s, 12, 6)]
    assert other, "fixture should have opponents holding players"
    for seat in other:
        got = {board["player"][int(i)] for i in rosters[seat - 1]}
        assert set(roster_for(st, seat, 12, 6)) <= got


def test_a_recorded_pick_not_on_the_board_is_skipped():
    """K and DST are drafted but deliberately off the board, and a typed pick can be
    misspelled. Raising here would make the simulator unusable from round 13 of every real
    draft; `suggest_unmatched` already flags a misspelling where a human can fix it."""
    board = _board()
    st = take(DraftState(), "P0", "Some Kicker", "P1")
    rosters = simulate_remaining_draft(board, st, my_slot=3, teams=12, rounds=6)
    assert sum(len(r) for r in rosters) > 0


def test_my_greedy_fills_its_starting_slots():
    """A roster that never drafts a QB cannot be a serious opponent model."""
    board = _board()
    rosters = simulate_remaining_draft(board, DraftState(), my_slot=3, teams=12, rounds=10)
    mine = [board["pos"][int(i)] for i in rosters[2]]
    for p, n in STARTERS.items():
        assert mine.count(p) >= min(n, 1), f"no {p} drafted"


# --- win probability ------------------------------------------------------

def test_returns_a_probability_per_candidate():
    out = win_probability(_board(), DraftState(), ["P0", "P1"], my_slot=3,
                          rounds=8, n_draft_sims=2, n_season_sims=40)
    assert out.height == 2
    assert ((out["p_win"] >= 0) & (out["p_win"] <= 1)).all()


def test_a_far_better_candidate_wins_more():
    """P0 is the best player available; P150 is replacement level."""
    out = win_probability(_board(), DraftState(), ["P0", "P150"], my_slot=3,
                          rounds=10, n_draft_sims=4, n_season_sims=120)
    best = out.filter(pl.col("player") == "P0")["p_win"][0]
    worst = out.filter(pl.col("player") == "P150")["p_win"][0]
    assert best > worst


def test_output_is_sorted_and_carries_a_field_baseline():
    out = win_probability(_board(), DraftState(), ["P100", "P0", "P50"], my_slot=3,
                          rounds=8, n_draft_sims=2, n_season_sims=40)
    assert out["lift"].to_list() == sorted(out["lift"].to_list(), reverse=True)
    assert out["lift"].sum() == pytest.approx(0.0, abs=1e-9)


def test_is_deterministic_under_a_fixed_seed():
    kw = dict(my_slot=3, rounds=8, n_draft_sims=2, n_season_sims=40, seed=7)
    a = win_probability(_board(), DraftState(), ["P0"], **kw)["p_win"][0]
    b = win_probability(_board(), DraftState(), ["P0"], **kw)["p_win"][0]
    assert a == b


def test_the_draft_optimizer_prices_stacks():
    """docs/correlation.md: a quarterback and his own pass catchers move together (+0.232),
    and the simulator was treating a stacked roster as independent -- understating its
    variance in exactly the weeks a stack is for. The pool carries NFL team already, so the
    only thing needed was to pass it through."""
    import inspect as _inspect
    from hub.draft import optimize as _opt
    src = _inspect.getsource(_opt.win_probability)
    assert "nfl_team" in src, "champion_probability must receive NFL team identity"


# --- saying only what the simulation can resolve --------------------------

def test_candidates_the_simulation_cannot_separate_are_marked_as_tied():
    """At pick 3 the top two differ by 0.17 points of championship equity with standard
    errors around 0.5. Printing a strict order there implies a distinction the simulation
    cannot make, and the objective is to pick the best player -- which is sometimes two
    players."""
    df = pl.DataFrame({"player": ["A", "B", "C"], "p_win": [0.48, 0.479, 0.40],
                       "lift": [0.035, 0.033, -0.050],
                       "lift_se": [0.005, 0.005, 0.006]})
    got = rank_tiers(df)
    lead = dict(zip(got["player"].to_list(), got["co_leader"].to_list()))
    assert lead["A"] and lead["B"], "0.002 apart with se 0.005 is not a distinction"
    assert not lead["C"]


def test_a_clear_winner_is_not_diluted_into_a_tie():
    """The other direction. A tiering rule that called everything tied would be as useless
    as one that called everything separable."""
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.50, 0.40],
                       "lift": [0.05, -0.05], "lift_se": [0.004, 0.004]})
    got = rank_tiers(df)
    assert got.filter(pl.col("co_leader"))["player"].to_list() == ["A"]


def test_the_gap_is_measured_in_pooled_standard_errors():
    """Both candidates carry sampling error, so the comparison uses the standard error of
    the difference, not the leader's alone."""
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.5, 0.45],
                       "lift": [0.02, 0.0], "lift_se": [0.01, 0.01]})
    got = rank_tiers(df)
    b = got.filter(pl.col("player") == "B")["gap_se"][0]
    assert b == pytest.approx(0.02 / (0.01 ** 2 + 0.01 ** 2) ** 0.5, rel=1e-6)


def test_the_leader_is_zero_standard_errors_from_itself():
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.5, 0.4],
                       "lift": [0.02, 0.0], "lift_se": [0.01, 0.01]})
    got = rank_tiers(df)
    assert got["gap_se"][0] == pytest.approx(0.0)


def test_zero_error_estimates_do_not_divide_by_zero():
    """One draft rollout gives a standard error of zero, and the CLI allows it."""
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.5, 0.4],
                       "lift": [0.02, 0.0], "lift_se": [0.0, 0.0]})
    got = rank_tiers(df)
    assert all(v is not None for v in got["gap_se"].to_list())


def test_a_significantly_positive_candidate_is_never_labelled_avoid():
    """Found on the live board: Ja'Marr Chase came out at +0.95 lift with a standard error
    of 0.37 and was tagged `avoid`, because the tag fired on "significantly different from
    zero" without checking the sign. He is significantly *better* than the field. On a draft
    board that is not a cosmetic bug."""
    df = pl.DataFrame({"player": ["A", "B", "C"], "p_win": [0.50, 0.46, 0.41],
                       "lift": [0.04, 0.0095, -0.030],
                       "lift_se": [0.004, 0.0037, 0.005]})
    r = rank_tiers(df)
    got = {x["player"]: tag_for(x["co_leader"], x["lift"], x["lift_se"])
           for x in r.iter_rows(named=True)}
    assert got["A"] == "TAKE"
    assert got["B"] != "avoid", "positive lift must never read as avoid"
    assert got["C"] == "avoid"


def test_a_candidate_indistinguishable_from_the_field_is_left_unmarked():
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.50, 0.45],
                       "lift": [0.04, 0.001], "lift_se": [0.004, 0.004]})
    r = rank_tiers(df)
    got = {x["player"]: tag_for(x["co_leader"], x["lift"], x["lift_se"])
           for x in r.iter_rows(named=True)}
    assert got["B"] == ""


# --- the market's pick, which is what the board now leads with -------------

def test_the_market_pick_fills_an_unfilled_starting_slot_first():
    """P0 measured this arm at +3.11 against the room, and championship equity at +3.15 --
    no detectable difference, n=36, CI [-3.64, +3.58]. The simpler one leads because the
    burden is on the complicated thing, not because the optimizer is bad."""
    pool = pl.DataFrame({"player": ["QB1", "RB1", "RB2"], "pos": ["QB", "RB", "RB"],
                         "adp": [1.0, 5.0, 6.0]})
    # already holding the quarterback, so the best ADP available should not be another
    got = market_pick(pool, {"QB": 1})
    assert got == "RB1"


def test_it_takes_the_best_available_once_every_slot_is_filled():
    pool = pl.DataFrame({"player": ["A", "B"], "pos": ["QB", "RB"], "adp": [9.0, 4.0]})
    full = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    assert market_pick(pool, full) == "B"


def test_a_missing_adp_does_not_win_the_pick():
    """Undrafted players carry a null ADP. Treating null as zero would put them first."""
    pool = pl.DataFrame({"player": ["Known", "Nobody"], "pos": ["RB", "RB"],
                         "adp": [12.0, None]})
    assert market_pick(pool, {}) == "Known"


def test_an_empty_pool_returns_nothing_rather_than_raising():
    pool = pl.DataFrame({"player": [], "pos": [], "adp": []},
                        schema={"player": pl.Utf8, "pos": pl.Utf8, "adp": pl.Float64})
    assert market_pick(pool, {}) is None


def test_no_adp_at_all_returns_nothing_rather_than_an_arbitrary_player():
    """ESPN is under load on draft night and ADP is the first thing to go. With no ADP the
    ranking has nothing to order by, so it returned whichever row came first -- a confident
    looking recommendation with nothing behind it, which is worse than saying so."""
    pool = pl.DataFrame({"player": ["A", "B"], "pos": ["RB", "WR"], "adp": [None, None]},
                        schema={"player": pl.Utf8, "pos": pl.Utf8, "adp": pl.Float64})
    assert market_pick(pool, {}) is None


def test_a_partial_adp_still_works():
    """Losing some ADP is normal -- deep players never have one. Only losing all of it means
    the market has nothing to say."""
    pool = pl.DataFrame({"player": ["A", "B"], "pos": ["RB", "RB"], "adp": [None, 12.0]})
    assert market_pick(pool, {}) == "B"


def test_a_board_with_no_adp_column_returns_nothing():
    pool = pl.DataFrame({"player": ["A"], "pos": ["RB"]})
    assert market_pick(pool, {}) is None


def _dboard(names, pos=None, **cols):
    """A tiny board for the decision tests, distinct from `_board()` above which builds the
    180-row fixture the simulation tests need."""
    n = len(names)
    df = pl.DataFrame({"player": names, "pos": pos or ["RB"] * n})
    return df.with_columns(**cols) if cols else df


# --- held_positions: the decision behind "filling a need" ------------------

def test_held_positions_counts_your_roster_by_position():
    b = _dboard(["A", "B", "C", "D"], pos=["RB", "RB", "WR", "TE"])
    st = DraftState(taken=["A", "B", "C"])
    got = optimize.held_positions(b, st, my_slot=1, teams=1)
    assert got == {"RB": 2, "WR": 1}


def test_held_positions_is_empty_before_you_have_picked():
    assert optimize.held_positions(_dboard(["A"]), DraftState(taken=[])) == {}


def test_held_positions_ignores_players_drafted_by_other_teams():
    """`my_roster` walks the snake; only your own picks count toward your need."""
    b = _dboard(["A", "B", "C", "D"], pos=["RB", "WR", "TE", "QB"])
    st = DraftState(taken=["A", "B", "C", "D"])
    got = optimize.held_positions(b, st, my_slot=1, teams=2)
    assert sum(got.values()) < 4


def test_a_pick_not_on_the_board_does_not_become_a_null_position():
    """Kickers and defences are taken but never on the board. Counting a null `pos` would
    put a phantom position into the need calculation."""
    b = _dboard(["A"], pos=["RB"])
    st = DraftState(taken=["A", "Some Kicker"])
    assert optimize.held_positions(b, st, my_slot=1, teams=1) == {"RB": 1}


def test_the_dead_guard_is_gone():
    """It read `(remaining(board, st).is_empty() and []) or [...]`, and `(x and []) or y`
    is `y` for every x -- so the guard never fired and `remaining()` ran for nothing. The
    behaviour it *looked* like it wanted, an empty count once the board is exhausted, was
    never what it did; this pins what it actually does."""
    b = _dboard(["A"], pos=["RB"])
    st = DraftState(taken=["A"])           # board fully exhausted
    assert optimize.held_positions(b, st, my_slot=1, teams=1) == {"RB": 1}


# --- pick_notes: what is worth interrupting a drafter with -----------------

def test_no_notes_for_a_clean_player():
    assert optimize.pick_notes({"td_luck": 0.1, "missed": 0, "injury_status": "ACTIVE"}) == []


def test_touchdown_luck_is_noted_in_both_directions():
    """An `avoid` tag that ignored sign is a bug this repo has already had once."""
    up = optimize.pick_notes({"td_luck": 1.2})
    down = optimize.pick_notes({"td_luck": -1.2})
    assert "+1.20" in up[0]
    assert "-1.20" in down[0]


def test_touchdown_luck_below_the_threshold_is_not_worth_saying():
    assert optimize.pick_notes({"td_luck": 0.4}) == []
    assert optimize.pick_notes({"td_luck": -0.4}) == []


def test_a_null_touchdown_luck_is_not_an_extreme_one():
    """`abs(None)` raises; a degraded board carries nulls here."""
    assert optimize.pick_notes({"td_luck": None}) == []


def test_missed_games_are_noted():
    assert "missed 6 last season" in optimize.pick_notes({"missed": 6})


def test_a_player_who_missed_nothing_gets_no_note():
    assert optimize.pick_notes({"missed": 0}) == []


def test_only_flagworthy_designations_are_surfaced():
    """ACTIVE is not news. QUESTIONABLE is, even though it is not priced."""
    assert optimize.pick_notes({"injury_status": "ACTIVE"}) == []
    assert optimize.pick_notes({"injury_status": "QUESTIONABLE"}) == ["QUESTIONABLE"]
    assert optimize.pick_notes({"injury_status": "OUT"}) == ["OUT"]


def test_an_empty_row_is_not_a_crash():
    """A board built with every optional stage degraded still has to reach THE PICK."""
    assert optimize.pick_notes({}) == []


def test_notes_stack_in_a_stable_order():
    got = optimize.pick_notes({"td_luck": 1.5, "missed": 4, "injury_status": "OUT"})
    assert len(got) == 3
    assert got[0].startswith("td luck") and got[2] == "OUT"
