"""Championship equity against the draft market, on realised outcomes.

P0 asked this question, answered it in an afternoon, and committed no code -- so the number
that demoted equity to a tiebreaker cannot be reproduced, and two departures from its own
pre-registered design went unrecorded. See ADR-0007.

Everything here runs offline. That is the point: a backtest whose statistics can only be
exercised by hitting ESPN is one nobody re-runs, which is how P0 ended up unreproducible.
"""
import numpy as np
import polars as pl
import pytest

from hub.draft import backtest as bt

# --- the pre-registered decision rule, as executable code ------------------
#
# The most valuable assertions in this file. The rule was fixed before the numbers; these
# make it impossible to quietly reinterpret afterwards.

def test_an_interval_above_zero_promotes_equity():
    assert bt.verdict({"lo": 0.4, "hi": 3.0}).startswith("PROMOTE")


def test_an_interval_below_zero_removes_equity():
    """Evidence demotes as well as promotes. A rule that only ever promotes is
    'heads I win, tails nothing changes'."""
    assert bt.verdict({"lo": -3.0, "hi": -0.4}).startswith("REMOVE")


def test_an_interval_containing_zero_changes_nothing():
    """The branch P0 landed on, and the one worth pre-registering: a null has an action
    rather than being a disappointment to explain away."""
    assert bt.verdict({"lo": -3.64, "hi": 3.58}).startswith("NO CHANGE")


def test_touching_zero_is_not_excluding_zero():
    """An interval whose bound sits exactly on zero has not excluded it."""
    assert bt.verdict({"lo": 0.0, "hi": 3.0}).startswith("NO CHANGE")
    assert bt.verdict({"lo": -3.0, "hi": 0.0}).startswith("NO CHANGE")


def test_p0s_own_numbers_still_read_as_no_change():
    """Regression on the historical result: +0.04, [-3.64, +3.58], n=36."""
    assert bt.verdict({"lo": -3.64, "hi": 3.58, "mean": 0.04}).startswith("NO CHANGE")


# --- the paired bootstrap --------------------------------------------------

def _paired(diffs):
    n = len(diffs)
    return pl.DataFrame({"season": [2022] * n, "draft": list(range(n)),
                         "market": [10.0] * n,
                         "optimizer": [10.0 + d for d in diffs]}).with_columns(
        (pl.col("optimizer") - pl.col("market")).alias("diff"))


def test_a_clear_advantage_produces_an_interval_above_zero():
    s = bt.summarise(_paired([5.0] * 40), bootstrap=500)
    assert s["mean"] == pytest.approx(5.0)
    assert s["lo"] > 0
    assert s["p_better"] == 1.0


def test_no_difference_produces_an_interval_spanning_zero():
    s = bt.summarise(_paired([2.0, -2.0] * 20), bootstrap=500)
    assert s["mean"] == pytest.approx(0.0, abs=1e-9)
    assert s["lo"] < 0 < s["hi"]


def test_more_observations_narrow_the_interval():
    """Why n=80 rather than P0's 36. The centre does not move; the precision does."""
    rng = np.random.default_rng(0)
    small = bt.summarise(_paired(list(rng.normal(0, 5, 20))), bootstrap=800)
    large = bt.summarise(_paired(list(rng.normal(0, 5, 200))), bootstrap=800)
    assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])


def test_an_empty_frame_does_not_raise():
    """A run that produced nothing must report nothing, not crash on the summary."""
    s = bt.summarise(_paired([]), bootstrap=100)
    assert s["n"] == 0


def test_the_bootstrap_is_deterministic_under_a_seed():
    a = bt.summarise(_paired([1.0, -3.0, 2.0] * 10), bootstrap=300, seed=7)
    b = bt.summarise(_paired([1.0, -3.0, 2.0] * 10), bootstrap=300, seed=7)
    assert a == b


# --- scoring a roster on what actually happened ---------------------------

def _realised(rows):
    """(player, week, points) triples, keyed the way `realised_ppg` keys them.

    Normalised here rather than raw, because that is the shape `score_roster` is actually
    handed in production. A fixture carrying raw names would have made the join look fine
    while silently matching nothing.
    """
    from hub.draft.state import _norm
    return pl.DataFrame({"player": [_norm(r[0]) for r in rows],
                         "week": [r[1] for r in rows],
                         "points": [r[2] for r in rows]},
                        schema={"player": pl.Utf8, "week": pl.Int64,
                                "points": pl.Float64})


def _full_roster():
    names = ["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "FLEX1"]
    pos = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR"]
    return names, pos


def test_a_roster_scores_its_best_legal_lineup_each_week():
    names, pos = _full_roster()
    rows = [(n, w, 10.0) for n in names for w in range(1, 15)]
    got = bt.score_roster(names, pos, _realised(rows), weeks=14)
    # eight starting slots at ten points each
    assert got == pytest.approx(80.0)


def test_a_player_with_no_realised_row_scores_zero():
    """He was hurt, cut, or never played. Zero rather than null, so the lineup rule can
    bench him -- a null would propagate and take the whole week with it."""
    names, pos = _full_roster()
    rows = [(n, w, 10.0) for n in names[:-1] for w in range(1, 15)]
    got = bt.score_roster(names, pos, _realised(rows), weeks=14)
    assert got == pytest.approx(70.0)


def test_weekly_scoring_is_not_season_totals():
    """Why the design scores weekly rather than on season totals.

    Nine players, so one of WR4/WR5 rides the bench each week. WR4 scores 140 either way and
    WR5 scores 70 either way, so *season totals are identical between the two cases and a
    totals-based lineup picks WR4 as the flex in both*. Weekly, they differ: when WR4 puts it
    all into one game, WR5 starts the other thirteen and those points are real.

    Season totals would erase exactly this -- and bye weeks and mid-season injuries with it,
    which are the things a roster is built to survive.
    """
    names = ["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "WR4", "WR5"]
    pos = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "WR"]
    base = [(n, w, 10.0) for n in names[:7] for w in range(1, 15)]
    wr5 = [("WR5", w, 5.0) for w in range(1, 15)]

    steady = [("WR4", w, 10.0) for w in range(1, 15)]      # 140 across the season
    spiky = [("WR4", 1, 140.0)]                             # 140, all in week one

    a = bt.score_roster(names, pos, _realised(base + wr5 + steady), weeks=14)
    b = bt.score_roster(names, pos, _realised(base + wr5 + spiky), weeks=14)
    assert a == pytest.approx(80.0)
    assert b == pytest.approx((210 + 13 * 75) / 14)
    assert b > a, "identical season totals, different weekly lineups"


def test_an_empty_roster_scores_nothing():
    assert bt.score_roster([], [], _realised([]), weeks=14) == 0.0


def test_points_are_per_team_game():
    """Comparable across seasons of different length, and the unit the P0 result is in."""
    names, pos = _full_roster()
    rows = [(n, w, 10.0) for n in names for w in range(1, 15)]
    seven = bt.score_roster(names, pos, _realised(rows), weeks=7)
    fourteen = bt.score_roster(names, pos, _realised(rows), weeks=14)
    assert seven == pytest.approx(fourteen)


# --- realised points come off nflverse under the board's own join key ------

def test_realised_points_are_keyed_the_way_the_board_joins():
    """nflverse and FantasyPros disagree about punctuation -- `A.J. Brown` against
    `AJ Brown` -- and an exact join drops him silently."""
    stats = pl.DataFrame({"player_display_name": ["A.J. Brown"], "week": [1],
                          "fantasy_points_ppr": [22.5]})
    got = bt.realised_ppg(stats)
    from hub.draft.state import _norm
    assert got["player"][0] == _norm("AJ Brown")


def test_a_null_score_is_zero_not_missing():
    stats = pl.DataFrame({"player_display_name": ["Guy"], "week": [3],
                          "fantasy_points_ppr": [None]},
                         schema={"player_display_name": pl.Utf8, "week": pl.Int64,
                                 "fantasy_points_ppr": pl.Float64})
    assert bt.realised_ppg(stats)["points"][0] == 0.0


def test_two_rows_for_one_player_week_are_summed():
    """A player traded mid-season can appear twice for one week."""
    stats = pl.DataFrame({"player_display_name": ["Guy", "Guy"], "week": [3, 3],
                          "fantasy_points_ppr": [10.0, 4.0]})
    assert bt.realised_ppg(stats)["points"][0] == 14.0


# --- arm A -----------------------------------------------------------------

def _board(n=40):
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": [["QB", "RB", "WR", "TE"][i % 4] for i in range(n)],
        "ecr": [float(i + 1) for i in range(n)],
        "vor": [float(n - i) for i in range(n)],
    })


def test_arm_a_takes_the_best_consensus_that_fills_a_need():
    """Lexicographic: an unfilled starting slot outranks any amount of consensus."""
    from hub.draft.optimize import market_pick
    pool = _board(8)
    # QB slot already full, so the best available QB must not be taken
    got = market_pick(pool, {"QB": 1}, by="ecr")
    assert got is not None and pool.filter(pl.col("player") == got)["pos"][0] != "QB"


def test_arm_a_ranks_on_consensus_when_there_is_no_draft_market():
    """The historical case. ESPN publishes ADP for the current season only, so a replay has
    no `adp` column at all -- and `by="adp"` would return None, silently handing arm A no
    opinion at every pick."""
    from hub.draft.optimize import market_pick
    pool = _board(8)
    assert "adp" not in pool.columns
    assert market_pick(pool, {}, by="adp") is None
    assert market_pick(pool, {}, by="ecr") == "P0"


# --- the room seam ---------------------------------------------------------

def test_a_pluggable_strategy_sits_in_my_seat_and_the_room_is_unchanged():
    from hub.draft.optimize import simulate_remaining_draft
    from hub.draft.state import DraftState
    board = _board(60)
    picked = []

    def always_last(pool, live, counts, taken):
        picked.append(len(taken))
        return int(live[-1])

    rosters = simulate_remaining_draft(board, DraftState(taken=[]), my_slot=3, teams=12,
                                       rounds=3, rng=np.random.default_rng(0),
                                       my_pick=always_last)
    assert len(picked) == 3, "the strategy should be consulted once per my pick"
    assert len(rosters[2]) == 3


def test_the_default_strategy_is_unchanged_by_the_seam():
    """Every existing caller of simulate_remaining_draft must be untouched."""
    from hub.draft.optimize import simulate_remaining_draft
    from hub.draft.state import DraftState
    board = _board(60)
    kw = {"my_slot": 3, "teams": 12, "rounds": 3}
    a = simulate_remaining_draft(board, DraftState(taken=[]),
                                 rng=np.random.default_rng(4), **kw)
    b = simulate_remaining_draft(board, DraftState(taken=[]), my_pick=None,
                                 rng=np.random.default_rng(4), **kw)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]


def test_a_strategy_that_returns_a_drafted_player_raises():
    """Silently allowing it would duplicate a player onto two rosters, which is how the
    rehearsal's duplicate-pick bug scored one player twice."""
    from hub.draft.optimize import simulate_remaining_draft
    from hub.draft.state import DraftState

    def always_zero(pool, live, counts, taken):
        return 0

    with pytest.raises(ValueError, match="already drafted"):
        simulate_remaining_draft(_board(60), DraftState(taken=[]), my_slot=3, teams=12,
                                 rounds=3, rng=np.random.default_rng(0),
                                 my_pick=always_zero)


def test_the_strategy_sees_picks_in_order_so_it_can_rebuild_the_state():
    """`roster_for` walks the snake to attribute picks to seats, so an unordered list would
    hand a strategy someone else's roster."""
    from hub.draft.optimize import simulate_remaining_draft
    from hub.draft.state import DraftState, roster_for
    board = _board(60)
    seen = {}

    def spy(pool, live, counts, taken):
        seen[len(taken)] = list(taken)
        return int(live[0])

    simulate_remaining_draft(board, DraftState(taken=[]), my_slot=3, teams=12, rounds=3,
                             rng=np.random.default_rng(0), my_pick=spy)
    # Slot 3 of 12 picks at 3, 22, 27. So the strategy is consulted with 2, 21 and 26 picks
    # already made -- which is itself the check that `taken` is in pick order and complete.
    assert sorted(seen) == [2, 21, 26]
    # By my second pick the state must attribute exactly one player to me, and by my third,
    # two. An unordered list would attribute someone else's.
    assert len(roster_for(DraftState(taken=seen[21]), 3, 12)) == 1
    assert len(roster_for(DraftState(taken=seen[26]), 3, 12)) == 2


# --- limitations are recorded before the numbers, not after ---------------

def test_the_named_gaps_between_harness_and_product_are_recorded():
    """A limitation discovered after the result is a rationalisation. These are the four
    the design fixed in advance."""
    assert len(bt.LIMITATIONS) == 5
    joined = " ".join(bt.LIMITATIONS)
    for expected in ("consensus", "xFP", "ties", "simulated", "POST-FIX"):
        assert expected in joined


# --- a defect this harness found in shipped code --------------------------

def test_equity_sees_the_roster_you_already_hold():
    """The defect this harness found on its first run, now fixed.

    `win_probability` scored equity on a roster that EXCLUDED your existing picks, because
    `simulate_remaining_draft` built rosters from a pool of only-available players. Holding a
    quarterback, it ranked a second one above a startable back -- and on the live 2026 board
    it named a third and fourth running back at three of your first six turns while QB, WR
    and TE sat empty.
    """
    from hub.draft.optimize import win_probability
    from hub.draft.state import DraftState, roster_for

    n = 48
    board = pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": [["QB", "RB", "WR", "TE"][i % 4] for i in range(n)],
        "ecr": [float(i + 1) for i in range(n)],
        "vor": [float(n - i) for i in range(n)],
        "proj_ppg": [float(max(20 - i * 0.3, 1.0)) for i in range(n)]})
    state = DraftState(taken=["P1", "P2", "P0"])
    assert roster_for(state, 3, 12) == ["P0"], "I hold exactly one quarterback"

    wp = win_probability(board, state, ["P4", "P5"], my_slot=3, teams=12, rounds=6,
                         n_draft_sims=8, n_season_sims=200, seed=0)
    by = dict(zip(wp["player"].to_list(), wp["lift"].to_list(), strict=True))
    # P4 is a second QB in a one-QB league; P5 is a running back.
    assert by["P5"] > by["P4"], "a redundant quarterback must not outrank a startable back"


# --- the gate on a change to the objective --------------------------------
#
# Fixed before the numbers. Deliberately not a threshold on how much the recommendation
# moved: a correctness fix that changes recommendations is doing its job, and gating on
# similarity would reject it for working.

def _diagnosed(rows):
    return pl.DataFrame(rows)


def test_the_tripwire_catches_a_filled_position_named_over_an_empty_one():
    """The defect's signature, and equally what a seat mis-attribution looks like after the
    fix -- rosters seeded, but with the wrong players."""
    got = bt.tripwire(_board(8), _diagnosed([
        {"pick": 22, "held": "QB1", "leader": "P4", "leader_pos": "QB", "lift": 0.01,
         "co_leaders": 1, "candidates": 8,
         "held_qb": 1, "held_rb": 0, "held_wr": 0, "held_te": 0,
         "need_co_led": False}]))
    assert len(got) == 1
    assert "QB is full" in got[0]


def test_the_tripwire_is_clear_when_the_leader_fills_a_need():
    got = bt.tripwire(_board(8), _diagnosed([
        {"pick": 22, "held": "QB1", "leader": "P5", "leader_pos": "RB", "lift": 0.01,
         "co_leaders": 1, "candidates": 8,
         "held_qb": 1, "held_rb": 0, "held_wr": 0, "held_te": 0,
         "need_co_led": False}]))
    assert got == []


def test_a_surplus_is_fine_once_every_required_slot_is_filled():
    """Depth is not a defect. A fourth WR with a full lineup is a legitimate pick; the
    tripwire only fires while a required slot still sits empty."""
    got = bt.tripwire(_board(8), _diagnosed([
        {"pick": 70, "held": "QB1, RB2, TE1, WR3", "leader": "P9", "leader_pos": "WR",
         "lift": 0.01, "co_leaders": 1, "candidates": 8,
         "held_qb": 1, "held_rb": 2, "held_wr": 3, "held_te": 1,
         "need_co_led": False}]))
    assert got == []


def test_the_tripwire_reads_typed_counts_not_the_display_string():
    """`held` is for a human to read. Parsing it back would break the first time somebody
    drafts ten running backs and the count needs two digits."""
    got = bt.tripwire(_board(8), _diagnosed([
        {"pick": 70, "held": "nonsense", "leader": "P4", "leader_pos": "QB", "lift": 0.01,
         "co_leaders": 1, "candidates": 8,
         "held_qb": 1, "held_rb": 0, "held_wr": 0, "held_te": 0,
         "need_co_led": False}]))
    assert len(got) == 1


def test_a_need_filling_co_leader_is_not_a_defence():
    """The clause that was added after the first run and then reverted.

    Read as a regression gate it looked right -- a tie is not a rejection. Read as a check on
    whether the objective is fit to pick with, it is backwards: an objective that cannot
    separate filling a hole from not filling one is telling you something, and P0b priced
    that same preference at -19.66 points per team game."""
    got = bt.tripwire(_board(8), _diagnosed([
        {"pick": 46, "held": "RB2, TE1", "leader": "P4", "leader_pos": "RB", "lift": 0.03,
         "co_leaders": 3, "candidates": 10,
         "held_qb": 0, "held_rb": 2, "held_wr": 0, "held_te": 1,
         "need_co_led": True}]))
    assert len(got) == 1
    assert "not a defence" in got[0]


def test_the_gate_catches_the_defect_it_was_written_for():
    """A second quarterback beating a startable back, with nothing tied. Arm B finished with
    four quarterbacks in a one-QB league."""
    got = bt.tripwire(_board(8), _diagnosed([
        {"pick": 22, "held": "QB1", "leader": "P4", "leader_pos": "QB", "lift": 0.01,
         "co_leaders": 1, "candidates": 10,
         "held_qb": 1, "held_rb": 0, "held_wr": 0, "held_te": 0,
         "need_co_led": False}]))
    assert len(got) == 1
    assert "QB is full" in got[0]


def test_the_diagnose_picks_are_your_first_six_turns():
    """Fixed rather than read from live state, because the two runs happen at two commits."""
    from hub.config import RosterConfig
    from hub.draft.picks import snake_picks
    cfg = RosterConfig()
    assert list(bt.DIAGNOSE_PICKS) == snake_picks(cfg.slot, cfg.teams, 16)[:6]


def test_diagnose_advances_the_draft_by_the_market_not_by_equity():
    """So the path through the draft is identical before and after the change, and the only
    thing that can differ is what equity says about the same situation."""
    import inspect
    src = inspect.getsource(bt.diagnose)
    assert "market_pick" in src
    assert "Advance by the market" in src


# --- the corrected-ADP gate (ADR-0011) ------------------------------------
#
# Fixed before the numbers, and deliberately NOT "did the recommendation change" -- it is
# supposed to change. That was the mistake made with the seeding tripwire earlier the same
# day: a gate that fires whenever the change works is not a gate.

def _corrected(moves):
    """A board with explicit (adp, adp_corrected, proj_correction) rows."""
    n = len(moves)
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": ["RB"] * n,
        "adp": [float(a) for a, _, _ in moves],
        "adp_corrected": [float(c) for _, c, _ in moves],
        "proj_correction": [float(p) for _, _, p in moves],
    })


def test_a_clean_board_trips_nothing():
    b = _corrected([(10.0, 11.5, -0.5), (50.0, 50.0, 0.0), (100.0, 92.0, +1.0)])
    assert bt.correction_tripwire(b) == []


def test_an_uncorrected_player_moving_is_a_bug():
    """The shift is a function of the correction, so zero in must give zero out."""
    b = _corrected([(50.0, 53.0, 0.0)])
    got = bt.correction_tripwire(b)
    assert len(got) == 1 and "no correction but moved" in got[0]


def test_a_move_past_the_clamp_is_a_bug():
    """The clamp is applied unconditionally, so exceeding it means the clamp is wrong."""
    b = _corrected([(10.0, 40.0, -5.0)])          # 30 picks on an ADP of 10, far past 20%
    got = bt.correction_tripwire(b)
    assert len(got) == 1 and "past the" in got[0]


def test_the_gate_does_not_fire_merely_because_the_pick_moved():
    """The lesson from the seeding tripwire, encoded. A large but legal move is fine."""
    b = _corrected([(100.0, 119.9, -2.0)])        # 19.9 picks, just inside 20% of 100
    assert bt.correction_tripwire(b) == []


def test_the_report_lists_only_players_who_moved():
    b = _corrected([(10.0, 10.0, 0.0), (20.0, 23.0, -1.0)])
    rep = bt.correction_report(b)
    assert rep["player"].to_list() == ["P1"]
    assert rep["move"][0] == pytest.approx(3.0)


def test_the_report_is_ordered_by_size_of_move():
    b = _corrected([(10.0, 11.0, -0.2), (50.0, 58.0, -2.0), (30.0, 31.5, -0.4)])
    assert bt.correction_report(b)["player"].to_list() == ["P1", "P2", "P0"]


def test_a_board_without_the_columns_reports_nothing():
    """A degraded board must not crash the gate."""
    b = pl.DataFrame({"player": ["A"], "pos": ["RB"], "adp": [1.0]})
    assert bt.correction_report(b).is_empty()
    assert bt.correction_tripwire(b) == []


# --- the orchestration, offline -------------------------------------------
#
# `compare`, `play`, `diagnose` and the two strategies were the untested half of this module,
# which is exactly the failure it exists to prevent: a harness nobody can run is a harness
# nobody re-runs. All of this uses synthetic frames and tiny sim counts.

def _full_board(n=80):
    """Everything `build()` emits that the harness reads."""
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": [["RB", "WR", "WR", "TE", "QB"][i % 5] for i in range(n)],
        "ecr": [float(i + 1) for i in range(n)],
        "adp": [float(i + 1) for i in range(n)],
        "vor": [float(n - i) for i in range(n)],
        "proj_ppg": [float(max(20 - i * 0.2, 1.0)) for i in range(n)],
        "xfp_per_game": [float(max(20 - i * 0.2, 1.0)) for i in range(n)],
        "games": pl.Series([16] * n, dtype=pl.UInt32),
    })


def _flat_realised(board, pts=10.0):
    from hub.draft.state import _norm
    names = board["player"].to_list()
    return pl.DataFrame({"player": [_norm(n) for n in names for _ in range(14)],
                         "week": [w for _ in names for w in range(1, 15)],
                         "points": [pts for _ in names for _ in range(14)]},
                        schema={"player": pl.Utf8, "week": pl.Int64, "points": pl.Float64})


def test_play_returns_my_roster_and_positions():
    board = _full_board()
    names, pos = bt.play(board, bt.market_strategy(), my_slot=3, teams=12, rounds=4,
                         rng=np.random.default_rng(0))
    assert len(names) == len(pos) == 4
    assert set(names) <= set(board["player"].to_list())


def test_the_market_arm_fills_its_starting_slots_before_taking_depth():
    board = _full_board()
    _, pos = bt.play(board, bt.market_strategy(), my_slot=3, teams=12, rounds=8,
                     rng=np.random.default_rng(0))
    assert "QB" in pos and "TE" in pos, "a lexicographic need gate must fill both"


def test_compare_produces_one_paired_row_per_draft():
    board = _full_board()
    real = _flat_realised(board)
    got = bt.compare({2024: board}, {2024: real}, n_drafts=2, rounds=4,
                     n_draft_sims=2, n_season_sims=10)
    assert got.height == 2
    assert set(got.columns) >= {"season", "draft", "market", "optimizer", "diff"}
    assert (got["diff"] == got["optimizer"] - got["market"]).all()


def test_compare_is_deterministic_under_a_seed():
    """Same seed, same rooms, same answer -- or the paired design means nothing."""
    board, real = _full_board(), None
    real = _flat_realised(board)
    kw = dict(n_drafts=1, rounds=4, n_draft_sims=2, n_season_sims=10, seed=7)
    a = bt.compare({2024: board}, {2024: real}, **kw)
    b = bt.compare({2024: board}, {2024: real}, **kw)
    assert a.equals(b)


def test_the_optimizer_arm_returns_a_live_player():
    """It ranks `recommend()`'s shortlist by equity and breaks ties on consensus. The failure
    it must not have is returning someone already drafted."""
    board = _full_board()
    strategy = bt.optimizer_strategy(board, my_slot=3, teams=12, rounds=4,
                                     n_draft_sims=2, n_season_sims=10, seed=0)
    live = np.arange(board.height)
    got = strategy(board, live, {}, [])
    assert 0 <= got < board.height


def test_diagnose_reports_one_row_per_requested_pick():
    board = _full_board(n=140)
    got = bt.diagnose(board, picks=(3, 22), my_slot=3, teams=12, rounds=3,
                      n_draft_sims=2, n_season_sims=10)
    assert set(got["pick"].to_list()) <= {3, 22}
    assert {"leader", "lift", "co_leaders", "need_co_led"} <= set(got.columns)


def test_diagnose_advances_by_the_market_so_both_runs_share_a_path():
    """Two runs at two commits must walk the same draft, or the comparison is not one."""
    board = _full_board(n=140)
    kw = dict(picks=(3, 22), my_slot=3, teams=12, rounds=3,
              n_draft_sims=2, n_season_sims=10, seed=0)
    assert bt.diagnose(board, **kw)["held"].to_list() == bt.diagnose(board, **kw)["held"].to_list()
