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
from hub.draft.board import BuildReport
from hub.names import player_key

# --- the pre-registered decision rule, as executable code ------------------
#
# The most valuable assertions in this file. The rule was fixed before the numbers; these
# make it impossible to quietly reinterpret afterwards.

# The three branches, and the boundary where an interval endpoint is exactly zero, are
# tested once in `test_experiment.py` -- there is one rule now (ADR-0019). What is this
# gate's own is which sentence each branch produces, and that its record still reads the same.
# Since #135 there is no per-gate `verdict` wrapper either: the rule is `experiment.gate` and
# the sentences are `ACTIONS`, and the two meet inside `experiment.run_gate`.

def _verdict(summary, seasons):
    from hub.models.experiment import gate
    return gate(summary, seasons, bt.ACTIONS)


def _yrs(gains):
    return pl.DataFrame({"season": list(range(2022, 2022 + len(gains))),
                         "gain": [float(g) for g in gains], "n": [10] * len(gains)})


def _sum(lo, hi):
    return {"n": 80.0, "clusters": 80.0, "mean": (lo + hi) / 2, "lo": lo, "hi": hi,
            "p_better": 0.5}


def test_an_interval_above_zero_in_every_season_promotes_equity():
    status, said = _verdict(_sum(0.4, 3.0), _yrs([0.5, 1.2, 0.9]))
    assert status == "ADOPT" and said.startswith("PROMOTE")


def test_an_interval_below_zero_in_every_season_removes_equity():
    """Evidence demotes as well as promotes. A rule that only ever promotes is
    'heads I win, tails nothing changes'."""
    status, said = _verdict(_sum(-3.0, -0.4), _yrs([-0.5, -1.2, -0.9]))
    assert status == "REMOVE" and said.startswith("REMOVE")


def test_an_interval_containing_zero_changes_nothing():
    """The branch P0 landed on, and the one worth pre-registering: a null has an action
    rather than being a disappointment to explain away."""
    status, said = _verdict(_sum(-3.64, 3.58), _yrs([0.5, -1.2, 0.9]))
    assert status == "SHOW" and said.startswith("NO CHANGE")


def test_p0s_own_numbers_still_read_as_no_change():
    """Regression on the historical result: +0.04, [-3.64, +3.58], n=36."""
    status, said = _verdict(_sum(-3.64, 3.58), _yrs([0.4, -0.3, 0.1]))
    assert status == "SHOW" and said.startswith("NO CHANGE")


def test_adr_0009s_own_numbers_still_remove_equity():
    """The published decision: -19.66, CI [-23.16, -16.20], losing in all four seasons.
    Unifying the rule tightened this gate, and it must not have moved what it published."""
    status, said = _verdict(_sum(-23.16, -16.20), _yrs([-19.0, -21.0, -18.0, -20.0]))
    assert status == "REMOVE" and said.startswith("REMOVE")


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
    return pl.DataFrame({"player": [player_key(r[0]) for r in rows],
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
    assert got["player"][0] == player_key("AJ Brown")


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
    """A limitation discovered after the result is a rationalisation. These are the five the
    design fixed in advance, plus one #196 measured.

    The sixth is a different kind and is counted here anyway. The first five are gaps between
    this harness and the tool it audits, fixed before the numbers. The row-coupling one was
    found afterwards, by probe -- which is exactly the position the docstring above calls a
    rationalisation, and the reason it is stated as a limitation with its consequence rather
    than argued away. It is not a gap against the product; it is a bound on which two runs of
    this harness may be compared at all.
    """
    assert len(bt.LIMITATIONS) == 7
    joined = " ".join(bt.LIMITATIONS)
    for expected in ("consensus", "xFP", "ties", "simulated", "POST-FIX", "board_digest",
                     # #199: the room's currency follows the draft-market stage, so a
                     # backtested room ranks on prior-season xFP and the live one does not.
                     "vor_proj"):
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
    from hub.draft.state import DraftState, roster_for
    from hub.exhibits.championship_equity import win_probability

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


def test_diagnose_advances_by_the_draft_market_not_by_equity():
    """So the path through the draft is identical before and after the change, and the only
    thing that can differ is what equity says about the same situation."""
    import inspect
    src = inspect.getsource(bt.diagnose)
    assert "market_pick" in src
    assert "Advance by the market" in src


def test_diagnose_asks_the_report_which_market_it_advances_by(monkeypatch):
    """Issue #131 at the site left out of it, because the file was owned elsewhere.

    Both directions, because the frame and the report agree on every board reachable today --
    so a test built from a reachable board would pass against the column-sniff it replaces
    and prove nothing. What is being held is the rule: which market this advances by is what
    `build` recorded, not what the frame it was handed happens to carry.
    """
    seen = []
    real = bt.market_pick

    def spy(pool, counts, by="adp"):
        seen.append(by)
        return real(pool, counts, by=by)

    monkeypatch.setattr(bt, "market_pick", spy)

    with_column = _board(60).with_columns(pl.col("ecr").alias("adp"))
    bt.diagnose(with_column, BuildReport(adp=False), picks=(), rounds=2)
    assert set(seen) == {"ecr"}, "the column is there and the report says the stage was not"

    seen.clear()
    # An *empty* draft market rather than an absent one, since #199. The report now travels
    # into the room as well -- `blended_adp` blends the column the report names -- so this
    # direction is stated as a column with nothing in it, which is the strongest form of the
    # contradiction a frame `build` could actually hand over. What is held is unchanged: the
    # `by` this advances on is what the report recorded, never what the column contains.
    bt.diagnose(_board(60).with_columns(pl.lit(None, pl.Float64).alias("adp")),
                BuildReport(adp=True), picks=(), rounds=2)
    assert set(seen) == {"adp"}, "the report is what is asked, not the column's contents"


def test_the_corrections_gate_says_when_it_could_not_run(monkeypatch, capsys):
    """A board no draft market reached has no Corrected ADP, so nothing moved -- and the
    lines below would render that as a clean ADR-0011 gate. The gate has to say it did not
    run and exit non-zero, which is the shape `hub.cli.unavailable` gives the fetch failures
    beside it."""
    from hub.draft import board as board_mod

    monkeypatch.setattr(board_mod, "build",
                        lambda *a, **k: (_board(8), board_mod.BuildReport(adp=False)))
    assert bt.main(["--diagnose-corrections"]) == 1
    said = capsys.readouterr().out
    assert "no draft market reached this board" in said
    assert "tripwire clear" not in said, "a gate that could not run has not passed"


def test_the_corrections_gate_runs_on_a_board_that_has_a_corrected_ranking(monkeypatch,
                                                                            capsys):
    """The other half, so the refusal above cannot be widened into a gate that never runs."""
    from hub.draft import board as board_mod

    b = _corrected([(10.0, 11.5, -0.5), (50.0, 50.0, 0.0)])
    monkeypatch.setattr(board_mod, "build",
                        lambda *a, **k: (b, board_mod.BuildReport(adp=True)))
    assert bt.main(["--diagnose-corrections"]) == 0
    assert "tripwire clear" in capsys.readouterr().out


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
        # `vor_proj` accompanies `adp`, not `vor`: both are `board.STAGE_COLUMNS["adp"]`, and
        # since #199 the room asks the report which currency it ranks in. Equal to `vor` so
        # the fixture stops contradicting its own report without moving any ordering here.
        "vor_proj": [float(n - i) for i in range(n)],
        "proj_ppg": [float(max(20 - i * 0.2, 1.0)) for i in range(n)],
        "xfp_per_game": [float(max(20 - i * 0.2, 1.0)) for i in range(n)],
        "games": pl.Series([16] * n, dtype=pl.UInt32),
    })


def _flat_realised(board, pts=10.0):
    names = board["player"].to_list()
    return pl.DataFrame({"player": [player_key(n) for n in names for _ in range(14)],
                         "week": [w for _ in names for w in range(1, 15)],
                         "points": [pts for _ in names for _ in range(14)]},
                        schema={"player": pl.Utf8, "week": pl.Int64, "points": pl.Float64})


def test_play_returns_my_roster_and_positions():
    board = _full_board()
    names, pos = bt.play(board, bt.market_strategy(), my_slot=3, teams=12, rounds=4,
                         rng=np.random.default_rng(0))
    assert len(names) == len(pos) == 4
    assert set(names) <= set(board["player"].to_list())


def test_arm_a_takes_the_first_live_player_when_its_market_has_nothing_to_say():
    """`market_pick` answers `None` to a pool whose ranking column is missing or all null,
    and the arm still has to make a pick: the first live index, rather than a crash in the
    middle of a simulated room."""
    pool = _board(6).drop("ecr")
    pick = bt.market_strategy()
    assert pick(pool, np.array([4, 2, 5]), {}, []) == 4


def test_the_draft_market_arm_fills_its_starting_slots_before_taking_depth():
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
    got = bt.diagnose(board, BuildReport(adp=True), picks=(3, 22), my_slot=3, teams=12,
                      rounds=3, n_draft_sims=2, n_season_sims=10)
    assert set(got["pick"].to_list()) <= {3, 22}
    assert {"leader", "lift", "co_leaders", "need_co_led"} <= set(got.columns)


def test_diagnose_advances_by_the_draft_market_so_both_runs_share_a_path():
    """Two runs at two commits must walk the same draft, or the comparison is not one."""
    board = _full_board(n=140)
    rep = BuildReport(adp=True)
    kw = {"picks": (3, 22), "my_slot": 3, "teams": 12, "rounds": 3,
              "n_draft_sims": 2, "n_season_sims": 10, "seed": 0}
    assert (bt.diagnose(board, rep, **kw)["held"].to_list()
            == bt.diagnose(board, rep, **kw)["held"].to_list())


# --- what produced these rows (issue #71) ---------------------------------

def test_the_paired_frame_names_the_config_and_the_data_that_made_it(monkeypatch):
    """A gate output that cannot name its data leaves a moved archive as a silently different
    number, which is the defect the whole pinning layer exists to remove."""
    from hub.config import UNPINNED
    from hub.draft import backtest as bt
    from hub.fetch import nflverse as nv

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    paired = pl.DataFrame({"season": [2024, 2025], "effect": [1.0, -2.0]})

    stamped, said = bt.stamped_for_publication(paired)
    assert stamped.height == paired.height
    assert set(stamped.columns) >= {"cfg_digest", "data_digest"}
    assert stamped["data_digest"].unique().to_list() == [UNPINNED], (
        "a run that pinned nothing must say so rather than publishing a digest that looks "
        "like data")
    assert "nothing was loaded through the pinning layer" in said


def test_a_pinned_load_changes_the_published_data_digest(monkeypatch):
    """The property a reader acts on: two runs over different data do not carry the same
    stamp. Without this the digest is decoration."""
    from hub.config import UNPINNED
    from hub.draft import backtest as bt
    from hub.fetch import nflverse as nv

    paired = pl.DataFrame({"season": [2024], "effect": [1.0]})
    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    unpinned, _ = bt.stamped_for_publication(paired)

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {
        "entry": nv.Pin(source="ff_opportunity", as_of="2026-09-04", digest="abcd1234",
                        rows=10, pinned_at=None)})
    pinned, said = bt.stamped_for_publication(paired)

    assert unpinned["data_digest"][0] == UNPINNED
    assert pinned["data_digest"][0] != UNPINNED
    assert "over 1 pinned source(s)" in said
    # And the config stamp is unmoved by the data changing -- they answer different questions.
    assert pinned["cfg_digest"][0] == unpinned["cfg_digest"][0]


# --- the ceiling a gate is measured against (issue #42) --------------------

def _season(n=48, weeks=4, flip=True):
    """A board whose consensus is *wrong*, and the season that shows it.

    Deliberately adversarial: realised points run opposite to ECR, so a drafter who knew the
    season would take players the market ranks last. A board where consensus is already right
    would let a broken foresight arm pass by doing nothing.
    """
    board = _board(n)
    rows = []
    for i in range(n):
        pts = float(i + 1) if flip else float(n - i)
        for w in range(1, weeks + 1):
            rows.append((f"P{i}", w, pts))
    return board, _realised(rows)


def test_the_foresight_market_ranks_on_what_actually_happened():
    from hub.draft.backtest import FORESIGHT, with_foresight
    board, real = _season(8, weeks=2)
    seeing = with_foresight(board, real)
    # P7 scored most, so it is rank 1 in a lower-is-better market; P0 scored least.
    order = seeing.sort(FORESIGHT)["player"].to_list()
    assert order[0] == "P7" and order[-1] == "P0"
    assert seeing.height == board.height, "the board's rows must not move; play indexes them"
    assert seeing["player"].to_list() == board["player"].to_list()


def test_a_player_with_no_realised_row_ranks_last_not_null():
    """He scored nothing, which is what `score_roster` already assumes. A null would sort
    into the middle of a lower-is-better column and hand him a mid-round pick."""
    from hub.draft.backtest import FORESIGHT, with_foresight
    board, real = _season(4, weeks=1)
    seeing = with_foresight(board, real.filter(pl.col("player") != player_key("P3")))
    assert seeing[FORESIGHT].null_count() == 0
    worst = seeing.sort(FORESIGHT, descending=True)["player"][0]
    assert worst == "P3"


def test_the_ceiling_arm_never_loses_to_the_draft_market_in_any_season():
    """Criterion one, and it is what makes the number a ceiling rather than a third arm.

    An arm that knows the season and still loses is not bounding anything, and would make
    every comparison against it meaningless in the safe-looking direction -- an effect could
    clear a ceiling that was simply low.
    """
    from hub.draft.backtest import ceiling
    boards, reals = {}, {}
    for season in (2023, 2024):
        b, r = _season()
        boards[season], reals[season] = b, r
    got = ceiling(boards, reals, n_drafts=4, my_slot=1, teams=4, rounds=6)
    by_season = got.group_by("season").agg(pl.col("diff").mean().alias("gain"))
    assert (by_season["gain"] > 0).all(), (
        f"the foresight arm lost a season: {by_season.to_dicts()}")


def test_the_ceiling_is_recomputed_for_the_season_set_it_is_given():
    """Criterion four. A ceiling served from a run over other seasons bounds a different
    question and is indistinguishable from a right answer -- same units, same shape,
    plausible size."""
    from hub.draft.backtest import ceiling
    b, r = _season()
    one = ceiling({2023: b}, {2023: r}, n_drafts=2, my_slot=1, teams=4, rounds=6)
    two = ceiling({2023: b, 2024: b}, {2023: r, 2024: r}, n_drafts=2, my_slot=1,
                  teams=4, rounds=6)
    assert sorted(one["season"].unique().to_list()) == [2023]
    assert sorted(two["season"].unique().to_list()) == [2023, 2024]
    assert two.height == 2 * one.height


def test_the_ceiling_bounds_the_arm_under_test_on_the_same_frame():
    """Criterion two. The ceiling and the gate share `season`, `draft` and their seeds, so
    they pair row for row -- and a ceiling that does not bound the effect means one of the two
    is measuring something the other is not.

    Asserted on the shared `market` column, which is the strongest form available: both frames
    play the same incumbent in the same rooms, so if that column disagrees the pairing is
    broken and neither difference is comparable.
    """
    from hub.draft.backtest import ceiling, compare
    board = _full_board()
    real = _flat_realised(board)
    boards, reals = {2024: board}, {2024: real}
    kw = {"n_drafts": 2, "rounds": 4, "seed": 0}
    gate = compare(boards, reals, n_draft_sims=2, n_season_sims=10, **kw)
    top = ceiling(boards, reals, **kw)

    assert gate["market"].to_list() == top["market"].to_list(), (
        "the two frames do not share an incumbent, so their differences are not comparable")
    assert float(top["diff"].to_numpy().mean()) >= float(gate["diff"].to_numpy().mean()), (
        "the ceiling does not bound the arm under test, so it bounds nothing")


def _gate_run(paired, bound):
    """This gate's call, as `main` spells it, with the width history pointed nowhere."""
    from hub.models.experiment import SEASON_CLUSTER, Ceiling, run_gate
    return run_gate(paired, cluster=SEASON_CLUSTER, actions=bt.ACTIONS, name="draft",
                    arm_a="optimizer", arm_b="market", bootstrap=200,
                    ceiling=Ceiling(bt.CEILING_ARM, bound["diff"]), record_width=False)


def test_a_ceiling_that_does_not_bound_says_so_loudly():
    """The number's whole use is as an upper bound. A ceiling below the effect means one of
    the two is measuring something the other is not, and publishing it quietly would let a
    reader take a broken bound for a tight one. `with_ceiling` said this for this gate alone
    until #135; the run says it for every gate."""
    run = _gate_run(_paired([5.0] * 8), pl.DataFrame({"diff": [1.0] * 8}))
    warning = "\n".join(run.lines)
    assert "CEILING BELOW THE EFFECT" in warning and "+1.00 < +5.00" in warning


def test_a_ceiling_that_bounds_carries_the_number_and_says_nothing():
    run = _gate_run(_paired([-19.66] * 8), pl.DataFrame({"diff": [40.0, 42.0] * 4}))
    assert run.summary["ceiling"] == 41.0
    assert "CEILING BELOW" not in "\n".join(run.lines)
    assert run.summary["mean"] == pytest.approx(-19.66), "the effect comes back intact"
    assert f"ceiling ({bt.CEILING_ARM}) +41.00" in "\n".join(run.lines), (
        "the line names this gate's declared arm, not the block's default")


# --- the seed lattice (issue #195) ----------------------------------------
#
# The defect these hold: `compare` used to build one integer per draft and hand the same
# integer to two different levels of the experiment -- to the room as a seed, and to
# `optimizer_strategy`, whose rollouts opened `default_rng(seed + k)`. At `k = 0` that was
# bit-for-bit the room being played, so one of arm B's evaluation futures WAS the room it was
# about to be scored in, at every pick. Arm A got nothing of the kind, which is why the leak
# is one-directional and the measured effect is if anything understated.
#
# These are asserted on the *streams*, not on the answers. Two runs whose answers differ is
# consistent with any seeding at all; what has to be true is that no generator opened at one
# level is a generator opened at another.


def _stream_spy(monkeypatch):
    """Record the state of every generator this package opens, by level.

    `simulate_remaining_draft` consumes its generator exactly once, at the top, before either
    `forced` or `state` is read -- so the state at entry identifies the stream, and two calls
    recording the same state are two plays of one room. `forced` is what separates the two
    kinds of call: an evaluation rollout always names the candidate it is testing, and the
    room being played never does.

    Both namespaces are patched because both hold a reference: `backtest.play` imported the
    function for the room, and the exhibit's `win_probability` imported it for the rollouts.
    """
    from hub.exhibits import championship_equity as ce

    seen: dict[str, list] = {"room": [], "rollout": [], "season": [], "log": []}
    real_draft = ce.simulate_remaining_draft
    real_season = ce.champion_probability

    def _key(rng):
        return repr(rng.bit_generator.state)

    def _note(level, rng):
        seen[level].append(_key(rng))
        seen["log"].append((level, _key(rng)))

    def draft_spy(board, state, *, forced=None, rng=None, **kw):
        _note("rollout" if forced is not None else "room", rng)
        return real_draft(board, state, forced=forced, rng=rng, **kw)

    def season_spy(rosters, mu, sd, pos, *, rng=None, **kw):
        _note("season", rng)
        return real_season(rosters, mu, sd, pos, rng=rng, **kw)

    monkeypatch.setattr(ce, "simulate_remaining_draft", draft_spy)
    monkeypatch.setattr(bt, "simulate_remaining_draft", draft_spy)
    monkeypatch.setattr(ce, "champion_probability", season_spy)
    return seen


def _rollouts_by_draft(log):
    """The evaluation futures each draft in a sweep opened, in sweep order.

    Segmented on the room calls rather than on the seeds, which is the point: `compare` plays
    each draft's room exactly twice -- once per arm -- before moving on, so every second room
    call opens a new draft. That boundary is a property of the loop and is the same before and
    after #195, which is what lets this same helper read both.
    """
    out: list[set] = []
    rooms = 0
    for level, key in log:
        if level == "room":
            if rooms % 2 == 0:
                out.append(set())
            rooms += 1
        elif level == "rollout" and out:
            out[-1].add(key)
    return out


_SMALL = {"rounds": 3, "n_draft_sims": 3, "n_season_sims": 5}


def _varied_realised(board):
    """Realised points that differ between players, so a roster's score identifies it.

    `_flat_realised` pays everyone the same, which makes `score_roster` a function of roster
    *size* alone -- fine for the tests that only need a number, and useless for any assertion
    about which players a room delivered. Anything comparing two rooms has to be able to tell
    two rosters apart.
    """
    names = board["player"].to_list()
    return pl.DataFrame(
        {"player": [player_key(n) for n in names for _ in range(14)],
         "week": [w for _ in names for w in range(1, 15)],
         "points": [float(20 - i % 17) for i, _ in enumerate(names) for _ in range(14)]},
        schema={"player": pl.Utf8, "week": pl.Int64, "points": pl.Float64})


def test_no_evaluation_future_is_the_room_it_is_scored_in(monkeypatch):
    """The load-bearing one, and it fails on the code that shipped before #195.

    Arm B chooses by playing futures forward and reading off how often it wins. If one of
    those futures is the very room the resulting roster is then graded in, arm B is being
    scored partly on a draft it has already seen, and arm A -- which plays no futures at all
    -- is not. That is foresight, it runs one way, and no amount of pairing removes it.
    """
    board = _full_board()
    seen = _stream_spy(monkeypatch)
    bt.compare({2024: board}, {2024: _flat_realised(board)}, n_drafts=1, seed=0, **_SMALL)

    assert seen["room"] and seen["rollout"], "the spy caught neither level"
    assert not set(seen["room"]) & set(seen["rollout"]), (
        "an evaluation future is the room it is scored in -- arm B is reading its own "
        "grading draft")


def test_a_seasons_rooms_are_not_the_next_seasons_simulated_seasons(monkeypatch):
    """The cross-level collision, asserted rather than assumed.

    The season-simulation seeds were `room + 1000 + k` and the rooms were
    `seed + 1000 * season + k`, so a season simulation inside 2024 landed exactly on a 2025
    room -- twenty times per season pair. ADR-0019 ties adoption to consistency across
    held-out seasons, which is the comparison this contaminates.
    """
    board = _full_board()
    real = _flat_realised(board)
    seen = _stream_spy(monkeypatch)
    bt.compare({2024: board, 2025: board}, {2024: real, 2025: real},
               n_drafts=2, seed=0, **_SMALL)

    assert seen["season"], "the spy caught no season simulation"
    assert not set(seen["room"]) & set(seen["season"]), (
        "a simulated season is drawn from a stream that is also somebody's room")


def test_consecutive_drafts_share_no_evaluation_futures(monkeypatch):
    """Rooms one apart used to share 11 of their 12 futures.

    `experiment.summarise` records the cluster choice as "one row per (season, draft),
    independent rooms: no cluster". The rooms were independent; arm B's evaluations of them
    were not, and that is the repeated-measures shape `docs/signal-screens.md` records as
    having once turned noise into a 4-sigma result.
    """
    board = _full_board()
    seen = _stream_spy(monkeypatch)
    bt.compare({2024: board}, {2024: _flat_realised(board)}, n_drafts=2, seed=0, **_SMALL)

    first, second = _rollouts_by_draft(seen["log"])
    assert first and second, "a draft opened no futures of its own"
    assert not first & second, (
        f"two drafts in one sweep share {len(first & second)} of their evaluation futures")


def test_both_arms_are_played_in_the_identical_room(monkeypatch):
    """The property the whole design rests on, and which #195 must not cost.

    Making the levels non-overlapping is easy to do by making everything different, which
    would also make the two arms face two different fields and turn a paired comparison into
    an unpaired one. So this is asserted, not assumed: one room per draft, played twice.
    """
    board = _full_board()
    seen = _stream_spy(monkeypatch)
    bt.compare({2024: board}, {2024: _flat_realised(board)}, n_drafts=2, seed=0, **_SMALL)

    assert len(seen["room"]) == 4, "two arms x two drafts is four plays of a room"
    assert len(set(seen["room"])) == 2, (
        "the two arms must face the same field -- one distinct room per draft, not four")
    assert seen["room"][0] == seen["room"][1] and seen["room"][2] == seen["room"][3]


def test_the_ceiling_is_played_in_compares_own_rooms():
    """`ceiling`'s incumbent column has to be `compare`'s incumbent column.

    It is the bound `docs/gate-power.md` prices every effect against, and a bound drawn
    against a different field is a bound on a different question -- same units, same shape,
    plausible size, and wrong. Both now reach the room through `draft_root`, so this is a
    property of one function rather than of two copies of an expression staying in step.
    """
    board = _full_board()
    # Varied, not flat: under flat realised points every roster scores the same and this
    # assertion holds for any two rooms whatever, which is an assertion about nothing.
    real = _varied_realised(board)
    kw = {"n_drafts": 2, "rounds": 6, "seed": 5}
    paired = bt.compare({2024: board}, {2024: real}, n_draft_sims=2, n_season_sims=5, **kw)
    bound = bt.ceiling({2024: board}, {2024: real}, **kw)
    assert bound["market"].to_list() == pytest.approx(paired["market"].to_list())


def test_a_negative_seed_still_runs():
    """`--seed` is a plain int and a negative one used to work. A seeding change is not the
    place to start rejecting one -- `SeedSequence` refuses negative entropy, so the root
    folds rather than raises."""
    board = _full_board()
    got = bt.compare({2024: board}, {2024: _flat_realised(board)}, n_drafts=1, seed=-7,
                     **_SMALL)
    assert got.height == 1


# --- which Board a run was measured on (issue #196) ------------------------
#
# A run stamped `cfg_digest` and `data_digest`, and the second is a digest of the upstream
# *source bytes* -- a good digest of the wrong object. The Board is what `compare` is handed,
# and nothing hashed it: `docs/gate-power.md` carries three runs at the data digest `621cb5dd`
# that were played on frames nobody can now name.


def test_the_frozen_board_digests_to_a_pinned_value():
    """Offline and pinned, because a digest nobody can reproduce is a decoration.

    The fixture is the Board already frozen under #108. If this value moves, either the
    fixture moved or the digest's own rule did, and both are things a reader of an old figure
    needs told rather than absorbed.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import panelarchive as arc

    from hub.config import frame_digest

    board = arc.frame("draft_board")
    assert (board.height, len(board.columns)) == (200, 32)
    assert frame_digest(board) == "f9fe3e88"


def test_two_boards_differing_by_one_player_do_not_share_a_digest():
    """The property the stamp exists for, and the reason it is worth having.

    `90a9bbb` dropped 814 of 1,372 players from 2025 at an unchanged data digest. Under the
    row-coupled draw that is a total re-draw of everyone's future, not a small perturbation --
    so two runs either side of it are not comparable, and nothing in the output said so.
    """
    from hub.config import frame_digest

    board = _full_board()
    assert frame_digest(board) != frame_digest(board.head(board.height - 1))


def test_row_order_is_in_the_digest():
    """Position pairs a player with his draw, so two orderings are two experiments.

    Both stochastic quantities are indexed by row: `simulate_remaining_draft` draws one
    pick-noise normal per Board row and `predict.correlated_normal` draws an array whose last
    axis is the Board's height. A digest that sorted the rows out would call two different
    experiments the same one.
    """
    from hub.config import frame_digest

    board = _full_board()
    assert frame_digest(board) != frame_digest(board.reverse())


def test_the_column_set_is_in_the_digest():
    """Which columns a frame carries decides which code runs on it.

    `correction_report` returns an empty frame when the corrected columns are absent, and
    `diagnose` advances by consensus rather than by the draft market on a board with no `adp`.
    A frame that lost a column is a different frame even where every retained value matches.

    The rename is the load-bearing half. Dropping a column also drops a cell from every row,
    so a digest over the cells alone would catch it and the column set would still be doing no
    work; renaming holds every value fixed and moves only the header. `adp` -> `adp2` also
    keeps its place under `sorted`, so the cells are not merely equal as a set but identical
    in order.
    """
    from hub.config import frame_digest

    board = _full_board()
    assert frame_digest(board) != frame_digest(board.drop("adp"))
    renamed = board.rename({"adp": "adp2"})
    assert renamed.select(sorted(renamed.columns)).rows() == \
        board.select(sorted(board.columns)).rows(), "the rename must move nothing but a name"
    assert frame_digest(board) != frame_digest(renamed)


def test_a_run_that_played_no_frames_says_so_rather_than_hashing_nothing():
    """A sha of the empty string is eight legitimate-looking characters that compare equal
    across every such run -- the false-provenance shape `data_digest` argues against."""
    from hub.config import NO_FRAMES, frames_digest

    assert frames_digest({}) == NO_FRAMES


def test_the_same_frames_under_different_seasons_are_different_runs():
    """The key is folded in beside the digest. One board played as 2024 and the same board
    played as 2025 are two experiments, and a digest over the values alone would miss it."""
    from hub.config import frames_digest

    board = _full_board()
    assert frames_digest({2024: board}) != frames_digest({2025: board})


def test_the_paired_frame_names_the_board_it_was_measured_on(monkeypatch):
    """Two runs on different Boards are distinguishable from their stamps alone, which is the
    criterion -- without reading the frames back."""
    from hub.draft import backtest as bt
    from hub.fetch import nflverse as nv

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    paired = pl.DataFrame({"season": [2024], "effect": [1.0]})
    board = _full_board()

    one, said = bt.stamped_for_publication(paired, {2024: board})
    two, _ = bt.stamped_for_publication(paired, {2024: board.head(board.height - 1)})

    assert {"board_digest", "commit"} <= set(one.columns)
    assert one["board_digest"][0] != two["board_digest"][0], (
        "two runs on different Boards carry the same stamp, so the stamp is decoration")
    assert one["cfg_digest"][0] == two["cfg_digest"][0], (
        "the Board is not the model -- a different frame must not move the model version")
    assert f"board: {one['board_digest'][0]}" in said


def test_a_run_that_hands_over_no_boards_stamps_the_sentinel(monkeypatch):
    """`stamped_for_publication` keeps its one-argument form for the callers that have no
    frames to give, and those runs must say `noframes` rather than a plausible hash."""
    from hub.config import NO_FRAMES
    from hub.draft import backtest as bt
    from hub.fetch import nflverse as nv

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    stamped, said = bt.stamped_for_publication(pl.DataFrame({"season": [2024]}))
    assert stamped["board_digest"].unique().to_list() == [NO_FRAMES]
    assert "did not hand over the frames it played" in said


def test_the_run_stamps_the_commit_that_produced_it(monkeypatch):
    """`docs/track-record.md` rule 1 makes these numbers commit-dated, and #190 records an
    effect moving 8.07 points across 270 commits with no owning commit -- because no run ever
    recorded which tree read the bytes."""
    from hub.config import NO_COMMIT
    from hub.draft import backtest as bt
    from hub.fetch import nflverse as nv

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    stamped, said = bt.stamped_for_publication(pl.DataFrame({"season": [2024]}))
    got = stamped["commit"][0]
    assert got, "the commit column is empty"
    assert got == NO_COMMIT or got[:8].isalnum()
    assert f"commit: {got}" in said


def test_an_unreachable_git_tree_degrades_rather_than_raising(monkeypatch):
    """A gate that dies because it could not find git is a gate that stops being run."""
    import subprocess

    from hub import config as cfg

    def boom(*a, **kw):
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", boom)
    assert cfg.commit() == cfg.NO_COMMIT


def test_a_dirty_tree_is_not_stamped_as_its_commit(monkeypatch):
    """A SHA claims "this tree is that commit". A tree with uncommitted changes is not one,
    and printing the SHA alone would be a stamp that names the wrong thing."""
    import subprocess

    from hub import config as cfg

    class Done:
        def __init__(self, out):
            self.stdout = out

    def fake(args, **kw):
        return Done("deadbeefcafe\n" if "rev-parse" in args else " M src/hub/config.py\n")

    monkeypatch.setattr(subprocess, "run", fake)
    assert cfg.commit() == "deadbeef-dirty"


def test_the_sentinels_are_not_mistakable_for_digests():
    """Eight characters so the stamps line up, and not eight *hex* characters, so a reader can
    tell a sentinel from a hash without being told. The same line `UNPINNED` holds."""
    from hub.config import NO_COMMIT, NO_FRAMES

    for sentinel in (NO_COMMIT, NO_FRAMES):
        assert len(sentinel) == 8
        with pytest.raises(ValueError):
            int(sentinel, 16)


# --- a join failure voids the run (issue #46) -----------------------------------------------
#
# `score_roster` scores a drafted player with no realised row as zero, which is right for a
# player who was hurt, cut or never played and wrong for one whose name a source spelled
# differently. The two arms draft different players, so a differential failure rate is a
# differential bias in the headline number, and the run refuses to report above a floor fixed
# before any run was made under it. Voiding is a property of `experiment.run_gate` since #135;
# what is this gate's own is how it counts, and the sentence it hands the run.


def _two_word_board(n=80):
    """`_full_board` with names a source could abbreviate -- `P0` has no initial to take.
    Eighty, so twelve teams over four rounds do not run the board dry before slot 3's
    fourth turn."""
    board = _full_board(n)
    return board.with_columns(pl.Series("player", [f"First{i} Last{i}" for i in range(n)]))


def test_a_roster_where_every_name_matches_reports_zero_failures():
    real = _realised([("Justin Jefferson", 1, 20.0), ("Ja'Marr Chase", 1, 18.0)])
    known = bt.realised_names(real)
    assert bt.join_failures(["Justin Jefferson", "JaMarr Chase"], known) == 0


def test_a_name_absent_but_matching_relaxed_is_a_join_failure():
    """The source abbreviated him: no player key matches, the relaxed key does. He played,
    the harness scored him zero, and that is the defect being counted."""
    real = _realised([("J. Jefferson", 1, 20.0), ("Ja'Marr Chase", 1, 18.0)])
    known = bt.realised_names(real)
    assert bt.join_failures(["Justin Jefferson", "Ja'Marr Chase"], known) == 1


def test_a_name_with_no_relaxed_match_is_never_played_not_a_failure():
    """A drafted rookie who never took a snap has no row under any spelling. Scoring him zero
    is the harness being right, and counting him would void a run for its own accuracy."""
    real = _realised([("Justin Jefferson", 1, 20.0)])
    known = bt.realised_names(real)
    assert bt.join_failures(["Bijan Robinson", "Justin Jefferson"], known) == 0
    assert bt.join_failures([], known) == 0


def test_compare_carries_both_arms_failure_counts_and_the_rates_read_off_them(monkeypatch):
    """Both arms' counts ride on the paired frame, so the rates are computed from what was
    actually drafted rather than re-drafted afterwards. Corrupting the realised spelling of
    the first five players -- the consensus arm's first picks -- gives arm A a failure rate,
    and arm B is pinned to the bottom of the pool so it cannot share one: the two counts
    have to be the two arms' own, not one arm's twice."""
    board = _two_word_board()
    real = _flat_realised(board)
    bad = {player_key(f"First{i} Last{i}"): player_key(f"F. Last{i}") for i in range(5)}
    corrupt = real.with_columns(pl.col("player").replace(bad))
    monkeypatch.setattr(bt, "optimizer_strategy",
                        lambda *a, **k: (lambda pool, live, counts, taken: int(live[-1])))
    kw = {"n_drafts": 2, "rounds": 4, "seed": 0, "n_draft_sims": 2, "n_season_sims": 10}
    clean = bt.compare({2024: board}, {2024: real}, **kw)
    dirty = bt.compare({2024: board}, {2024: corrupt}, **kw)
    for frame in (clean, dirty):
        assert {"market_failed", "optimizer_failed", "picks"} <= set(frame.columns)
        assert frame["picks"].to_list() == [4] * frame.height
    assert clean["market_failed"].sum() == 0 and clean["optimizer_failed"].sum() == 0
    assert dirty["market_failed"].sum() > 0
    assert dirty["optimizer_failed"].sum() == 0, "arm B drafted from the bottom; its names join"
    rates = bt.join_failure_rates(dirty)
    assert rates["picks"] == 8.0
    assert rates["market"] == pytest.approx(int(dirty["market_failed"].sum()) / 8.0)
    assert rates["optimizer"] == pytest.approx(int(dirty["optimizer_failed"].sum()) / 8.0)
    assert 0.0 < rates["market"] <= 1.0
    # And the columns the verdict reads are the ones they were: the counts are beside the
    # scores, not inside them.
    assert clean.select("season", "draft", "market", "optimizer", "diff").columns == \
        ["season", "draft", "market", "optimizer", "diff"]


def test_the_verdict_voids_above_the_floor_and_names_it():
    rates = {"picks": 320.0, "market": 0.05, "optimizer": 0.0}
    said = bt.void_condition(rates)
    assert said is not None and said.startswith("VOID")
    assert "5.0%" in said and "0.0%" in said and f"{bt.VOID_FLOOR:.0%}" in said
    # Either arm over the floor voids: the bias is differential, and it does not matter
    # which arm carries it.
    assert (bt.void_condition({"picks": 320.0, "market": 0.0, "optimizer": 0.03}) or "") \
        .startswith("VOID")


def test_the_floor_itself_and_an_empty_run_are_not_a_void():
    at = {"picks": 100.0, "market": bt.VOID_FLOOR, "optimizer": bt.VOID_FLOOR}
    assert bt.void_condition(at) is None
    assert bt.void_condition({"picks": 0.0, "market": float("nan"),
                              "optimizer": float("nan")}) is None
    empty = pl.DataFrame({"picks": pl.Series([], dtype=pl.Int64),
                          "market_failed": pl.Series([], dtype=pl.Int64),
                          "optimizer_failed": pl.Series([], dtype=pl.Int64)})
    assert bt.join_failure_rates(empty)["picks"] == 0.0


def test_below_the_floor_the_verdict_is_unchanged_from_today_for_the_same_inputs():
    """The acceptance criterion: a clean join hands the run no void, and the run with no void
    is the run there was. Asserted on the summary, the season table and the verdict."""
    from hub.models.experiment import SEASON_CLUSTER, run_gate

    paired = pl.DataFrame({"season": [2022, 2022, 2023, 2023, 2024, 2024, 2025, 2025],
                           "draft": [0, 1] * 4,
                           "diff": [-19.66 + 0.4 * i for i in range(8)]})
    rates = {"picks": 128.0, "market": 0.01, "optimizer": 0.005}
    assert bt.void_condition(rates) is None
    kw = {"cluster": SEASON_CLUSTER, "actions": bt.ACTIONS, "name": "draft",
          "arm_a": "optimizer", "arm_b": "market", "bootstrap": 200, "record_width": False}
    with_void = run_gate(paired, void=bt.void_condition(rates), **kw)
    before = run_gate(paired, **kw)
    assert with_void.summary == before.summary
    assert with_void.seasons.equals(before.seasons)
    assert with_void.verdict == before.verdict == ("REMOVE", before.verdict[1])


def test_both_arms_rates_are_reported_and_a_difference_is_said():
    same = bt.join_report({"picks": 320.0, "market": 0.01, "optimizer": 0.01})
    assert len(same) == 1
    assert "market 1.0%" in same[0] and "optimizer 1.0%" in same[0]
    assert "320 drafted names" in same[0] and f"floor {bt.VOID_FLOOR:.0%}" in same[0]
    differ = bt.join_report({"picks": 320.0, "market": 0.01, "optimizer": 0.0})
    assert len(differ) == 2
    assert "differ" in differ[1] and "differential" in differ[1]
    assert bt.join_report({"picks": 0.0, "market": float("nan"), "optimizer": float("nan")}) \
        == []
# --- the arm under test has a frozen identity (issue #197) ------------------

# `test_compare_is_deterministic_under_a_seed` used to live here. It asserted that two calls
# in one process agree, which is true at every commit in history while the measured effect
# moved eight points -- the seed-to-outcome map is a function of the code, and a test that
# only compares the code with itself cannot see the code change. The pin below proves the
# same property more strongly (a value that matches across processes is a value that matches
# within one) and adds the half that was missing: *which* draft the seed maps to. Its second
# half replays the pinned draft through `compare` in the same process as the direct plays, so
# the cross-call statelessness the old test named is asserted rather than dropped.

# The recipe. Fixed here rather than read from the CLI defaults, because the pin is a claim
# about this Board at this seed and this budget, and a default that moved would move the
# claim without moving the code the claim is about. The budget is a fraction of the shipped
# 12 x 250 so the test costs seconds; `compare`'s docstring says why the *published* number
# may not be measured at a reduced budget, and this is not that number -- it is the arm's
# identity, and a changed objective moves the roster at any budget.
FROZEN_BOARD_DIGEST = "f9fe3e88"
FROZEN_SEASON, FROZEN_SEED, FROZEN_DRAFT = 2025, 7, 0
FROZEN_BUDGET = {"rounds": 6, "n_draft_sims": 2, "n_season_sims": 10}

# What each arm drafted from slot 3 on that Board, at that root, in pick order. Pinned at
# commit 9732519 (2026-09-11). A commit that changes either list must say so, because the
# arm's published verdict (ADR-0009) was measured on the arm that produced these.
#
# Moved once already, the same day: `e9360c7` (#235) refitted `TALENT_CV` net of the absence
# the season simulator now draws, and arm B -- whose objective is that simulator -- took
# CeeDee Lamb over Christian McCaffrey at pick 1 and Austin Ekeler over David Njoku at pick
# 6, with the room unchanged. That commit did not say the arm moved because this pin was
# not yet on `main` to say it; ADR-0009's -17.30 was measured on the arm before it.
#
# What the pin sees is the roster, not every constant behind it. Mutation-proved against the
# tie-break direction, the lift ordering, the seeding root and `compare`'s own stream; it
# did *not* move when the co-leader bar was widened from 2 to 200 se, because at this budget
# the consensus-best co-leader is the same player either way. A change to the tiering rule
# is `test_optimize.py`'s to catch; this catches a change to what the arm drafts.
FROZEN_ARM_A = ["Christian McCaffrey", "Drake London", "Travis Etienne Jr.",
                "Patrick Mahomes II", "D.K. Metcalf", "Terry McLaurin"]
# Three running backs in six picks is the preference ADR-0009 describes.
FROZEN_ARM_B = ["CeeDee Lamb", "Jalen Hurts", "Kyren Williams", "Josh Jacobs",
                "Ken Walker III", "Austin Ekeler"]


def _frozen_board():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import panelarchive as arc

    return arc.frame("draft_board")


def _frozen_realised(board):
    """Every player scores his own xFP per game, every week. Deterministic and derived from
    the Board, so the pinned scores below are a function of the pinned rosters alone."""
    return pl.DataFrame({
        "player": [player_key(n) for n in board["player"].to_list() for _ in range(14)],
        "week": [w for _ in range(board.height) for w in range(1, 15)],
        "points": [float(x) for x in board["xfp_per_game"].to_list() for _ in range(14)],
    }, schema={"player": pl.Utf8, "week": pl.Int64, "points": pl.Float64})


def test_the_arm_under_test_is_pinned_on_the_frozen_board():
    """A fixed Board, seed and budget produce a fixed roster from each arm, and the failure
    names which arm moved rather than which number did.

    Three things can move this and the message tells them apart. The Board digest moving is
    the fixture, not the code. Arm A moving is the room -- the simulator or the seeding tree,
    which both arms share, so arm B's roster is expected to move with it and says nothing on
    its own. Arm B moving alone is the objective: `win_probability`, `rank_tiers`, the season
    simulator underneath, or the shortlist -- the thing ADR-0009's verdict was measured on.
    """
    from hub.config import frame_digest
    from hub.draft.optimize import ROOM, stream

    board = _frozen_board()
    assert frame_digest(board) == FROZEN_BOARD_DIGEST, (
        "the frozen Board moved, so nothing below is about the arm -- re-pin the fixture "
        "first (test_the_frozen_board_digests_to_a_pinned_value)")
    report = BuildReport.of_served(board)
    root = bt.draft_root(FROZEN_SEED, FROZEN_SEASON, FROZEN_DRAFT)
    a_names, _ = bt.play(board, bt.market_strategy(), my_slot=3, teams=12,
                         rounds=FROZEN_BUDGET["rounds"], rng=stream(root, ROOM),
                         report=report)
    arm_b = bt.optimizer_strategy(board, my_slot=3, teams=12, seed=root, report=report,
                                  **FROZEN_BUDGET)
    b_names, _ = bt.play(board, arm_b, my_slot=3, teams=12,
                         rounds=FROZEN_BUDGET["rounds"], rng=stream(root, ROOM),
                         report=report)
    assert a_names == FROZEN_ARM_A, (
        f"THE ROOM MOVED: arm A (the draft market, which the product ships) drafted a "
        f"different roster from the frozen Board at the same root. The simulator or the "
        f"seeding tree changed, and every recorded room is now a different room -- say so "
        f"in the commit and re-pin both arms.\n  pinned: {FROZEN_ARM_A}\n  now:    {a_names}")
    assert b_names == FROZEN_ARM_B, (
        f"THE ARM UNDER TEST MOVED: championship equity drafted a different roster from the "
        f"frozen Board at the same root and budget, while the room did not move. The "
        f"objective ADR-0009's verdict was measured on is not the one in this tree -- say so "
        f"in the commit and re-pin.\n  pinned: {FROZEN_ARM_B}\n  now:    {b_names}")


def test_compare_plays_the_pinned_draft():
    """`compare`'s row is the score of exactly the rosters pinned above.

    This is what ties the pin to the harness: the two plays above reach the room through
    `draft_root` and `stream(root, ROOM)`, and this asserts that `compare` does too, on a
    Board whose rosters are known -- so a seeding change inside `compare` that the direct
    plays cannot see fails here. It also runs after the direct plays in the same process, so
    a room advanced by module-level state would score a different draft.
    """
    board = _frozen_board()
    real = _frozen_realised(board)
    got = bt.compare({FROZEN_SEASON: board}, {FROZEN_SEASON: real}, n_drafts=1,
                     seed=FROZEN_SEED, my_slot=3, teams=12, **FROZEN_BUDGET)
    assert got.height == 1
    row = got.row(0, named=True)
    a_pos = [board.filter(pl.col("player") == n)["pos"][0] for n in FROZEN_ARM_A]
    b_pos = [board.filter(pl.col("player") == n)["pos"][0] for n in FROZEN_ARM_B]
    assert row["market"] == pytest.approx(bt.score_roster(FROZEN_ARM_A, a_pos, real)), (
        "compare's market column is not the score of the pinned arm-A roster: compare is "
        "playing a different room from the one the pin was taken in")
    assert row["optimizer"] == pytest.approx(bt.score_roster(FROZEN_ARM_B, b_pos, real)), (
        "compare's optimizer column is not the score of the pinned arm-B roster: compare is "
        "playing a different arm B from the one the pin was taken in")


# --- the gate names its own reads, however it is invoked (issue #192) ---------------------
#
# #165 scoped the process-global read set with `reads_of_one_run` and left this gate unwired,
# because a process that does one thing has nothing else to fold in. That is true only by
# accident of how it is invoked: called in-process by a harness running two gates, or by a
# notebook, it inherited the enclosing run's reads and published a digest over bytes it never
# touched -- a failure that looks exactly like a clean digest.

def _in_process_gate(monkeypatch, tmp_path, *, inner, argv=()):
    """Drive `main`'s gate path with the network replaced.

    The walk-forward loader records `inner` as the one read it made and hands back a
    synthetic season; `compare` returns a small paired frame, since the room is exercised
    above and the seam here is what the stamp names. The width history is pointed nowhere.
    """
    from functools import partial

    from hub.fetch import nflverse as nv
    from hub.models.experiment import run_gate

    board = _full_board(24)
    real = _flat_realised(board)

    def loads(seasons, load, *, on_season=None):
        nv._remember(tmp_path / "the-gates-own-entry.parquet", inner)
        return {2024: board}, {2024: real}

    paired = pl.DataFrame({"season": [2024] * 4, "draft": [0, 1, 2, 3],
                           "market": [10.0, 11.0, 9.0, 10.5],
                           "optimizer": [9.0, 10.0, 8.5, 9.0],
                           "market_failed": [0] * 4, "optimizer_failed": [0] * 4,
                           "picks": [4] * 4})
    paired = paired.with_columns((pl.col("optimizer") - pl.col("market")).alias("diff"))
    monkeypatch.setattr(bt, "walk_forward_inputs", loads)
    monkeypatch.setattr(bt, "compare", lambda *a, **k: paired)
    monkeypatch.setattr(bt, "run_gate", partial(run_gate, record_width=False, bootstrap=100))
    out = tmp_path / "paired.parquet"
    assert bt.main(["--seasons", "2024", "--drafts", "4", "--out", str(out), *argv]) == 0
    return pl.read_parquet(out)


def test_the_gate_called_in_process_names_only_its_own_reads(monkeypatch, tmp_path, capsys):
    """Called from inside a run that has already read something, the gate's published digest
    covers the gate's reads and not the enclosing run's -- and the enclosing run still ends
    up holding both, because a scope narrows what a component reports and must never be a
    way for a run to lose a read."""
    from hub.config import data_digest
    from hub.fetch import nflverse as nv

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    outer = nv.Pin(source="player_stats", as_of=None, digest="0ut51de0", rows=1,
                   pinned_at=None)
    inner = nv.Pin(source="ff_opportunity", as_of="2024-09-01", digest="1n51de01", rows=1,
                   pinned_at=None)
    # The enclosing run: something else in this process has read a source already.
    nv._remember(tmp_path / "the-enclosing-runs-entry.parquet", outer)

    stamped = _in_process_gate(monkeypatch, tmp_path, inner=inner)

    assert stamped["data_digest"].unique().to_list() == [data_digest([inner])], (
        "the gate's digest is not a digest over the gate's own read")
    assert stamped["data_digest"][0] != data_digest([outer, inner]), (
        "the gate published a digest over the enclosing run's reads as well as its own")
    assert "over 1 pinned source(s)" in capsys.readouterr().out
    assert sorted(p.source for p in nv.pins_this_run()) == ["ff_opportunity",
                                                            "player_stats"], (
        "the gate's read did not reach the run around it: scoping lost a read")
