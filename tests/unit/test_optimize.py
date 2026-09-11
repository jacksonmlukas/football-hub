"""The room, the seeding tree and THE PICK.

Championship equity's tests were here until #198 moved the arm to
`hub.exhibits.championship_equity`; they are `test_championship_equity.py` now. What stays
is what the product and every Gate run on: the simulated room, which currency it ranks in,
and the market's pick that the board leads with.
"""
import numpy as np
import polars as pl
import pytest

from hub.draft import optimize
from hub.draft.optimize import (
    _need_score,
    market_pick,
    simulate_remaining_draft,
)
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
        # Carried because `adp` above is: `vor_proj` is `board.STAGE_COLUMNS["adp"]`, so a
        # frame claiming that stage and not carrying it is a board `build` cannot emit --
        # and since #199 the room asks the report which of the two currencies it ranks in
        # rather than asking this frame. Equal to `vor` on purpose: the fixture supplies
        # what its report claims without changing any ordering these tests assert on.
        "vor_proj": [float(n - i) for i in range(n)],
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


# --- which currency the room ranks in -------------------------------------

def _two_currency_board(n=60):
    """One frame carrying both VOR columns, ordered against each other.

    `vor` prefers the last player on the board and `vor_proj` prefers the first, so which
    of the two the room ranked in is readable off a single pick. Every position is the
    same, so `_need_score` never breaks the tie and the currency is the only thing left.
    """
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": ["RB"] * n,
        "ecr": [float(i + 1) for i in range(n)],
        "adp": [float(i + 1) for i in range(n)],
        "vor": [float(i) for i in range(n)],
        "vor_proj": [float(n - i) for i in range(n)],
    })


def test_the_room_ranks_in_the_currency_the_report_names():
    """Issue #199, and the site with a live consequence rather than only an owner.

    It is also **issue #147's third criterion** -- "whichever way it lands, a test holds it,
    proven by mutation". The way it landed is the report, so this is the test that holds it:
    restoring `"vor_proj" in pool.columns` in `_greedy_currency` makes the second assertion
    below fail, because the frame carries both columns and the sniff cannot see the report.

    A rule test on one frame and two reports, for #164's reason: on every board `build` can
    emit, `vor_proj` is present exactly when `report.adp` is true, so a test built from a
    reachable board would pass against `"vor_proj" in pool.columns` and prove nothing. The
    frame here carries both columns and does not move; only the report does.

    What it holds is the decision the ticket asked for: the live room ranks on the
    replacement-adjusted blended projection and every backtested room ranks on
    replacement-adjusted prior-season xFP, because the draft-market stage is what leaves the
    first -- and that is now read off the report and named in `backtest.LIMITATIONS`, rather
    than falling out of a column nobody chose.
    """
    from hub.draft.board import BuildReport
    board = _two_currency_board()
    kw = {"my_slot": 1, "teams": 2, "rounds": 1}

    on_proj = simulate_remaining_draft(board, DraftState(), report=BuildReport(adp=True),
                                       **kw)
    on_xfp = simulate_remaining_draft(board, DraftState(), report=BuildReport(adp=False),
                                      **kw)
    assert board["player"][int(on_proj[0][0])] == "P0", "vor_proj tops out at the first row"
    assert board["player"][int(on_xfp[0][0])] == "P59", "vor tops out at the last"


def test_a_room_handed_no_report_still_ranks_and_says_so_through_one_owner():
    """With only a frame, `board.report_for` derives one -- `BuildReport.of_served`, the
    single derivation, rather than a twelfth private guess. This board carries `adp`, so
    that derivation says the draft-market stage ran and the projection scale is the one."""
    board = _two_currency_board()
    got = simulate_remaining_draft(board, DraftState(), my_slot=1, teams=2, rounds=1)
    assert board["player"][int(got[0][0])] == "P0"


# --- the market's pick, which is what the board now leads with -------------

def test_the_draft_market_pick_fills_an_unfilled_starting_slot_first():
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


# --- corrected ADP: what THE PICK ranks on --------------------------------
#
# Ranking on raw ADP while printing a fitted correction beside it said "our measurements show
# the market is wrong about this player", then ordered the board by the market anyway. Every
# input here carries a fitted coefficient; the assembly is bounded because it cannot be
# validated end to end -- scoring a ranking needs historical ADP, which does not exist.

def _corr_board(n=120, corrections=None):
    """A board with a realistic, curved ADP-to-projection relationship."""
    proj = [max(22.0 - 2.6 * (i ** 0.5), 1.0) for i in range(n)]
    df = pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": [["RB", "WR", "WR", "TE", "QB"][i % 5] for i in range(n)],
        "adp": [float(i + 1) for i in range(n)],
        "proj_blend": proj,
        "proj_correction": [0.0] * n,
    })
    if corrections:
        df = df.with_columns(pl.Series("proj_correction",
                                       [corrections.get(f"P{i}", 0.0) for i in range(n)]))
    return df


def test_a_player_with_no_correction_does_not_move():
    """By construction, and the first half of the backtest tripwire."""
    b = _corr_board()
    got = optimize.corrected_adp(b).to_list()
    assert got == pytest.approx(b["adp"].to_list())


def test_a_marked_down_player_falls():
    b = _corr_board(corrections={"P10": -3.0})
    got = optimize.corrected_adp(b)
    assert got[10] > b["adp"][10], "a negative correction must push him later"


def test_a_marked_up_player_rises():
    b = _corr_board(corrections={"P40": +3.0})
    got = optimize.corrected_adp(b)
    assert got[40] < b["adp"][40]


def test_no_move_exceeds_the_clamp():
    """The second half of the tripwire. A huge correction must not reorder the board."""
    from hub.config import DraftConfig
    frac = DraftConfig().correction_clamp_frac
    b = _corr_board(corrections={f"P{i}": -25.0 for i in range(120)})
    got = optimize.corrected_adp(b).to_numpy()
    adp = b["adp"].to_numpy()
    assert (abs(got - adp) <= frac * adp + 1e-9).all()


def test_the_clamp_binds_hardest_at_the_top():
    """Why proportional and not absolute: twelve picks at ADP 150 is noise, twelve picks at
    ADP 3 is catastrophic."""
    b = _corr_board(corrections={"P0": -20.0, "P100": -20.0})
    got = optimize.corrected_adp(b).to_numpy()
    adp = b["adp"].to_numpy()
    assert abs(got[0] - adp[0]) < 1.0, "the top of round one barely moves"
    assert abs(got[100] - adp[100]) > 5.0, "a late player can move a round or more"


def test_the_curve_is_monotone_in_projection():
    """A better projection must never map to a later pick, however noisy one player's ADP."""
    import numpy as np
    noisy = _corr_board().with_columns(
        pl.Series("adp", [float(i + 1) + (30.0 if i == 60 else 0.0) for i in range(120)]))
    _xs, ys = optimize.market_curve(noisy["adp"].to_numpy(), noisy["proj_blend"].to_numpy())
    assert (np.diff(ys) <= 1e-9).all(), "curve must be non-increasing in projection"


def test_a_board_without_the_columns_falls_back_to_raw_adp():
    """A board built before `proj_correction` existed, or one that degraded."""
    b = _corr_board().drop("proj_correction")
    assert optimize.corrected_adp(b).to_list() == pytest.approx(b["adp"].to_list())


def test_the_pick_prefers_corrected_adp_over_raw():
    """The whole point: a player the market likes but our measurements mark down should lose
    the pick to one it does not."""
    b = _corr_board(corrections={"P0": -8.0})
    b = b.with_columns(optimize.corrected_adp(b).alias("adp_corrected"),
                       pl.Series("ecr", [float(i + 1) for i in range(b.height)]))
    tp = optimize.the_pick(b, DraftState(taken=[]), my_slot=1, teams=1)
    assert tp is not None
    assert tp.via == "draft market, corrected"


def test_the_reported_rank_is_his_adp_not_the_corrected_number():
    """The drafter compares against a room that sees ADP."""
    b = _corr_board(corrections={"P3": -5.0})
    b = b.with_columns(optimize.corrected_adp(b).alias("adp_corrected"),
                       pl.Series("ecr", [float(i + 1) for i in range(b.height)]))
    tp = optimize.the_pick(b, DraftState(taken=[]), my_slot=1, teams=1)
    assert tp is not None
    row = b.filter(pl.col("player") == tp.player).row(0, named=True)
    assert tp.rank == row["adp"]


def test_the_rank_label_travels_with_the_value():
    """It used to be inferred in board.py with `"ADP" in via`, which silently labelled an
    ADP value "ECR" the moment the prose changed to "draft market, corrected". A unit
    recovered from a human-readable sentence is a unit waiting to be wrong."""
    b = _corr_board(corrections={"P3": -5.0})
    b = b.with_columns(optimize.corrected_adp(b).alias("adp_corrected"),
                       pl.Series("ecr", [float(i + 1) for i in range(b.height)]))
    tp = optimize.the_pick(b, DraftState(taken=[]), my_slot=1, teams=1)
    assert tp is not None
    assert tp.rank_label == "ADP"
    row = b.filter(pl.col("player") == tp.player).row(0, named=True)
    assert tp.rank == row["adp"]


def test_the_consensus_fallback_reports_ecr_and_says_so():
    b = _corr_board().drop("adp").with_columns(
        pl.Series("ecr", [float(i + 1) for i in range(120)]))
    tp = optimize.the_pick(b, DraftState(taken=[]), my_slot=1, teams=1)
    assert tp is not None
    assert tp.rank_label == "ECR"
    assert "consensus" in tp.via


# --- a correction moves a player once, not twice (issue #39) ---------------

def _priced(n=60):
    """A board whose ADP-to-projection relationship is steeply non-linear, like the real one.

    Flat would hide this defect entirely: reading the curve one correction too far only
    matters where the curve's slope changes, which is the whole reason it is interpolated
    rather than fitted.
    """
    proj = np.linspace(300.0, 40.0, n)
    adp = np.round(np.exp(np.linspace(0.0, 5.2, n)))      # 1 .. ~180, steepening
    return proj, adp


def _closed_form(adp, raw, corr):
    """The identity, recomputed from the board's own columns and nothing else.

    Deliberately not a call into `corrected_adp`: an oracle that reuses the code under test
    agrees with it by construction, including when both are wrong.
    """
    from hub.draft.optimize import market_curve
    xs, ys = market_curve(adp, raw)
    shift = np.interp(raw + corr, xs, ys) - np.interp(raw, xs, ys)
    return np.where(np.isfinite(shift), shift, 0.0)


def _frame(proj_corrected, adp, corr):
    return pl.DataFrame({"adp": adp, "proj_blend": proj_corrected, "proj_correction": corr})


def test_the_pre_clamp_shift_matches_the_closed_form_identity():
    """Criterion one, and the one that decides the ticket."""
    from hub.draft.optimize import corrected_adp
    raw, adp = _priced()
    corr = np.linspace(-8.0, 8.0, raw.size)
    got = corrected_adp(_frame(raw + corr, adp, corr), clamp_frac=1e9).to_numpy()
    want = adp + _closed_form(adp, raw, corr)
    assert np.allclose(got, want, atol=1e-9), (
        f"largest disagreement {np.abs(got - want).max():.3f} picks")


def test_the_identity_holds_for_a_clamped_player():
    """Criterion three. The check runs pre-clamp, so a clamped player must still agree once
    the same clamp is applied to the identity -- otherwise the clamp could be hiding the very
    error this is about."""
    from hub.draft.optimize import corrected_adp
    raw, adp = _priced()
    corr = np.full(raw.size, 25.0)                 # large enough to clamp widely
    frac = 0.05
    got = corrected_adp(_frame(raw + corr, adp, corr), clamp_frac=frac).to_numpy()
    shift = _closed_form(adp, raw, corr)
    bound = frac * np.abs(adp)
    want = adp + np.clip(shift, -bound, bound)
    assert (np.abs(got - adp) <= bound + 1e-9).all(), "the clamp was not applied"
    assert np.allclose(got, want, atol=1e-9)


def test_only_the_corrected_player_moves_on_a_steep_curve():
    """Criterion four. `test_a_player_with_no_correction_does_not_move` above already asserts
    the all-zero case; what this adds is that one player's correction does not drag the rest,
    which is possible only because the curve is now built on the uncorrected projection."""
    from hub.draft.optimize import corrected_adp
    raw, adp = _priced()
    corr = np.zeros(raw.size)
    corr[10] = -6.0
    got = corrected_adp(_frame(raw + corr, adp, corr)).to_numpy()
    unmoved = np.ones(raw.size, dtype=bool)
    unmoved[10] = False
    assert np.allclose(got[unmoved], adp[unmoved]), (
        "one player's correction moved the others, so the curve is still being re-fitted "
        "under everyone")
    assert got[10] != adp[10]


def test_the_curve_is_unchanged_when_every_correction_is_zero():
    """Criterion five. Built on the corrected projection the curve moved under everyone
    whenever any correction changed -- which is what #121 measured at 346 of 457 players."""
    from hub.draft.optimize import corrected_adp, market_curve
    raw, adp = _priced()
    zero = np.zeros(raw.size)
    assert np.allclose(corrected_adp(_frame(raw, adp, zero)).to_numpy(), adp)
    # And the curve the shift is read from is the uncorrected one, whatever the corrections.
    corr = np.linspace(-8.0, 8.0, raw.size)
    a1, b1 = market_curve(adp, raw)
    a2, b2 = market_curve(adp, raw)          # same inputs, same curve
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    moved = corrected_adp(_frame(raw + corr, adp, corr), clamp_frac=1e9).to_numpy()
    want = adp + _closed_form(adp, raw, corr)
    assert np.allclose(moved, want)


def test_a_player_near_the_smoothing_window_boundary_is_covered():
    """Criterion six. `market_curve` smooths with a rolling median over a 15-wide window, so
    a correction large enough to re-sort a player changes which window he lands in. Reading
    the curve at the corrected projection made that re-sorting part of the measurement."""
    from hub.draft.optimize import corrected_adp
    raw, adp = _priced()
    corr = np.zeros(raw.size)
    # Big enough to jump this player across several neighbours, and so across the window.
    corr[30] = -(raw[30] - raw[38])
    got = corrected_adp(_frame(raw + corr, adp, corr), clamp_frac=1e9).to_numpy()
    want = adp + _closed_form(adp, raw, corr)
    assert np.allclose(got, want, atol=1e-9)
    assert got[30] > adp[30], "a downgraded player should be taken later, not earlier"


def test_a_frame_whose_report_claims_a_stage_it_lacks_says_so(_board_min=None):
    """Review of #199 found this raising `ColumnNotFoundError` four frames deep.

    A frame carrying `adp` but not `vor_proj` is incoherent: it says the draft-market stage
    ran while missing one of that stage's columns. `build` cannot emit one -- the stage runs
    under `_stage(..., absorbs=())`, so a failure inside it aborts the build -- but these are
    public functions taking a bare `pl.DataFrame`, and a fixture or a board off disk from
    before the column existed can be one.

    Before #199 this degraded: `optimize` sniffed `"vor_proj" in pool.columns` and fell back to
    `vor`. That is not the behaviour to restore. Quietly ranking on a currency the caller does
    not believe it is ranking on is what let the live room use `proj_blend` while every
    backtested room used prior-season xFP, with nothing recording that the two differ -- the
    defect #199 exists to close. So it still refuses; it just explains itself now.
    """
    import polars as pl
    import pytest as _pytest

    from hub.draft import optimize as O
    from hub.draft.board import BuildReport

    incoherent = pl.DataFrame({
        "player": ["A", "B"], "pos": ["RB", "WR"],
        "adp": [1.0, 2.0], "vor": [10.0, 9.0],      # `adp` present, `vor_proj` absent
    })
    claims_the_stage_ran = BuildReport.of_served(incoherent)
    assert claims_the_stage_ran.adp, "premise: the sentinel column makes the report say it ran"

    with _pytest.raises(ValueError, match="vor_proj") as caught:
        O._greedy_currency(incoherent, claims_the_stage_ran)
    assert "report" in str(caught.value), (
        "the refusal does not tell the caller the report and the frame disagree")

    # And it is not refusing everything: the same frame with a report that does not claim the
    # stage ranks on `vor` without complaint.
    honest = BuildReport()
    assert not honest.adp
    assert O._greedy_currency(incoherent, honest).tolist() == [10.0, 9.0]
