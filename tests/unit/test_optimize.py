"""Championship-equity ranking.

The properties worth pinning are the ones that make this different from VOR: it must
respond to roster construction and to who is already gone, and it must return a real
probability rather than a score.
"""
import numpy as np
import polars as pl
import pytest

from hub.draft import optimize
from hub.draft.optimize import (
    _lift_frame,
    _need_score,
    market_pick,
    rank_tiers,
    simulate_remaining_draft,
    tag_for,
    win_probability,
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
    kw = {"my_slot": 3, "rounds": 8, "n_draft_sims": 2, "n_season_sims": 40, "seed": 7}
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


def test_the_run_can_be_told_how_much_of_it_was_actually_correlated():
    """Issue #171. A run makes candidates x draft-sims simulations, and a block that will
    not factor is silently independent in every one of them. The caller owns the report, so
    the count survives the calls rather than dying with each stack frame."""
    board = _board(n=60).with_columns(
        pl.Series("team", [f"T{i % 8}" for i in range(60)]))
    report = optimize.CorrelationReport()
    # `correlation`, not `report`: since #199 this function takes two reports about two runs
    # -- the `BuildReport` describing the board, and this one describing the simulation.
    win_probability(board, DraftState(), ["P0", "P1"], my_slot=3, rounds=6,
                    n_draft_sims=2, n_season_sims=20, correlation=report)
    assert report.blocks > 0, "no block was counted, so nothing could have been reported"
    assert not report.degraded()
    assert report.note() == f"correlation: all {report.blocks} team blocks factored."


# --- saying only what the simulation can resolve --------------------------
#
# Two fixtures carry the whole argument for the paired standard error, and they are built to
# DISAGREE with each other -- one where the two formulas differ by a factor of two and one
# where they agree to five figures. A single fixture on which both give the same answer would
# have passed the old code and the new one alike and proved nothing about either.
#
# Each is four candidates scored over ten shared simulated futures, the shape `win_probability`
# produces: row i column k is candidate i's championship equity in future k, and column k is
# the same future for everyone.

# A and B are helped by the same futures -- the rollouts where the run on their position comes
# late. Their lifts correlate at +0.80, so the gap between them is steadier than either lift.
CORRELATED_FUTURES = [
    [0.4634, 0.4800, 0.4604, 0.4488, 0.4518, 0.4697, 0.4657, 0.4642, 0.4478, 0.4482],
    [0.4447, 0.4836, 0.4432, 0.4383, 0.4439, 0.4787, 0.4663, 0.4375, 0.4435, 0.4403],
    [0.3966, 0.3800, 0.3996, 0.4112, 0.4082, 0.3903, 0.3943, 0.3958, 0.4122, 0.4118],
    [0.3973, 0.3584, 0.3988, 0.4037, 0.3981, 0.3633, 0.3757, 0.4045, 0.3985, 0.4017],
]
# Same shape, but A's and B's lifts are independent across futures -- correlation +0.00004.
UNCORRELATED_FUTURES = [
    [0.4361, 0.4404, 0.4755, 0.4963, 0.4690, 0.4527, 0.5025, 0.4467, 0.4389, 0.4419],
    [0.4184, 0.4431, 0.4623, 0.4604, 0.4568, 0.4610, 0.4228, 0.4758, 0.4331, 0.4664],
    [0.4239, 0.4196, 0.3845, 0.3637, 0.3910, 0.4073, 0.3575, 0.4133, 0.4211, 0.4181],
    [0.4216, 0.3969, 0.3777, 0.3796, 0.3832, 0.3790, 0.4172, 0.3642, 0.4069, 0.3736],
]


def test_a_single_future_reports_no_error_bar_rather_than_a_spurious_one(): 
    """One simulated future cannot separate anybody, and must not pretend to.

    `ddof=1` over a single column is a division by zero, so both error bars are set to
    exactly zero here and every gap `rank_tiers` computes from them is undefined rather than
    infinite. The branch existed before #167 for `lift_se`; that ticket gave it a second
    column to zero and no test reached either, which the coverage ratchet caught on the
    merge. It is reachable in earnest: a caller asking for one simulation gets this frame.
    """
    got = _lift_frame(["A", "B"], np.array([[1.0], [0.0]], dtype=float))
    assert got["lift_se"].to_list() == [0.0, 0.0]
    assert got["lead_gap_se"].to_list() == [0.0, 0.0], (
        "a single future produced a non-zero standard error for the paired difference"
    )
    # The lift itself is still real -- it is the mean of one number, not an estimate of one.
    assert got.sort("player")["lift"].to_list() == pytest.approx([0.5, -0.5])


def _futures(rows):
    """The candidate frame `win_probability` would return for these simulated futures."""
    return _lift_frame(["A", "B", "C", "D"], np.array(rows, dtype=float))


def _lift_correlation(rows, i, j):
    mat = np.array(rows, dtype=float)
    diff = mat - mat.mean(axis=0)[None, :]
    return float(np.corrcoef(diff[i], diff[j])[0, 1])


def _quadrature_gap_se(frame, player):
    """The superseded expression: two `lift_se` values combined as if unrelated."""
    lead = float(frame["lift_se"][0])
    mine = float(frame.filter(pl.col("player") == player)["lift_se"][0])
    return (lead ** 2 + mine ** 2) ** 0.5


def test_candidates_the_simulation_cannot_separate_are_marked_as_tied():
    """Two candidates 0.002 of championship equity apart, with the paired error on that gap
    at 0.007, are 0.29 standard errors apart and not a distinction. Printing a strict order
    there implies one the simulation cannot make, and the objective is to pick the best
    player -- which is sometimes two players.

    The frame is written out rather than simulated, so what is asserted is the tiering rule
    and not a board. This docstring used to describe a pick-3 board instead -- *"the top two
    differ by 0.17 points of championship equity with standard errors around 0.5"* -- which
    was never this fixture's numbers, and was in any case a tie decided by the quadrature
    expression #167 replaced. Superseded under issue #189; `docs/decisions.md` restates it."""
    df = pl.DataFrame({"player": ["A", "B", "C"], "p_win": [0.48, 0.479, 0.40],
                       "lift": [0.035, 0.033, -0.050],
                       "lift_se": [0.005, 0.005, 0.006],
                       "lead_gap_se": [0.0, 0.007, 0.008]})
    got = rank_tiers(df)
    lead = dict(zip(got["player"].to_list(), got["co_leader"].to_list(), strict=True))
    # The 0.007 is `lead_gap_se`, which is the only error bar the tier reads. The message
    # used to name `lift_se` 0.005, which is the quadrature-era confusion this test's own
    # docstring is restated for.
    assert lead["A"] and lead["B"], "0.002 apart with a paired se of 0.007 is not a distinction"
    assert not lead["C"]


def test_a_clear_winner_is_not_diluted_into_a_tie():
    """The other direction. A tiering rule that called everything tied would be as useless
    as one that called everything separable."""
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.50, 0.40],
                       "lift": [0.05, -0.05], "lift_se": [0.004, 0.004],
                       "lead_gap_se": [0.0, 0.006]})
    got = rank_tiers(df)
    assert got.filter(pl.col("co_leader"))["player"].to_list() == ["A"]


def test_the_gap_is_the_standard_error_of_the_paired_difference():
    """The two lifts come off the same simulated seasons, so the error on their difference is
    the spread of that difference -- one number per future, taken before it is averaged.

    On `CORRELATED_FUTURES` the leader A and the rival B differ by 0.008 of championship
    equity. The paired standard error of that gap is 0.0034421, so the gap is 2.32 of them.
    The superseded quadrature expression put the error at 0.0064602 and the gap at 1.24,
    which is a different answer, not a rounder one."""
    got = rank_tiers(_futures(CORRELATED_FUTURES))
    b = got.filter(pl.col("player") == "B")
    assert float(b["lead_gap_se"][0]) == pytest.approx(0.0034421, abs=5e-7)
    assert float(b["gap_se"][0]) == pytest.approx(2.3242, abs=5e-4)
    assert _quadrature_gap_se(got, "B") == pytest.approx(0.0064602, abs=5e-7)


def test_the_gap_is_measured_against_the_leader_not_the_first_row():
    """Whose gap it is depends on who leads on lift, not on where the caller put him. The
    same futures with the rows reversed must give B the same 2.32."""
    got = rank_tiers(_lift_frame(["D", "C", "B", "A"],
                                 np.array(CORRELATED_FUTURES[::-1], dtype=float)))
    assert got["player"].to_list() == ["A", "B", "C", "D"]
    b = got.filter(pl.col("player") == "B")
    assert float(b["gap_se"][0]) == pytest.approx(2.3242, abs=5e-4)


def test_correlated_lifts_are_tighter_paired_than_in_quadrature():
    """The claim the fix rests on, asserted rather than assumed. A's and B's lifts rise and
    fall together (+0.80 across the ten futures), and quadrature has no way to know that: it
    is the formula for two unrelated estimates. It reports 0.0064602 where the paired
    difference reports 0.0034421 -- 53% of it, a whole factor, not a rounding."""
    assert _lift_correlation(CORRELATED_FUTURES, 0, 1) == pytest.approx(0.8010, abs=5e-4)
    got = _futures(CORRELATED_FUTURES)
    paired = float(got.filter(pl.col("player") == "B")["lead_gap_se"][0])
    quad = _quadrature_gap_se(got, "B")
    assert paired < quad
    assert paired / quad == pytest.approx(0.5328, abs=5e-4)


def test_uncorrelated_lifts_land_on_the_same_number_either_way():
    """The control, and the reason this is a correction rather than a rescale. On
    `UNCORRELATED_FUTURES` A's and B's lifts are independent (+0.00004), which is the one case
    quadrature is the right formula for -- and there the two agree to five figures. A fix that
    moved every number by a constant would fail here, and so would a fixture chosen to make
    both formulas agree everywhere: the correlated pair above separates them by a factor of
    two on the same code path."""
    assert _lift_correlation(UNCORRELATED_FUTURES, 0, 1) == pytest.approx(0.0, abs=1e-3)
    got = _futures(UNCORRELATED_FUTURES)
    paired = float(got.filter(pl.col("player") == "B")["lead_gap_se"][0])
    quad = _quadrature_gap_se(got, "B")
    assert paired == pytest.approx(0.0099086, abs=5e-7)
    assert quad == pytest.approx(0.0099087, abs=5e-7)
    assert paired == pytest.approx(quad, rel=1e-3)


def test_the_tier_that_moves_is_the_one_with_correlated_lifts():
    """Which boundary the change actually moves, named once rather than asserted in general.

    B is the only candidate to cross the two-standard-error line on either fixture. On
    `CORRELATED_FUTURES` he was tied for the lead at 1.24 quadrature errors and is 2.32 paired
    ones away, so he leaves the tier: the drafter is told to take A. On `UNCORRELATED_FUTURES`
    he sits at 1.008 either way and stays in it. C and D are 3.9 or more errors adrift under
    both expressions on both fixtures and never move."""
    moved = {}
    for name, rows in (("correlated", CORRELATED_FUTURES),
                       ("uncorrelated", UNCORRELATED_FUTURES)):
        got = rank_tiers(_futures(rows))
        lead_lift = float(got["lift"][0])
        for r in got.iter_rows(named=True):
            quad = _quadrature_gap_se(got, r["player"])
            was = ((lead_lift - r["lift"]) / quad) < 2.0 if quad else True
            if was != r["co_leader"]:
                moved.setdefault(name, []).append(r["player"])
    assert moved == {"correlated": ["B"]}


def test_the_leader_is_zero_standard_errors_from_itself():
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.5, 0.4],
                       "lift": [0.02, 0.0], "lift_se": [0.01, 0.01],
                       "lead_gap_se": [0.0, 0.014]})
    got = rank_tiers(df)
    assert got["gap_se"][0] == pytest.approx(0.0)


def test_zero_error_estimates_do_not_divide_by_zero():
    """One draft rollout gives a standard error of zero, and the CLI allows it."""
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.5, 0.4],
                       "lift": [0.02, 0.0], "lift_se": [0.0, 0.0],
                       "lead_gap_se": [0.0, 0.0]})
    got = rank_tiers(df)
    assert all(v is not None for v in got["gap_se"].to_list())


def test_a_frame_without_the_paired_error_is_refused():
    """The quantity cannot be reconstructed from `lift_se`, so a frame that lacks it is not
    something to fall back on -- it is a caller who has not been through `win_probability`."""
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.5, 0.4],
                       "lift": [0.02, 0.0], "lift_se": [0.01, 0.01]})
    with pytest.raises(ValueError, match="lead_gap_se"):
        rank_tiers(df)


def test_the_simulation_hands_the_tiering_its_paired_error():
    """The two halves meet: `win_probability` measures the column `rank_tiers` divides by."""
    out = win_probability(_board(), DraftState(), ["P0", "P1", "P50"], my_slot=3,
                          rounds=8, n_draft_sims=3, n_season_sims=40)
    assert "lead_gap_se" in out.columns
    assert float(out["lead_gap_se"][0]) == pytest.approx(0.0), "the leader against itself"
    assert (out["lead_gap_se"] >= 0).all()
    assert rank_tiers(out)["co_leader"].any()


def test_a_significantly_positive_candidate_is_never_labelled_avoid():
    """Found on the live board: Ja'Marr Chase came out at +0.95 lift with a standard error
    of 0.37 and was tagged `avoid`, because the tag fired on "significantly different from
    zero" without checking the sign. He is significantly *better* than the field. On a draft
    board that is not a cosmetic bug."""
    df = pl.DataFrame({"player": ["A", "B", "C"], "p_win": [0.50, 0.46, 0.41],
                       "lift": [0.04, 0.0095, -0.030],
                       "lift_se": [0.004, 0.0037, 0.005],
                       "lead_gap_se": [0.0, 0.0054, 0.0064]})
    r = rank_tiers(df)
    got = {x["player"]: tag_for(x["co_leader"], x["lift"], x["lift_se"])
           for x in r.iter_rows(named=True)}
    assert got["A"] == "TAKE"
    assert got["B"] != "avoid", "positive lift must never read as avoid"
    assert got["C"] == "avoid"


def test_a_candidate_indistinguishable_from_the_field_is_left_unmarked():
    df = pl.DataFrame({"player": ["A", "B"], "p_win": [0.50, 0.45],
                       "lift": [0.04, 0.001], "lift_se": [0.004, 0.004],
                       "lead_gap_se": [0.0, 0.0057]})
    r = rank_tiers(df)
    got = {x["player"]: tag_for(x["co_leader"], x["lift"], x["lift_se"])
           for x in r.iter_rows(named=True)}
    assert got["B"] == ""


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
