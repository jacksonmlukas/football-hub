"""Championship equity, the exhibit: the removed arm's own arithmetic.

These tests lived in `test_optimize.py` while `win_probability`, `_lift_frame` and
`rank_tiers` lived in `hub.draft.optimize`; #198 moved the code to
`hub.exhibits.championship_equity`, where a reader can tell it from what the product reads,
and the tests came with it unchanged. The properties worth pinning are the ones that make
this different from VOR: it must respond to roster construction and to who is already gone,
it must return a real probability rather than a score, and it must say only what the
simulation can resolve -- the paired standard error and the co-leader tier.

`tag_for` is not here. It labelled the equity column the board stopped printing under
ADR-0009, had no caller and no measurement behind it, and was deleted with the move rather
than exhibited; its two tests went with it.
"""
import numpy as np
import polars as pl
import pytest

from hub.draft.state import DraftState
from hub.exhibits.championship_equity import (
    _lift_frame,
    rank_tiers,
    win_probability,
)
from hub.models.predict import CorrelationReport


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

    from hub.exhibits import championship_equity as _opt
    src = _inspect.getsource(_opt.win_probability)
    assert "nfl_team" in src, "champion_probability must receive NFL team identity"


def test_the_room_hands_the_simulator_a_durability_history_when_the_stage_ran(monkeypatch):
    """Issue #183's wiring, which is the half that would fail silently.

    `season.simulate_weeks` reads `missed=None` as "absence not modelled" -- correctly, for a
    board whose advisory durability stage was absorbed. So a `win_probability` that never
    passed the column would leave the whole games-played draw as dead code, every P(win)
    would be the pre-#183 one, and nothing in `test_season.py` would notice, because it
    exercises the simulator directly.

    Which of the two it does is a **provenance** question -- did the stage run -- so it is
    read off the `BuildReport` rather than sniffed off the frame, per #199. Both directions
    are asserted: the flag off must reach the simulator as `None`, or "not modelled" would
    quietly become "everybody is healthy".
    """
    from hub.draft.board import BuildReport
    from hub.exhibits import championship_equity as _opt

    seen: list[object] = []
    real = _opt.champion_probability

    def spy(*a, **k):
        seen.append(k.get("missed"))
        return real(*a, **k)

    monkeypatch.setattr(_opt, "champion_probability", spy)
    board = _board().with_columns(
        pl.Series("missed", [float(i % 9) for i in range(180)]))
    kw = {"my_slot": 3, "rounds": 8, "n_draft_sims": 1, "n_season_sims": 20}

    win_probability(board, DraftState(), ["P0"], report=BuildReport(adp=True,
                                                                   durability=True), **kw)
    assert seen and seen[0] is not None, (
        "the durability stage ran and the simulator was still handed no history, so absence "
        "is not modelled for any board this repo actually builds")
    assert len(seen[0]) == board.height          # type: ignore[arg-type]

    seen.clear()
    win_probability(board, DraftState(), ["P0"], report=BuildReport(adp=True), **kw)
    assert seen and seen[0] is None, (
        "a board whose durability stage did not run was still given a history, so the "
        "report is not what decides and the frame is")


def test_the_room_hands_the_simulator_each_players_bye_when_the_stage_ran(monkeypatch):
    """#226's wiring, the same shape as #183's above. The stage decides, not the frame; a
    board whose bye stage was absorbed reaches the simulator as `None`, and one whose stage
    ran reaches it as one week per player with the unplaced at 0, which zeroes nothing."""
    from hub.draft.board import BuildReport
    from hub.exhibits import championship_equity as _opt

    seen: list[object] = []
    real = _opt.champion_probability

    def spy(*a, **k):
        seen.append(k.get("bye_week"))
        return real(*a, **k)

    monkeypatch.setattr(_opt, "champion_probability", spy)
    board = _board().with_columns(
        pl.Series("bye_week", [(i % 10) + 5 if i % 7 else None for i in range(180)],
                  dtype=pl.Int64))
    kw = {"my_slot": 3, "rounds": 8, "n_draft_sims": 1, "n_season_sims": 20}

    win_probability(board, DraftState(), ["P0"], report=BuildReport(adp=True, bye=True), **kw)
    assert seen and seen[0] is not None, "the bye stage ran and the simulator got no byes"
    got = np.asarray(seen[0])
    assert got.shape == (board.height,) and (got[::7] == 0).all() and (got[1::7] > 0).all()

    seen.clear()
    win_probability(board, DraftState(), ["P0"], report=BuildReport(adp=True), **kw)
    assert seen and seen[0] is None, "the report is not what decides and the frame is"


def test_the_run_can_be_told_how_much_of_it_was_actually_correlated():
    """Issue #171. A run makes candidates x draft-sims simulations, and a block that will
    not factor is silently independent in every one of them. The caller owns the report, so
    the count survives the calls rather than dying with each stack frame."""
    board = _board(n=60).with_columns(
        pl.Series("team", [f"T{i % 8}" for i in range(60)]))
    report = CorrelationReport()
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


def test_the_room_widens_the_talent_spread_of_imputed_players(monkeypatch):
    """#87's wiring: the flag on the frame reaches the simulator as a per-player talent
    spread, wider for the imputed, and a board without the flag is unchanged."""
    from hub.draft.board import BuildReport
    from hub.draft.season import talent_cv_for
    from hub.exhibits import championship_equity as _ce

    seen: list[object] = []
    real = _ce.champion_probability

    def spy(*a, **k):
        seen.append(k.get("talent_cv"))
        return real(*a, **k)

    monkeypatch.setattr(_ce, "champion_probability", spy)
    board = _board().with_columns(pl.Series("xfp_imputed", [i % 5 == 0 for i in range(180)]))
    kw = {"my_slot": 3, "rounds": 8, "n_draft_sims": 1, "n_season_sims": 20}
    win_probability(board, DraftState(), ["P0"], report=BuildReport(adp=True), **kw)
    got = np.asarray(seen[0])
    plain = talent_cv_for(board["pos"].to_numpy())
    assert got.shape == plain.shape and (got[::5] > plain[::5]).all() and (got[1::5] == plain[1::5]).all()

    seen.clear()
    win_probability(_board(), DraftState(), ["P0"], report=BuildReport(adp=True), **kw)
    assert seen[0] is None, "no flag on the frame: the simulator's own default"
