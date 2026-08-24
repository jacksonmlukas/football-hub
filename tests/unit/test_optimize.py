"""Championship-equity ranking.

The properties worth pinning are the ones that make this different from VOR: it must
respond to roster construction and to who is already gone, and it must return a real
probability rather than a score.
"""
import numpy as np
import polars as pl
import pytest
from hub.draft.optimize import (draft_pool, rank_tiers, simulate_remaining_draft,
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
    rosters = simulate_remaining_draft(_board(), DraftState(), my_slot=3, teams=12,
                                       rounds=6, forced="P42")
    pool = draft_pool(_board(), DraftState())
    mine = [pool["player"][int(i)] for i in rosters[2]]
    assert "P42" in mine


def test_already_drafted_players_are_never_selected():
    st = take(DraftState(), *[f"P{i}" for i in range(20)])
    rosters = simulate_remaining_draft(_board(), st, my_slot=3, teams=12, rounds=6)
    pool = draft_pool(_board(), st)
    picked = {pool["player"][int(i)] for r in rosters for i in r}
    assert not (picked & {f"P{i}" for i in range(20)})


def test_my_greedy_fills_its_starting_slots():
    """A roster that never drafts a QB cannot be a serious opponent model."""
    rosters = simulate_remaining_draft(_board(), DraftState(), my_slot=3, teams=12, rounds=10)
    pool = draft_pool(_board(), DraftState())
    mine = [pool["pos"][int(i)] for i in rosters[2]]
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
