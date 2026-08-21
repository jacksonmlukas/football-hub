"""Championship-equity ranking.

The properties worth pinning are the ones that make this different from VOR: it must
respond to roster construction and to who is already gone, and it must return a real
probability rather than a score.
"""
import numpy as np
import polars as pl
import pytest
from hub.draft.optimize import (draft_pool, simulate_remaining_draft,
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
