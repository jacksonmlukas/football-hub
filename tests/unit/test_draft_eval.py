"""Evaluating projection_lambda over simulated drafts.

Every earlier metric scored one static ordering. A draft is not an ordering -- it is a
sequence of decisions against a pool other people are emptying, and the value of a board
is the roster it actually produces.

This is the evaluation the top-50 metric could not be: a full 16-round snake per trial,
scored on starters only, repeated across all twelve slots. There is no boundary for one
injured player to sit on, and every pick reads the whole board.

The load-bearing tests are the same discipline as the sweep: recover a planted signal,
return nothing on noise, and refuse to be moved by a single player.
"""
import numpy as np
import polars as pl
import pytest

from hub.draft import evaluate as ev


def _pool(n=250, *, signal_strength=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cycle = ["RB", "WR", "WR", "TE", "QB"]
    ecr = np.arange(1, n + 1, dtype=float)
    z = rng.normal(0, 1, n)
    true_rank = ecr - signal_strength * z * 25 + rng.normal(0, 8, n)
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": [cycle[i % len(cycle)] for i in range(n)],
        "ecr": ecr, "z_regress": z,
        "actual_points": np.clip(320.0 - true_rank, 0, None),
    })


# --- scoring a roster -----------------------------------------------------

def test_only_starters_count():
    """A surplus player scores zero. You can start three WRs."""
    roster = pl.DataFrame({"pos": ["WR"] * 6, "actual_points": [100.0] * 6})
    assert ev.starter_points(roster) == pytest.approx(400.0)  # 3 WR + 1 flex


def test_an_empty_slot_scores_nothing_rather_than_borrowing():
    roster = pl.DataFrame({"pos": ["RB", "RB"], "actual_points": [100.0, 90.0]})
    # RB2 filled, flex takes nothing left over; no QB/WR/TE at all
    assert ev.starter_points(roster) == pytest.approx(190.0)


def test_the_flex_takes_the_best_leftover():
    roster = pl.DataFrame({"pos": ["RB", "RB", "RB"], "actual_points": [10.0, 10.0, 50.0]})
    assert ev.starter_points(roster) == pytest.approx(70.0)


def test_a_full_lineup_uses_eight_players():
    roster = pl.DataFrame({
        "pos": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB"],
        "actual_points": [10.0] * 8})
    assert ev.starter_points(roster) == pytest.approx(80.0)


# --- the draft itself -----------------------------------------------------

def test_every_team_gets_a_full_roster_and_nobody_is_drafted_twice():
    rosters = ev.simulate_draft(_pool(), lam=0.0, my_slot=3,
                                rng=np.random.default_rng(0))
    picked = [p for r in rosters for p in r]
    assert len(picked) == len(set(picked))
    assert all(len(r) == ev.ROUNDS for r in rosters)


def test_my_slot_gets_the_picks_the_snake_says():
    rosters = ev.simulate_draft(_pool(), lam=0.0, my_slot=1,
                                rng=np.random.default_rng(0))
    assert len(rosters[0]) == ev.ROUNDS


def test_a_draft_is_deterministic_under_a_seed():
    a = ev.simulate_draft(_pool(), 0.0, 3, rng=np.random.default_rng(5))
    b = ev.simulate_draft(_pool(), 0.0, 3, rng=np.random.default_rng(5))
    assert a == b


def test_drafting_earlier_is_worth_more():
    """Sanity: if slot did not matter the simulation would not be modelling a draft."""
    pool = _pool()
    early = np.mean([ev.trial(pool, 0.0, 1, np.random.default_rng(s)) for s in range(12)])
    late = np.mean([ev.trial(pool, 0.0, 12, np.random.default_rng(s)) for s in range(12)])
    assert early > late


# --- the evaluation -------------------------------------------------------

def test_a_planted_signal_is_recovered():
    """The whole point: a board that genuinely knows something should draft better."""
    pool = _pool(signal_strength=1.2, seed=1)
    got = ev.evaluate(pool, lams=[0.0, 0.15], n_sims=6)
    lift = got.filter(pl.col("lam") == 0.15)["lift"][0]
    assert lift > 0


def test_noise_produces_no_lift():
    pool = _pool(signal_strength=0.0, seed=2)
    got = ev.evaluate(pool, lams=[0.0, 0.15], n_sims=6)
    assert ev.best_lambda(got) == 0.0


def test_one_huge_player_does_not_move_the_answer():
    """The 2023-24 failure mode. A single enormous outcome must not decide this."""
    pool = _pool(signal_strength=0.0, seed=3)
    spiked = pool.with_columns(
        pl.when(pl.col("player") == "P60").then(9999.0)
          .otherwise(pl.col("actual_points")).alias("actual_points"))
    assert ev.best_lambda(ev.evaluate(spiked, lams=[0.0, 0.15], n_sims=6)) == 0.0


def test_lambda_zero_is_exactly_the_baseline():
    got = ev.evaluate(_pool(), lams=[0.0, 0.1], n_sims=4)
    assert got.filter(pl.col("lam") == 0.0)["lift"][0] == pytest.approx(0.0)


def test_every_slot_is_sampled():
    """Averaging over all twelve seats removes the seat as a confound."""
    got = ev.evaluate(_pool(), lams=[0.0], n_sims=3)
    assert got["n_trials"][0] == 3 * ev.TEAMS
