"""Ranking the board for one specific pick.

Slot 3 of 12 alternates 19-pick and 5-pick waits, and that single fact should drive
the pick. After a 19-pick wait the question is "who will not survive it", so
cost_of_waiting (VOR x P(gone)) rules. Before a 5-pick wait almost anyone you like
survives, so availability is noise and raw VOR rules.

Ranking by `edge` alone would draft replacement-level players: the biggest edges sit
on players consensus does not rate, which is exactly the mixed-room dynamic.
"""
import polars as pl
import pytest

from hub.draft.board import recommend


def _board(n=60):
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": ["RB" if i % 2 else "WR" for i in range(n)],
        "ecr": [float(i + 1) for i in range(n)],
        "sd": [3.0] * n,
        "adp": [float(i + 1) for i in range(n)],
        # VOR deliberately anti-correlated with ECR so the two modes cannot agree
        # by accident: the highest-VOR player is ranked near the bottom by consensus.
        "vor": [float(i) for i in range(n)],
    })


def test_first_pick_is_scarcity_mode():
    mode, _ = recommend(_board(), current_pick=3)
    assert mode == "scarcity"


def test_second_round_pick_is_value_mode():
    """Pick 22 is followed by pick 27 -- a 5-pick wait, so availability is noise."""
    mode, _ = recommend(_board(), current_pick=22)
    assert mode == "value"


def test_value_mode_ranks_by_raw_vor():
    _, out = recommend(_board(), current_pick=22, top=3)
    assert out["vor"].to_list() == sorted(out["vor"].to_list(), reverse=True)
    assert out["player"][0] == "P59"


def test_scarcity_mode_ranks_by_cost_of_waiting():
    _, out = recommend(_board(), current_pick=3, top=5)
    assert "cost_of_waiting" in out.columns
    cw = out["cost_of_waiting"].to_list()
    assert cw == sorted(cw, reverse=True)


def test_the_two_modes_disagree():
    """If they agreed the mode switch would be decoration."""
    _, scarce = recommend(_board(), current_pick=3, top=5)
    _, value = recommend(_board(), current_pick=22, top=5)
    assert scarce["player"].to_list() != value["player"].to_list()


def test_top_is_respected():
    _, out = recommend(_board(), current_pick=3, top=4)
    assert out.height == 4


def test_scarcity_mode_drops_players_who_will_be_gone():
    """No point ranking someone with a 5% chance of lasting to your turn."""
    _, out = recommend(_board(120), current_pick=75, top=200)
    assert out.height < 120


def test_works_without_adp_in_ecr_only_mode():
    b = _board().drop("adp")
    mode, out = recommend(b, current_pick=3, top=3)
    assert mode == "scarcity" and out.height == 3


def test_pick_not_on_your_schedule_is_rejected():
    with pytest.raises(ValueError):
        recommend(_board(), current_pick=4)
