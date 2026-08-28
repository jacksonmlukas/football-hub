"""Scoring the half of objective 1 that is reconstructible.

Which players you took follows from your slot and the snake order; what THE PICK would have
said follows from the board plus the picks made by then. Why you deviated follows from nothing,
which is why the runbook has you write it on paper.

All offline.
"""
import json

import polars as pl

from hub.draft import adherence
from hub.draft.picks import my_picks
from hub.draft.state import DraftState


def _board(names, pos, adp=None, ecr=None):
    n = len(names)
    return pl.DataFrame({
        "player": names, "pos": pos,
        "adp": adp if adp is not None else [float(i + 1) for i in range(n)],
        "ecr": ecr if ecr is not None else [float(i + 1) for i in range(n)],
        "vor": [float(n - i) for i in range(n)],
    })


def _full_draft(board, upto):
    """A pure-ADP room: everyone takes the best remaining by ADP."""
    return DraftState(taken=board.sort("adp")["player"].to_list()[:upto])


def test_the_pick_is_computed_against_the_state_before_your_turn():
    """Recomputing against the finished draft asks a different question and flatters every
    answer -- the players you passed on are gone by then."""
    import inspect
    src = inspect.getsource(adherence.replay)
    assert "overall - 1" in src


def test_only_your_turns_are_scored():
    names = [f"P{i}" for i in range(200)]
    board = _board(names, ["RB", "WR", "TE", "QB"] * 50)
    scored = adherence.replay(board, _full_draft(board, 192))
    assert scored["overall"].to_list() == my_picks(adherence.ROUNDS)


def test_turns_the_draft_never_reached_are_not_scored():
    """A draft abandoned in round 4 should report four turns, not sixteen blanks."""
    names = [f"P{i}" for i in range(200)]
    board = _board(names, ["RB", "WR", "TE", "QB"] * 50)
    scored = adherence.replay(board, _full_draft(board, 50))
    assert scored.height == sum(1 for p in my_picks(adherence.ROUNDS) if p <= 50)


def test_a_name_that_differs_only_in_punctuation_is_not_a_deviation():
    """The board matches picks loosely; grading must too, or an apostrophe reads as a
    deviation you then cannot explain."""
    names = ["Ja'Marr Chase"] + [f"P{i}" for i in range(199)]
    board = _board(names, ["WR"] + ["RB", "WR", "TE", "QB"] * 49 + ["RB", "WR", "TE"])
    taken = board.sort("adp")["player"].to_list()[:192]
    taken[my_picks(adherence.ROUNDS)[0] - 1] = "JaMarr Chase Jr."
    scored = adherence.replay(board, DraftState(taken=taken))
    assert scored["followed"][0] or scored["the_pick"][0] != "Ja'Marr Chase"


# --- the pre-registered threshold ------------------------------------------

def _scored(followed):
    n = len(followed)
    return pl.DataFrame({"round": list(range(1, n + 1)), "overall": list(range(1, n + 1)),
                         "the_pick": [f"A{i}" for i in range(n)],
                         "via": ["x"] * n, "took": [f"B{i}" for i in range(n)],
                         "followed": followed})


def test_the_threshold_is_the_one_fixed_before_the_draft():
    assert adherence.THRESHOLD == 12 and adherence.ROUNDS == 16


def test_twelve_of_sixteen_meets_it():
    ok, text = adherence.verdict(_scored([True] * 12 + [False] * 4))
    assert ok and text.startswith("MET")


def test_eleven_of_sixteen_misses_it():
    """Eleven is what a pure-ADP draft scores: THE PICK fills a need and ADP does not, so the
    threshold is not met by following the market."""
    ok, text = adherence.verdict(_scored([True] * 11 + [False] * 5))
    assert not ok and text.startswith("MISSED")


def test_every_deviation_is_located_so_it_can_be_paired_with_its_note():
    _, text = adherence.verdict(_scored([True] * 14 + [False] * 2))
    assert text.count("round") >= 2
    assert "note you wrote at the time" in text


def test_a_partial_draft_says_so_rather_than_scoring_it_as_a_miss():
    _, text = adherence.verdict(_scored([True] * 5))
    assert "partial" in text


def test_no_turns_is_reported_not_scored():
    ok, text = adherence.verdict(_scored([]))
    assert not ok and "no turns" in text


# --- the CLI ----------------------------------------------------------------

def test_help_needs_no_board():
    import pytest
    with pytest.raises(SystemExit) as e:
        adherence.main(["--help"])
    assert e.value.code == 0


def test_an_absent_board_is_a_sentence(tmp_path, capsys):
    assert adherence.main(["--board", str(tmp_path / "nope.parquet")]) == 1
    assert "no board at" in capsys.readouterr().out


def test_replaying_against_the_live_board_warns(tmp_path, capsys):
    """Every --pick rebuilds it, so it has drifted since the draft. The runbook copies an
    AS-DRAFTED board before the first pick; without it this is approximate."""
    names = [f"P{i}" for i in range(200)]
    board = _board(names, ["RB", "WR", "TE", "QB"] * 50)
    bp = tmp_path / "b.parquet"
    board.write_parquet(bp)
    sp = tmp_path / "s.json"
    sp.write_text(json.dumps({"taken": board.sort("adp")["player"].to_list()[:192]}))
    import pytest

    from hub.draft import board as board_mod
    monkey = pytest.MonkeyPatch()
    monkey.setattr(adherence, "BOARD_PARQUET", bp)
    monkey.setattr(adherence, "AS_DRAFTED", tmp_path / "absent.parquet")
    assert adherence.main(["--state", str(sp)]) == 0
    assert "drifted since the draft" in capsys.readouterr().out
    monkey.undo()
    assert board_mod.BOARD_PARQUET
