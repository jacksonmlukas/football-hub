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


# --- a stale as-drafted copy (found rehearsing the runbook, 2026-08-27) -----
#
# A *missing* copy is reported loudly. A *stale* one was not, and that is the
# silent-plausible-wrong case: a copy left behind by a rehearsal grades you against a board
# you never saw, carrying exactly the ADP drift the copy exists to eliminate.


def _touch(tmp_path, hours_old):
    import os
    import time
    p = tmp_path / "draft_board.AS-DRAFTED.parquet"
    p.write_bytes(b"x")
    t = time.time() - hours_old * 3600
    os.utime(p, (t, t))
    return p


def test_a_fresh_copy_reports_its_age_and_nothing_else(tmp_path):
    lines = adherence.age_note(_touch(tmp_path, 2.0))
    assert len(lines) == 1 and "2.0h old" in lines[0]


def test_a_copy_from_last_week_says_it_is_a_rehearsal_leftover(tmp_path):
    lines = adherence.age_note(_touch(tmp_path, 7 * 24.0))
    assert len(lines) == 2
    assert "WARNING" in lines[1] and "7.0 days" in lines[1]
    assert "--board" in lines[1], "and says how to override it"


def test_the_boundary_is_a_day(tmp_path):
    assert len(adherence.age_note(_touch(tmp_path, adherence.STALE_HOURS - 0.1))) == 1
    assert len(adherence.age_note(_touch(tmp_path, adherence.STALE_HOURS + 0.1))) == 2


def test_the_live_board_gets_the_drift_warning_not_an_age(tmp_path, monkeypatch, capsys):
    """The two warnings are for different mistakes and must not be swapped."""
    import polars as pl
    board = tmp_path / "draft_board.parquet"
    pl.DataFrame({"player": ["A"], "pos": ["RB"], "adp": [1.0], "ecr": [1.0],
                  "vor": [1.0], "proj_blend": [10.0]}).write_parquet(board)
    monkeypatch.setattr(adherence, "BOARD_PARQUET", board)
    monkeypatch.setattr(adherence, "AS_DRAFTED", tmp_path / "nope.parquet")
    monkeypatch.setattr(adherence, "load_state", lambda *a, **k: DraftState(taken=[]))
    adherence.main([])
    out = capsys.readouterr().out
    assert "every --pick rebuilds" in out and "as-drafted copy:" not in out
