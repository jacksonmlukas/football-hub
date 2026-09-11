"""The money layer's entry point, and the journal's (#163).

Neither module had a `main`, an importer or a target, which exempted both from the contract
every other component is held to: absent input answered with a sentence, and last-good
state served where there is any. `tests/contracts/test_cli_surface.py` now drives both
against a fresh clone; these drive them against a board, end to end, through the same
`main` an operator types.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from hub.config import PoolConfig, pool_digest
from hub.season import journal, pool
from hub.season import survivor as sv


def _grid() -> pl.DataFrame:
    """Week 1 is over -- it carries a result -- and week 2 is the one being decided. KC is
    barely the best pick in week 2 and a near-lock in week 3, so the recommendation departs
    from the free pick, which is the case both the journal and the axis have something to
    say about."""
    rows = []
    spec = {1: [("DAL", "NYG", 0.80), ("MIA", "NE", 0.60)],
            2: [("KC", "LV", 0.70), ("SF", "SEA", 0.68), ("BUF", "NYJ", 0.66)],
            3: [("KC", "LV", 0.97), ("SF", "SEA", 0.55), ("BUF", "NYJ", 0.54)]}
    for wk, games in spec.items():
        for i, (a, b, p) in enumerate(games):
            res = 1.0 if wk == 1 else None
            rows += [(wk, a, p, f"{wk}-{i}", res), (wk, b, 1 - p, f"{wk}-{i}", res)]
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [r[2] for r in rows], "game_id": [r[3] for r in rows],
                         "result": [r[4] for r in rows]},
                        schema={"week": pl.Int64, "team": pl.Utf8, "win_prob": pl.Float64,
                                "game_id": pl.Utf8, "result": pl.Float64})


@pytest.fixture
def board(monkeypatch):
    """The schedule answers with the grid above, and the published plan spent nothing."""
    monkeypatch.setattr(sv, "grid_from_schedule", lambda season, cache=None: _grid())
    monkeypatch.setattr(sv, "published_plan", lambda path=None: [])


def _run(tmp_path: Path, *extra: str) -> list[str]:
    return ["--season", "2026", "--entries", "12", "--pot", "420", "--trials", "40",
            "--at", "1,4", "--store", str(tmp_path), *extra]


def test_the_weekly_figure_runs_end_to_end_and_is_a_range_over_the_axis(board, tmp_path,
                                                                        capsys):
    """The target's contract: one command prices the week, reports it across the
    concentration knob rather than at a point, names the rules and the board it ran under,
    and -- asked to -- records the decision with what it takes to run it again."""
    code = pool.main(_run(tmp_path, "--record"))
    out = capsys.readouterr().out
    assert code == 0
    assert "week 2:" in out, "the first week still ahead, with week 1 played"
    assert "across the field-concentration axis, 2 points" in out
    assert " to $" in out and "across the axis against $" in out
    assert f"rules {pool_digest(PoolConfig())}" in out and "board " in out
    assert "recorded 2026-w02-pick-" in out

    got = journal.read(2026, base=tmp_path)
    assert got.height == 1
    row = got.to_dicts()[0]
    assert row["week"] == 2 and row["kind"] == "pick"
    assert row["pool_digest"] == pool_digest(PoolConfig())
    assert row["grid_digest"] == pool.grid_digest(_grid())
    assert row["trials"] == 40 and row["entries"] == 12 and row["pot"] == 420.0
    assert row["outlay"] == PoolConfig().entry_fee, "net of the entry fee by default"


def test_a_failed_fetch_serves_the_last_recorded_decision_rather_than_erroring(
        board, tmp_path, capsys, monkeypatch):
    """`CLAUDE.md`'s degradation rule, for the module that most needs it. The journal is
    this module's last-good state: once a week has been recorded, a schedule that cannot be
    read hands back that row, under the columns that say what it was conditional on."""
    assert pool.main(_run(tmp_path, "--record")) == 0
    capsys.readouterr()

    def down(season, cache=None):
        raise OSError("the network is absent")
    monkeypatch.setattr(sv, "grid_from_schedule", down)
    code = pool.main(_run(tmp_path, "--week", "2"))
    out = capsys.readouterr()
    assert code == 0
    assert "serving the last decision recorded for week 2" in out.err
    assert "pick" in out.out and "Traceback" not in out.out + out.err


def test_a_failed_fetch_with_nothing_recorded_is_reported_the_way_every_cli_reports_it(
        tmp_path, capsys, monkeypatch):
    """No row is the fresh-clone case: a sentence naming what could not be read, non-zero."""
    def down(season, cache=None):
        raise OSError("the network is absent")
    monkeypatch.setattr(sv, "grid_from_schedule", down)
    code = pool.main(_run(tmp_path, "--week", "2"))
    out = capsys.readouterr()
    assert code == 1
    assert "unavailable" in out.err and "OSError" in out.err
    assert "Traceback" not in out.err


def test_a_week_the_board_cannot_play_is_refused_with_a_sentence(board, tmp_path, capsys):
    code = pool.main(_run(tmp_path, "--week", "9"))
    out = capsys.readouterr()
    assert code == 1
    assert "week 9 is not among the weeks the board can play" in out.err


def test_the_buyback_has_a_production_caller_and_records_its_verdict(board, tmp_path,
                                                                     capsys):
    """`buyback` had no caller outside the tests. `--eliminated` prices the re-entry after
    going out in `--week`, across the same axis, and records the decision as a buyback row
    -- the second `kind` the journal was built to hold and nothing wrote."""
    code = pool.main(_run(tmp_path, "--eliminated", "--week", "2", "--record"))
    out = capsys.readouterr().out
    assert code == 0
    assert "BUY BACK" in out and "net $" in out and "across the axis" in out
    assert "recorded 2026-w02-buyback-" in out
    row = journal.read(2026, base=tmp_path).to_dicts()[0]
    assert row["kind"] == "buyback" and row["chose"] in ("buy back", "stay out")
    assert row["expected_dollars"] is not None
    assert row["pool_digest"] == pool_digest(PoolConfig()) and row["trials"] == 40


def test_the_journal_reads_back_and_settles_from_the_terminal(board, tmp_path, capsys):
    assert pool.main(_run(tmp_path, "--record")) == 0
    key = journal.read(2026, base=tmp_path)["key"][0]
    capsys.readouterr()

    assert journal.main(["--season", "2026", "--store", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 decision(s); departed from auto-pick in 1 week: 2" in out
    assert "seed" in out, "a row that can be re-derived says under what"

    assert journal.main(["--settle", key, "--survived", "--dollars", "0",
                         "--store", str(tmp_path)]) == 0
    assert "settled" in capsys.readouterr().out
    assert journal.read(2026, base=tmp_path)["survived"][0] is True
    assert journal.main(["--season", "2026", "--store", str(tmp_path)]) == 0
    assert "survived" in capsys.readouterr().out


def test_settling_a_key_nothing_wrote_is_a_sentence_and_a_non_zero_exit(tmp_path, capsys):
    code = journal.main(["--settle", "2026-w02-pick-20260907T120000", "--survived",
                         "--store", str(tmp_path)])
    out = capsys.readouterr()
    assert code == 1 and "unavailable" in out.err and "Traceback" not in out.err


def test_an_empty_journal_names_what_would_write_one(tmp_path, capsys):
    code = journal.main(["--season", "2026", "--store", str(tmp_path)])
    out = capsys.readouterr()
    assert code == 1
    assert "no decision recorded for 2026" in out.err and "--record" in out.err


def test_a_row_from_before_the_rerun_columns_reports_itself_as_not_re_derivable():
    """`report` is what the pool CLI serves as last-good, so the blank where a digest would
    go has to say what it means rather than look like a missing digest."""
    old = {c: t for c, t in journal.SCHEMA.items() if c not in journal.RERUN_COLUMNS}
    row = pl.DataFrame({c: [None] for c in old}, schema=old).with_columns(
        pl.lit(2).alias("week"), pl.lit("pick").alias("kind"), pl.lit("SF").alias("chose"),
        pl.lit("KC").alias("fallback"), pl.lit(dt.datetime(2026, 9, 7)).alias("at"))
    lines = journal.report(row)
    assert len(lines) == 2 and "not re-derivable" in lines[-1]
    assert journal.report(journal.read(2026, base=Path("/nonexistent"))) == [
        "\n  no decisions recorded"]


def test_the_axis_report_says_when_the_pick_moves_with_the_assumption():
    """A pick that holds across the knob is a pick about the pool; one that moves is a pick
    about the assumption, and only the line at the bottom can say which."""
    def hand(pick: str, dollars: float) -> pool.Weekly:
        cands = [pool.Candidate(pick, 0.7, 0.3, dollars, pick == "KC"),
                 pool.Candidate("KC" if pick != "KC" else "SF", 0.7, 0.3, dollars - 1.0,
                                pick != "KC")]
        return pool.Weekly(2, pick, "KC", pick == "KC", 0.0, 420.0, 0.5, cands)
    same = pool.axis_report({1.0: hand("SF", 10.0), 4.0: hand("SF", 12.0)})
    assert same[-1].endswith("SF at every point") and "$10.00 to $12.00" in same[-1]
    moves = pool.axis_report({1.0: hand("SF", 10.0), 4.0: hand("KC", 9.0)})
    assert "the pick moves with the assumption: KC, SF" in moves[-1]
    assert pool.axis_report({}) == ["\n  no concentrations swept"]


def test_a_buyback_with_nothing_ahead_is_reported_as_unavailable_not_priced(board, tmp_path,
                                                                            capsys):
    """Week 3 is the last priced week, so there is nothing to re-enter for. `buyback` says
    so and the CLI prints the reason rather than a range over nothing."""
    code = pool.main(_run(tmp_path, "--eliminated", "--week", "3"))
    out = capsys.readouterr().out
    assert code == 0
    assert "no buyback: nothing is priced after week 3" in out
    assert "across the axis" not in out


def test_a_journal_that_cannot_be_read_is_a_sentence_on_both_entry_points(tmp_path, capsys,
                                                                          monkeypatch):
    """A store that raises is not an empty store, and neither CLI may turn it into a
    traceback. The pool falls through to the fresh-clone report; the journal names it."""
    def broken(season, week=None, base=None):
        raise RuntimeError("the catalog is corrupt")
    monkeypatch.setattr(journal, "read", broken)

    def down(season, cache=None):
        raise OSError("the network is absent")
    monkeypatch.setattr(sv, "grid_from_schedule", down)
    assert pool.main(_run(tmp_path, "--week", "2")) == 1
    out = capsys.readouterr()
    assert "unavailable" in out.err and "Traceback" not in out.err

    assert journal.main(["--season", "2026", "--store", str(tmp_path)]) == 1
    out = capsys.readouterr()
    assert "catalog is corrupt" in out.err and "Traceback" not in out.err
