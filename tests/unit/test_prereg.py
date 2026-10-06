"""`n_preregistered` is counted from the commit history, and unreadable history is unknown (#422).

The count was the literal 0 for four weeks while 64 predictions the scheduled slate committed
before kickoff sat in `main`. Every control here runs against a real throwaway repository built
in `tmp_path` with set commit dates -- never against `data/`, and never against this repo's own
history, which would make the answer a fact about today.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import polars as pl
import pytest

from hub import prereg, publish

KICK = dt.datetime(2026, 9, 6, 17, 0)  # naive UTC, as `schedule.kickoff_expr` yields
BEFORE = "2026-09-05T12:00:00+00:00"
AFTER = "2026-09-07T12:00:00+00:00"


def _git(cwd: Path, *args: str, when: str | None = None) -> None:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    if when:
        env.update(GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "commit.gpgsign=false", *args],
                   check=True, capture_output=True, env=env)


def _row(game_id: str, prob: float, priced_at: str = "2026-09-05 11:00:00") -> dict:
    return {"game_id": game_id, "season": 2026, "week": 1, "home_win_prob": prob,
            "priced_at": priced_at}


def _publish(site: Path, rows: list[dict], when: str, name: str = "preds_2026_wk01.json"):
    (site / name).write_text(json.dumps({"rows": rows}))
    _git(site, "add", "-A")
    _git(site, "commit", "-m", f"slate: {name}", when=when)


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "repo"
    (root / "site" / "data").mkdir(parents=True)
    _git(root, "init", "-q")
    return root / "site" / "data"


def _scored(*rows: dict) -> list[dict]:
    return [{**r, "kickoff": KICK} for r in rows]


def test_a_commit_before_kickoff_counts(site):
    _publish(site, [_row("g1", 0.6)], BEFORE)
    history = prereg.read_history(site)
    assert history is not None
    got = prereg.classify(_scored(_row("g1", 0.6)), prereg.first_seen(history))
    assert got.n_preregistered == 1 and got.late == [] and got.unverified == []


def test_a_commit_after_kickoff_does_not_count_and_is_named(site):
    _publish(site, [_row("g1", 0.6)], AFTER)
    history = prereg.read_history(site)
    assert history is not None
    got = prereg.classify(_scored(_row("g1", 0.6)), prereg.first_seen(history))
    assert got.n_preregistered == 0 and got.late == ["g1"]


def test_deleting_the_commit_that_pre_registered_a_game_drops_its_count(site, tmp_path):
    """The positive control #422 asks for: the same scored game, with and without the
    commit that pre-registered it."""
    _publish(site, [_row("g1", 0.6)], BEFORE)
    _publish(site, [_row("g1", 0.6), _row("g2", 0.7)], AFTER)
    scored = _scored(_row("g1", 0.6), _row("g2", 0.7))
    with_it = prereg.classify(scored, prereg.first_seen(prereg.read_history(site) or []))
    assert with_it.n_preregistered == 1 and with_it.late == ["g2"]

    other = tmp_path / "other" / "site" / "data"
    other.mkdir(parents=True)
    _git(other.parents[1], "init", "-q")
    _publish(other, [_row("g1", 0.6), _row("g2", 0.7)], AFTER)  # the early commit never was
    without = prereg.classify(scored, prereg.first_seen(prereg.read_history(other) or []))
    assert without.n_preregistered == 0, "removing the pre-registering commit changed nothing"


def test_a_re_priced_game_is_a_new_prediction(site):
    """The scored price is what counts: an early commit of a different price does not vouch
    for the one that was scored after kickoff."""
    _publish(site, [_row("g1", 0.6)], BEFORE)
    _publish(site, [_row("g1", 0.65, priced_at="2026-09-07 09:00:00")], AFTER)
    seen = prereg.first_seen(prereg.read_history(site) or [])
    got = prereg.classify(_scored(_row("g1", 0.65, "2026-09-07 09:00:00")), seen)
    assert got.n_preregistered == 0 and got.late == ["g1"]


def test_a_commit_at_the_kickoff_instant_is_not_before_it():
    seen = {prereg.key_of(_row("g1", 0.6)): KICK}
    assert prereg.classify(_scored(_row("g1", 0.6)), seen).n_preregistered == 0


def test_a_game_no_commit_or_kickoff_can_place_is_unverified_not_late():
    seen = {prereg.key_of(_row("g1", 0.6)): KICK - dt.timedelta(days=1)}
    unplaced = [*_scored(_row("g2", 0.5)), {**_row("g1", 0.6), "kickoff": None}]
    got = prereg.classify(unplaced, seen)
    assert got.n_preregistered == 0 and got.late == [] and got.unverified == ["g1", "g2"]


def test_history_that_cannot_be_read_is_none_never_empty(site, tmp_path):
    _publish(site, [_row("g1", 0.6)], BEFORE)
    _publish(site, [_row("g1", 0.6)], AFTER, name="preds_2026_wk02.json")
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{site.parents[1]}",
                    str(shallow)], check=True, capture_output=True)
    assert prereg.read_history(shallow / "site" / "data") is None, "shallow read as history"
    elsewhere = tmp_path / "not-a-repo"
    elsewhere.mkdir()
    assert prereg.read_history(elsewhere) is None


def _finished(game_ids, probs, priced="2026-09-05 11:00:00"):
    return pl.DataFrame(
        {"game_id": game_ids, "season": [2026] * len(game_ids), "week": [1] * len(game_ids),
         "home_win_prob": probs, "priced_at": [priced] * len(game_ids),
         "result": [7] * len(game_ids), "kickoff": [KICK] * len(game_ids)})


def test_the_record_counts_pre_registration_and_keeps_every_other_number(site, monkeypatch):
    _publish(site, [_row("g1", 0.6)], BEFORE)
    _publish(site, [_row("g1", 0.6), _row("g2", 0.7)], AFTER)
    monkeypatch.setattr(publish, "_finished", lambda out: _finished(["g1", "g2"], [0.6, 0.7]))
    got = publish.track_record(out=site)
    assert isinstance(got, dict)
    assert got["n_preregistered"] == 1 and got["n_scored"] == 2
    assert got["is_backtest"] is True and got["late_game_ids"] == ["g2"]
    assert got["seasons"][0]["n_preregistered"] == 1 and got["seasons"][0]["n_backtest"] == 1
    assert "1 of 2 scored" in got["note"]


def test_everything_pre_registered_is_not_a_backtest(site, monkeypatch):
    _publish(site, [_row("g1", 0.6)], BEFORE)
    monkeypatch.setattr(publish, "_finished", lambda out: _finished(["g1"], [0.6]))
    got = publish.track_record(out=site)
    assert isinstance(got, dict)
    assert got["n_preregistered"] == 1 and got["is_backtest"] is False


def test_the_record_publishes_unknown_not_zero_when_history_is_unreadable(tmp_path, monkeypatch):
    """The defect: a 0 that means "could not tell". Outside a repository the count is null,
    the record still carries its calibration, and it is not called a record."""
    out = tmp_path / "plain" / "data"
    out.mkdir(parents=True)
    (out / "preds_2026_wk01.json").write_text(json.dumps({"rows": [_row("g1", 0.6)]}))
    monkeypatch.setattr(publish, "_finished", lambda o: _finished(["g1"], [0.6]))
    got = publish.track_record(out=out)
    assert isinstance(got, dict)
    assert got["n_preregistered"] is None, "unknown was published as a number"
    assert got["is_backtest"] is True and got["n_scored"] == 1
    assert got["seasons"][0]["n_preregistered"] is None
    assert "could not be checked" in got["note"]
    assert got["log_loss"] is not None


def test_a_schedule_without_kickoffs_leaves_the_games_unverified(site, monkeypatch):
    """No kickoff column means no comparison can be made: nothing is counted, and nothing is
    called late either."""
    _publish(site, [_row("g1", 0.6)], BEFORE)
    monkeypatch.setattr(publish, "_finished",
                        lambda out: _finished(["g1"], [0.6]).drop("kickoff"))
    got = publish.track_record(out=site)
    assert isinstance(got, dict)
    assert got["n_preregistered"] == 0 and got["unverified_game_ids"] == ["g1"]
    assert got["late_game_ids"] == []


def test_a_deleted_or_malformed_artifact_in_the_history_is_skipped(site):
    """The history is read commit by commit; a commit that deleted a file, or wrote one that
    is not JSON, must not take the reader down or hide the commit that held the rows."""
    _publish(site, [_row("g1", 0.6)], BEFORE)
    (site / "preds_2026_wk02.json").write_text("{not json")
    _git(site, "add", "-A")
    _git(site, "commit", "-m", "malformed", when=BEFORE)
    _git(site, "rm", "-q", "preds_2026_wk02.json")
    _git(site, "commit", "-m", "deleted", when=AFTER)
    history = prereg.read_history(site)
    assert history is not None
    seen = prereg.first_seen(history)
    assert prereg.classify(_scored(_row("g1", 0.6)), seen).n_preregistered == 1
