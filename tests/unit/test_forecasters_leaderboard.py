"""`hub.forecasters.leaderboard`: the interim leaderboard on a fixture week, end to end (#453).

Stubs only: the schedule is injected, the pins and the burn-in record are written here. Each
guard has its planted failure in the test beside it (`docs/method.md` rule 18).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from hub.forecasters import config as study_config
from hub.forecasters import leaderboard
from hub.forecasters.score import Analysis

CFG = study_config.load()
AWARE = [f"{m.name}:line_aware" for m in CFG.llms]
PINNED = Analysis(delta=0.02, h1_min_deviation_sd=0.05,
                  h2_implementation="hub.forecasters.score@abcdef0")
SNAP = "20261108T140000Z-0123456789ab"
RAW_SENTINEL = "RAW-ODDS-PAYLOAD-SENTINEL"


def _burn_in(tmp_path, week=9):
    p = tmp_path / "burn-in.json"
    p.write_text(json.dumps({"season": 2026, "burn_in_week": week, "h1_min_deviation_sd": 0.05,
                             "deviation_sd": dict.fromkeys(AWARE, 0.3)}))
    return p


def _schedule(played=True):
    ids = ["g10a", "g10b", "g10c", "g10d", "burn9"]
    weeks = [10, 10, 10, 10, 9]
    hs = [24, 17, 31, 10, 21] if played else [None] * 5
    as_ = [20, 21, 14, 13, 17] if played else [None] * 5
    frame = pl.DataFrame({"game_id": ids, "season": [2026] * 5, "game_type": ["REG"] * 5,
                          "week": weeks, "home_score": hs, "away_score": as_})
    return lambda season: frame


def _forecasts(tmp_path) -> Path:
    out = tmp_path / "forecasts"
    if out.exists():
        return out
    out.mkdir()
    rows = []
    for c in [*AWARE, "coin_flip"]:
        for gid, mk in [("g10a", 0.6), ("g10b", 0.45), ("g10c", 0.7), ("g10d", 0.5)]:
            if c == AWARE[1] and gid == "g10d":
                continue                                    # llm2 misses one game
            rows.append({"game_id": gid, "contestant": c, "p": 0.55, "excluded": None,
                         "market_home_win_prob": mk, "market_snapshot_id": SNAP,
                         "run_start": "2026-11-08 14:00:00+00:00",
                         "raw_odds_payload": RAW_SENTINEL})
    (out / "day.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def _run(tmp_path, *, forecasts=None, burn=None, schedule=None, analysis=PINNED):
    lines: list[str] = []
    out = tmp_path / "board" / "interim.json"
    rc = leaderboard.main(
        ["--forecasts", str(forecasts or _forecasts(tmp_path)), "--out", str(out),
         "--burn-in", str(burn or _burn_in(tmp_path))],
        echo=lines.append, config=CFG, analysis=analysis, schedule=schedule or _schedule())
    return rc, lines, out


def test_a_fixture_week_end_to_end(tmp_path):
    rc, lines, out = _run(tmp_path)
    assert rc == 0 and len(lines) == 1
    doc = json.loads(out.read_text())
    assert doc["interim"] is True and doc["label"] == "Interim: descriptive, not a test."
    assert doc["scored_weeks"] == [10] and doc["through_week"] == 10
    assert doc["first_scored_week"] == 10                  # the week 9 game is the burn-in's
    assert doc["h2_implementation"] == PINNED.h2_implementation
    by = {r["contestant"]: r for r in doc["contestants"]}
    assert set(by) == {*AWARE, "coin_flip"}
    assert by[AWARE[0]]["missing"] == 0 and by[AWARE[0]]["scorable"] == 4
    assert by[AWARE[0]]["log_loss_diff"]["games"] == 4
    assert by[AWARE[1]]["missing"] == 1 and by[AWARE[1]]["missing_games"] == ["g10d"]
    assert doc["missing_games"] == [{"game_id": "g10d", "week": 10, "contestants": [AWARE[1]]}]
    assert doc["market_snapshot_ids"] == [SNAP]


def test_descriptive_only_no_test_and_no_verdict(tmp_path):
    """The confirmatory analysis runs once after Week 18: nothing here may read as a test."""
    doc = json.loads(_run(tmp_path)[2].read_text())
    text = json.dumps(doc)
    for banned in ("h1", "h2\"", "holm", "p_noninferior", "claim", "detected", "sentence"):
        assert banned not in text.lower().replace("h2_implementation", "")


def test_no_raw_odds_payload_reaches_the_leaderboard(tmp_path):
    """Planted: every record carries a raw payload field. Only snapshot ids (hashes) go out."""
    rc, _, out = _run(tmp_path)
    assert rc == 0
    assert RAW_SENTINEL not in out.read_text()
    assert "raw" not in out.read_text().lower()


def test_it_refuses_before_the_first_scored_week_has_finished(tmp_path):
    rc, lines, out = _run(tmp_path, schedule=_schedule(played=False))
    assert rc == leaderboard.NO_SCORED_WEEK and not out.exists()
    assert len(lines) == 1 and "Week 10" in lines[0]
    rc, lines, out = _run(tmp_path, burn=_burn_in(tmp_path, week=10))   # weeks 11+: none yet
    assert rc == leaderboard.NO_SCORED_WEEK and not out.exists()


def test_it_refuses_without_the_burn_in_record(tmp_path):
    rc, lines, out = _run(tmp_path, burn=tmp_path / "absent.json")
    assert rc == leaderboard.NO_BURN_IN and not out.exists()
    assert len(lines) == 1 and lines[0].startswith("refused: no burn-in estimability record")


def test_it_refuses_while_the_analysis_is_unpinned(tmp_path):
    rc, lines, out = _run(tmp_path, analysis=Analysis(delta=None, h1_min_deviation_sd=None,
                                                      h2_implementation=None))
    assert rc == leaderboard.NOT_PINNED and not out.exists() and len(lines) == 1


def test_an_unreadable_forecasts_checkout_is_an_error_not_a_refusal(tmp_path):
    rc, _, _ = _run(tmp_path, forecasts=tmp_path / "nope")
    assert rc == leaderboard.UNREADABLE


@pytest.mark.parametrize("rc", [leaderboard.NOT_PINNED, leaderboard.NO_BURN_IN,
                                leaderboard.NO_SCORED_WEEK])
def test_the_refusal_codes_are_the_ones_the_workflow_treats_as_clean(rc):
    wf = (Path(__file__).resolve().parents[2] / ".github" / "workflows"
          / "forecast-score.yml").read_text()
    assert str(rc) in wf.split("case")[1].split("*)")[0]
