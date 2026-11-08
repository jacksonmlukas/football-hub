"""`hub.forecasters.burn_in`: the burn-in estimability record and report, on a synthetic week (#451).

Stubs only: the schedule is injected, the records are written here, nothing touches the network.
Each guard has its planted failure in the test beside it (`docs/method.md` rule 18).
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from hub.forecasters import burn_in, score
from hub.forecasters import config as study_config
from hub.forecasters.score import Analysis

CFG = study_config.load()
AWARE = [f"{m.name}:line_aware" for m in CFG.llms]
PINNED = Analysis(delta=0.02, h1_min_deviation_sd=0.25,
                  h2_implementation="hub.forecasters.score@abcdef0")
GAMES = ["g9a", "g9b", "g9c", "g9d"]
MARKET = {"g9a": 0.6, "g9b": 0.45, "g9c": 0.7, "g9d": 0.5}
# A deviation of +-d in log-odds on alternating games: mean 0, sample SD d*sqrt(n/(n-1)).
DEV = {AWARE[0]: 0.5, AWARE[1]: 0.3, AWARE[2]: 0.1, AWARE[3]: 0.0}     # llm3, llm4 are < 0.25


def _expected_sd(d: float) -> float:
    return d * math.sqrt(4 / 3)


def _p(market: float, dev: float) -> float:
    return float(1 / (1 + math.exp(-(math.log(market / (1 - market)) + dev))))


def _schedule(week=9):
    ids = [*GAMES, "g10a", "pre8"]
    weeks = [week] * 4 + [10, 8]
    frame = pl.DataFrame({"game_id": ids, "season": [2026] * 6, "game_type": ["REG"] * 6,
                          "week": weeks, "home_score": [None] * 6, "away_score": [None] * 6,
                          "gameday": ["2026-11-05", "2026-11-08", "2026-11-08", "2026-11-09",
                                      "2026-11-12", "2026-10-29"],
                          "gametime": ["20:15", "09:30", "13:00", "20:15", "20:15", "20:15"]})
    return lambda season: frame


AFTER_WEEK = datetime(2026, 11, 10, 15, 0, tzinfo=UTC)                 # Tuesday 10:00 ET


def _rows(skip: tuple[str, str] | None = None, null_market: str | None = None):
    rows = []
    for c, d in zip(AWARE, DEV.values(), strict=True):
        for i, g in enumerate(GAMES):
            if skip == (c, g):
                continue
            dev = d if i % 2 == 0 else -d
            rows.append({"game_id": g, "contestant": c, "p": _p(MARKET[g], dev),
                         "excluded": None, "market_snapshot_id": "S",
                         "market_home_win_prob": None if g == null_market else MARKET[g],
                         "run_start": "2026-11-05 14:00:00+00:00"})
    # the blind arm and the week-10 game must play no part in the measurement
    rows.append({"game_id": "g9a", "contestant": "llm1:blind", "p": 0.99, "excluded": None,
                 "market_snapshot_id": "S", "market_home_win_prob": 0.6,
                 "run_start": "2026-11-05 14:00:00+00:00"})
    rows.append({"game_id": "g10a", "contestant": AWARE[0], "p": 0.99, "excluded": None,
                 "market_snapshot_id": "T", "market_home_win_prob": 0.5,
                 "run_start": "2026-11-12 14:00:00+00:00"})
    return rows


def _forecasts(tmp_path, **kw) -> Path:
    out = tmp_path / "forecasts"
    out.mkdir(exist_ok=True)
    (out / "day.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows(**kw)))
    return out


def _run(tmp_path, *, forecasts=None, out=None, schedule=None, analysis=PINNED, sims=200,
         now=AFTER_WEEK):
    lines: list[str] = []
    dest = out or tmp_path / "prereg" / "burn-in.json"
    rc = burn_in.main(["--forecasts", str(forecasts or _forecasts(tmp_path)),
                       "--out", str(dest), "--week", "9"],
                      echo=lines.append, config=CFG, analysis=analysis,
                      schedule=schedule or _schedule(), simulations=sims, now=lambda: now)
    return rc, lines, dest


def test_the_written_record_round_trips_through_score(tmp_path):
    rc, _, dest = _run(tmp_path)
    assert rc == 0
    rec = json.loads(dest.read_text())
    assert set(rec) == {"season", "burn_in_week", "h1_min_deviation_sd", "deviation_sd"}
    assert (rec["season"], rec["burn_in_week"], rec["h1_min_deviation_sd"]) == (2026, 9, 0.25)
    b = score.load_burn_in(dest)
    assert (b.season, b.week, b.threshold, b.first_scored_week) == (2026, 9, 0.25, 10)
    for c, d in zip(AWARE, DEV.values(), strict=True):
        assert b.deviation_sd[c] == pytest.approx(_expected_sd(d), abs=1e-5)
    # planted: a SD measured over the wrong games (the blind 0.99 or the week-10 row) would
    # move llm1 far from the closed form above, and the closed-form assertion would go red.


def test_h1_family_follows_the_measured_sds(tmp_path):
    _, _, dest = _run(tmp_path)
    fam = score.h1_family(score.load_burn_in(dest), AWARE, 0.25)
    # d=0.5 -> 0.577, d=0.3 -> 0.346 clear 0.25; d=0.1 -> 0.115 and d=0 (copies the line) do not
    assert fam == frozenset(AWARE[:2])


def test_the_threshold_comes_from_the_pin_not_an_argument(tmp_path):
    other = Analysis(delta=0.02, h1_min_deviation_sd=0.5,
                     h2_implementation="hub.forecasters.score@abcdef0")
    _, _, dest = _run(tmp_path, analysis=other)
    assert json.loads(dest.read_text())["h1_min_deviation_sd"] == 0.5
    with pytest.raises(score.BurnInRefused):          # score refuses a record at another pin
        score.h1_family(score.load_burn_in(dest), AWARE, 0.25)
    assert CFG.analysis.h1_min_deviation_sd == 0.25    # and the committed pin is the prereg's


def test_the_report_names_each_model_against_the_threshold_and_labels_the_simulation(tmp_path):
    _, lines, _ = _run(tmp_path)
    text = "\n".join(lines)
    assert "SIMULATION" in text and "delta 0.02" in text and "Week 9" in text
    for c in AWARE:
        assert any(x.strip().startswith(f"{c}: SD") for x in lines)
        assert any(x.strip().startswith(f"{c}: power") for x in lines)
    assert text.count(score.NOT_ESTIMABLE) == 2
    assert "H1 family fixed at 2 of 4" in text


@pytest.mark.parametrize("kw", [{"skip": (AWARE[2], "g9c")}, {"null_market": "g9d"}])
def test_it_refuses_in_one_sentence_when_the_week_is_incomplete(tmp_path, kw):
    rc, lines, dest = _run(tmp_path, forecasts=_forecasts(tmp_path, **kw))
    assert rc == burn_in.INCOMPLETE
    assert len(lines) == 1 and lines[0].startswith("refused: ") and lines[0].count(". ") == 0
    assert not dest.exists()                           # planted: writing anyway would fail this


def test_a_week_with_no_records_is_refused_and_nothing_is_written(tmp_path):
    empty = tmp_path / "forecasts"
    empty.mkdir()
    rc, lines, dest = _run(tmp_path, forecasts=empty)
    assert rc == burn_in.INCOMPLETE and len(lines) == 1 and not dest.exists()


def test_a_week_with_no_scheduled_games_is_refused(tmp_path):
    rc, lines, dest = _run(tmp_path, schedule=_schedule(week=12))
    assert rc == burn_in.INCOMPLETE and len(lines) == 1 and not dest.exists()


def test_a_game_the_runner_refused_does_not_make_the_week_incomplete(tmp_path):
    out = _forecasts(tmp_path, skip=(AWARE[0], "g9a"))
    # g9a is gone for the one model; a logged refusal removes it from the week for everyone
    (out / score.EXCLUSIONS).write_text(json.dumps({"game_id": "g9a"}) + "\n")
    rc, _, dest = _run(tmp_path, forecasts=out)
    assert rc == 0 and dest.exists()


def _without(out: Path, keep) -> Path:
    day = out / "day.jsonl"
    rows = [json.loads(x) for x in day.read_text().splitlines()]
    day.write_text("".join(json.dumps(r) + "\n" for r in rows if keep(r)))
    return out


def test_a_game_no_run_reached_is_left_out_and_named(tmp_path):
    # 2026-11-08: no Saturday run, so the London game has no record from any contestant
    out = _without(_forecasts(tmp_path), lambda r: r["game_id"] != "g9d")
    rc, lines, dest = _run(tmp_path, forecasts=out)
    assert rc == 0 and dest.exists()
    assert any("g9d" in x and "no run reached" in x for x in lines)
    # measured as if g9d were never scheduled: the same SDs, the same power
    frame = _schedule()(2026).filter(pl.col("game_id") != "g9d")
    _, alone, ref = _run(tmp_path / "ref", forecasts=out, schedule=lambda season: frame)
    assert json.loads(dest.read_text()) == json.loads(ref.read_text())
    assert [x for x in lines if "power" in x] == [x for x in alone if "power" in x]


@pytest.mark.parametrize("now", [
    datetime(2026, 11, 9, 14, 0, tzinfo=UTC),       # Monday 09:00 ET: g9d's run has not started
    datetime(2026, 11, 9, 16, 29, tzinfo=UTC),      # Monday 11:29 ET: its window is still open
])
def test_a_game_whose_window_has_not_closed_is_pending_not_unreached(tmp_path, now):
    # run before Monday's forecast, the Monday night game must refuse the week, not leave it:
    # the record is a v1 artifact that is never rewritten
    out = _without(_forecasts(tmp_path), lambda r: r["game_id"] != "g9d")
    rc, lines, dest = _run(tmp_path, forecasts=out, now=now)
    assert rc == burn_in.INCOMPLETE and len(lines) == 1 and not dest.exists()
    assert "g9d" in lines[0]


def test_the_window_closes_at_half_past_eleven_on_the_run_day(tmp_path):
    out = _without(_forecasts(tmp_path), lambda r: r["game_id"] != "g9d")
    rc, _, dest = _run(tmp_path, forecasts=out, now=datetime(2026, 11, 9, 16, 30, tzinfo=UTC))
    assert rc == 0 and dest.exists()


def test_a_game_only_some_contestants_reached_still_refuses(tmp_path):
    # the blind arm reached g9a, so a run did; the line-aware records are missing, not unreached
    out = _without(_forecasts(tmp_path),
                   lambda r: not (r["game_id"] == "g9a" and r["contestant"] in AWARE))
    rc, lines, dest = _run(tmp_path, forecasts=out)
    assert rc == burn_in.INCOMPLETE and len(lines) == 1 and not dest.exists()


def test_a_week_no_run_reached_at_all_is_refused_by_name(tmp_path):
    out = _without(_forecasts(tmp_path), lambda r: r["game_id"] not in GAMES)
    rc, lines, dest = _run(tmp_path, forecasts=out)
    assert rc == burn_in.INCOMPLETE and not dest.exists()
    assert len(lines) == 1 and "no run reached" in lines[0]


def test_a_different_record_is_never_overwritten(tmp_path):
    dest = tmp_path / "prereg" / "burn-in.json"
    dest.parent.mkdir()
    committed = json.dumps({"season": 2026, "burn_in_week": 9, "h1_min_deviation_sd": 0.25,
                            "deviation_sd": dict.fromkeys(AWARE, 0.9)})
    dest.write_text(committed)
    rc, lines, _ = _run(tmp_path, out=dest)
    assert rc == burn_in.EXISTS and len(lines) == 1 and "never overwritten" in lines[0]
    assert dest.read_text() == committed               # planted: an overwrite changes the bytes


def test_an_unreadable_existing_file_is_not_replaced_either(tmp_path):
    dest = tmp_path / "prereg" / "burn-in.json"
    dest.parent.mkdir()
    dest.write_text("{ not json")
    rc, lines, _ = _run(tmp_path, out=dest)
    assert rc == burn_in.EXISTS and len(lines) == 1
    assert dest.read_text() == "{ not json"


def test_the_same_record_again_is_a_no_op_that_still_reports(tmp_path):
    _, _, dest = _run(tmp_path)
    before = dest.stat().st_mtime_ns
    rc, lines, _ = _run(tmp_path, out=dest)
    assert rc == 0 and dest.stat().st_mtime_ns == before
    assert lines[0].startswith("already holds this exact record")


def test_the_write_is_atomic_and_leaves_no_scratch(tmp_path):
    _, _, dest = _run(tmp_path)
    assert [p.name for p in dest.parent.iterdir()] == [dest.name]


def test_an_unreadable_schedule_is_a_sentence_and_writes_nothing(tmp_path, capsys):
    def down(season):
        raise OSError("the nflverse cache is absent")
    rc, lines, dest = _run(tmp_path, schedule=down)
    assert rc == 1 and not dest.exists() and lines == []
    err = capsys.readouterr().err
    assert "schedule unavailable: OSError" in err and "Traceback" not in err


def test_it_refuses_while_the_analysis_is_unpinned_or_the_checkout_is_unreadable(tmp_path):
    rc, lines, dest = _run(tmp_path, analysis=Analysis())
    assert rc == burn_in.NOT_PINNED and len(lines) == 1 and not dest.exists()
    rc, lines, _ = _run(tmp_path, forecasts=tmp_path / "nope")
    assert rc == burn_in.UNREADABLE and len(lines) == 1


# --- the power simulation -----------------------------------------------------------------

def test_power_is_reproducible_and_depends_on_delta():
    mk = list(MARKET.values()) * 4
    sds = {"a": 0.4, "b": 0.3}
    kw = {"games_per_week": 16, "weeks": 9, "simulations": 300}
    one = burn_in.simulate_power(mk, sds, delta=0.02, **kw)
    assert one == burn_in.simulate_power(mk, sds, delta=0.02, **kw)
    wide = burn_in.simulate_power(mk, sds, delta=1.0, **kw)
    tiny = burn_in.simulate_power(mk, sds, delta=1e-6, **kw)
    assert wide["all"]["parity"] == 1.0                    # a huge margin is always cleared
    assert tiny["a"]["parity"] < 0.25                      # a vanishing one is not, at parity
    assert wide["a"]["noise"] >= one["a"]["noise"] >= tiny["a"]["noise"]
    assert 0.0 <= one["all"]["noise"] <= min(one["a"]["noise"], one["b"]["noise"])


def test_no_model_means_no_claim():
    out = burn_in.simulate_power([0.5, 0.6], {}, delta=0.02, games_per_week=16, weeks=9,
                                 simulations=10)
    assert out == {"all": {"noise": 0.0, "parity": 0.0}}


def test_without_week_the_burn_in_week_is_the_preregistered_week_nine(tmp_path):
    """#448 (adopted 2026-10-21): the burn-in is week 9, the first scored week 10. Run without
    `--week`, the CLI must read week 9. Found by the 2026-11-01 goal-check (default 9->10
    survived)."""
    lines: list[str] = []
    dest = tmp_path / "prereg" / "burn-in.json"
    rc = burn_in.main(["--forecasts", str(_forecasts(tmp_path)), "--out", str(dest)],
                      echo=lines.append, config=CFG, analysis=PINNED,
                      schedule=_schedule(), simulations=200)
    assert rc == 0
    assert json.loads(dest.read_text())["burn_in_week"] == 9
    assert score.load_burn_in(dest).first_scored_week == 10


def test_the_power_line_says_it_is_not_preregistered(tmp_path):
    _, lines, _ = _run(tmp_path)
    assert any("not pre-registered" in ln for ln in lines)
