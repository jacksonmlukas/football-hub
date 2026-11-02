"""`hub.forecasters.close_line`: the study's closing line, one call per kickoff time (#450).

No Odds API credit: the transport is a counting stub answering from a synthetic slate below,
whose Sunday shares a kickoff (two games at 13:00 ET) and spreads the rest over 09:30, 16:25
and 20:20, with a Thursday and a Monday game around it -- six distinct kickoff times, seven
games. The slate is local rather than `forecasters_slate`'s because the guard harness
(`test_guards_are_load_bearing.py`) runs this file with only the mutated `src/` on the path.
What the ticket asks:

* one call per kickoff time, however many triggers land in the window;
* a missed window is not retried after kickoff;
* the study floor refuses before any call when the balance is low;
* writes go to the study's path, never the hub's `state/`;
* a rerun spends nothing.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from hub import closing
from hub.fetch import odds
from hub.forecasters import close_line, market, pipeline, runner, sources

NAMES = {"Kansas City Chiefs": "KC", "Buffalo Bills": "BUF", "Baltimore Ravens": "BAL",
         "Cincinnati Bengals": "CIN", "Dallas Cowboys": "DAL", "Philadelphia Eagles": "PHI",
         "Los Angeles Rams": "LA", "Washington Commanders": "WAS", "Jacksonville Jaguars": "JAX",
         "New York Jets": "NYJ", "New England Patriots": "NE", "Detroit Lions": "DET",
         "Miami Dolphins": "MIA", "Green Bay Packers": "GB"}
FULL = {v: k for k, v in NAMES.items()}
# (game_id, week, gameday, gametime ET, home, away)
_ROWS = [
    ("2026_06_KC_BUF", 6, "2026-10-11", "16:25", "KC", "BUF"),
    ("2026_07_NE_DET", 7, "2026-10-15", "20:15", "NE", "DET"),          # Thursday
    ("2026_07_JAX_NYJ", 7, "2026-10-18", "09:30", "JAX", "NYJ"),        # London
    ("2026_07_BUF_KC", 7, "2026-10-18", "13:00", "BUF", "KC"),
    ("2026_07_CIN_BAL", 7, "2026-10-18", "13:00", "CIN", "BAL"),
    ("2026_07_PHI_DAL", 7, "2026-10-18", "16:25", "PHI", "DAL"),
    ("2026_07_WAS_LA", 7, "2026-10-18", "20:20", "WAS", "LA"),
    ("2026_07_MIA_GB", 7, "2026-10-19", "20:15", "MIA", "GB"),          # Monday
]
# Decimal prices (home, away) for the games the stubbed feed quotes.
PRICES = {"2026_07_BUF_KC": (1.80, 2.10), "2026_07_CIN_BAL": (2.40, 1.60),
          "2026_07_PHI_DAL": (1.50, 2.70), "2026_07_WAS_LA": (2.00, 1.85)}


def schedule_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [{"game_id": g, "season": 2026, "game_type": "REG", "week": w, "gameday": d,
          "gametime": t, "home_team": h, "away_team": a, "home_score": None,
          "away_score": None, "location": "Home"} for g, w, d, t, h, a in _ROWS],
        schema={"game_id": pl.Utf8, "season": pl.Int32, "game_type": pl.Utf8,
                "week": pl.Int32, "gameday": pl.Utf8, "gametime": pl.Utf8,
                "home_team": pl.Utf8, "away_team": pl.Utf8, "home_score": pl.Int32,
                "away_score": pl.Int32, "location": pl.Utf8})


def _book(key: str, home: str, away: str, hp: float, ap: float) -> dict:
    return {"key": key, "title": key, "markets": [
        {"key": "h2h", "outcomes": [{"name": home, "price": hp}, {"name": away, "price": ap}]}]}


def odds_payload() -> list[dict]:
    out = []
    for gid, _, d, t, h, a in _ROWS:
        if gid not in PRICES:
            continue
        hp, ap = PRICES[gid]
        out.append({"id": f"ev-{gid}", "home_team": FULL[h], "away_team": FULL[a],
                    "commence_time": sources.kickoff(d, t).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "bookmakers": [_book("a", FULL[h], FULL[a], hp, ap),
                                   _book("b", FULL[h], FULL[a], hp + 0.02, ap - 0.02)]})
    return out


KICKOFFS = sources.game_infos(schedule_frame())
SUNDAY_1PM = datetime(2026, 10, 18, 17, 0, tzinfo=UTC)               # 13:00 EDT, two games
SHARED = ("2026_07_BUF_KC", "2026_07_CIN_BAL")


class Odds:
    """The Odds API, stubbed: counts calls and answers like the real one, one credit a call."""

    def __init__(self, remaining: int = 400, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.remaining = remaining
        self.fail = fail

    def __call__(self, params, key):
        self.calls.append(dict(params))
        if self.fail:
            raise ConnectionError("upstream down")
        self.remaining -= 1
        return odds_payload(), {"x-requests-remaining": str(self.remaining)}


def _names(games):
    return dict(NAMES)


@pytest.fixture(autouse=True)
def hub_state(tmp_path, monkeypatch) -> Path:
    """The hub's own `state/`, redirected so any write to it is visible and harmless."""
    hub = tmp_path / "hub" / "state"
    monkeypatch.setattr(odds, "STATE", hub / "odds.json")
    monkeypatch.setattr(closing, "ATTEMPTS_FILE", hub / "odds_close_attempts.json")
    return hub


def _capture(out: Path, now: datetime, get: Odds, **kw: Any) -> close_line.Capture:
    return close_line.capture(out, now=now, kickoffs=KICKOFFS, names=_names, get=get,
                              key="test-key", **kw)


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()} if root.exists() else {}


def _distinct_kickoffs(lo: datetime, hi: datetime) -> set[datetime]:
    return {g.kickoff for g in KICKOFFS.values() if lo < g.kickoff <= hi}


# --- one call per kickoff time ------------------------------------------------------------

@pytest.mark.parametrize("every_min", [5, 3, 15])
def test_one_call_per_kickoff_time_over_a_whole_week_of_triggers(tmp_path, every_min):
    """Triggers every few minutes from Wednesday to Tuesday: several land inside each
    30-minute window, the two 13:00 games share one, and the count is the kickoff times."""
    out, get = tmp_path / "forecasts", Odds()
    start = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    end = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
    t, outcomes = start, []
    while t <= end:
        outcomes.append((t, _capture(out, t, get)))
        t += timedelta(minutes=every_min)
    kicks = _distinct_kickoffs(start, end)
    games = [g for g in KICKOFFS.values() if start < g.kickoff <= end]
    assert len(kicks) == 6 and len(games) == 7                     # 13:00 is shared
    assert len(get.calls) == len(kicks)
    called = [t for t, c in outcomes if c.calls]
    for k in kicks:                      # each window got exactly one call, inside it
        assert sum(k - timedelta(minutes=close_line.WINDOW_MIN) <= t < k for t in called) == 1
    assert all(p["markets"] == "h2h" for p in get.calls)           # one credit a call


def test_the_shared_kickoff_is_one_call_and_closes_both_games(tmp_path):
    out, get = tmp_path / "forecasts", Odds()
    got = _capture(out, SUNDAY_1PM - timedelta(minutes=25), get)
    assert got.status == "captured" and got.calls == 1 and len(get.calls) == 1
    assert set(got.due) == set(SHARED) and set(got.closed) == set(SHARED)
    assert got.path is not None
    doc = json.loads(got.path.read_text())
    assert got.path.parent == out / "closing" / "2026"
    assert {r["game_id"] for r in doc["closes"]} == set(SHARED)      # not the 16:25 game
    for r in doc["closes"]:
        assert 0 < r["lead_minutes"] <= close_line.WINDOW_MIN and r["books"] == 2
    kc = next(r for r in doc["closes"] if r["game_id"] == "2026_07_BUF_KC")
    # Two books, 1.80/2.10 and 1.82/2.08, de-vigged one at a time, then the median.
    assert kc["home_win_prob"] == pytest.approx((2.10 / 3.90 + 2.08 / 3.90) / 2)
    assert "raw" not in json.dumps(doc).replace("raw_sha256", "")   # the payload as a hash only


def test_a_second_trigger_in_the_window_spends_nothing(tmp_path):
    out, get = tmp_path / "forecasts", Odds()
    _capture(out, SUNDAY_1PM - timedelta(minutes=29), get)
    for m in (24, 15, 5, 1):
        assert _capture(out, SUNDAY_1PM - timedelta(minutes=m), get).status == "nothing_due"
    assert len(get.calls) == 1


def test_a_rerun_writes_nothing_and_spends_nothing(tmp_path):
    out, get = tmp_path / "forecasts", Odds()
    now = SUNDAY_1PM - timedelta(minutes=10)
    _capture(out, now, get)
    before = _files(out)
    again = Odds()
    assert _capture(out, now, again) == close_line.Capture("nothing_due")
    assert again.calls == [] and _files(out) == before


# --- the clock: a late trigger ------------------------------------------------------------

@pytest.mark.parametrize("late_min", [0, 1, 90, 6 * 60])
def test_a_missed_window_is_not_retried_after_kickoff(tmp_path, late_min):
    """A scheduled start delivered late (#468) lands at or after kickoff: nothing is due."""
    out, get = tmp_path / "forecasts", Odds()
    got = _capture(out, SUNDAY_1PM + timedelta(minutes=late_min), get)
    # 6h after 13:00 the 20:20 game is not yet in its window either.
    assert got.status == "nothing_due" and get.calls == [] and _files(out) == {}


def test_a_trigger_before_the_window_spends_nothing(tmp_path):
    get = Odds()
    got = _capture(tmp_path / "forecasts", SUNDAY_1PM - timedelta(minutes=31), get)
    assert got.status == "nothing_due" and get.calls == []


# --- refusals before any call -------------------------------------------------------------

def _state(path: Path, remaining: int, checked_at: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"remaining": remaining,
                                "checked_at": checked_at.replace(tzinfo=None).isoformat()}))


def test_the_study_floor_refuses_before_any_call(tmp_path):
    out, get = tmp_path / "forecasts", Odds()
    now = SUNDAY_1PM - timedelta(minutes=10)
    _state(out / pipeline.ODDS_STATE, market.STUDY_FLOOR - 1, now - timedelta(hours=3))
    with pytest.raises(odds.QuotaFloor, match="study floor"):
        _capture(out, now, get)
    assert get.calls == [] and not (out / close_line.ATTEMPTS).exists()


def test_a_fresher_low_balance_in_the_hubs_file_refuses_too(tmp_path, hub_state):
    """One key, one account: the hub polled since the study last did, and its reading wins."""
    out, get = tmp_path / "forecasts", Odds()
    now = SUNDAY_1PM - timedelta(minutes=10)
    _state(out / pipeline.ODDS_STATE, 300, now - timedelta(days=2))
    _state(hub_state / "odds.json", 4, now - timedelta(hours=1))
    before = (hub_state / "odds.json").read_bytes()
    with pytest.raises(odds.QuotaFloor):
        _capture(out, now, get)
    assert get.calls == [] and (hub_state / "odds.json").read_bytes() == before


def test_an_older_low_reading_does_not_refuse_over_a_newer_healthy_one(tmp_path, hub_state):
    out, get = tmp_path / "forecasts", Odds()
    now = SUNDAY_1PM - timedelta(minutes=10)
    _state(out / pipeline.ODDS_STATE, 300, now - timedelta(hours=1))
    _state(hub_state / "odds.json", 4, now - timedelta(days=2))
    assert _capture(out, now, get).calls == 1


def test_the_balance_helper_takes_the_newest_reading_and_none_when_unknown(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    when = datetime(2026, 10, 18, 16, 0)
    assert close_line.balance([a, b], when) is None
    _state(a, 50, when - timedelta(hours=2))
    _state(b, 40, when - timedelta(hours=1))
    assert close_line.balance([a, b], when) == 40
    b.write_text(json.dumps({"remaining": 400}))                    # undated: the oldest
    assert close_line.balance([a, b], when) == 50


def test_the_weekly_cap_refuses_before_any_call(tmp_path):
    out, get = tmp_path / "forecasts", Odds()
    now = SUNDAY_1PM - timedelta(minutes=10)
    for h in range(closing.CLOSE_CALLS_PER_WEEK):
        closing.record_attempt((now - timedelta(days=1, hours=h)).replace(tzinfo=None),
                               out / close_line.ATTEMPTS)
    with pytest.raises(closing.CloseBudgetExceeded):
        _capture(out, now, get)
    assert get.calls == []


def test_no_key_refuses_without_recording_an_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(odds, "_api_key", lambda: None)
    out, get = tmp_path / "forecasts", Odds()
    with pytest.raises(odds.QuotaFloor, match="ODDS_API_KEY"):
        close_line.capture(out, now=SUNDAY_1PM - timedelta(minutes=5), kickoffs=KICKOFFS,
                           names=_names, get=get)
    assert get.calls == [] and not (out / close_line.ATTEMPTS).exists()


# --- the attempt is on the record before the request --------------------------------------

def test_a_failed_request_still_marks_the_window_spent(tmp_path):
    out, down = tmp_path / "forecasts", Odds(fail=True)
    now = SUNDAY_1PM - timedelta(minutes=20)
    with pytest.raises(odds.MarketUnreachable):
        _capture(out, now, down)
    assert len(down.calls) == 1
    again = Odds()
    assert _capture(out, now + timedelta(minutes=5), again).status == "nothing_due"
    assert again.calls == []


def test_an_unpublishable_attempt_makes_no_request(tmp_path):
    out, get = tmp_path / "forecasts", Odds()

    def push() -> None:
        raise RuntimeError("push rejected")
    with pytest.raises(closing.AttemptNotRecorded):
        _capture(out, SUNDAY_1PM - timedelta(minutes=5), get, publish_attempt=push)
    assert get.calls == []


def test_the_attempt_is_published_before_the_request(tmp_path):
    out, order = tmp_path / "forecasts", []

    class Recording(Odds):
        def __call__(self, params, key):
            order.append("request")
            return super().__call__(params, key)
    _capture(out, SUNDAY_1PM - timedelta(minutes=5), Recording(),
             publish_attempt=lambda: order.append(
                 "publish" if (out / close_line.ATTEMPTS).exists() else "publish-empty"))
    assert order == ["publish", "request"]


# --- where it writes ----------------------------------------------------------------------

def test_writes_go_to_the_studys_path_never_the_hubs_state(tmp_path, hub_state):
    out, get = tmp_path / "forecasts", Odds(remaining=321)
    _capture(out, SUNDAY_1PM - timedelta(minutes=10), get)
    assert _files(hub_state) == {}                                  # neither balance nor attempts
    written = set(_files(out))
    assert written >= {"closing/attempts.json", "state/odds.json"}
    assert all(p.startswith(("closing/", "state/")) for p in written)
    own = json.loads((out / pipeline.ODDS_STATE).read_text())
    assert own["remaining"] == 320 and own["asked_for"] == "study:closing:h2h"
    assert own["declared_cost"] == 1


def test_the_closes_are_not_forecast_records_to_the_runner(tmp_path):
    """The runner reads every `*.jsonl` under the checkout as a forecast record."""
    out = tmp_path / "forecasts"
    _capture(out, SUNDAY_1PM - timedelta(minutes=10), Odds())
    assert runner.written(out) == set() and not list(out.rglob("*.jsonl"))


# --- the CLI ------------------------------------------------------------------------------

def _deps(now: datetime, get: Odds | None = None, *, broken: bool = False,
          published: list[Path] | None = None) -> close_line.Deps:
    def frame(season: int, refresh: bool):
        if broken:
            raise OSError("nflverse unreachable")
        return schedule_frame()
    return close_line.Deps(schedule_frame=frame, team_names=_names, odds_get=get or Odds(),
                           odds_key="test-key", clock=lambda: now,
                           publish=(published.append if published is not None else None))


def test_cli_due_answers_0_inside_an_uncovered_window_and_1_after(tmp_path):
    out, now = tmp_path / "forecasts", SUNDAY_1PM - timedelta(minutes=10)
    assert close_line.main(["due", "--out", str(out)], deps=_deps(now)) == 0
    published: list[Path] = []
    lines: list[str] = []
    assert close_line.main(["capture", "--out", str(out), "--commit-attempt"], echo=lines.append,
                           deps=_deps(now, published=published)) == 0
    assert json.loads(lines[-1])["calls"] == 1 and published == [out]
    assert close_line.main(["due", "--out", str(out)], deps=_deps(now)) == 1


def test_cli_cannot_tell_without_a_schedule_and_spends_nothing(tmp_path):
    get = Odds()
    for cmd in ("due", "capture"):
        assert close_line.main([cmd, "--out", str(tmp_path)], echo=lambda _: None,
                               deps=_deps(SUNDAY_1PM, get, broken=True)) == 2
    assert get.calls == []


def test_cli_refusal_is_one_line_and_exit_1(tmp_path):
    out, now = tmp_path / "forecasts", SUNDAY_1PM - timedelta(minutes=10)
    _state(out / pipeline.ODDS_STATE, 2, now)
    lines: list[str] = []
    get = Odds()
    assert close_line.main(["capture", "--out", str(out)], echo=lines.append,
                           deps=_deps(now, get)) == 1
    assert lines and "study floor" in lines[-1] and get.calls == []
