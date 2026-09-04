"""The Odds API fetch layer.

`weekly-slate/SKILL.md:18` invokes this and it did not exist.

The Odds API charges **markets x regions per request**, so a two-market two-region pull
costs four credits, not one. `docs/decisions.md` records that this multiplier is why props
are roadmap-only: one props pull would cost more than a month of the free tier. The plan's
requirement is that this "cannot silently burn quota", which means two things -- the
multiplier can never be triggered by accident, and the credit balance is never a mystery.

The other half is what the snapshot is *for*. `hub.store.AS_OF_LINES` joins lines to
predictions on nflverse `game_id`, and this is the only thing that will ever produce more
than one line per game. If it keys on The Odds API's own ids the whole as-of machinery
proven in 1.6 has nothing to join to, so the mapping is not a nicety.
"""
import datetime as dt
import json

import polars as pl
import pytest

from hub.fetch import odds


@pytest.fixture
def paths(tmp_path):
    return {"state": tmp_path / "odds.json", "store": tmp_path / "processed"}


@pytest.fixture
def teams(monkeypatch):
    monkeypatch.setattr(odds, "_team_abbrs", lambda: {
        "Philadelphia Eagles": "PHI", "Dallas Cowboys": "DAL",
        "Kansas City Chiefs": "KC", "Los Angeles Chargers": "LAC",
    })


@pytest.fixture
def schedule(monkeypatch):
    monkeypatch.setattr(odds, "_schedule", lambda season: pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI", "2025_01_KC_LAC"],
        "season": [2025, 2025],
        "week": [1, 1],
        "gameday": ["2025-09-04", "2025-09-05"],
        "home_team": ["PHI", "LAC"],
        "away_team": ["DAL", "KC"],
    }))


def _event(home, away, day, points, books=2):
    return {
        "id": f"{away}@{home}",
        "commence_time": f"{day}T20:00:00Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {"key": f"book{i}", "markets": [{"key": "spreads", "outcomes": [
                {"name": home, "point": points + i * 0.5},
                {"name": away, "point": -(points + i * 0.5)},
            ]}]}
            for i in range(books)
        ],
    }


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def _install(events=None, remaining=400):
        def _fake(params, key):
            calls.append(dict(params))
            payload = events if events is not None else [
                _event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04", -8.5)]
            return payload, {"x-requests-remaining": str(remaining),
                             "x-requests-used": "1"}
        monkeypatch.setattr(odds, "_http_get", _fake)
        monkeypatch.setattr(odds, "_api_key", lambda: "test-key")
        return calls
    return _install


# --- the multiplier -------------------------------------------------------

def test_exactly_one_market_and_one_region(transport, teams, schedule, paths):
    """Cost is markets x regions. Two of either doubles the bill for the same game."""
    calls = transport()
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert len(calls) == 1
    assert calls[0]["markets"] == odds.MARKET
    assert calls[0]["regions"] == odds.REGION
    assert "," not in calls[0]["markets"] and "," not in calls[0]["regions"]


def test_asking_for_a_second_market_is_refused(transport, teams, schedule, paths):
    transport()
    with pytest.raises(odds.MultiplierRefused):
        odds.snapshot(season=2025, markets="spreads,totals",
                      state_path=paths["state"], base=paths["store"])


def test_asking_for_a_second_region_is_refused(transport, teams, schedule, paths):
    transport()
    with pytest.raises(odds.MultiplierRefused):
        odds.snapshot(season=2025, regions="us,uk",
                      state_path=paths["state"], base=paths["store"])


def test_the_refusal_explains_the_cost(transport, teams, schedule, paths):
    transport()
    with pytest.raises(odds.MultiplierRefused) as e:
        odds.snapshot(season=2025, markets="spreads,totals,h2h",
                      state_path=paths["state"], base=paths["store"])
    assert "credit" in str(e.value).lower()


# --- the credit floor -----------------------------------------------------

def test_remaining_credits_are_recorded(transport, teams, schedule, paths):
    transport(remaining=372)
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert odds.credits_remaining(paths["state"]) == 372


def test_remaining_credits_are_printed(transport, teams, schedule, paths, capsys):
    """'Cannot silently burn quota' means the balance is never a mystery."""
    transport(remaining=372)
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert "372" in capsys.readouterr().out


def test_a_balance_below_the_floor_refuses_before_calling(transport, teams, schedule, paths):
    calls = transport()
    paths["state"].write_text(json.dumps({"remaining": 3}))
    with pytest.raises(odds.QuotaFloor):
        odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert not calls, "refusing after spending the credit would defeat the point"


def test_the_first_ever_pull_is_allowed(transport, teams, schedule, paths):
    """No stored balance is unknown, not empty. Refusing would be unrecoverable."""
    calls = transport()
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert len(calls) == 1


def test_dropping_below_the_floor_warns_loudly(transport, teams, schedule, paths, capsys):
    transport(remaining=odds.CREDIT_FLOOR - 1)
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert "floor" in capsys.readouterr().out.lower()


def test_a_corrupt_state_file_does_not_block_fetching(transport, teams, schedule, paths):
    paths["state"].write_text("not json")
    calls = transport()
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert len(calls) == 1


# --- mapping to nflverse game ids ----------------------------------------

def test_snapshot_keys_on_the_nflverse_game_id(transport, teams, schedule, paths):
    """Otherwise AS_OF_LINES has nothing to join to and 1.6 was for nothing."""
    transport()
    got = odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert got["game_id"].to_list() == ["2025_01_DAL_PHI"]


def test_an_unmatched_event_is_reported_not_dropped_silently(transport, teams, schedule,
                                                             paths, capsys):
    transport(events=[_event("Kansas City Chiefs", "Dallas Cowboys", "2199-01-01", -3.0)])
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    out = capsys.readouterr().out.lower()
    assert "no nflverse game" in out, "a dropped event must say so"
    assert "1 events" in out


def test_a_matched_game_with_no_posted_line_is_counted_apart(transport, teams, schedule,
                                                             paths, capsys):
    """The two causes used to share one tally reported as a team/date mismatch, which
    asserted the wrong one for half the cases it covered."""
    ev = _event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04", -3.0)
    ev["bookmakers"] = []                        # the game is real; nobody has priced it
    transport(events=[ev])
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    out = capsys.readouterr().out.lower()
    assert "no posted spread" in out
    assert "no nflverse game" not in out, "a priced-less game is not a mapping failure"


# --- the kickoff date, which is not the UTC date ---

def test_a_primetime_kickoff_keeps_the_date_the_game_is_played_on():
    """A 20:20 ET Sunday kickoff is 00:20 UTC on Monday. Slicing the raw string put it on
    Monday and it matched no nflverse game -- 55 of 272 games in 2026 kick off at or after
    20:00 ET, which is every Sunday, Monday and Thursday night game of the season."""
    assert odds._game_date("2026-09-14T00:20:00Z") == "2026-09-13"


def test_an_afternoon_kickoff_is_unchanged():
    assert odds._game_date("2026-09-13T17:00:00Z") == "2026-09-13"


def test_the_conversion_follows_daylight_saving_rather_than_a_fixed_offset():
    """The season crosses out of DST in November. A constant -4 would fix September and
    break December; a constant -5 would do the reverse."""
    assert odds._game_date("2026-09-14T00:20:00Z") == "2026-09-13"   # EDT, UTC-4
    assert odds._game_date("2026-12-08T01:15:00Z") == "2026-12-07"   # EST, UTC-5


def test_an_unparseable_timestamp_falls_back_rather_than_crashing():
    assert odds._game_date("not-a-timestamp") == "not-a-time"
    assert odds._game_date("") == ""


def test_books_are_combined_by_median(transport, teams, schedule, paths):
    """One book's outlier should not become the line of record."""
    transport(events=[_event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04",
                             -8.0, books=3)])
    got = odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    # books quote the home handicap at -8.0, -7.5, -7.0; median -7.5, stored as +7.5
    assert got["close_spread"][0] == pytest.approx(7.5)


def test_a_home_favourite_is_stored_positive(transport, teams, schedule, paths):
    """The sign trap, pinned.

    The Odds API reports a handicap: a home favourite is -8.5, because they must win by
    more than 8.5. nflverse `spread_line` is the reverse -- positive means the home team
    is favoured. The `lines` table speaks nflverse, since that is what `AS_OF_LINES` and
    `store.verify` already assume. Store the raw handicap and every backtest is exactly
    backwards while still looking calibrated in aggregate.
    """
    transport(events=[_event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04",
                             -8.5, books=1)])
    got = odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert got["close_spread"][0] == pytest.approx(8.5)


def test_a_home_underdog_is_stored_negative(transport, teams, schedule, paths):
    transport(events=[_event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04",
                             3.0, books=1)])
    got = odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert got["close_spread"][0] == pytest.approx(-3.0)


# --- what it writes -------------------------------------------------------

def test_it_writes_the_lines_table_the_asof_join_reads(transport, teams, schedule, paths):
    transport()
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"],
                  now=dt.datetime(2025, 9, 3, 12))
    from hub import store
    got = store.sql("SELECT * FROM lines", base=paths["store"])
    assert {"game_id", "close_spread", "captured_at"} <= set(got.columns)


def test_repeated_snapshots_accumulate_rather_than_overwrite(transport, teams, schedule,
                                                             paths):
    """The entire point: AS_OF_LINES needs more than one line per game to be meaningful."""
    transport()
    for hour in (9, 12, 15):
        odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"],
                      now=dt.datetime(2025, 9, 3, hour))
    from hub import store
    assert store.sql("SELECT count(*) n FROM lines", base=paths["store"])["n"][0] == 3


# --- degradation ----------------------------------------------------------

def test_missing_key_degrades_loudly(paths, monkeypatch, capsys):
    monkeypatch.setattr(odds, "_api_key", lambda: None)
    assert odds.main(["--snapshot", "--state-path", str(paths["state"])]) == 1
    assert "ODDS_API_KEY" in capsys.readouterr().err


def test_credits_command_works_without_a_key(paths, monkeypatch, capsys):
    monkeypatch.setattr(odds, "_api_key", lambda: None)
    paths["state"].write_text(json.dumps({"remaining": 372}))
    assert odds.main(["--credits", "--state-path", str(paths["state"])]) == 0
    assert "372" in capsys.readouterr().out
