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
import math
import statistics

import polars as pl
import pytest

from hub import store
from hub.contracts import ContractViolation
from hub.fetch import odds


@pytest.fixture
def paths(tmp_path):
    return {"state": tmp_path / "odds.json", "store": tmp_path / "processed"}


@pytest.fixture
def teams(monkeypatch):
    monkeypatch.setattr(odds, "_team_abbrs", lambda valid=None: {
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


def _markets(home, away, points, i, total, price, total_price):
    """One book's markets. `total=None` is a book that has posted a spread and no total."""
    out = [{"key": "spreads", "outcomes": [
        {"name": home, "price": price, "point": points + i * 0.5},
        {"name": away, "price": price, "point": -(points + i * 0.5)},
    ]}]
    if total is not None:
        out.append({"key": "totals", "outcomes": [
            {"name": "Over", "price": total_price, "point": total + i * 0.5},
            {"name": "Under", "price": total_price, "point": total + i * 0.5},
        ]})
    return out


def _event(home, away, day, points, books=2, total=44.5, price=-110, total_price=-110):
    return {
        "id": f"{away}@{home}",
        "commence_time": f"{day}T20:00:00Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {"key": f"book{i}",
             "markets": _markets(home, away, points, i, total, price, total_price)}
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

def test_both_declared_markets_ride_on_one_request(transport, teams, schedule, paths):
    """Cost is markets x regions, and the second market is the point of #211.

    One request, two markets, one region: two credits for a call that returns the whole
    season. Two *requests* would be the same two credits and twice the latency, and would
    also be the shape that grows -- so the thing pinned here is that the number of calls
    does not track the number of markets.
    """
    calls = transport()
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert len(calls) == 1, "a market is a parameter of one call, never a second call"
    assert calls[0]["markets"] == "spreads,totals"
    assert calls[0]["regions"] == odds.REGION
    assert calls[0]["oddsFormat"] == "american", "the price columns are on this staying true"


def test_a_market_outside_the_declared_pair_is_refused(transport, teams, schedule, paths):
    """The refusal that matters, and the one a comma count could never make.

    `player_pass_tds` has no comma in it and runs about four credits an *event* --
    `docs/decisions.md` puts a full week of props at 64. Counting separators waves it
    through; checking the name does not.
    """
    for market in ("h2h", "player_pass_tds", "spreads,h2h", "alternate_spreads"):
        calls = transport()
        with pytest.raises(odds.MultiplierRefused):
            odds.snapshot(season=2025, markets=market,
                          state_path=paths["state"], base=paths["store"])
        assert not calls, f"{market} was refused only after the credit was spent"


def test_a_market_named_twice_is_refused(transport, teams, schedule, paths):
    """Two markets to a biller counting what was asked for, one to a reader skimming it."""
    calls = transport()
    with pytest.raises(odds.MultiplierRefused):
        odds.snapshot(season=2025, markets="spreads,spreads",
                      state_path=paths["state"], base=paths["store"])
    assert not calls


def test_asking_for_a_second_region_is_refused(transport, teams, schedule, paths):
    calls = transport()
    with pytest.raises(odds.MultiplierRefused):
        odds.snapshot(season=2025, regions="us,uk",
                      state_path=paths["state"], base=paths["store"])
    assert not calls


def test_the_refusal_explains_the_cost(transport, teams, schedule, paths):
    transport()
    with pytest.raises(odds.MultiplierRefused) as e:
        odds.snapshot(season=2025, markets="spreads,totals,h2h",
                      state_path=paths["state"], base=paths["store"])
    assert "credit" in str(e.value).lower()


def test_the_declared_budget_is_what_the_guard_allows(transport, teams, schedule, paths):
    """The guard's premise. It permits exactly `MARKETS` x `REGIONS` and no other set, so
    widening the budget is editing those tuples rather than finding a spelling that slips
    past -- which is what "the refusal on a comma exists for a reason" has to mean once the
    refusal is no longer on a comma."""
    assert odds.MARKETS == ("spreads", "totals")
    assert odds.REGIONS == ("us",)
    for good in ("spreads", "totals", "spreads,totals", "totals,spreads"):
        assert odds._budgeted(good, odds.MARKETS, "markets")
    for bad in ("", " spreads", "spreads,", ",spreads", "spreads, totals"):
        with pytest.raises(odds.MultiplierRefused):
            odds._budgeted(bad, odds.MARKETS, "markets")


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


# --- one team, two abbreviations, and a dict that quietly kept the wrong one ---

def test_a_team_listed_under_two_abbreviations_resolves_to_the_one_in_play(monkeypatch):
    """`load_teams` lists "Los Angeles Rams" twice, as LA and as LAR. `dict(zip(...))` keeps
    the last, which is LAR; the 2026 schedule uses LA. Every Rams game therefore matched
    nothing -- 17 of them, the whole residual after the timezone fix, looking like a rounding
    error rather than one team's entire season."""
    import polars as pl

    monkeypatch.setattr(odds, "_schedule", lambda season: pl.DataFrame({}))
    fake = pl.DataFrame({"team_name": ["Los Angeles Rams", "Los Angeles Chargers",
                                       "Los Angeles Rams"],
                         "team_abbr": ["LA", "LAC", "LAR"]})

    class _Nfl:
        @staticmethod
        def load_teams():
            return fake

    monkeypatch.setitem(__import__("sys").modules, "nflreadpy", _Nfl)
    assert odds._team_abbrs({"LA", "LAC"})["Los Angeles Rams"] == "LA"
    assert odds._team_abbrs({"LAR", "LAC"})["Los Angeles Rams"] == "LAR"
    assert odds._team_abbrs({"LAC"})["Los Angeles Rams"] == "LA", "no signal: keep the first"


def test_the_strict_zip_never_protected_against_this(monkeypatch):
    """`strict=True` checks that the two columns are the same length -- they were -- and says
    nothing about duplicate keys. It is why the collapse looked safe."""
    import polars as pl

    fake = pl.DataFrame({"team_name": ["A", "A"], "team_abbr": ["X", "Y"]})

    class _Nfl:
        @staticmethod
        def load_teams():
            return fake

    monkeypatch.setitem(__import__("sys").modules, "nflreadpy", _Nfl)
    assert odds._team_abbrs({"Y"})["A"] == "Y"
    assert len(odds._team_abbrs({"Y"})) == 1, "two rows, one name, one entry"


# --- whose failure was it (issue #119) ------------------------------------

def test_the_betting_market_not_answering_is_reported_as_the_markets(monkeypatch, paths, capsys):
    """The half that was always right, kept as the control. Without it the test below passes
    on a CLI that calls everything a repo-side defect, which is the same error inverted."""
    def _down(params, key):
        raise ConnectionError("odds.api unreachable")
    monkeypatch.setattr(odds, "_http_get", _down)
    monkeypatch.setattr(odds, "_api_key", lambda: "test-key")

    assert odds.main(["--snapshot", "--state-path", str(paths["state"])]) == 1
    err = capsys.readouterr().err
    assert "the betting market's prices unavailable" in err
    assert "ConnectionError" in err


def test_a_failure_after_the_credit_is_spent_is_not_the_markets(
        transport, teams, monkeypatch, paths, capsys):
    """Issue #119. The guard spanned the whole snapshot, so the schedule join against a
    different provider, the contract check and the write all printed as the betting market
    being unavailable -- naming the one source that had just answered.

    The credit is the reason this matters rather than being a wording complaint. It is spent
    and recorded by the time any of these can fail, so the operator is told to retry a fetch
    that worked, and the retry spends another against a metered monthly quota.
    """
    transport()
    monkeypatch.setattr(odds, "_schedule", lambda season: (_ for _ in ()).throw(
        RuntimeError("nflverse schedules did not load")))

    assert odds.main(["--snapshot", "--state-path", str(paths["state"])]) == 1
    err = capsys.readouterr().err
    assert "unavailable" not in err, "the market answered; it is not the thing that failed"
    assert "a credit was spent" in err and "nflverse schedules did not load" in err
    assert "not the betting market's" in err


def test_a_contract_failure_leaves_no_half_written_snapshot(
        transport, teams, monkeypatch, paths):
    """The write is all-or-nothing. Interleaved, a contract failure on the second week left
    the first on disk with a fresh timestamp while the CLI reported nothing was fetched --
    and the as-of join reads a half-written snapshot as a whole one."""
    # Two weeks, because one partition cannot be written partially.
    monkeypatch.setattr(odds, "_schedule", lambda season: pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI", "2025_02_KC_LAC"],
        "season": [2025, 2025], "week": [1, 2],
        "gameday": ["2025-09-04", "2025-09-11"],
        "home_team": ["PHI", "LAC"], "away_team": ["DAL", "KC"]}))
    transport([_event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04", -8.5),
               _event("Los Angeles Chargers", "Kansas City Chiefs", "2025-09-11", -3.0)])
    seen: list = []
    real = odds.ODDS_SNAPSHOT

    class _FailsOnTheSecond:
        """The contract is a frozen dataclass, so the module name is what moves."""
        def validate(self, part):
            seen.append(part)
            if len(seen) > 1:
                raise ValueError("contract violated on the second partition")
            return real.validate(part)
    monkeypatch.setattr(odds, "ODDS_SNAPSHOT", _FailsOnTheSecond())
    wrote: list = []
    monkeypatch.setattr(odds.store, "write",
                        lambda *a, **k: wrote.append(a) or (paths["store"] / "x"))

    with pytest.raises(odds.SnapshotIncomplete):
        odds.snapshot(2025, state_path=paths["state"], base=paths["store"])
    assert wrote == [], "a partition was written before every partition had been checked"


# --- the cost model is checked against the account, not assumed -------------

def test_a_poll_charged_more_than_declared_says_so(transport, teams, schedule, paths, capsys):
    """`_budgeted` refuses a market on a budget of markets x regions. Nothing checked it.

    The allowlist's whole safety argument is that one call costs exactly `len(MARKETS) *
    len(REGIONS)` credits. That is a claim about the vendor's pricing, not about this code, and
    this repo cannot verify it without spending the quota it is protecting. What it *can* do is
    notice when the account disagrees -- and then say so, before the next poll rather than after
    the balance is gone.

    Reached through the balance rather than by calling `_record`: the guard's value is that it
    fires on a real poll, and a test that hand-built the arguments would pass with the guard
    wired to nothing.
    """
    paths["state"].write_text(json.dumps({"remaining": 400,
                                          "checked_at": "2026-09-09T12:00:00"}))
    transport(remaining=397)          # charged 3; markets x regions declares 2
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])

    said = capsys.readouterr().out
    assert "WARNING" in said, f"a poll charged 3 against a declared 2 said nothing: {said}"
    assert "3" in said and "2" in said, f"the warning names neither figure: {said}"


def test_a_poll_charged_what_it_declared_stays_quiet(transport, teams, schedule, paths, capsys):
    """The other half. A guard that warned on every poll would be noise, and noise is how the
    warning stops being read -- which this repo has already paid for once in `preflight_public`.
    """
    paths["state"].write_text(json.dumps({"remaining": 400,
                                          "checked_at": "2026-09-09T12:00:00"}))
    transport(remaining=398)          # charged 2, exactly what markets x regions declares
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])

    said = capsys.readouterr().out
    assert "WARNING: the account was charged" not in said, (
        f"warned about a cost that matched the declaration: {said}")


def test_a_monthly_reset_is_not_reported_as_a_cost(transport, teams, schedule, paths, capsys):
    """The balance going *up* is a reset, not a negative charge, and must not trip the guard.

    The prior balance has to clear the floor, or the poll refuses before it can be charged at
    all and the test passes on the refusal rather than on the guard.
    """
    paths["state"].write_text(json.dumps({"remaining": 60,
                                          "checked_at": "2026-09-09T12:00:00"}))
    transport(remaining=500)
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])

    said = capsys.readouterr().out
    assert "WARNING: the account was charged" not in said, (
        f"a monthly reset was reported as a mismatched charge: {said}")


# --- a book with no posted total, and a price that is not American ---------

def test_a_game_with_a_spread_and_no_total_keeps_the_spread(transport, teams, schedule,
                                                            paths, capsys):
    """Storing totals must not cost a spread that was already being stored.

    Books post a total later than a spread and sometimes not at all. `close_spread` is the
    only non-null priced column in the contract precisely so this case has somewhere to land:
    the row keeps its spread and carries a null total, and the count is said out loud rather
    than left for a reader to notice a shorter frame.
    """
    events = [_event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04", -8.5, total=None)]
    transport(events=events)
    got = odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])

    assert got.height == 1, "the game was dropped for having no total"
    row = got.row(0, named=True)
    assert row["close_spread"] is not None, "the spread was lost with the total"
    assert row["close_total"] is None and row["total_price"] is None
    said = capsys.readouterr().out
    assert "no posted total" in said, f"the missing total was not reported: {said}"


@pytest.mark.parametrize("price", [0, 50, -99, 99.9, "even", None])
def test_a_price_that_is_not_american_becomes_null_not_a_number(price):
    """American odds have a hole between -100 and +100; decimal odds do not.

    So a price inside the hole is the shape of a quote on some *other* scale, and storing it
    as if it were American would put a plausible-looking number in a priced column. It becomes
    null instead, which the contract allows and a reader can see.
    """
    assert odds._american(price) is None, f"{price!r} was accepted as an American price"


@pytest.mark.parametrize(("price", "expected"), [(-110, -110.0), (110, 110.0), (100, 100.0),
                                                 (-100, -100.0), ("-115", -115.0)])
def test_a_real_american_price_survives(price, expected):
    """The other half: a guard that nulled everything would pass the test above."""
    assert odds._american(price) == pytest.approx(expected)


def test_two_books_at_minus_110_and_plus_110_do_not_median_to_zero():
    """The reason the median is taken in decimal space, asserted on the pair that shows it.

    In American space these median to **0**, which is not a price any book can quote and would
    sit unremarked inside any plausibility bound a contract could put on the column.
    """
    assert odds._median_price([-110.0, 110.0]) == pytest.approx(100.45, abs=0.01)
    assert odds._median_price([]) is None


def test_an_outcome_with_no_point_is_skipped_rather_than_stored_as_zero(transport, teams,
                                                                        schedule, paths):
    """A book that lists the side and posts no number yet.

    `point` is the whole content of the row; `float(None)` would raise and `or 0.0` would
    store a pick'em that nobody quoted. It is skipped, and if that leaves no quotes at all the
    game is counted as having no line rather than stored at zero.
    """
    event = _event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04", -8.5, books=1)
    for m in event["bookmakers"][0]["markets"]:
        for outcome in m["outcomes"]:
            outcome["point"] = None
    transport(events=[event])
    got = odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])
    assert got.height == 0, f"a book with no posted number produced a row: {got}"


def test_a_point_with_no_readable_price_keeps_the_point(transport, teams, schedule,
                                                        paths, capsys):
    """The degradation the price column exists to allow.

    The point is unaffected by how the price was quoted, so an unreadable price nulls the
    price and no more -- refusing the frame would throw away a snapshot whose credit is
    already spent, over the half of it that parsed.
    """
    events = [_event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04", -8.5,
                     books=1, price=0, total_price=0)]      # 0 is inside the American hole
    transport(events=events)
    got = odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"])

    assert got.height == 1, "the game was dropped for an unreadable price"
    row = got.row(0, named=True)
    assert row["close_spread"] is not None, "the point was lost with the price"
    assert row["spread_price"] is None
    said = capsys.readouterr().out
    assert "no American price" in said, f"the null price was not reported: {said}"


# --- staleness (#210) -------------------------------------------------------

def _quotes(*polls, game="g1"):
    """One game's polls as (day, spread, price) triples, in the order given."""
    return pl.DataFrame(
        {"game_id": [game] * len(polls),
         "close_spread": [p[1] for p in polls],
         "spread_price": [p[2] for p in polls],
         "captured_at": [dt.datetime(2026, 8, 25) + dt.timedelta(days=p[0]) for p in polls]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "spread_price": pl.Float64, "captured_at": pl.Datetime})


def test_a_first_poll_is_a_run_of_one():
    got = odds.staleness(_quotes((0, -3.0, -110.0)))
    assert got["polls_unmoved"].to_list() == [1]
    assert got["unmoved_since"].to_list() == [dt.datetime(2026, 8, 25)]


def test_polls_at_one_number_count_up_and_a_move_starts_the_count_again():
    """The measure #210 asks for: consecutive polls at this quote, and since when."""
    got = odds.staleness(_quotes((0, -3.0, -110.0), (1, -3.0, -110.0), (2, -3.0, -110.0),
                                 (5, -3.5, -110.0), (6, -3.5, -110.0)))
    assert got["polls_unmoved"].to_list() == [1, 2, 3, 1, 2]
    assert got["unmoved_since"].to_list() == [dt.datetime(2026, 8, 25)] * 3 + [
        dt.datetime(2026, 8, 30)] * 2


def test_never_moved_is_distinguishable_from_polled_once():
    """A game polled eight times at one number and a game polled once both have an
    `unmoved_since` equal to their first capture. Only the count tells them apart, which is
    why there are two columns and not one."""
    eight = _quotes(*[(d, 7.0, -110.0) for d in range(8)], game="frozen")
    once = _quotes((0, 7.0, -110.0), game="fresh")
    got = odds.staleness(pl.concat([eight, once]))
    last = got.group_by("game_id").agg(pl.col("polls_unmoved").last()).sort("game_id")
    assert last["polls_unmoved"].to_list() == [1, 8]


def test_a_price_moving_on_a_fixed_point_is_a_move():
    """-7 at -110 becoming -7 at -120 is the betting market leaning without crossing the
    number. A frozen lookahead is a quote nobody has touched; a shaded price is a touch."""
    got = odds.staleness(_quotes((0, -7.0, -110.0), (1, -7.0, -120.0), (2, -7.0, -120.0)))
    assert got["polls_unmoved"].to_list() == [1, 1, 2]


def test_a_missing_price_is_no_evidence_either_way():
    """Every snapshot before #211 has no price. The poll that first records one is the
    archive gaining a column, not the quote moving, so the point decides alone there."""
    got = odds.staleness(_quotes((0, -7.0, None), (1, -7.0, None), (2, -7.0, -110.0),
                                 (3, -7.0, None), (4, -7.5, None)))
    assert got["polls_unmoved"].to_list() == [1, 2, 3, 4, 1]


def test_a_frame_with_no_price_column_at_all_is_measured_on_the_point():
    """What the archive looks like read back through the store before #211's first poll."""
    got = odds.staleness(_quotes((0, -7.0, None), (1, -7.0, None)).drop("spread_price"))
    assert got["polls_unmoved"].to_list() == [1, 2]
    assert "spread_price" not in got.columns, "a column the caller did not pass was invented"


def test_games_are_measured_apart():
    a = _quotes((0, -3.0, -110.0), (1, -3.0, -110.0), game="a")
    b = _quotes((0, 2.5, -110.0), (1, 3.0, -110.0), game="b")
    got = odds.staleness(pl.concat([b, a]))
    assert got.sort("game_id", "captured_at")["polls_unmoved"].to_list() == [1, 2, 1, 1]


def test_a_snapshot_is_stamped_against_the_polls_already_stored(transport, teams, schedule,
                                                                paths):
    """The guard: the third poll of an unmoved number is written as the third, not as a
    first. `_record` sees only the rows it is about to write, so the count has to come from
    reading the archive back -- and a row that says 1 when the archive says 3 is exactly the
    labelling error #210 names."""
    transport()
    for day in (1, 2, 3):
        odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"],
                      now=dt.datetime(2025, 9, day, 12))
    from hub import store
    got = store.sql("SELECT polls_unmoved, unmoved_since FROM lines ORDER BY captured_at",
                    base=paths["store"])
    assert got["polls_unmoved"].to_list() == [1, 2, 3]
    assert got["unmoved_since"].to_list() == [dt.datetime(2025, 9, 1, 12)] * 3


def test_a_moved_number_is_written_as_a_fresh_run(transport, teams, schedule, paths):
    transport(events=[_event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04", -8.5)])
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"],
                  now=dt.datetime(2025, 9, 1, 12))
    transport(events=[_event("Philadelphia Eagles", "Dallas Cowboys", "2025-09-04", -9.5)])
    got = odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"],
                        now=dt.datetime(2025, 9, 2, 12))
    assert got["polls_unmoved"].to_list() == [1]
    assert got["unmoved_since"].to_list() == [dt.datetime(2025, 9, 2, 12)]


def test_the_archive_on_disk_is_not_rewritten_to_carry_the_columns(transport, teams,
                                                                   schedule, paths):
    """Immutability is the as-of guarantee. Older rows get the columns on read, from the
    same derivation, never from a backfill."""
    from hub import store
    old = pl.DataFrame({"game_id": ["2025_01_DAL_PHI"], "close_spread": [8.0],
                        "captured_at": [dt.datetime(2025, 8, 30, 12)]})
    p = store.write(old, "lines", "nfl", 2025, 1, base=paths["store"], name="snap-old")
    before = p.read_bytes()
    transport()
    odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"],
                  now=dt.datetime(2025, 9, 1, 12))
    assert p.read_bytes() == before
    got = store.sql("SELECT polls_unmoved FROM lines ORDER BY captured_at", base=paths["store"])
    assert got["polls_unmoved"].to_list() == [None, 1], (
        "the old partition must read back with a null, and the new row a run of one -- "
        "8.0 is not the 8.25 the two fixture books median to")


def test_the_staleness_report_names_the_frozen_games_per_week(transport, teams, schedule,
                                                              paths, capsys):
    transport()
    for day in (1, 2):
        odds.snapshot(season=2025, state_path=paths["state"], base=paths["store"],
                      now=dt.datetime(2025, 9, day, 12))
    assert odds.main(["--staleness", "--season", "2025", "--base", str(paths["store"])]) == 0
    said = capsys.readouterr().out
    assert "1 games, 2 polls" in said, said
    assert "     1      1      2       1" in said, said


def test_the_staleness_report_on_a_fresh_clone_says_so(paths, capsys):
    assert odds.main(["--staleness", "--base", str(paths["store"])]) == 0
    assert "no snapshot archive" in capsys.readouterr().out


# --- the noise floor for line movement (#214) --------------------------------

def _polls(*rows, week=1):
    """Archive rows as (game_id, spread, captured_at) already stamped by `staleness`."""
    df = pl.DataFrame(
        {"game_id": [r[0] for r in rows], "close_spread": [float(r[1]) for r in rows],
         "captured_at": [r[2] for r in rows], "week": [week] * len(rows)},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime, "week": pl.Int64})
    return odds.staleness(df)


def _starters(*rows):
    return pl.DataFrame({"dt": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "qb": [r[2] for r in rows]},
                        schema={"dt": pl.Datetime, "team": pl.Utf8, "qb": pl.Utf8})


# Tuesday 2026-08-25 00:58 ET, Thursday 2026-08-27 16:27 and 22:01 ET (the second is past
# midnight UTC), Friday 2026-09-04 11:17 ET, Sunday 2026-09-06 07:44 ET.
TUE = dt.datetime(2026, 8, 25, 4, 58)
THU_A = dt.datetime(2026, 8, 27, 20, 27)
THU_B = dt.datetime(2026, 8, 28, 2, 1)
FRI = dt.datetime(2026, 9, 4, 15, 17)
SUN = dt.datetime(2026, 9, 6, 11, 44)
G = "2026_01_DAL_PHI"


def test_polls_on_one_eastern_date_are_one_poll():
    """Two polls an hour apart on a Thursday evening, one of them past midnight UTC, are one
    Thursday poll -- the later one. Counting the interval between them would report a floor
    that is a fact about the polling, not the line."""
    got = odds.poll_days(_polls((G, -3.0, TUE), (G, -3.0, THU_A), (G, -3.5, THU_B)))
    assert got["captured_at"].to_list() == [TUE, THU_B]
    assert got["close_spread"].to_list() == [-3.0, -3.5], "the last poll of the day stands"


def test_the_first_interval_of_a_game_carries_its_own_change():
    """The bug the report's own sanity count found: shifting after the first poll day is
    dropped nulls every game's first interval and misaligns the rest."""
    got = odds.line_moves(_polls((G, -3.0, TUE), (G, -3.5, THU_B), (G, -2.0, FRI)))
    assert got["delta"].to_list() == [-0.5, 1.5]
    assert got["from"].to_list() == [TUE, THU_B]
    assert got["to"].to_list() == [THU_B, FRI]


def test_an_interval_is_named_by_the_eastern_day_the_later_poll_fell_on():
    got = odds.line_moves(_polls((G, -3.0, TUE), (G, -3.0, THU_B), (G, -3.0, FRI),
                                 (G, -3.0, SUN)))
    assert got["day"].to_list() == ["Thu", "Fri", "Sun"]
    assert got["days"].to_list() == pytest.approx([2.877, 7.553, 1.852], abs=0.001)


def test_a_game_that_never_moved_is_flagged_on_every_interval():
    got = odds.line_moves(pl.concat([
        _polls((G, -3.0, TUE), (G, -3.0, THU_B), (G, -3.0, FRI)),
        _polls(("2026_01_KC_LAC", 2.5, TUE), ("2026_01_KC_LAC", 3.0, THU_B),
               ("2026_01_KC_LAC", 3.0, FRI))]))
    assert got.filter(pl.col("game_id") == G)["frozen"].to_list() == [True, True]
    assert got.filter(pl.col("game_id") != G)["frozen"].to_list() == [False, False]


def test_a_quarterback_change_between_polls_marks_the_interval():
    """Dallas lists a new starter on the Wednesday. The Tuesday-to-Thursday interval spans
    it; Thursday-to-Friday does not."""
    charts = _starters((dt.datetime(2026, 8, 1), "PHI", "qb-phi"),
                       (dt.datetime(2026, 8, 1), "DAL", "qb-dal-1"),
                       (dt.datetime(2026, 8, 26), "DAL", "qb-dal-2"))
    got = odds.line_moves(_polls((G, -3.0, TUE), (G, -3.0, THU_B), (G, -3.0, FRI)), charts)
    assert got["same_qb"].to_list() == [False, True]


def test_a_poll_before_any_chart_is_unknown_not_a_change():
    charts = _starters((dt.datetime(2026, 8, 26), "PHI", "qb-phi"),
                       (dt.datetime(2026, 8, 26), "DAL", "qb-dal"))
    got = odds.line_moves(_polls((G, -3.0, TUE), (G, -3.0, THU_B), (G, -3.0, FRI)), charts)
    assert got["same_qb"].to_list() == [None, True]


def test_without_a_chart_the_condition_is_unknown_throughout():
    got = odds.line_moves(_polls((G, -3.0, TUE), (G, -3.0, THU_B)))
    assert got["same_qb"].to_list() == [None]


def test_an_empty_archive_has_no_intervals():
    assert odds.line_moves(pl.DataFrame(schema={**odds._QUOTE_SCHEMA, "week": pl.Int64,
                                                "polls_unmoved": pl.Int64})).is_empty()
    assert odds.noise_floor(pl.DataFrame(schema=odds.MOVE_COLUMNS))["intervals"] == 0


def _moves(*rows):
    """(game_id, day, delta) as `line_moves` would return them, everything else fixed."""
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows], "week": [1] * len(rows),
         "from": [TUE] * len(rows), "to": [THU_B] * len(rows), "days": [2.0] * len(rows),
         "day": [r[1] for r in rows], "delta": [float(r[2]) for r in rows],
         "frozen": [False] * len(rows), "same_qb": [True] * len(rows)},
        schema=odds.MOVE_COLUMNS)


def test_the_per_day_error_is_clustered_by_game():
    """Two games, two intervals each, all closing on a Thursday. Game A moved +1 twice and
    game B -1 twice: the day's mean is 0, and the two intervals of one game are not two
    readings. Row-wise the se would be sd/sqrt(4) = 0.577; over games each game's residuals
    sum to +/-2, so the sandwich gives sqrt(4 + 4) / 4 = 0.707. The rule cuts both ways,
    but this is the way it cuts here."""
    f = odds.noise_floor(_moves(("a", "Thu", 1), ("a", "Thu", 1),
                                ("b", "Thu", -1), ("b", "Thu", -1)), bootstrap=10)
    (thu,) = f["by_day"]
    assert thu["day"] == "Thu"
    assert thu["intervals"] == 4 and thu["games"] == 2
    assert thu["mean"] == pytest.approx(0.0)
    assert thu["se"] == pytest.approx(math.sqrt(8) / 4)
    assert thu["moved"] == 1.0 and thu["max_abs"] == 1.0


def test_the_floor_is_the_pooled_sd_with_games_resampled():
    f = odds.noise_floor(_moves(("a", "Thu", 1), ("a", "Fri", 0),
                                ("b", "Thu", -1), ("b", "Fri", 0.5),
                                ("c", "Sun", 0), ("c", "Fri", 0)), bootstrap=500)
    deltas = [1, 0, -1, 0.5, 0, 0]
    assert f["sd"] == pytest.approx(statistics.stdev(deltas))
    assert f["games"] == 3 and f["intervals"] == 6
    assert f["sd_lo"] <= f["sd"] <= f["sd_hi"]
    assert f["mean_abs"] == pytest.approx(2.5 / 6)
    assert f["moved"] == pytest.approx(3 / 6)
    assert [d["day"] for d in f["by_day"]] == ["Thu", "Fri", "Sun"], "calendar order"


def test_a_floor_that_never_moved_is_zero_with_a_zero_interval():
    f = odds.noise_floor(_moves(("a", "Thu", 0), ("b", "Thu", 0)), bootstrap=50)
    assert f["sd"] == 0.0 and f["sd_lo"] == 0.0 and f["sd_hi"] == 0.0


def _archive_of(store_dir, *snaps):
    """Write snapshots (captured_at, {game_id: spread}) as the poller would, weeks by id."""
    from hub import store
    for when, spreads in snaps:
        rows = pl.DataFrame({"game_id": list(spreads), "close_spread": list(spreads.values()),
                             "captured_at": [when] * len(spreads)},
                            schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                    "captured_at": pl.Datetime})
        for gid in spreads:
            store.write(rows.filter(pl.col("game_id") == gid), "lines", "nfl", 2026,
                        int(gid.split("_")[1]), base=store_dir, name=f"snap-{when:%Y%m%dT%H%M%S}")


def test_the_noise_floor_report_excludes_frozen_games_and_says_how_many(paths, capsys):
    """Three games. Two never move -- frozen lookaheads, out of the sample and counted --
    and one moves, which is the floor. The cluster unit is named on the line that carries
    the number, and the rule-8 sentence is printed."""
    frozen_a, frozen_b, live = "2026_09_DAL_PHI", "2026_12_KC_LAC", "2026_01_KC_LAC"
    _archive_of(paths["store"],
                (TUE, {frozen_a: -3.0, frozen_b: 7.0, live: 2.5}),
                (THU_A, {frozen_a: -3.0, frozen_b: 7.0, live: 2.5}),
                (THU_B, {frozen_a: -3.0, frozen_b: 7.0, live: 3.0}),
                (FRI, {frozen_a: -3.0, frozen_b: 7.0, live: 2.5}))
    charts = _starters((dt.datetime(2026, 8, 1), "KC", "qb-kc"),
                       (dt.datetime(2026, 8, 1), "LAC", "qb-lac"))
    assert odds.noise_floor_report(2026, base=paths["store"], starters=charts,
                                   bootstrap=20) == 0
    said = capsys.readouterr().out
    assert "3 games, 4 polls on 3 days, weeks 1-12" in said, said
    assert "frozen lookaheads excluded: 2 games" in said, said
    assert "1 games kept" in said, said
    assert "quarterback changes excluded: 0 intervals" in said, said
    assert "over 2 intervals of 1 games (cluster = game)" in said, said
    assert "ceiling for every line-movement question" in said, said
    assert "no poll closed on: Mon, Tue, Wed, Sat, Sun" in said, said


def test_the_noise_floor_report_drops_the_interval_a_quarterback_changed_in(paths, capsys):
    """Two live games, two intervals each. Dallas changes its starter on the Wednesday, so
    the Tuesday-to-Thursday interval of its game is not noise -- it is the study #221 will
    run -- and leaves the sample, counted. Three intervals of two games remain."""
    kc, dal = "2026_01_KC_LAC", "2026_02_DAL_PHI"
    _archive_of(paths["store"],
                (TUE, {kc: 2.5, dal: -3.0}), (THU_B, {kc: 3.0, dal: -6.0}),
                (FRI, {kc: 2.5, dal: -6.5}))
    charts = _starters((dt.datetime(2026, 8, 1), "KC", "qb-kc"),
                       (dt.datetime(2026, 8, 1), "LAC", "qb-lac"),
                       (dt.datetime(2026, 8, 1), "PHI", "qb-phi"),
                       (dt.datetime(2026, 8, 1), "DAL", "qb-dal-1"),
                       (dt.datetime(2026, 8, 26), "DAL", "qb-dal-2"))
    assert odds.noise_floor_report(2026, base=paths["store"], starters=charts,
                                   bootstrap=20) == 0
    said = capsys.readouterr().out
    assert "quarterback changes excluded: 1 intervals" in said, said
    assert "floor: sd 0.577 [" in said, "the +0.5, -0.5, -0.5 that remain; the -3.0 is out"
    assert "over 3 intervals of 2 games (cluster = game)" in said, said


def test_the_noise_floor_report_without_a_chart_measures_and_says_so(paths, capsys,
                                                                     monkeypatch):
    def _no_chart(season):
        raise ConnectionError("no network")
    monkeypatch.setattr(odds, "_qb_starters", _no_chart)
    _archive_of(paths["store"], (TUE, {G: -3.0}), (THU_B, {G: -3.5}))
    assert odds.main(["--noise-floor", "--base", str(paths["store"])]) == 0
    said = capsys.readouterr().out
    assert "same-quarterback condition NOT applied: ConnectionError" in said, said
    assert "quarterback changes excluded" not in said, "a condition not applied is not counted"
    assert "floor: sd" in said, said


def test_the_noise_floor_report_on_a_fresh_clone_says_so(paths, capsys):
    assert odds.main(["--noise-floor", "--base", str(paths["store"])]) == 0
    assert "no snapshot archive" in capsys.readouterr().out


def test_both_reports_name_a_season_the_archive_does_not_hold(paths, capsys):
    """An archive of last season answers a question about this one by saying so, not with
    a table of nothing and not with last season's numbers under this season's label."""
    _archive_of(paths["store"], (TUE, {G: -3.0}))          # written under season 2026
    for flag in ("--staleness", "--noise-floor"):
        assert odds.main([flag, "--season", "2027", "--base", str(paths["store"])]) == 0
        assert "no 2027 snapshots in the archive" in capsys.readouterr().out, flag


# --- player props (#217): the recording half, with no pull behind it ------------------

def _prop_event(home="Philadelphia Eagles", away="Dallas Cowboys", day="2025-09-04", *,
                books=(("Jalen Hurts", "player_pass_yds", 250.5, -115, -105),
                       ("Jalen Hurts", "player_pass_yds", 251.5, -110, -110))):
    """One per-event props response in the documented shape: the side in `name`, the player in
    `description`. Each book carries one (player, market) quote from `books`."""
    return {
        "id": f"{away}@{home}", "commence_time": f"{day}T20:00:00Z",
        "home_team": home, "away_team": away,
        "bookmakers": [
            {"key": f"book{i}", "markets": [{"key": market, "outcomes": [
                {"name": "Over", "description": player, "price": over, "point": point},
                {"name": "Under", "description": player, "price": under, "point": point},
            ]}]}
            for i, (player, market, point, over, under) in enumerate(books)
        ],
    }


NOON = dt.datetime(2025, 9, 3, 12)
_PKEY = ("game_id", "player_key", "market")


def test_prop_quotes_read_the_documented_shape_and_combine_books_by_median():
    rows = odds.prop_quotes(_prop_event(), "2025_01_DAL_PHI", 1, NOON)
    assert len(rows) == 1
    r = rows[0]
    assert r["player_key"] == "jalen hurts" and r["player"] == "Jalen Hurts"
    assert r["market"] == "player_pass_yds" and r["point"] == 251.0
    assert r["over_price"] == pytest.approx(-112.44, abs=0.01), "the median in decimal space"
    assert r["under_price"] == pytest.approx(-107.44, abs=0.01)
    assert r["game_id"] == "2025_01_DAL_PHI" and r["week"] == 1 and r["captured_at"] == NOON


def test_the_anytime_market_has_no_point_and_its_yes_is_the_over():
    ev = _prop_event(books=())
    ev["bookmakers"] = [{"key": "b", "markets": [{"key": "player_anytime_td", "outcomes": [
        {"name": "Yes", "description": "Saquon Barkley", "price": -150},
        {"name": "No", "description": "Saquon Barkley", "price": +120},
        {"name": "Yes", "description": "A Lineman", "price": +2500},
    ]}]}]
    rows = {r["player_key"]: r for r in odds.prop_quotes(ev, "g", 1, NOON)}
    assert rows["saquon barkley"]["point"] is None
    assert rows["saquon barkley"]["over_price"] == pytest.approx(-150.0)
    assert rows["saquon barkley"]["under_price"] == pytest.approx(120.0)
    assert rows["a lineman"]["under_price"] is None, "a one-sided quote keeps its one side"


def test_a_point_market_with_no_point_or_no_over_is_skipped_not_stored():
    ev = _prop_event(books=())
    ev["bookmakers"] = [{"key": "b", "markets": [
        {"key": "player_receptions", "outcomes": [
            {"name": "Over", "description": "No Point", "price": -110},
            {"name": "Under", "description": "No Point", "price": -110}]},
        {"key": "player_receptions", "outcomes": [
            {"name": "Under", "description": "Under Only", "price": -110, "point": 4.5}]},
        {"key": "player_receptions", "outcomes": [
            {"name": "Over", "description": "Priced", "price": -110, "point": 4.5},
            {"name": "Under", "description": "Priced", "price": "even", "point": 4.5}]},
        {"key": "h2h", "outcomes": [{"name": "Over", "description": "Wrong Market",
                                     "price": -110, "point": 1.5}]},
    ]}]
    rows = odds.prop_quotes(ev, "g", 1, NOON)
    assert [r["player_key"] for r in rows] == ["priced"]
    assert rows[0]["under_price"] is None, "an unreadable price is a null beside the point"


def test_no_props_market_is_budgeted_and_every_one_is_refused_by_name(transport, teams,
                                                                       schedule, paths):
    """The line this ticket must not cross. A props market runs about four credits an event,
    and the guard refuses each of them before a request is formed."""
    assert not set(odds.PROP_MARKETS) & set(odds.MARKETS)
    for market in odds.PROP_MARKETS:
        calls = transport()
        with pytest.raises(odds.MultiplierRefused, match="four credits an event"):
            odds.snapshot(season=2025, markets=f"spreads,{market}",
                          state_path=paths["state"], base=paths["store"])
        assert not calls


def test_record_props_writes_prop_lines_and_stamps_staleness_against_the_archive(
        teams, schedule, paths):
    first = odds.record_props([_prop_event()], 2025, NOON, base=paths["store"])
    assert first["polls_unmoved"].to_list() == [1]
    assert "prop_lines" in store.tables(paths["store"])
    # The same quote two hours on is the second poll of a run; a moved quote starts one.
    same = odds.record_props([_prop_event()], 2025, NOON + dt.timedelta(hours=2),
                             base=paths["store"])
    assert same["polls_unmoved"].to_list() == [2]
    assert same["unmoved_since"].to_list() == [NOON]
    moved = odds.record_props(
        [_prop_event(books=(("Jalen Hurts", "player_pass_yds", 255.5, -110, -110),))],
        2025, NOON + dt.timedelta(hours=4), base=paths["store"])
    assert moved["polls_unmoved"].to_list() == [1]
    archive = store.sql("SELECT * FROM prop_lines", base=paths["store"])
    assert archive.height == 3, "snapshots append"
    assert set(archive["week"].cast(pl.Int64)) == {1}


def test_record_props_counts_an_unmatched_event_and_writes_nothing_for_it(teams, schedule,
                                                                           paths, capsys):
    got = odds.record_props([_prop_event(home="Nobody FC")], 2025, NOON, base=paths["store"])
    assert got.is_empty()
    assert "1 events with no nflverse game" in capsys.readouterr().out
    assert "prop_lines" not in store.tables(paths["store"])


def test_the_props_archive_is_refused_under_its_contract(teams, schedule, paths):
    ev = _prop_event(books=(("Jalen Hurts", "player_pass_yds", 900.5, -110, -110),))
    with pytest.raises(ContractViolation, match="point"):
        odds.record_props([ev], 2025, NOON, base=paths["store"])


def test_a_null_point_against_a_null_point_is_not_a_move():
    """The anytime market has no point. Two polls at a null point and one price are one run;
    a plain `!=` on nulls would call every one of them a move and every run a run of one."""
    polls = pl.DataFrame(
        {"game_id": ["g"] * 3, "player_key": ["p"] * 3, "market": ["player_anytime_td"] * 3,
         "point": [None, None, None], "over_price": [-150.0, -150.0, -140.0],
         "captured_at": [dt.datetime(2026, 9, 1), dt.datetime(2026, 9, 2),
                         dt.datetime(2026, 9, 3)]},
        schema={"game_id": pl.Utf8, "player_key": pl.Utf8, "market": pl.Utf8,
                "point": pl.Float64, "over_price": pl.Float64, "captured_at": pl.Datetime})
    got = odds.staleness(polls, key=_PKEY, point="point", price="over_price")
    assert got["polls_unmoved"].to_list() == [1, 2, 1]


def test_prop_staleness_is_measured_per_player_and_market_not_per_game():
    polls = pl.DataFrame(
        {"game_id": ["g"] * 4, "player_key": ["a", "b", "a", "b"],
         "market": ["player_receptions"] * 4, "point": [4.5, 5.5, 4.5, 6.5],
         "over_price": [-110.0] * 4,
         "captured_at": [dt.datetime(2026, 9, 1)] * 2 + [dt.datetime(2026, 9, 2)] * 2},
        schema={"game_id": pl.Utf8, "player_key": pl.Utf8, "market": pl.Utf8,
                "point": pl.Float64, "over_price": pl.Float64, "captured_at": pl.Datetime})
    got = odds.staleness(polls, key=_PKEY, point="point", price="over_price")
    assert got.sort("player_key", "captured_at")["polls_unmoved"].to_list() == [1, 2, 1, 1]


def test_the_record_props_command_reads_a_payload_file_and_spends_nothing(teams, schedule,
                                                                           paths, tmp_path,
                                                                           monkeypatch):
    def _no_http(*a, **k):
        raise AssertionError("a props payload on disk must not reach the betting market")
    monkeypatch.setattr(odds, "_http_get", _no_http)
    payload = tmp_path / "props.json"
    payload.write_text(json.dumps([_prop_event()]))
    assert odds.main(["--record-props", str(payload), "--season", "2025",
                      "--base", str(paths["store"])]) == 0
    assert "prop_lines" in store.tables(paths["store"])
    assert odds.credits_remaining(paths["state"]) is None, "no balance was touched"


def test_a_missing_payload_file_is_a_sentence(paths, tmp_path, capsys):
    code = odds.main(["--record-props", str(tmp_path / "nope.json"),
                      "--base", str(paths["store"])])
    assert code == 1
    err = capsys.readouterr().err
    assert "props payload" in err and "Traceback" not in err
