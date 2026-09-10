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
