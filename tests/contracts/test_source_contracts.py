"""Every source, against a frozen payload.

`data-contracts/SKILL.md` prescribes the shape of this: fetch once by hand, freeze the
response, write the contract, test the parser against the frozen copy. Contract tests never
touch the network -- live-API tests in CI are flaky and prove nothing about our parsing.
`tests/golden/` is the nightly job that hits the real API and diffs, and that is what would
actually catch a rename.

What each fixture can and cannot prove is not uniform, and it matters:

  * nflverse and ESPN fixtures are **real captures**, so a passing test here means the
    contract holds against what those APIs actually returned on 2026-08-23. The ESPN one is
    a *trimmed* capture -- see `_scoreboard_frame` below, which is where that costs
    something.
  * CFBD and Odds fixtures are **hand-built**, because neither key exists on this machine.
    They prove the parser handles the shape we *believe* is returned. A synthetic fixture
    cannot catch a rename, which is the whole reason contracts exist -- so those two
    sources are structurally covered and empirically unverified. See fixtures/README.md.

Stating that here rather than in a commit message, because the gap is invisible from a
green test run.
"""
import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from hub.contracts import (
    CFBD_GAMES,
    CFBD_LINES,
    ESPN_SCOREBOARD,
    FF_OPPORTUNITY,
    ODDS_SNAPSHOT,
    PBP,
    SCHEDULES,
    ContractViolation,
)

FIXTURES = Path(__file__).resolve().parents[1] / "golden" / "fixtures"


def load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())


def frame(name: str) -> pl.DataFrame:
    return pl.DataFrame(load(name), infer_schema_length=None)


def shape_only(contract):
    """The same contract with its volume floor removed.

    A fixture is a handful of rows on purpose -- committing a thousand would be
    redistribution and unreadable besides. `min_rows` is a production concern (an outage
    that returns three rows is not a good week), and it is tested separately against an
    empty frame. Everything else -- names, nulls, uniqueness, ranges -- is what a fixture
    is for, and those apply unchanged.
    """
    return replace(contract, min_rows=1)


# --- captured fixtures: these prove we parse reality ----------------------

def test_pbp_contract_holds_on_the_real_slice():
    df = frame("nflverse_pbp.json").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    assert shape_only(PBP).validate(df).height == df.height


def test_ff_opportunity_contract_holds_on_the_real_slice():
    df = frame("nflverse_ff_opportunity.json")
    assert shape_only(FF_OPPORTUNITY).validate(df).height == df.height


def test_schedules_contract_holds_on_the_real_slice():
    df = frame("nflverse_schedules.json").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    assert SCHEDULES.validate(df).height == df.height


def _scoreboard_frame(payload: dict) -> pl.DataFrame:
    """The four columns `ESPN_SCOREBOARD` types, lifted out of a raw scoreboard response.

    **Not `espn.live_state`, which would be the better route**, since that is the function the
    contract is actually applied in. Feeding it this capture raises `KeyError: 'status'`: it
    reads the state from `competitions[0]["status"]["type"]` and the capture carries `status`
    on the event with nothing at all on the competition. Measured 2026-09-05 by trying it.

    Recorded as its own test below rather than resolved here. The capture is trimmed to the
    fields the test that froze it asserted, so its silence about a competition-level `status`
    is not evidence that ESPN omits one -- telling those two apart needs a re-capture or a
    change to `hub.fetch.espn`, and neither is a contracts test's call to make.

    `_sides` is imported rather than reimplemented. Reading the side off the array's order
    instead of off the `homeAway` field swaps both teams and both scores, which is the one
    part of this flattening with an incident behind it, and a second copy here would be a
    second place to get it wrong. Events it cannot name are dropped exactly as production
    drops them, so a payload that stops naming its sides reaches the contract as a short
    frame rather than as a wrong one.
    """
    from hub.fetch.espn import SCOREBOARD_TYPES, _sides

    rows: list[dict] = []
    for ev in payload.get("events", []):
        sides = _sides(ev["competitions"][0].get("competitors", []))
        if sides is None:
            continue
        home, away = sides
        rows.append({"id": ev["id"], "state": ev["status"]["type"]["state"],
                     "home": home["team"]["abbreviation"],
                     "away": away["team"]["abbreviation"]})
    return (pl.DataFrame(rows, schema_overrides=SCOREBOARD_TYPES) if rows
            else pl.DataFrame(schema=SCOREBOARD_TYPES))


def test_espn_scoreboard_contract_holds_on_the_real_capture():
    """The capture, through the contract instead of through a list of hand-written asserts.

    This test used to read the same file and check four fields by hand, and that is why
    `ESPN_SCOREBOARD` declared its provenance unmeasured while a real capture of the endpoint
    sat two directories away: nothing routed the payload through a validation, so the resolver
    in `test_every_contract_is_applied.py` had no evidence to see.

    Four shape changes were tried against this on 2026-09-05 and all four are caught: a
    renamed `team.abbreviation` (the lift raises), a renamed `homeAway` (every event drops out
    and the frame arrives empty), two events sharing an id, and a state arriving null. The
    height is asserted at 2 rather than at `df.height` because of the second: `min_rows=0` is
    right for this contract -- February has no slate -- so an empty frame is a shape the
    contract accepts and only the count can tell it from a Sunday that vanished.

    One change is *not* caught, measured the same way: an `id` arriving as a number. Both this
    and `live_state` build the frame with `SCOREBOARD_TYPES` as `schema_overrides`, which
    coerces it back to `Utf8` before the contract sees it, so the dtype check cannot fire on
    these four columns from this direction. That is production's behaviour too, not something
    the test introduces, and it is written down here rather than left as a surprise.
    """
    df = _scoreboard_frame(json.loads((FIXTURES / "espn_scoreboard.json").read_text()))
    assert ESPN_SCOREBOARD.validate(df).height == 2


def test_the_espn_capture_does_not_carry_what_the_poller_reads_its_state_from():
    """The finding from wiring that capture into a real validation, kept where it was made.

    An assertion about the frozen file and not about ESPN: the capture is trimmed, so it
    cannot say whether the endpoint sends a competition-level `status`, and this does not
    claim it does not. What it pins is why `_scoreboard_frame` exists at all. The day someone
    re-captures with that field present this goes red, and the fix is to delete both this and
    the flattening and validate through `espn.live_state`.
    """
    event = json.loads((FIXTURES / "espn_scoreboard.json").read_text())["events"][0]
    assert event["status"]["type"]["state"], "the path this capture carries the state on"
    assert "status" not in event["competitions"][0], (
        "the capture now carries the field `hub.fetch.espn.live_state` reads the state from, "
        "so the flattening in `_scoreboard_frame` has stopped earning its place -- validate "
        "through `live_state` and delete both")


# --- synthetic fixtures: these prove we parse the documented shape --------

def test_cfbd_games_contract_holds_on_the_documented_shape():
    assert CFBD_GAMES.validate(frame("cfbd_games.synthetic.json")).height == 2


def test_cfbd_lines_contract_holds_on_the_documented_shape():
    assert CFBD_LINES.validate(frame("cfbd_lines.synthetic.json")).height == 1


def test_odds_fixture_parses_to_the_lines_table_shape():
    """The parser, not just the contract: the snapshot has to land in `lines`."""
    import datetime as dt

    from hub.fetch.odds import _median_home_spread

    events = load("odds_spreads.synthetic.json")
    rows = [{"game_id": "2025_01_DAL_PHI",
             "close_spread": _median_home_spread(e, e["home_team"]),
             "captured_at": dt.datetime(2025, 9, 4, 18)} for e in events]
    df = pl.DataFrame(rows, schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                    "captured_at": pl.Datetime})
    assert ODDS_SNAPSHOT.validate(df).height == 1
    # books at -8.5 and -8.0; median -8.25, stored positive because home is favoured
    assert df["close_spread"][0] == pytest.approx(8.25)


# --- the contracts have teeth --------------------------------------------

def test_a_renamed_column_fails_every_source():
    """The Week 7 failure this repo names: a field quietly renamed upstream."""
    for contract, fixture in ((SCHEDULES, "nflverse_schedules.json"),
                              (CFBD_GAMES, "cfbd_games.synthetic.json")):
        df = frame(fixture)
        key = next(iter(contract.required))
        with pytest.raises(ContractViolation, match="missing columns"):
            contract.validate(df.rename({key: f"{key}_v2"}))


def test_a_plausible_but_wrong_number_fails():
    """Structural breakage is easy to catch. This is the dangerous case."""
    df = frame("nflverse_schedules.json").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
        pl.lit(120.0).alias("spread_line"))
    with pytest.raises(ContractViolation, match="range"):
        SCHEDULES.validate(df)


def test_an_empty_payload_fails_rather_than_passing_vacuously():
    """An outage that returns [] must not read as a clean week."""
    empty = pl.DataFrame(schema={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                                 "home_team": pl.Utf8, "away_team": pl.Utf8})
    with pytest.raises(ContractViolation):
        SCHEDULES.validate(empty)


def test_duplicate_games_fail():
    df = frame("cfbd_games.synthetic.json")
    with pytest.raises(ContractViolation, match="not unique"):
        CFBD_GAMES.validate(df.head(1).vstack(df.head(1)))


def test_odds_allows_repeated_games_by_design():
    """Several snapshots per game is the point, so uniqueness here would be wrong."""
    import datetime as dt
    df = pl.DataFrame(
        {"game_id": ["g1", "g1"], "close_spread": [-3.0, -4.5],
         "captured_at": [dt.datetime(2025, 9, 1), dt.datetime(2025, 9, 3)]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime})
    assert ODDS_SNAPSHOT.validate(df).height == 2
