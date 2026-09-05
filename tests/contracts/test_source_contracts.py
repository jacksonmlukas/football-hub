"""Every source, against a frozen payload.

`data-contracts/SKILL.md` prescribes the shape of this: fetch once by hand, freeze the
response, write the contract, test the parser against the frozen copy. Contract tests never
touch the network -- live-API tests in CI are flaky and prove nothing about our parsing.
`tests/golden/` is the nightly job that hits the real API and diffs, and that is what would
actually catch a rename.

What each fixture can and cannot prove is not uniform, and it matters:

  * nflverse and ESPN fixtures are **real captures**, so a passing test here means the
    contract holds against what those APIs actually returned -- the nflverse three on
    2026-08-23, the ESPN scoreboard on 2026-09-05. The ESPN one is trimmed, and the rule
    the re-capture in #73 established is that a trim may remove anything except a path the
    production reader takes: the capture before it had been cut past the competition-level
    status `live_state` reads a game's state from, so the fixture could not fail on the one
    path the code actually walks.
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


def _live_frame(monkeypatch, fixture: str) -> pl.DataFrame:
    """The frozen capture put through `espn.live_state`, as the frame the contract sees.

    **Through the production reader**, which is the point. This file used to lift the four
    contracted columns out of the payload itself, because the capture carried a game's state
    on the *event* while `live_state` reads it off the *competition* -- so handing the reader
    its own fixture raised `KeyError: 'status'`. A fixture validated through a shape
    production does not build cannot fail when production's shape changes, which is the one
    job a frozen payload has. #73 re-captured the endpoint with the competition-level path
    intact and the lifting went with it.

    `espn.scoreboard_frame` rather than a `pl.DataFrame(...)` written out here: the typing is
    what `live_state` asserts the contract on, and a second copy of it beside the first is how
    the two halves drifted apart in the first place.
    """
    from hub.fetch import espn

    payload = json.loads((FIXTURES / fixture).read_text())
    monkeypatch.setattr(espn, "scoreboard", lambda league="nfl": payload)
    return espn.scoreboard_frame(espn.live_state())


def test_espn_scoreboard_contract_holds_on_the_real_capture(monkeypatch):
    """The capture, through the reader that reads it, through the contract it asserts.

    Four shape changes were tried against this on 2026-09-05 and all four are caught, though
    not all by the contract: a renamed `team.abbreviation`, a renamed `homeAway`, and a
    competition that stops carrying `status.type.state` each drop every event, and
    `live_state` raises rather than publishing an empty board as a quiet day. Two events
    sharing an id fails the contract's uniqueness. A single event losing its state is
    *dropped* rather than raised -- that is the degradation this module wants -- and it
    reaches this test as a short frame, which is why the height is asserted at 4 rather than
    at `df.height`. `min_rows=0` is right for this contract (February has no slate), so only
    the count can tell a Sunday that vanished from a Tuesday.

    One change is not caught, measured the same way: an `id` arriving as a number.
    `scoreboard_frame` builds with `SCOREBOARD_TYPES` as `schema_overrides`, which coerces it
    back to `Utf8` before the contract sees it. That is production's behaviour, not something
    the test introduces, and it is written down here rather than left as a surprise.
    """
    assert ESPN_SCOREBOARD.validate(
        _live_frame(monkeypatch, "espn_scoreboard.json")).height == 4


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
