"""Every source, against a frozen payload.

`data-contracts/SKILL.md` prescribes the shape of this: fetch once by hand, freeze the
response, write the contract, test the parser against the frozen copy. Contract tests never
touch the network -- live-API tests in CI are flaky and prove nothing about our parsing.
`tests/golden/` is the nightly job that hits the real API and diffs, and that is what would
actually catch a rename.

What each fixture can and cannot prove is not uniform, and it matters:

  * nflverse and ESPN fixtures are **real captures**, so a passing test here means the
    contract holds against what those APIs actually returned -- three of the nflverse six on
    2026-08-23, the three #33 added on 2026-09-05, and both ESPN scoreboards the same day.
    The two ESPN ones are trimmed, and the rule the re-capture in #73 established is that a
    trim may remove anything except a path the production reader takes: the capture before it
    had been cut past the competition-level status `live_state` reads a game's state from, so
    the fixture could not fail on the one path the code actually walks. #75 added the
    NFL board beside the college one and the sentence the college trim had left
    unevidenced -- see `tests/golden/fixtures/README.md`, which records what each trim keeps
    and, for the NFL board, what a capture taken before kickoff cannot show.
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
from polars.testing import assert_frame_equal

from hub.contracts import (
    BIGTEN_CAPTURES,
    CFBD_GAMES,
    CFBD_LINES,
    ESPN_SCOREBOARD,
    FF_OPPORTUNITY,
    FF_RANKINGS,
    INJURIES,
    NFELOQB,
    ODDS_SNAPSHOT,
    PBP,
    POOL_STATE,
    SCHEDULES,
    SNAP_COUNTS,
    ContractViolation,
)
from hub.models.spread import snap_usage

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


def test_ff_rankings_contract_holds_on_the_real_slice():
    """Eight rows of the archive as it stood on 2025-08-29, six from the page this league
    drafts on and two from another -- `page_type` varies in the fixture because 47 pages are
    stacked in the real frame and telling them apart is the column's whole job.

    `best` and `worst` come back `Float64` from the archive and `Int64` from the `draft`
    page. Both are the numeric family, which is the level `_family` compares at, so one
    declaration covers the two pages `load_rankings` serves.
    """
    df = frame("nflverse_ff_rankings.json")
    assert shape_only(FF_RANKINGS).validate(df).height == 8
    assert df["page_type"].n_unique() > 1, "the capture no longer proves pages are told apart"


def test_injuries_contract_holds_on_the_real_slice():
    """Five rows carrying a game designation and three carrying none.

    The three are the reason `report_status` is required and not `non_null`: a player on the
    injury report with no designation is the ordinary case (21,490 of 40,204 rows over
    2019-25), and a contract that refused them would fail every honest refresh. The five are
    the reason the column is not all-null in the fixture -- an all-null column arrives typed
    `Null`, which is its own family and would fail this rather than prove anything.
    """
    df = frame("nflverse_injuries.json").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    assert shape_only(INJURIES).validate(df).height == 8
    assert df["report_status"].null_count() == 3


def test_snap_counts_contract_holds_on_the_real_slice():
    """Six starters and two players who took no offensive snap.

    The zeroes are deliberate: `offense_pct` is a fraction, the range is what would catch
    PFR switching to whole percents, and a fixture of starters alone would never exercise the
    bottom of it.
    """
    df = frame("nflverse_snap_counts.json").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    assert shape_only(SNAP_COUNTS).validate(df).height == 8
    assert df["offense_pct"].min() == 0.0


def test_a_whole_percent_capture_reaches_one_answer_down_one_path():
    """The units change the declaration exists for, on the real capture rather than a
    made-up frame, and read from both ends of it.

    Until #132 this was two assertions in two files with two different expected results and
    nothing checking that they agreed. Here the contract *refused* the whole-percent frame,
    because every snap share multiplied by a hundred is a plausible number in a column
    `hub.models.panel.snap_share` reads and nothing prints. Forty lines from the contract,
    `hub.models.spread.snap_usage` divided the identical case by a hundred and carried on,
    and `tests/unit/test_spread.py` asserted that it did. Whichever was right, routing this
    source through the contract would have made the other unreachable.

    The repair is declared beside the bound now, so there is one answer, and this is where
    the two ends are tied to it: the contract hands back the capture's own numbers, and the
    consumer computes what it computes from the frame that never varied.
    """
    pcts = ("offense_pct", "defense_pct", "st_pct")
    df = frame("nflverse_snap_counts.json").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    percents = df.with_columns([pl.col(c) * 100 for c in pcts])

    repaired = shape_only(SNAP_COUNTS).validate(percents)
    for c in pcts:
        assert repaired[c].to_list() == pytest.approx(df[c].to_list()), c

    xw = pl.DataFrame({"pfr_id": df["pfr_player_id"], "gsis_id": df["pfr_player_id"]})
    read = snap_usage(percents, xw).sort("player_id")
    assert read.height == 1, "the capture no longer carries a drafted-position row to read"
    assert_frame_equal(read, snap_usage(df, xw).sort("player_id"))


def test_one_corrupt_share_in_the_real_capture_is_refused_and_not_rescaled():
    """The units change above, and the thing it must not be confused with, on the same
    capture (#149).

    The row set here is fractions: six starters at 1.0 and two players who took no
    offensive snap. Putting 87.0 into one row leaves a column that is fractions everywhere
    else, and a trigger reading only the frame's tallest value cannot tell that from the
    percent refresh above -- it rescales all three columns by a hundred, the bad row lands
    at 0.87, and the frame passes the very bound that exists to refuse it. 87.0 is chosen
    to be a plausible percent; an impossible number would be refused whatever fired.
    """
    df = frame("nflverse_snap_counts.json").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    holed = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(pl.lit(87.0))
          .otherwise(pl.col("offense_pct")).alias("offense_pct"))
    assert shape_only(SNAP_COUNTS).repairs(holed) == {}, (
        "a boundary would pin this archive as rescaled on ingest, and serve it in units "
        "nothing upstream sent")
    with pytest.raises(ContractViolation, match=r"offense_pct range \[0.0, 87.0\]"):
        shape_only(SNAP_COUNTS).validate(holed)


def test_a_snap_share_no_rescaling_can_rescue_is_still_refused():
    """The repair widens what this contract can say and does not soften what it refuses.
    A hundredth of 500 is 5, which is not a snap share however it was published."""
    df = frame("nflverse_snap_counts.json").with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
        (pl.col("offense_pct") + 500).alias("offense_pct"))
    with pytest.raises(ContractViolation, match="offense_pct range"):
        shape_only(SNAP_COUNTS).validate(df)


def _live_frame(monkeypatch, fixture: str, league: str = "cfb") -> pl.DataFrame:
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
    # Tie the capture to the league it was taken from, by name. The first version of this
    # asserted the requested league against the league this function was *given*, which is
    # the same value twice -- a guard that compares an argument to itself and can never fire.
    # What actually needs checking is the pair: `espn_scoreboard_nfl.json` served as "cfb"
    # would otherwise validate happily and evidence nothing about either board.
    captured = fixture.removesuffix(".json").rsplit("_", 1)[-1]
    assert captured == league, (
        f"{fixture} is a capture of the {captured} board and this call declares it {league}. "
        f"The frame would validate either way, which is why the pair is checked here rather "
        f"than left to match by convention.")

    def _scoreboard(league: str = "nfl", date: str | None = None, *,
                    allow_cache: bool = True) -> dict:
        assert league == captured, (
            f"the reader asked for the {league} board while holding the {captured} capture")
        return payload

    monkeypatch.setattr(espn, "scoreboard", _scoreboard)
    return espn.scoreboard_frame(espn.live_state(league))


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
        _live_frame(monkeypatch, "espn_scoreboard_cfb.json", "cfb")).height == 4


def test_the_nfl_board_has_a_frozen_shape_too(monkeypatch):
    """The other league the poller serves, which had no frozen shape at all until now.

    `espn_scoreboard_cfb.json` is the college board, frozen because it is the one that had
    games in every state on the afternoon it was taken -- and the poller *defaults* to
    `nfl`. So the endpoint the overlay reads most had nothing offline to compare against, and
    a rename would have waited for the nightly canary. This is the same endpoint under
    `LEAGUE_PATHS["nfl"]`, captured the same day, through the same reader.

    **What it evidences and what it cannot.** The 2026 NFL season had not kicked off on
    2026-09-05, so all sixteen Week 1 events on that board were `pre` and an in-progress
    NFL game did not exist to capture. Every path this file's counterpart exercises
    for a live game -- `situation.possession`, `downDistanceText` -- is therefore absent here,
    and no fixture in this repo asserts a shape of the NFL board that anybody
    observed in progress. Freezing one would have meant editing a `pre` event into an `in`
    one, which is a fixture asserting a shape nobody saw: worse than the gap. The gap is
    covered live instead, by `tests/golden/test_golden.py`, which runs both leagues through
    `live_state` nightly.

    What is left is not nothing, and one part of it is covered nowhere else. Sixteen of
    sixteen events resolve, so the field names the reader walks are evidenced on the
    NFL board rather than assumed from the college one. That is the whole of what it earns.

    It is *not* here for `possession` and `down_distance` arriving all-null on a quiet board.
    That was the original justification and it is false: neither column is in
    `SCOREBOARD_TYPES` or the contract's `required`, so nothing validates their dtype and
    dropping both outright still passes. Asserting `pl.Null` here would have tested polars'
    inference and called it contract coverage -- a check that cannot fail, which is the
    defect this repo keeps paying for. It is written down rather than quietly deleted
    because the same sentence sat in `espn.py` for months.
    """
    df = ESPN_SCOREBOARD.validate(
        _live_frame(monkeypatch, "espn_scoreboard_nfl.json", "nfl"))
    assert df.height == 16, (
        f"the NFL capture resolved {df.height} of its events; every one of them "
        f"resolved when it was frozen, so a drop is the reader refusing a shape")
    assert set(df["state"]) == {"pre"}, (
        "the NFL capture is documented as pre-only -- it was taken before the "
        "season kicked off -- and this test's claim about what it cannot evidence depends "
        "on that staying true")
    from hub.fetch.espn import SCOREBOARD_TYPES

    assert set(SCOREBOARD_TYPES) == set(ESPN_SCOREBOARD.required), (
        f"the frame builder types {sorted(SCOREBOARD_TYPES)} and the contract requires "
        f"{sorted(ESPN_SCOREBOARD.required)}. They are declared apart and have to agree: a "
        f"column typed but not required is unvalidated, and one required but not typed "
        f"vanishes from an empty board, which is what the explicit typing is for.")


# --- synthetic fixtures: these prove we parse the documented shape --------

def test_cfbd_games_contract_holds_on_the_documented_shape():
    assert CFBD_GAMES.validate(frame("cfbd_games.synthetic.json")).height == 2


def test_cfbd_lines_contract_holds_on_the_documented_shape():
    assert CFBD_LINES.validate(frame("cfbd_lines.synthetic.json")).height == 1


def test_bigten_captures_contract_holds_on_the_hand_built_index():
    """Hand-built for a different reason than the two above: the shape is this repo's own,
    and the 2026 page held no report on the day it was written, so no capture existed to
    freeze. Four rows over two deadlines, one of each kind, and a lines row with the two
    nullable columns null -- which is the case an inferred schema turns into `Null` and the
    contract has to be shown to accept when the writer declares the dtypes."""
    df = frame("bigten_captures.synthetic.json")
    got = BIGTEN_CAPTURES.validate(df)
    assert got.height == 4
    assert set(got["kind"].to_list()) == {"article", "file", "lines"}
    assert got.filter(pl.col("kind") == "lines")["label"].null_count() == 1


def test_pool_state_contract_holds_on_the_hand_built_payload():
    """Hand-built for the Big Ten reason and one more: the shape is what `hub.fetch.pool`
    *stores*, and the host's payload it is parsed from is documented nowhere in this repo --
    so the fixture is the assumed payload, and the contract meets it only through the
    parser. Five entries, one out, one that has spent nothing, and a week-3 pick in an open
    week that must not be in any Ledger."""
    from hub.fetch import pool

    # One name, not a tuple unpack: `test_every_contract_is_applied` follows the payload
    # from a single-name binding to the `.validate` argument, and a tuple target is invisible
    # to it -- which would leave POOL_STATE reading as evidenced by nothing.
    parsed = pool.parse_payload(load("pool_payload.synthetic.json"), ours="ent-8841")
    got = POOL_STATE.validate(pool.to_frame(parsed[0]))
    assert got.height == 5
    assert got["used"].dtype == pl.List(pl.Utf8), "an empty Ledger inferred as List(Null)"
    assert got.filter(pl.col("entry") == 0)["used"].to_list() == [["DAL", "KC"]]
    assert got["alive"].sum() == 4


def test_nfeloqb_contract_holds_on_the_hand_built_shape():
    """Hand-built in 538's published `qb_elo` schema (#218): no pull of `greerreNFL/nfeloqb`'s
    file has been made from this repo and the suite refuses the network. Seven game rows,
    both sides on each, the last two unplayed with null scores -- the row where a backup is
    first named, and the one the reader must accept without a result."""
    df = frame("nfeloqb_qb_elos.synthetic.json")
    got = NFELOQB.validate(df)
    assert got.height == 7
    assert got["score1"].null_count() == 2
    assert got.schema["qb1_adj"] == pl.Float64


def test_nfeloqb_contract_holds_on_the_captured_file():
    """The first live pull, 2026-09-12: the last 300 rows of `greerreNFL/nfeloqb`'s
    `qb_elos.csv`, seasons 2025-2026. Two played 2026 games carry no quarterback yet --
    the reader drops those before the contract sees them, so here they are dropped the same
    way. The hand-built shape above was right; this is what lets the contract say so."""
    df = frame("nfeloqb_qb_elos.json").filter(
        pl.col("qb1").is_not_null() & pl.col("qb2").is_not_null())
    got = NFELOQB.validate(df)
    assert got.height == 298
    assert set(got["season"].to_list()) == {2025, 2026}
    assert got.schema["qb1_adj"] == pl.Float64


def test_odds_fixture_parses_to_the_lines_table_shape():
    """The parser, not just the contract: the snapshot has to land in `lines`.

    Both markets the poller asks for, because both come back in the one response it pays
    for. The price columns are why the two markets are not the whole of #211: -8.25 at -109
    and -8.25 at -105 are the same point and different prices, and before it the table could
    not tell them apart.
    """
    import datetime as dt

    from hub.fetch.odds import _median_game_total, _median_home_spread, staleness

    events = load("odds_spreads.synthetic.json")
    rows = []
    for e in events:
        spread, spread_price = _median_home_spread(e, e["home_team"])
        total, total_price = _median_game_total(e)
        rows.append({"game_id": "2025_01_DAL_PHI", "close_spread": spread,
                     "spread_price": spread_price, "close_total": total,
                     "total_price": total_price,
                     "captured_at": dt.datetime(2025, 9, 4, 18)})
    df = pl.DataFrame(rows, schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                    "spread_price": pl.Float64, "close_total": pl.Float64,
                                    "total_price": pl.Float64,
                                    "captured_at": pl.Datetime})
    # The two staleness columns are derived, never parsed: `staleness` is the one producer.
    assert ODDS_SNAPSHOT.validate(staleness(df)).height == 1
    # books at -8.5 and -8.0; median -8.25, stored positive because home is favoured
    assert df["close_spread"][0] == pytest.approx(8.25)
    # books at 47.5 and 47.0. A total is a sum, so it is not negated the way a spread is.
    assert df["close_total"][0] == pytest.approx(47.25)
    # -110 and -108 on the home side, -110 and -105 on the Over. Both medians are taken in
    # decimal space, so each lands between its two quotes rather than in the hole American
    # odds have between -100 and +100.
    assert df["spread_price"][0] == pytest.approx(-108.99, abs=0.01)
    assert df["total_price"][0] == pytest.approx(-107.44, abs=0.01)


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


ODDS_COLUMNS = {"game_id": pl.Utf8, "close_spread": pl.Float64,
                "spread_price": pl.Float64, "close_total": pl.Float64,
                "total_price": pl.Float64, "captured_at": pl.Datetime}


def _stamped(df: pl.DataFrame) -> pl.DataFrame:
    """The frame as `hub.fetch.odds._record` writes it: staleness derived, never typed in."""
    from hub.fetch.odds import staleness
    return staleness(df)


def test_odds_allows_repeated_games_by_design():
    """Several snapshots per game is the point, so uniqueness here would be wrong."""
    import datetime as dt
    df = pl.DataFrame(
        {"game_id": ["g1", "g1"], "close_spread": [-3.0, -4.5],
         "spread_price": [-110.0, -105.0], "close_total": [44.5, 45.0],
         "total_price": [-110.0, -110.0],
         "captured_at": [dt.datetime(2025, 9, 1), dt.datetime(2025, 9, 3)]},
        schema=ODDS_COLUMNS)
    assert ODDS_SNAPSHOT.validate(_stamped(df)).height == 2


def test_a_snapshot_that_does_not_say_how_long_its_number_has_stood_is_refused():
    """#210: a dated capture that ranks above the schedule field on provenance must carry
    what says whether the number has moved. The six-column shape is the labelling error."""
    import datetime as dt
    df = pl.DataFrame(
        {"game_id": ["g1"], "close_spread": [-3.0], "spread_price": [-110.0],
         "close_total": [44.5], "total_price": [-110.0],
         "captured_at": [dt.datetime(2025, 9, 1)]},
        schema=ODDS_COLUMNS)
    with pytest.raises(ContractViolation, match=r"polls_unmoved.*unmoved_since"):
        ODDS_SNAPSHOT.validate(df)
    with pytest.raises(ContractViolation, match="polls_unmoved"):
        ODDS_SNAPSHOT.validate(_stamped(df).with_columns(
            pl.lit(None, dtype=pl.Int64).alias("polls_unmoved")))


def test_a_snapshot_of_the_pre_totals_shape_reports_an_absent_total(tmp_path):
    """#211's compatibility clause, held on the store rather than on the contract.

    A partition written before the totals columns existed has three columns where a new one
    has six. `hub.store.connect` reads a table with `union_by_name`, so the old rows come
    back with the three they never had set to null -- absent, which is true, rather than a
    number derived from the spread beside them, which would be invented. Without it DuckDB
    refuses the whole glob and every snapshot ever taken becomes unreadable the day a column
    is added, which is the opposite of what an append-only dated store is for.
    """
    import datetime as dt

    from hub import store

    old = pl.DataFrame(
        {"game_id": ["2025_01_DAL_PHI"], "close_spread": [8.25],
         "captured_at": [dt.datetime(2025, 9, 3, 9)]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                "captured_at": pl.Datetime})
    new = pl.DataFrame(
        {"game_id": ["2025_01_DAL_PHI"], "close_spread": [8.5],
         "spread_price": [-110.0], "close_total": [47.25], "total_price": [-107.44],
         "captured_at": [dt.datetime(2025, 9, 3, 12)]},
        schema=ODDS_COLUMNS)
    store.write(old, "lines", "nfl", 2025, 1, base=tmp_path, name="snap-old")
    store.write(ODDS_SNAPSHOT.validate(_stamped(new)), "lines", "nfl", 2025, 1,
                base=tmp_path, name="snap-new")

    got = store.sql("SELECT close_spread, close_total, total_price FROM lines "
                    "ORDER BY captured_at", base=tmp_path)
    assert got.height == 2, "the older partition has to still be readable, not skipped"
    assert got["close_total"].to_list() == [None, 47.25]
    assert got["total_price"][0] is None, "an unpriced snapshot is null, never zero"
