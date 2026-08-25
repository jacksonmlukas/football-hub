"""The storage layer, which until now had no tests at all.

`hub.store` is what every Phase 1 fetch module is specified to write through, and its
docstring names ASOF JOIN as the reason DuckDB was chosen over SQLite. Neither claim had
ever executed: coverage was 0% and no test imported the module.

The property that matters is the one the docstring asserts -- that matching a prediction
to the line that was *live when it was made* is an as-of problem, and that approximating
it with a join on week is how lookahead gets into a backtest. So the central test builds
the case that breaks the naive join and shows both behaviours side by side.
"""
import datetime as dt

import polars as pl
import pytest

from hub import store


@pytest.fixture
def base(tmp_path):
    """An isolated processed-data root. Real one is never touched by tests."""
    return tmp_path


def _lines(rows):
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows],
         "close_spread": [r[1] for r in rows],
         "captured_at": [r[2] for r in rows]},
        schema={"game_id": pl.Utf8, "close_spread": pl.Float64, "captured_at": pl.Datetime})


def _preds(rows):
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows],
         "model": ["m"] * len(rows),
         "version": ["v1"] * len(rows),
         "home_win_prob": [r[1] for r in rows],
         "predicted_at": [r[2] for r in rows]},
        schema={"game_id": pl.Utf8, "model": pl.Utf8, "version": pl.Utf8,
                "home_win_prob": pl.Float64, "predicted_at": pl.Datetime})


# --- write / read round trip ---------------------------------------------

def test_write_lands_in_a_hive_partition(base):
    p = store.write(_lines([("g1", -3.0, dt.datetime(2025, 9, 1))]),
                    "lines", "nfl", 2025, 1, base=base)
    assert p.relative_to(base).as_posix() == "lines/league=nfl/season=2025/week=01/part.parquet"


def test_partition_keys_come_back_as_columns(base):
    store.write(_preds([("g1", 0.6, dt.datetime(2025, 9, 4))]), "preds", "nfl", 2025, 3,
                base=base)
    got = store.sql("SELECT * FROM preds", base=base)
    assert {"league", "season", "week"} <= set(got.columns)
    assert got["league"][0] == "nfl"


def test_weeks_order_correctly_across_the_single_digit_boundary(base):
    """The `:02d` in LAYOUT is load-bearing.

    Hive keys arrive as strings, so an unpadded week 9 would sort after week 10
    lexically and quietly reorder every weekly sequence. Nothing documented this.
    """
    for wk in (2, 9, 10, 18):
        store.write(_preds([(f"g{wk}", 0.5, dt.datetime(2025, 9, 4))]),
                    "preds", "nfl", 2025, wk, base=base)
    weeks = store.sql("SELECT week FROM preds ORDER BY week", base=base)["week"].to_list()
    assert weeks == sorted(weeks), "weeks must sort in calendar order"
    assert weeks == ["02", "09", "10", "18"]


def test_a_correction_does_not_overwrite_the_original(base):
    """Immutable partitions: a backfill must stay reconstructible and auditable."""
    store.write(_preds([("g1", 0.60, dt.datetime(2025, 9, 4))]), "preds", "nfl", 2025, 1,
                base=base, name="part")
    store.write(_preds([("g1", 0.65, dt.datetime(2025, 9, 5))]), "preds", "nfl", 2025, 1,
                base=base, name="correction")
    got = store.sql("SELECT home_win_prob FROM preds ORDER BY home_win_prob", base=base)
    assert got["home_win_prob"].to_list() == [0.60, 0.65]


def test_querying_an_absent_table_fails_loudly(base):
    """Named, not blind. `pytest.raises(Exception)` also passes when the import is wrong or
    the fixture is broken, so it can go green while testing nothing."""
    import duckdb

    with pytest.raises(duckdb.CatalogException, match="lines"):
        store.sql("SELECT * FROM lines", base=base)


# --- the as-of join, and what it exists to prevent ------------------------

@pytest.fixture
def three_snapshots(base):
    """One game, three line snapshots, one prediction made between the 2nd and 3rd.

    This is the shape `hub.fetch.odds` will produce: the line moves during the week, and
    a prediction is only entitled to the number that existed when it was made.
    """
    store.write(_lines([
        ("g1", -3.0, dt.datetime(2025, 9, 1, 9)),   # Monday
        ("g1", -4.5, dt.datetime(2025, 9, 3, 9)),   # Wednesday  <- live at prediction time
        ("g1", -6.0, dt.datetime(2025, 9, 6, 9)),   # Saturday   <- the future
    ]), "lines", "nfl", 2025, 1, base=base)
    store.write(_preds([("g1", 0.6, dt.datetime(2025, 9, 4, 12))]),  # Thursday
                "preds", "nfl", 2025, 1, base=base)
    return base


def test_asof_matches_the_line_that_was_live(three_snapshots):
    got = store.sql(store.AS_OF_LINES, params=["nfl"], base=three_snapshots)
    assert got.height == 1
    assert got["close_spread"][0] == -4.5
    assert got["captured_at"][0] == dt.datetime(2025, 9, 3, 9)


def test_asof_never_reaches_a_line_captured_after_the_prediction(three_snapshots):
    """The lookahead this module exists to prevent."""
    got = store.sql(store.AS_OF_LINES, params=["nfl"], base=three_snapshots)
    assert got["captured_at"][0] < got["predicted_at"][0]
    assert -6.0 not in got["close_spread"].to_list()


def test_the_naive_week_level_join_gets_it_wrong(three_snapshots):
    """Stated as an executable claim, because the docstring rests on it.

    Joining on game_id alone fans out to every snapshot and admits the Saturday line.
    """
    naive = store.sql(
        "SELECT l.close_spread FROM preds p JOIN lines l ON p.game_id = l.game_id",
        base=three_snapshots)
    assert naive.height == 3, "the naive join fans out over snapshots"
    assert -6.0 in naive["close_spread"].to_list(), "and admits a line from the future"


def test_a_prediction_older_than_every_line_gets_null(base):
    """A LEFT join, so an unmatched prediction survives with a null rather than vanishing."""
    store.write(_lines([("g1", -3.0, dt.datetime(2025, 9, 5))]), "lines", "nfl", 2025, 1,
                base=base)
    store.write(_preds([("g1", 0.6, dt.datetime(2025, 9, 1))]), "preds", "nfl", 2025, 1,
                base=base)
    got = store.sql(store.AS_OF_LINES, params=["nfl"], base=base)
    assert got.height == 1
    assert got["close_spread"][0] is None


def test_asof_is_per_game_not_global(base):
    """Two games moving on different schedules must not borrow each other's lines."""
    store.write(_lines([
        ("g1", -3.0, dt.datetime(2025, 9, 1)),
        ("g2", +7.0, dt.datetime(2025, 9, 2)),
    ]), "lines", "nfl", 2025, 1, base=base)
    store.write(_preds([
        ("g1", 0.6, dt.datetime(2025, 9, 3)),
        ("g2", 0.4, dt.datetime(2025, 9, 3)),
    ]), "preds", "nfl", 2025, 1, base=base)
    got = store.sql(store.AS_OF_LINES, params=["nfl"], base=base).sort("game_id")
    assert got["close_spread"].to_list() == [-3.0, 7.0]


def test_league_filter_is_honoured(base):
    for lg, gid in (("nfl", "n1"), ("cfb", "c1")):
        store.write(_lines([(gid, -1.0, dt.datetime(2025, 9, 1))]), "lines", lg, 2025, 1,
                    base=base)
        store.write(_preds([(gid, 0.5, dt.datetime(2025, 9, 2))]), "preds", lg, 2025, 1,
                    base=base)
    assert store.sql(store.AS_OF_LINES, params=["nfl"], base=base)["game_id"].to_list() == ["n1"]
    assert store.sql(store.AS_OF_LINES, params=["cfb"], base=base)["game_id"].to_list() == ["c1"]


# --- scale ----------------------------------------------------------------

def test_holds_up_at_a_real_season_of_games(base):
    """285 games is the real 2025 count with a published spread."""
    day = dt.datetime(2025, 9, 1)
    rows = [(f"g{i}", -3.0 + i % 7, day + dt.timedelta(hours=i)) for i in range(285)]
    store.write(_lines(rows), "lines", "nfl", 2025, 1, base=base)
    store.write(_preds([(g, 0.5, t + dt.timedelta(hours=1)) for g, _, t in rows]),
                "preds", "nfl", 2025, 1, base=base)
    got = store.sql(store.AS_OF_LINES, params=["nfl"], base=base)
    assert got.height == 285
    assert got["close_spread"].null_count() == 0


# --- the verify entry point ----------------------------------------------

def _fake_schedule(rows):
    return pl.DataFrame(
        {"season": [2025] * len(rows),
         "game_id": [r[0] for r in rows],
         "spread_line": [r[1] for r in rows],
         "gameday": [r[2] for r in rows],
         "gametime": [r[3] for r in rows]},
        schema={"season": pl.Int64, "game_id": pl.Utf8, "spread_line": pl.Float64,
                "gameday": pl.Utf8, "gametime": pl.Utf8})


@pytest.fixture
def fake_nflverse(monkeypatch):
    import nflreadpy as nfl

    def _install(rows):
        monkeypatch.setattr(nfl, "load_schedules", lambda *a, **k: _fake_schedule(rows))
    return _install


def test_verify_passes_on_well_formed_games(fake_nflverse, base, capsys):
    fake_nflverse([("2025_01_DAL_PHI", 8.5, "2025-09-04", "20:20"),
                   ("2025_01_KC_LAC", -3.0, "2025-09-05", "20:00")])
    assert store.verify(season=2025, base=base) == 0
    out = capsys.readouterr().out
    assert "2 matched a line" in out and "0 looked ahead" in out and "OK" in out


def test_verify_tolerates_a_missing_kickoff_time(fake_nflverse, base):
    """gametime is null for some historical rows; a null there must not drop the game."""
    fake_nflverse([("2025_01_DAL_PHI", 8.5, "2025-09-04", None)])
    assert store.verify(season=2025, base=base) == 0


def test_verify_reports_rather_than_crashes_when_there_is_nothing(fake_nflverse, base, capsys):
    fake_nflverse([])
    assert store.verify(season=2025, base=base) == 1
    assert "nothing to verify" in capsys.readouterr().out


def test_verify_converts_spread_to_a_probability_on_the_right_side(fake_nflverse, base):
    """Pins the sign convention, which is the easy thing to get backwards.

    In nflverse a POSITIVE spread_line means the home team is favoured -- 2025_01_DAL_PHI
    is home PHI at +8.5, and PHI won. Invert it and every home favourite lands below 50%:
    a backtest would still run and still look calibrated in aggregate, while being exactly
    wrong on every game.
    """
    fake_nflverse([("fav", 7.0, "2025-09-04", "13:00"),
                   ("dog", -7.0, "2025-09-04", "13:00")])
    store.verify(season=2025, base=base)
    got = store.sql("SELECT game_id, home_win_prob FROM preds ORDER BY game_id", base=base)
    probs = dict(zip(got["game_id"].to_list(), got["home_win_prob"].to_list(), strict=True))
    assert probs["fav"] > 0.5 > probs["dog"]


def test_main_without_verify_prints_help(capsys):
    assert store.main([]) == 0
    assert "verify" in capsys.readouterr().out


def test_main_routes_to_verify(fake_nflverse, monkeypatch, tmp_path):
    fake_nflverse([("g", 1.0, "2025-09-04", "13:00")])
    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "CATALOG", tmp_path / "hub.duckdb")
    assert store.main(["--verify", "--season", "2025"]) == 0


# --- catalog discovery ----------------------------------------------------

def test_any_table_written_becomes_queryable(base):
    """write() accepts any table name, so connect() must expose any table name.

    These were out of step: connect() enumerated four tables while write() took anything,
    so `hub.fetch.nflverse --refresh` wrote 54,402 rows the catalog could not see.
    """
    store.write(pl.DataFrame({"game_id": ["g1"], "epa": [0.1]}),
                "pbp", "nfl", 2025, 1, base=base)
    assert store.sql("SELECT count(*) n FROM pbp", base=base)["n"][0] == 1


def test_several_discovered_tables_coexist(base):
    for table in ("pbp", "ff_opportunity", "preds"):
        store.write(pl.DataFrame({"game_id": ["g1"]}), table, "nfl", 2025, 1, base=base)
    with store.connect(base=base) as con:
        names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"pbp", "ff_opportunity", "preds"} <= names


def test_a_directory_with_no_parquet_is_not_a_table(base):
    (base / "scratch").mkdir()
    with store.connect(base=base) as con:
        assert "scratch" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}


def test_a_directory_that_is_not_a_valid_identifier_is_skipped(base):
    """Otherwise a stray folder name becomes an unquoted SQL identifier."""
    odd = base / "not-an-identifier" / "league=nfl" / "season=2025" / "week=01"
    odd.mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(odd / "part.parquet")
    with store.connect(base=base) as con:
        con.execute("SELECT 1")  # must not have raised while building views


# --- what the catalog can actually see -------------------------------------

def test_an_empty_store_has_no_tables(tmp_path):
    """Not an empty `preds` table -- no `preds` view at all, because `connect` builds one per
    directory that exists. Querying it raises CatalogException, which is how a fresh clone
    answered `conformal --recalibrate` with a stack trace."""
    assert store.tables(tmp_path) == set()


def test_tables_agrees_with_what_connect_actually_creates(base):
    """The whole point of `tables`. If it and `connect` disagree about what exists, a caller
    that checks before querying is checking the wrong thing -- which is worse than not
    checking, because it looks safe."""
    (base / "scratch").mkdir(exist_ok=True)          # a directory with no parquet
    with store.connect(base=base) as con:
        seen = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert store.tables(base) == seen


def test_an_absent_root_is_no_tables_not_a_crash(tmp_path):
    assert store.tables(tmp_path / "nothing") == set()
