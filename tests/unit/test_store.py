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


def test_one_predicate_decides_what_a_table_is(tmp_path):
    """`connect` and `tables` each carried a copy, and `tables`' docstring claimed the two
    "cannot disagree" -- an invariant asserted in prose and enforced by nothing. Write and read
    have to agree on what a table is: a hardcoded list here against a `write()` that accepted
    any name is how 54,402 rows landed in the store and invisible to the catalog."""
    (tmp_path / "preds" / "league=nfl").mkdir(parents=True)
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "preds" / "league=nfl" / "x.parquet")
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "not-an-identifier").mkdir()
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "not-an-identifier" / "x.parquet")

    assert store.is_table(tmp_path / "preds")
    assert not store.is_table(tmp_path / "empty_dir"), "a directory with no parquet is not one"
    assert not store.is_table(tmp_path / "not-an-identifier"), "nor is an unquotable name"
    assert store.tables(tmp_path) == {"preds"}
    with store.connect(base=tmp_path) as con:
        # `internal` excludes DuckDB's own information_schema views, which are not ours.
        views = {r[0] for r in con.execute(
            "SELECT view_name FROM duckdb_views() WHERE internal = false").fetchall()}
    assert views == store.tables(tmp_path), "connect and tables cannot disagree"


def test_the_catalog_spans_partitions_written_under_different_schemas(tmp_path):
    """The store is append-only and its schemas evolve -- "immutable dated partitions;
    corrections write a new file" -- so a partition written before a column existed has fewer
    columns than a later one. Without `union_by_name` DuckDB refuses the whole glob rather
    than filling nulls, and it took `publish --all` down on a real store: a 2026 week-1 file
    with twelve columns against a 2025 week-18 file with fourteen.
    """
    old = pl.DataFrame({"game_id": ["g1"], "home_win_prob": [0.6]})
    new = pl.DataFrame({"game_id": ["g2"], "home_win_prob": [0.4],
                        "cfg_digest": ["abc123"]})
    store.write(old, "preds", "nfl", 2025, 18, base=tmp_path)
    store.write(new, "preds", "nfl", 2026, 1, base=tmp_path)

    got = store.sql("SELECT game_id, cfg_digest FROM preds ORDER BY game_id", base=tmp_path)
    assert got.height == 2, "both partitions read, not one or an exception"
    assert got["cfg_digest"].to_list() == [None, "abc123"], \
        "the older partition fills null for a column it predates"


# --- the immutability the docstring used to only promise ---

def _part(v: float) -> pl.DataFrame:
    return pl.DataFrame({"player": ["x"], "rec_yards": [v]})


def test_an_unguarded_overwrite_of_differing_data_is_refused(tmp_path):
    """`write_parquet` truncates and the path is a pure function of its key, so a caller
    reusing a name silently destroyed the previous partition. `fetch.nflverse` did exactly
    that on every weekly refresh, losing the record of any upstream stat correction."""
    store.write(_part(50.0), "player_stats", "nfl", 2025, 3, base=tmp_path)
    with pytest.raises(FileExistsError, match="already holds different data"):
        store.write(_part(72.0), "player_stats", "nfl", 2025, 3, base=tmp_path)
    assert pl.read_parquet(
        store.write(_part(50.0), "player_stats", "nfl", 2025, 3,
                    base=tmp_path))["rec_yards"].to_list() == [50.0]


def test_rewriting_identical_data_is_a_no_op_not_an_error(tmp_path):
    """`make slate` re-runs must stay idempotent, so identical bytes are not a collision."""
    p1 = store.write(_part(50.0), "player_stats", "nfl", 2025, 3, base=tmp_path)
    p2 = store.write(_part(50.0), "player_stats", "nfl", 2025, 3, base=tmp_path)
    assert p1 == p2
    assert len(list(p1.parent.iterdir())) == 1


def test_replace_is_available_but_has_to_be_asked_for(tmp_path):
    """The point is not that replacement is forbidden -- `fetch.nflverse` genuinely mirrors a
    table that revises in place -- it is that it has to be visible in the call."""
    store.write(_part(50.0), "player_stats", "nfl", 2025, 3, base=tmp_path)
    p = store.write(_part(72.0), "player_stats", "nfl", 2025, 3, base=tmp_path, replace=True)
    assert pl.read_parquet(p)["rec_yards"].to_list() == [72.0]


def test_a_distinct_name_keeps_both_partitions(tmp_path):
    """What board, ratings and odds do, and what the error message points a caller at."""
    a = store.write(_part(50.0), "lines", "nfl", 2025, 3, base=tmp_path, name="snap-a")
    b = store.write(_part(72.0), "lines", "nfl", 2025, 3, base=tmp_path, name="snap-b")
    assert a != b
    assert len(list(a.parent.iterdir())) == 2


# --- the line that was live at a moment ------------------------------------

def test_lines_as_of_returns_the_snapshot_that_was_live(base):
    """The as-of question asked forwards. `AS_OF_LINES` prices a prediction that already
    exists; a fit has to ask the same question before it has written anything."""
    store.write(_lines([("g1", -3.0, dt.datetime(2025, 9, 1, 9)),
                        ("g1", -6.5, dt.datetime(2025, 9, 1, 15))]),
                "lines", "nfl", 2025, 1, base=base)
    got = store.lines_as_of(dt.datetime(2025, 9, 1, 12), season=2025, base=base)
    assert got["close_spread"].to_list() == [-3.0]
    assert got["captured_at"].to_list() == [dt.datetime(2025, 9, 1, 9)]


def test_lines_as_of_takes_the_latest_snapshot_not_the_first(base):
    store.write(_lines([("g1", -3.0, dt.datetime(2025, 9, 1, 9)),
                        ("g1", -6.5, dt.datetime(2025, 9, 1, 15))]),
                "lines", "nfl", 2025, 1, base=base)
    got = store.lines_as_of(dt.datetime(2025, 9, 2), season=2025, base=base)
    assert got["close_spread"].to_list() == [-6.5]


def test_a_snapshot_captured_after_the_moment_prices_nothing(base):
    """A line that did not exist yet cannot have priced anything, and returning it would be
    the same lookahead `AS_OF_LINES` exists to prevent."""
    store.write(_lines([("g1", -3.0, dt.datetime(2025, 9, 10))]),
                "lines", "nfl", 2025, 1, base=base)
    assert store.lines_as_of(dt.datetime(2025, 9, 1), season=2025, base=base).height == 0


def test_lines_as_of_on_a_store_with_no_lines_is_empty_rather_than_an_error(base):
    """A fresh clone has no `lines` view at all, and querying one raises rather than
    returning nothing -- the failure mode `tables()` was written for."""
    got = store.lines_as_of(dt.datetime(2025, 9, 1), season=2025, base=base)
    assert got.height == 0
    assert set(got.columns) == {"game_id", "close_spread", "captured_at"}


def test_lines_as_of_is_scoped_to_the_season_asked_for(base):
    store.write(_lines([("g1", -3.0, dt.datetime(2024, 9, 1))]), "lines", "nfl", 2024, 1,
                base=base)
    store.write(_lines([("g2", -7.0, dt.datetime(2025, 9, 1))]), "lines", "nfl", 2025, 1,
                base=base)
    got = store.lines_as_of(dt.datetime(2025, 12, 1), season=2025, base=base)
    assert got["game_id"].to_list() == ["g2"]


# --- one prediction per game ------------------------------------------------
#
# The store keeps every version on purpose: two configurations both survive, which is what
# `docs/foundation-plan.md` 3.5 asks for. Nothing that *read* it knew that. On the live store
# 2026 week 1 held five fitted versions of the same sixteen games, and every consumer treated
# them as eighty independent rows -- the site listed each game five times, `eval._paired`
# built a 25-fold cross product, and `conformal` counted each residual five times while
# reporting the inflated n as its calibration window.

def _versioned(game, version, at, prob=0.6, model="market_baseline"):
    return pl.DataFrame(
        {"game_id": [game], "model": [model], "version": [version],
         "home_win_prob": [prob], "predicted_at": [at]},
        schema={"game_id": pl.Utf8, "model": pl.Utf8, "version": pl.Utf8,
                "home_win_prob": pl.Float64, "predicted_at": pl.Datetime})


def test_predictions_returns_one_row_per_game_not_one_per_version(base):
    store.write(_versioned("g1", "v1", dt.datetime(2026, 9, 1)), "preds", "nfl", 2026, 1,
                base=base, name="v1")
    store.write(_versioned("g1", "v2", dt.datetime(2026, 9, 2)), "preds", "nfl", 2026, 1,
                base=base, name="v2")
    assert store.sql("SELECT * FROM preds", base=base).height == 2, "both are kept"
    got = store.predictions(base=base)
    assert got.height == 1


def test_the_latest_prediction_is_the_one_that_wins(base):
    """The rule, stated: a later write under the same game is a correction -- refreshed odds,
    a new price source -- and the most recent is what the model now says."""
    store.write(_versioned("g1", "old", dt.datetime(2026, 9, 1), prob=0.10), "preds", "nfl",
                2026, 1, base=base, name="a")
    store.write(_versioned("g1", "new", dt.datetime(2026, 9, 3), prob=0.90), "preds", "nfl",
                2026, 1, base=base, name="b")
    got = store.predictions(base=base)
    assert got["version"].to_list() == ["new"]
    assert got["home_win_prob"].to_list() == [0.90]


def test_versions_written_in_the_same_run_resolve_to_one_deterministically(base):
    """Determinism matters more than which one wins. `ratings.fit` writes a partition per
    price source in a single run, so the same game genuinely can carry two rows stamped the
    same second -- and a reader returning whichever DuckDB happened to scan first would give
    a different page on every refresh."""
    at = dt.datetime(2026, 9, 1)
    store.write(_versioned("g1", "market-x-schedule", at), "preds", "nfl", 2026, 1,
                base=base, name="a")
    store.write(_versioned("g1", "market-x-snapshot", at), "preds", "nfl", 2026, 1,
                base=base, name="b")
    first = store.predictions(base=base)["version"].to_list()
    assert len(first) == 1
    assert first == store.predictions(base=base)["version"].to_list()


def test_two_different_games_both_survive(base):
    store.write(_versioned("g1", "v1", dt.datetime(2026, 9, 1)), "preds", "nfl", 2026, 1,
                base=base, name="a")
    store.write(_versioned("g2", "v1", dt.datetime(2026, 9, 1)), "preds", "nfl", 2026, 1,
                base=base, name="b")
    assert sorted(store.predictions(base=base)["game_id"].to_list()) == ["g1", "g2"]


def test_the_same_game_in_two_weeks_is_two_predictions(base):
    """Partitioned by game *and* week: a game re-predicted in a later week is a different
    forecast, not a correction of the first."""
    store.write(_versioned("g1", "v1", dt.datetime(2026, 9, 1)), "preds", "nfl", 2026, 1,
                base=base, name="a")
    store.write(_versioned("g1", "v1", dt.datetime(2026, 9, 8)), "preds", "nfl", 2026, 2,
                base=base, name="a")
    assert store.predictions(base=base).height == 2


def test_predictions_can_be_narrowed_to_a_week_a_season_and_a_model(base):
    store.write(_versioned("g1", "v1", dt.datetime(2026, 9, 1)), "preds", "nfl", 2026, 1,
                base=base, name="a")
    store.write(_versioned("g2", "v1", dt.datetime(2026, 9, 8)), "preds", "nfl", 2026, 2,
                base=base, name="a")
    assert store.predictions(week=1, base=base)["game_id"].to_list() == ["g1"]
    assert store.predictions(season=2026, base=base).height == 2
    assert store.predictions(model="nobody", base=base).is_empty()


def test_the_week_filter_takes_an_int_not_a_padded_key(base):
    """`week_key` padding is the store's business. A caller passing 1 and silently matching
    nothing is the footgun this removes."""
    store.write(_versioned("g1", "v1", dt.datetime(2026, 9, 1)), "preds", "nfl", 2026, 1,
                base=base, name="a")
    assert store.predictions(week=1, base=base).height == 1


def test_predictions_on_a_store_with_none_is_empty_rather_than_an_error(base):
    got = store.predictions(base=base)
    assert got.is_empty()


def test_latest_week_reads_the_padded_partition_key_correctly(base):
    """`max(week)` is a string comparison over `week=01`, and gives the right answer only
    because `week_key` pads. Week 9 sorting above week 10 is what an unpadded key would do."""
    for wk in (1, 9, 10):
        store.write(_versioned(f"g{wk}", "v1", dt.datetime(2026, 9, wk)), "preds", "nfl",
                    2026, wk, base=base, name="a")
    assert store.latest_week(2026, base=base) == 10


def test_latest_week_is_none_when_the_season_has_no_predictions(base):
    assert store.latest_week(2026, base=base) is None
    store.write(_versioned("g1", "v1", dt.datetime(2025, 9, 1)), "preds", "nfl", 2025, 1,
                base=base, name="a")
    assert store.latest_week(2026, base=base) is None


# --- what a reader gets back is what the schema declared (issue #25) ---------
#
# Hive partitioning infers league/season/week from the *path*, and those inferred columns
# shadow the written ones -- so `week`, declared `Int32` and validated on the way in, came
# back as the zero-padded string `"01"` and `season` as `Int64`. Both consumers that treat
# week numerically (`conformal.rolling_coverage`, `eval.compare`) were tested only against
# hand-built frames carrying the declared types, so neither had ever met the store's own
# output. A schema that promises a type and a reader that believes it is the whole point.

def _one_prediction(week=1, season=2026):
    import datetime as dt
    return pl.DataFrame(
        {"game_id": [f"{season}_{week:02d}_LV_KC"], "league": ["nfl"],
         "season": pl.Series([season], dtype=pl.Int32),
         "week": pl.Series([week], dtype=pl.Int32),
         "home_win_prob": [0.6], "margin_mean": [3.0], "margin_lo": [-14.0],
         "margin_hi": [20.0], "model": ["market_baseline"], "version": ["v1"],
         "fit_through_week": pl.Series([week - 1], dtype=pl.Int32),
         "predicted_at": [dt.datetime(2026, 9, 1)]})


def test_a_prediction_reads_back_with_the_dtypes_it_was_written_with(tmp_path):
    """Against `PREDICTION_SCHEMA` itself rather than a repeated literal, so the assertion
    cannot drift away from the declaration it exists to check."""
    from hub.models.base import PREDICTION_SCHEMA
    store.write(_one_prediction(), "preds", "nfl", 2026, 1, base=tmp_path)
    got = store.predictions(season=2026, base=tmp_path)
    diverged = {c: (PREDICTION_SCHEMA[c], got.schema[c]) for c in got.columns
                if c in PREDICTION_SCHEMA and got.schema[c] != PREDICTION_SCHEMA[c]}
    assert not diverged, f"declared vs returned: {diverged}"


def test_the_week_read_back_is_the_number_not_the_padded_key(tmp_path):
    """`week_key` pads for the path. A caller filtering `week > 4` on `"01"` is doing string
    comparison, and a caller passing the int to `is_in` raises."""
    store.write(_one_prediction(week=9), "preds", "nfl", 2026, 9, base=tmp_path)
    got = store.predictions(season=2026, base=tmp_path)
    assert got["week"].to_list() == [9]
    assert got.filter(pl.col("week").is_in([9])).height == 1


def test_reading_a_narrowed_week_still_returns_every_declared_column(tmp_path):
    store.write(_one_prediction(week=2), "preds", "nfl", 2026, 2, base=tmp_path)
    got = store.predictions(season=2026, week=2, base=tmp_path)
    assert got.height == 1 and got["season"].to_list() == [2026]
