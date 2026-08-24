"""The nflverse fetch wrapper.

`Makefile:10` and `weekly-slate/SKILL.md:16` both invoke this and it did not exist.

Its reason to exist is narrow and specific: 2025 play-by-play is 48,771 rows by **372
columns**, and CLAUDE.md rule 1 exists because pulling that whole frame anywhere near a
context window costs a session. nflreadpy has no column selection, so the narrowing has to
happen at this boundary or not at all. The load-bearing test is therefore the refusal:
asking for pbp without naming columns must raise rather than quietly hand back 372 of them.

Everything here runs against an injected fake nflreadpy. Tests that download three seasons
of play-by-play are tests nobody runs.
"""
import polars as pl
import pytest

from hub.contracts import ContractViolation
from hub.fetch import nflverse as nv


@pytest.fixture
def fake_pbp(monkeypatch):
    """A frame with the shape that matters: far more columns than anyone wants."""
    def _install(rows: int = 1200, cols: int = 372):
        named = {
            "game_id": ["2025_01_A_B"] * rows,
            "season": pl.Series([2025] * rows, dtype=pl.Int32),
            "week": pl.Series([1] * rows, dtype=pl.Int32),
            "posteam": ["A"] * rows,
            "defteam": ["B"] * rows,
            "play_type": ["pass"] * rows,
            "epa": [0.1] * rows,
            "wp": [0.5] * rows,
            "yards_gained": [7.0] * rows,
            "success": [1.0] * rows,
        }
        frame = pl.DataFrame({
            **named,
            **{f"filler_{i}": [0.0] * rows for i in range(cols - len(named))},
        })
        monkeypatch.setattr(nv, "_raw_pbp", lambda seasons: frame)
        return frame
    return _install


@pytest.fixture
def fake_ffo(monkeypatch):
    def _install(rows: int = 1200):
        frame = pl.DataFrame({
            "player_id": [f"p{i}" for i in range(rows)],
            "position": ["WR"] * rows,
            "total_fantasy_points_exp": [10.0] * rows,
            "season": pl.Series([2025] * rows, dtype=pl.Int32),
            "week": pl.Series([1] * rows, dtype=pl.Int32),
        })
        monkeypatch.setattr(nv, "_raw_ff_opportunity", lambda seasons: frame)
        return frame
    return _install


# --- the refusal ----------------------------------------------------------

def test_pbp_without_columns_raises(fake_pbp, tmp_path):
    """The whole point of the module."""
    fake_pbp()
    with pytest.raises(nv.WideFrameRefused):
        nv.load("pbp", seasons=[2025], cache=tmp_path)


def test_the_refusal_says_how_to_proceed(fake_pbp, tmp_path):
    fake_pbp()
    with pytest.raises(nv.WideFrameRefused) as e:
        nv.load("pbp", seasons=[2025], cache=tmp_path)
    msg = str(e.value)
    assert "372" in msg, "name the actual width so the number is not abstract"
    assert "cols" in msg


def test_pbp_with_columns_returns_only_those(fake_pbp, tmp_path):
    fake_pbp()
    got = nv.load("pbp", seasons=[2025], cols=["game_id", "season", "week", "epa"],
                  cache=tmp_path)
    assert got.columns == ["game_id", "season", "week", "epa"]


def test_narrow_sources_do_not_require_columns(fake_ffo, tmp_path):
    """Refusing everything would just move the friction rather than remove it."""
    fake_ffo()
    assert nv.load("ff_opportunity", seasons=[2025], cache=tmp_path).height == 1200


def test_asking_for_a_column_that_does_not_exist_raises(fake_pbp, tmp_path):
    fake_pbp()
    with pytest.raises(nv.WideFrameRefused):
        nv.load("pbp", seasons=[2025], cols=["game_id", "nope"], cache=tmp_path)


def test_unknown_source_names_the_known_ones(tmp_path):
    with pytest.raises(nv.WideFrameRefused) as e:
        nv.load("not_a_source", seasons=[2025], cache=tmp_path)
    assert "pbp" in str(e.value)


# --- contracts ------------------------------------------------------------

def test_contract_runs_on_the_way_out(fake_pbp, tmp_path, monkeypatch):
    """A silently renamed upstream column is the Week 7 failure this repo names."""
    frame = fake_pbp()
    monkeypatch.setattr(nv, "_raw_pbp", lambda seasons: frame.drop("game_id"))
    with pytest.raises(ContractViolation):
        nv.load("pbp", seasons=[2025], cols=["season", "week", "epa"], cache=tmp_path)


def test_contract_catches_an_out_of_range_value(fake_ffo, tmp_path, monkeypatch):
    frame = fake_ffo()
    bad = frame.with_columns(pl.lit(500.0).alias("total_fantasy_points_exp"))
    monkeypatch.setattr(nv, "_raw_ff_opportunity", lambda seasons: bad)
    with pytest.raises(ContractViolation):
        nv.load("ff_opportunity", seasons=[2025], cache=tmp_path)


# --- caching --------------------------------------------------------------

def test_second_call_is_served_from_cache(fake_pbp, tmp_path, monkeypatch):
    fake_pbp()
    cols = ["game_id", "season", "week", "epa"]
    nv.load("pbp", seasons=[2025], cols=cols, cache=tmp_path)

    def _boom(seasons):
        raise AssertionError("should have been served from cache")
    monkeypatch.setattr(nv, "_raw_pbp", _boom)
    assert nv.load("pbp", seasons=[2025], cols=cols, cache=tmp_path).height == 1200


def test_refresh_bypasses_the_cache(fake_pbp, tmp_path, monkeypatch):
    fake_pbp()
    cols = ["game_id", "season", "week", "epa"]
    nv.load("pbp", seasons=[2025], cols=cols, cache=tmp_path)
    calls = []
    monkeypatch.setattr(nv, "_raw_pbp", lambda seasons: (calls.append(1), fake_pbp())[1])
    nv.load("pbp", seasons=[2025], cols=cols, cache=tmp_path, refresh=True)
    assert calls, "refresh must go back to the source"


def test_a_different_column_set_is_a_different_cache_entry(fake_pbp, tmp_path):
    """Otherwise the second caller silently gets the first caller's columns."""
    fake_pbp()
    a = nv.load("pbp", seasons=[2025], cols=["game_id", "season", "week"], cache=tmp_path)
    b = nv.load("pbp", seasons=[2025], cols=["game_id", "season", "week", "epa"],
                cache=tmp_path)
    assert a.columns != b.columns


# --- the refresh command --------------------------------------------------

def test_refresh_writes_week_partitions_through_the_store(fake_pbp, fake_ffo, tmp_path):
    fake_pbp()
    fake_ffo()
    store_root = tmp_path / "processed"
    assert nv.refresh(season=2025, cache=tmp_path / "raw", base=store_root) == 0
    written = sorted(p.relative_to(store_root).as_posix()
                     for p in store_root.rglob("*.parquet"))
    assert "pbp/league=nfl/season=2025/week=01/part.parquet" in written
    assert "ff_opportunity/league=nfl/season=2025/week=01/part.parquet" in written


def test_refresh_prints_a_summary_not_a_frame(fake_pbp, fake_ffo, tmp_path, capsys):
    """CLAUDE.md rule 1: a fetch path may print counts, never rows."""
    fake_pbp()
    fake_ffo()
    nv.refresh(season=2025, cache=tmp_path / "raw", base=tmp_path / "processed")
    out = capsys.readouterr().out
    assert len(out.splitlines()) <= 12
    assert "filler_0" not in out, "column names of a 372-wide frame are themselves a dump"


def test_refresh_splits_multi_week_data_into_separate_partitions(fake_ffo, fake_pbp,
                                                                 tmp_path, monkeypatch):
    frame = fake_ffo()
    multi = pl.concat([frame, frame.with_columns(pl.lit(2, dtype=pl.Int32).alias("week"))])
    monkeypatch.setattr(nv, "_raw_ff_opportunity", lambda seasons: multi)
    fake_pbp()
    store_root = tmp_path / "processed"
    nv.refresh(season=2025, cache=tmp_path / "raw", base=store_root)
    weeks = sorted(p.parent.name for p in (store_root / "ff_opportunity").rglob("*.parquet"))
    assert weeks == ["week=01", "week=02"]


def test_main_routes_to_refresh(fake_pbp, fake_ffo, tmp_path, monkeypatch):
    fake_pbp()
    fake_ffo()
    monkeypatch.setattr(nv, "RAW", tmp_path / "raw")
    monkeypatch.setattr("hub.store.DATA", tmp_path / "processed")
    assert nv.main(["--refresh", "--season", "2025"]) == 0


def test_main_without_refresh_prints_help(capsys):
    assert nv.main([]) == 0
    assert "refresh" in capsys.readouterr().out


# --- unattributed ff_opportunity rows ------------------------------------

def test_unattributed_rows_are_dropped():
    """2025 ships 423 of 6,054 rows with no player_id, position or name at all.

    They carry real expected points belonging to nobody. Admitting them would force the
    FF_OPPORTUNITY contract to drop its non-null guarantee on player_id, which is the one
    thing that would catch a genuine upstream rename.
    """
    df = pl.DataFrame({
        "player_id": ["p1", None, "p2"],
        "position": ["WR", None, "RB"],
        "total_fantasy_points_exp": [10.0, 13.31, 8.0],
    })
    got = nv._clean_ff_opportunity(df)
    assert got["player_id"].to_list() == ["p1", "p2"]


def test_cleaning_keeps_everything_when_nothing_is_unattributed():
    df = pl.DataFrame({"player_id": ["p1", "p2"], "total_fantasy_points_exp": [1.0, 2.0]})
    assert nv._clean_ff_opportunity(df).height == 2
