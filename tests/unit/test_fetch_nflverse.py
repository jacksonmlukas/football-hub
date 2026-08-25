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


# --- weekly player stats ---------------------------------------------------

def test_player_stats_is_refused_whole():
    """150 columns. Same rule as play-by-play: a source wide enough that handing it back
    entire is the mistake has to be asked for by column."""
    with pytest.raises(nv.WideFrameRefused):
        nv.load("player_stats", seasons=[2024])


def test_player_stats_has_a_named_default_slice():
    """The CLI has to stay usable. `PLAYER_STATS_COLS` is what `--refresh` takes, and it
    has to carry what the weekly-spread fit needs: who, when, and how much he scored."""
    assert {"player_id", "position", "season", "week", "fantasy_points_ppr"} <= set(
        nv.PLAYER_STATS_COLS)


def _weekly(**over):
    base = {"player_id": ["a", "b"], "player_display_name": ["A", "B"],
            "position": ["RB", "WR"], "season": pl.Series([2024, 2024], dtype=pl.Int32),
            "week": pl.Series([1, 1], dtype=pl.Int32), "season_type": ["REG", "REG"],
            "fantasy_points_ppr": [12.5, 8.0]}
    base.update(over)
    return pl.DataFrame(base)


def test_player_stats_passes_its_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(nv, "_raw_player_stats", lambda seasons: _weekly())
    got = nv.load("player_stats", seasons=[2024], cols=nv.PLAYER_STATS_COLS,
                  cache=tmp_path)
    assert got.height == 2


def test_a_renamed_points_column_is_caught(monkeypatch, tmp_path):
    """The failure this exists for: nv renames the column between releases and every
    weekly-spread number goes quietly wrong for a month."""
    bad = _weekly().rename({"fantasy_points_ppr": "fantasy_points_full_ppr"})
    monkeypatch.setattr(nv, "_raw_player_stats", lambda seasons: bad)
    with pytest.raises(nv.WideFrameRefused):
        # tmp cache, or the previous test's cached parquet is served and nothing is
        # validated -- which is how this test first passed for the wrong reason.
        nv.load("player_stats", seasons=[2024], cols=nv.PLAYER_STATS_COLS,
                cache=tmp_path)


def test_rows_belonging_to_no_player_are_dropped(monkeypatch, tmp_path):
    """nflverse ships exactly 22 rows a season with player_id, position and name all null
    and zero points -- residue, not players. The PLAYER_STATS contract declares player_id
    non-null and is right to, so these go at the boundary rather than weakening it. Same
    call as `_clean_ff_opportunity` makes for the same reason."""
    raw = pl.concat([_weekly(), pl.DataFrame({
        "player_id": [None], "player_display_name": [None], "position": [None],
        "season": pl.Series([2024], dtype=pl.Int32),
        "week": pl.Series([1], dtype=pl.Int32),
        "season_type": ["REG"], "fantasy_points_ppr": [0.0]})])
    monkeypatch.setattr(nv, "_raw_player_stats",
                        lambda seasons: nv._clean_player_stats(raw))
    got = nv.load("player_stats", seasons=[2024], cols=nv.PLAYER_STATS_COLS, cache=tmp_path)
    assert got.height == 2


def test_a_real_player_carrying_points_is_never_dropped(monkeypatch):
    """The guard on the guard. If upstream ever ships a scoring row with a null id, that is
    a break to surface, not residue to sweep -- and the drop must not hide it."""
    raw = pl.concat([_weekly(), pl.DataFrame({
        "player_id": [None], "player_display_name": ["Somebody"], "position": ["WR"],
        "season": pl.Series([2024], dtype=pl.Int32),
        "week": pl.Series([1], dtype=pl.Int32),
        "season_type": ["REG"], "fantasy_points_ppr": [14.0]})])
    with pytest.raises(nv.UnattributedPoints):
        nv._clean_player_stats(raw)


# --- the scheme layer -----------------------------------------------------
#
# `participation` (personnel, formation, box count, coverage) and `ftn_charting` (motion,
# play action, screens, blitzers) are the foundation for the game-level model. Added
# 2026-08-24; neither is WIDE, so both come back whole.

def _participation_rows(n=1200, **over):
    base = {
        "nflverse_game_id": ["2024_01_A_B" for _ in range(n)],
        "play_id": [float(i) for i in range(n)],
        "offense_personnel": ["1 RB, 1 TE, 3 WR"] * n,
        "defense_personnel": ["4 DL, 2 LB, 5 DB"] * n,
        "defenders_in_box": pl.Series([6] * n, dtype=pl.Int32),
        "offense_formation": ["SHOTGUN"] * n,
    }
    base.update(over)
    return pl.DataFrame(base)


def test_participation_is_a_known_source():
    from hub.fetch.nflverse import SOURCES
    assert "participation" in SOURCES and "ftn_charting" in SOURCES


def test_participation_tolerates_a_null_formation():
    """~20% of plays have none -- 9,237 of 45,919 in 2024, special teams and plays the
    charter did not resolve. Requiring it non-null would fail every honest refresh."""
    from hub.contracts import PARTICIPATION
    df = _participation_rows(offense_formation=[None] * 1200)
    assert PARTICIPATION.validate(df).height == 1200


def test_participation_requires_the_play_key():
    """There is no `season` column -- rows key on game and play, and the season is a
    load-time argument -- so these two are the only identity the frame has."""
    from hub.contracts import PARTICIPATION, ContractViolation
    df = _participation_rows().drop("play_id")
    with pytest.raises(ContractViolation, match="missing columns"):
        PARTICIPATION.validate(df)


def test_an_impossible_box_count_is_caught():
    """Eleven defenders on the field, so a box of 40 means the upstream shape changed."""
    from hub.contracts import PARTICIPATION, ContractViolation
    df = _participation_rows(defenders_in_box=pl.Series([40] * 1200, dtype=pl.Int32))
    with pytest.raises(ContractViolation, match="range"):
        PARTICIPATION.validate(df)


def test_ftn_charting_contract_holds_on_a_well_formed_frame():
    from hub.contracts import FTN_CHARTING
    n = 1200
    df = pl.DataFrame({
        "nflverse_game_id": ["2024_01_A_B"] * n,
        "nflverse_play_id": pl.Series(range(n), dtype=pl.Int32),
        "season": pl.Series([2024] * n, dtype=pl.Int32),
        "week": pl.Series([1] * n, dtype=pl.Int32),
        "is_play_action": [False] * n,
        "is_motion": [True] * n,
        "n_defense_box": pl.Series([6] * n, dtype=pl.Int32),
    })
    assert FTN_CHARTING.validate(df).height == n


def test_the_scheme_sources_are_not_wide():
    """26 and 29 columns. `pbp` is 372 and `player_stats` 150, and those must be narrowed;
    these need not be, so a caller is not forced to guess a column list."""
    from hub.fetch.nflverse import WIDE
    assert "participation" not in WIDE and "ftn_charting" not in WIDE
