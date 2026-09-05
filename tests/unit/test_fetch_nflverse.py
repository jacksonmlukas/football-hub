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
import json
from datetime import date

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
    charter did not resolve. Requiring it non-null would fail every honest refresh.

    Typed explicitly, because that is what the real frame looks like: a `Utf8` column with
    nulls in it. This fixture used to pass `[None] * 1200`, which polars types as `Null` --
    a column with no type at all, which is a different failure and is now caught below.
    """
    from hub.contracts import PARTICIPATION
    df = _participation_rows(
        offense_formation=pl.Series([None] * 1200, dtype=pl.Utf8))
    assert PARTICIPATION.validate(df).height == 1200


def test_a_column_that_arrives_untyped_is_caught():
    """The silent degradation the dtype check exists for. A source returning nothing, or a
    join that matched nothing, yields an all-null column typed `Null`. It passes every
    column-presence check and then produces nulls wherever it is used --
    `fetch/espn.py:161` hand-wrote a `schema=` to work around exactly this."""
    from hub.contracts import PARTICIPATION, ContractViolation
    df = _participation_rows(offense_formation=[None] * 1200)
    assert df.schema["offense_formation"] == pl.Null, "the fixture must be untyped"
    with pytest.raises(ContractViolation, match="declared string"):
        PARTICIPATION.validate(df)


def test_a_retyped_column_is_caught():
    """The other half. Renaming was already caught; retyping was not, and the module
    docstring names both as the same Week 7 failure."""
    from hub.contracts import PARTICIPATION, ContractViolation
    df = _participation_rows(play_id=[str(i) for i in range(1200)])
    with pytest.raises(ContractViolation, match="declared numeric"):
        PARTICIPATION.validate(df)


def test_a_wider_integer_is_not_a_violation():
    """Families, not exact dtypes. nflverse ships a count as Int32 one season and Int64 the
    next; nothing downstream cares, and failing on it would be noise."""
    from hub.contracts import PARTICIPATION
    df = _participation_rows(defenders_in_box=pl.Series([6] * 1200, dtype=pl.Int64))
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


# --- the as-of pin and the data digest ------------------------------------
#
# U1 of `docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md`. Two mechanisms,
# because the sources differ: an append-only archive carrying a scrape date is *filtered* at
# the as-of, so a later fetch of a grown archive yields the same rows; a source that revises
# in place can only be *labelled*, so its pin carries the moment it was taken.
#
# The failure being closed is the one the plan opens on: no gate in the tree re-runs to its
# own number while the FantasyPros archive it scores against is refetched live every time.

@pytest.fixture
def fake_archive(monkeypatch):
    """An append-only source, faked onto `ff_opportunity`.

    `ff_rankings` is the archive the filter exists for and it has no loader until U2 of the
    same plan -- it is page-typed rather than season-parameterised and needs a contract
    first. So the registry is monkeypatched onto a source that *is* wired up: what is under
    test here is the filter, not a loader that does not exist yet.
    """
    def _install(dates, per_date: int = 1200):
        n = len(dates) * per_date
        frame = pl.DataFrame({
            "player_id": [f"p{i}" for i in range(n)],
            "position": ["WR"] * n,
            "total_fantasy_points_exp": [10.0] * n,
            "season": pl.Series([2025] * n, dtype=pl.Int32),
            "week": pl.Series([1] * n, dtype=pl.Int32),
            "scrape_date": pl.Series(
                [d for d in dates for _ in range(per_date)]).str.to_date(),
        })
        monkeypatch.setattr(nv, "APPEND_ONLY", {"ff_opportunity": "scrape_date"})
        monkeypatch.setattr(nv, "_raw_ff_opportunity", lambda seasons: frame)
        return frame
    return _install


def test_an_as_of_writes_a_dated_cache_path(fake_ffo, tmp_path):
    fake_ffo()
    nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")
    written = [p.name for p in (tmp_path / "ff_opportunity").glob("*.parquet")]
    assert written and all("2026-09-04" in name for name in written), written


def test_two_as_ofs_are_two_entries_and_neither_is_served_the_others_rows(
        fake_ffo, tmp_path, monkeypatch):
    """The reason the column set is already part of the key, applied to the date."""
    frame = fake_ffo()
    early = nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-08-01")
    monkeypatch.setattr(nv, "_raw_ff_opportunity", lambda seasons: frame.head(1100))
    late = nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")
    assert (early.height, late.height) == (1200, 1100)
    again = nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-08-01")
    assert again.height == 1200, "the earlier pin was overwritten by the later one"


def test_a_second_load_at_one_as_of_hits_the_network_once(fake_ffo, tmp_path, monkeypatch):
    fake_ffo()
    nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")

    def _boom(seasons):
        raise AssertionError("a pinned load must be served from its dated entry")
    monkeypatch.setattr(nv, "_raw_ff_opportunity", _boom)
    assert nv.load("ff_opportunity", seasons=[2025], cache=tmp_path,
                   as_of="2026-09-04").height == 1200


def test_a_load_with_no_as_of_uses_the_undated_path_for_read_and_write(
        fake_ffo, tmp_path, monkeypatch):
    """What keeps this a prefactor: `make slate` passes no as-of and must not move."""
    fake_ffo()
    nv.load("ff_opportunity", seasons=[2025], cache=tmp_path)
    names = [p.name for p in (tmp_path / "ff_opportunity").glob("*.parquet")]
    assert names == [nv._cache_path("ff_opportunity", [2025], None, tmp_path).name]

    def _boom(seasons):
        raise AssertionError("the undated entry must still be read back")
    monkeypatch.setattr(nv, "_raw_ff_opportunity", _boom)
    assert nv.load("ff_opportunity", seasons=[2025], cache=tmp_path).height == 1200


def test_refresh_rewrites_the_entry_matching_the_as_of_it_was_given(
        fake_ffo, tmp_path, monkeypatch):
    frame = fake_ffo()
    nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")
    nv.load("ff_opportunity", seasons=[2025], cache=tmp_path)
    monkeypatch.setattr(nv, "_raw_ff_opportunity", lambda seasons: frame.head(1100))
    pinned = nv.load("ff_opportunity", seasons=[2025], cache=tmp_path,
                     as_of="2026-09-04", refresh=True)
    assert pinned.height == 1100
    assert nv.load("ff_opportunity", seasons=[2025], cache=tmp_path).height == 1200, (
        "a refresh at an as-of must not rewrite the undated entry")


def test_an_append_only_archive_is_filtered_at_the_as_of(fake_archive, tmp_path):
    fake_archive(["2026-08-01", "2026-09-10"])
    got = nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")
    assert got.height == 1200
    assert got["scrape_date"].max() == date(2026, 8, 1), "a later scrape survived the as-of"


def test_a_grown_archive_still_yields_the_same_rows_at_one_as_of(
        fake_archive, tmp_path, monkeypatch):
    """The whole point of R2: the archive grows and the pinned frame does not move."""
    original = fake_archive(["2026-08-01"])
    first = nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")
    grown = pl.concat([original, original.with_columns(
        pl.lit("2026-09-20").str.to_date().alias("scrape_date"))])
    monkeypatch.setattr(nv, "_raw_ff_opportunity", lambda seasons: grown)
    second = nv.load("ff_opportunity", seasons=[2025], cache=tmp_path,
                     as_of="2026-09-04", refresh=True)
    assert second.equals(first)
    pin = nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04")
    assert pin is not None
    assert pin.digest == nv.pin_digest("ff_opportunity", "2026-09-04", first), (
        "the archive grew and the digest of the pinned frame moved with it")


def test_ff_rankings_is_declared_append_only():
    """The archive the filter exists for. `hub.draft.tune.holdout` already bounds
    `scrape_date` on this source by hand; U2 gives it a loader that reads this registry."""
    assert nv.APPEND_ONLY["ff_rankings"] == "scrape_date"


def test_a_vanished_scrape_date_column_is_a_contract_violation(
        fake_ffo, tmp_path, monkeypatch):
    """A source declared append-only whose date column is gone cannot be pinned, and
    silently returning the unfiltered archive would be the drift this unit exists to catch."""
    fake_ffo()
    monkeypatch.setattr(nv, "APPEND_ONLY", {"ff_opportunity": "scrape_date"})
    with pytest.raises(ContractViolation, match="scrape_date"):
        nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")


def test_two_runs_at_one_as_of_that_fetched_different_bytes_disagree(
        fake_ffo, tmp_path, monkeypatch):
    """The digest hashes content, not labels. Hashing the source name and the as-of alone
    would be invariant to exactly the drift it exists to catch."""
    frame = fake_ffo()
    nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")
    before = nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04")
    monkeypatch.setattr(nv, "_raw_ff_opportunity", lambda seasons: frame.with_columns(
        pl.lit(11.0).alias("total_fantasy_points_exp")))
    nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04",
            refresh=True)
    after = nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04")
    assert before is not None and after is not None
    assert before.digest != after.digest, "same as-of, different bytes, same digest"


def test_the_digest_moves_with_the_as_of_on_identical_content(fake_ffo, tmp_path):
    """Folded with the source and the as-of, so two pins of the same rows are still two
    pins -- `hub.fetch.odds` names a snapshot by when it was taken for the same reason."""
    fake_ffo()
    for stamp in ("2026-08-01", "2026-09-04"):
        nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of=stamp)
    a = nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-08-01")
    b = nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04")
    assert a is not None and b is not None and a.digest != b.digest


def test_the_digest_is_reproducible_in_a_fresh_process(fake_ffo, tmp_path):
    """Python's `hash()` is salted per process, so a digest resting on it would change on
    every run and could not name the data a published gate scored against.

    Proved rather than asserted: the digest is recomputed in a subprocess, from the cache
    file this process wrote, under two `PYTHONHASHSEED` values that differ from each other.
    A literal in this file would prove nothing -- it would pin whatever this process
    happened to produce.
    """
    import os
    import subprocess
    import sys

    fake_ffo()
    nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")
    pin = nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04")
    assert pin is not None
    path = nv._cache_path("ff_opportunity", [2025], None, tmp_path,
                          date(2026, 9, 4))
    script = ("import sys, polars as pl\n"
              "from hub.fetch.nflverse import pin_digest\n"
              "print(pin_digest('ff_opportunity', '2026-09-04', pl.read_parquet(sys.argv[1])))")
    seen = set()
    for seed in ("0", "524287"):
        env = {**os.environ, "PYTHONHASHSEED": seed,
               "PYTHONPATH": str(nv.ROOT / "src")}
        out = subprocess.run([sys.executable, "-c", script, str(path)],
                             capture_output=True, text=True, env=env, timeout=120)
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert seen == {pin.digest}, seen


def test_a_source_that_revises_in_place_carries_a_pinned_at(monkeypatch, tmp_path):
    """`player_stats` mirrors an upstream that rewrites -- `_write_by_week` says so where it
    passes `replace=True`. An as-of cannot reproduce those rows, so the pin records when they
    were taken instead of claiming they can be fetched again."""
    monkeypatch.setattr(nv, "_raw_player_stats", lambda seasons: _weekly())
    nv.load("player_stats", seasons=[2024], cols=nv.PLAYER_STATS_COLS, cache=tmp_path,
            as_of="2026-09-04")
    pin = nv.data_pin("player_stats", [2024], cols=nv.PLAYER_STATS_COLS, cache=tmp_path,
                      as_of="2026-09-04")
    assert pin is not None and pin.pinned_at is not None
    assert pin.as_of == "2026-09-04" and pin.rows == 2


def test_an_archive_filtered_at_its_as_of_needs_no_pinned_at(fake_archive, tmp_path):
    """The other half of the same distinction: these rows *are* reproducible from the
    as-of, so a stamp saying when they were taken would claim less than is true."""
    fake_archive(["2026-08-01"])
    nv.load("ff_opportunity", seasons=[2025], cache=tmp_path, as_of="2026-09-04")
    pin = nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04")
    assert pin is not None and pin.pinned_at is None


def test_a_pin_that_was_never_written_reads_as_none(tmp_path):
    """Graceful degradation: caches written before this unit have no pin beside them, and
    asking for one must answer "nothing recorded" rather than raise on a cold tree."""
    assert nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04") is None


# --- what is reachable, and what the docstrings say is -----------------------
#
# `APPEND_ONLY` declares a property of a *source*, which is the right thing to declare early
# and the reason `hub.draft.tune.holdout` already bounds `scrape_date` by hand. Until #33 the
# one source declared there had no loader, so `load` refused it at the `SOURCES` check before
# the as-of filter was ever consulted, and every pin a real caller could write carried a
# `pinned_at`. Two tests stood here saying so, and the second asserted that three docstrings
# still admitted it -- a canary, written to go red the day a loader landed.
#
# It went red on #33, which is the canary working. Both are rewritten rather than deleted,
# because the property each was protecting outlived the state it described: that the
# reproducible half of the pin is reached by a *real load* and not only by a fixture that
# substitutes the registry, and that the module's prose still matches what a caller can do.
# The second is now an equivalence rather than a one-way check, so it fails in both
# directions -- if the loader is ever pulled back out of `SOURCES`, the wording has to come
# back with it.

def test_the_append_only_path_is_reachable_through_load(fake_rankings, tmp_path):
    """The half of `Pin` no caller could reach before #33, reached the way a caller would.

    This is what `test_no_declared_append_only_source_can_be_loaded_yet` was asking for when
    it went red. `fake_archive` above proves the *filter* works by monkeypatching
    `APPEND_ONLY` onto a source that has a loader; that is honest about the filter and says
    nothing about whether anything reaches it. Here the registry is the real one, the source
    is the one declared in it, and only the network is faked.
    """
    assert sorted(set(nv.APPEND_ONLY) & set(nv.SOURCES)) == ["ff_rankings"]
    fake_rankings(["2026-08-01", "2026-09-10"])
    got = nv.load_rankings("all", as_of="2026-09-04", cache=tmp_path)
    assert got["scrape_date"].unique().to_list() == ["2026-08-01"], (
        "a scrape from after the as-of survived a pinned load")
    pin = nv.data_pin("ff_rankings", ["all"], cache=tmp_path, as_of="2026-09-04")
    assert pin is not None and pin.pinned_at is None, (
        "these rows reproduce from the as-of alone, so a stamp would claim less than is true")


def test_the_docstrings_match_what_a_caller_can_reach():
    """The wording is the deliverable, so it is asserted rather than trusted -- in both
    directions now. A field whose docstring describes a state its callers cannot produce is
    how a reader comes to believe the archive is pinned reproducibly when every pin says
    otherwise; a module that has quietly lost its only append-only loader while still
    promising the path is the same defect with the sign flipped.

    The whole module text, not `Pin.__doc__` and `load.__doc__` alone: the third statement of
    this lived in the comment above `APPEND_ONLY`, where no attribute lookup reaches it, and
    the canary's own instructions named all three.
    """
    reachable = bool(set(nv.APPEND_ONLY) & set(nv.SOURCES))
    from pathlib import Path
    promised_unreachable = "not yet reachable" in Path(nv.__file__).read_text()
    assert reachable is not promised_unreachable, (
        f"append-only sources reachable through load: {reachable}; module still says "
        f"'not yet reachable': {promised_unreachable}. Those have to disagree -- the wording "
        f"exists to tell a reader which half of `Pin` a caller can actually produce.")


# --- a sidecar is read back leniently ----------------------------------------
#
# `data_pin` reconstructs a fixed record type from JSON. A pin written by a later version of
# this module -- an extra field, say -- would raise on the way in, and a half-written file
# would raise differently. Both are the same call the module already makes for a cold tree:
# answer "nothing recorded" or answer with what is understood, never take down the caller
# asking about provenance.

def _write_pin(tmp_path, payload: str) -> None:
    path = nv._pin_path(nv._cache_path("ff_opportunity", [2025], None, tmp_path,
                                       date(2026, 9, 4)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def test_a_pin_written_by_a_later_version_degrades_rather_than_raising(tmp_path):
    """A field this version has never heard of is not a reason to lose the digest it can
    read. The pin sits in a tree that outlives any one version of this module."""
    _write_pin(tmp_path, json.dumps({
        "source": "ff_opportunity", "as_of": "2026-09-04", "digest": "aaaaaaaa",
        "rows": 1200, "pinned_at": None, "upstream_release": "2026.09.1"}))
    pin = nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04")
    assert pin is not None
    assert (pin.source, pin.as_of, pin.digest, pin.rows) == (
        "ff_opportunity", "2026-09-04", "aaaaaaaa", 1200)


def test_a_pin_that_cannot_be_parsed_reads_as_none(tmp_path):
    """Same answer as a cold tree, for the same reason: an interrupted write is not an error
    a gate asking what it scored against should die on."""
    _write_pin(tmp_path, '{"source": "ff_opportunity", "as_of": "2026-')
    assert nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04") is None


def test_a_pin_missing_what_identifies_it_reads_as_none(tmp_path):
    """A record without a digest names no data, so it is nothing recorded rather than a Pin
    with a hole in it."""
    _write_pin(tmp_path, json.dumps({"source": "ff_opportunity", "rows": 1200}))
    assert nv.data_pin("ff_opportunity", [2025], cache=tmp_path, as_of="2026-09-04") is None


# --- something reads the digest ----------------------------------------------
#
# Until this unit, nothing in `src/` computed a data digest: the pins were written and never
# read back, so a moved archive still surfaced as a silently different number, which is the
# whole defect. `refresh` is the path that actually pulls nflverse, so it is the path that
# has to name what it pulled.

def test_refresh_names_the_data_it_pulled(fake_pbp, fake_ffo, tmp_path, capsys):
    """The digest of the two frames just written, printed beside the digests of the code
    that will read them."""
    from hub.config import HubConfig, config_digest, data_digest, fitted_digest
    fake_pbp()
    fake_ffo()
    nv.refresh(season=2025, cache=tmp_path, base=tmp_path / "store")
    out = capsys.readouterr().out
    pins = [nv.data_pin(t, [2025], cols=c, cache=tmp_path)
            for t, c in (("pbp", list(nv.PBP_COLS)), ("ff_opportunity", None))]
    assert all(p is not None for p in pins)
    expected = data_digest([p for p in pins if p is not None])
    assert expected != "unpinned"
    assert f"data {expected}" in out
    assert f"cfg {config_digest(HubConfig())}" in out
    assert f"fitted {fitted_digest()}" in out


def test_a_moved_archive_shows_up_in_what_refresh_prints(fake_pbp, fake_ffo, tmp_path,
                                                         capsys, monkeypatch):
    """The claim R1 makes, end to end on the one path that fetches: the bytes change, the
    printed digest changes, and no other digest moves."""
    import re
    fake_pbp()
    frame = fake_ffo()
    nv.refresh(season=2025, cache=tmp_path, base=tmp_path / "store")
    first = capsys.readouterr().out
    monkeypatch.setattr(nv, "_raw_ff_opportunity", lambda seasons: frame.with_columns(
        pl.lit(11.0).alias("total_fantasy_points_exp")))
    nv.refresh(season=2025, cache=tmp_path, base=tmp_path / "store")
    second = capsys.readouterr().out

    def digests(text):
        return dict(re.findall(r"(cfg|fitted|data) ([0-9a-f]{8}|unpinned)", text))
    a, b = digests(first), digests(second)
    assert set(a) == {"cfg", "fitted", "data"}, a
    assert a["data"] != b["data"], "the archive moved and the printed digest did not"
    assert (a["cfg"], a["fitted"]) == (b["cfg"], b["fitted"])


# --- the three sources #33 added ---------------------------------------------
#
# U2 of `docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md`. The fetch layer
# could not serve the rankings archive, the injury report or snap counts, so the eight call
# sites that read them imported `nflreadpy` directly and nothing validated, cached or pinned
# what came back. This unit is the loaders and the contracts; the call sites move in #34-36.
#
# Two of the three are ordinary season-partitioned sources. `ff_rankings` is not: it is one
# table of every scrape of 47 ranking pages, so its loader takes a page type and an as-of,
# and it is the source `APPEND_ONLY` was declared for.

def _rankings_frame(dates=("2026-08-28",), per_date: int = 4, **over) -> pl.DataFrame:
    """A well-formed slice of the archive: the nine columns `FF_RANKINGS` declares.

    `scrape_date` is a string here because it is a string upstream -- `_as_of_filter` parses
    it, `hub.draft.board.consensus` compares it as text, and a fake that handed back a real
    date would test neither.
    """
    n = len(dates) * per_date
    return pl.DataFrame({
        "page_type": ["redraft-overall"] * n,
        "player": [f"Player {i}" for i in range(n)],
        "pos": ["WR"] * n,
        "team": ["PHI"] * n,
        "ecr": [float(i + 1) for i in range(n)],
        "sd": [1.5] * n,
        "best": [1.0] * n,
        "worst": [20.0] * n,
        "scrape_date": [d for d in dates for _ in range(per_date)],
    } | dict(over))


def _injuries_frame(rows: int = 4, **over) -> pl.DataFrame:
    """`report_status` is null on one row on purpose: it is null on 21,490 of 40,204 rows
    over 2019-25, so a fake without one would be testing a shape nflverse does not send."""
    return pl.DataFrame({
        "season": pl.Series([2024] * rows, dtype=pl.Int32),
        "week": pl.Series([1] * rows, dtype=pl.Int32),
        "team": ["PHI"] * rows,
        "game_type": ["REG"] * rows,
        "gsis_id": [f"00-00{i:05d}" for i in range(rows)],
        "position": ["WR"] * rows,
        "full_name": [f"Player {i}" for i in range(rows)],
        "report_status": ["Questionable"] * (rows - 1) + [None],
        "practice_status": ["Limited Participation in Practice"] * rows,
    } | dict(over))


def _snaps_frame(rows: int = 4, **over) -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": [f"2024_01_PHI_DAL_{i}" for i in range(rows)],
        "season": pl.Series([2024] * rows, dtype=pl.Int32),
        "week": pl.Series([1] * rows, dtype=pl.Int32),
        "game_type": ["REG"] * rows,
        "player": [f"Player {i}" for i in range(rows)],
        "pfr_player_id": [f"Play00{i:02d}" for i in range(rows)],
        "position": ["WR"] * rows,
        "team": ["PHI"] * rows,
        "opponent": ["DAL"] * rows,
        "offense_snaps": [55.0] * rows,
        "offense_pct": [0.82] * rows,
        "defense_pct": [0.0] * rows,
        "st_pct": [0.14] * rows,
    } | dict(over))


@pytest.fixture
def fake_rankings(monkeypatch):
    """The archive, faked onto the real `ff_rankings` source.

    Unlike `fake_archive` above, `APPEND_ONLY` is left alone: this is the source declared in
    it, so the as-of filter runs because the registry says so rather than because a test said
    so. Only the network is replaced.
    """
    def _install(dates=("2026-08-28",), per_date: int = 4, **over):
        frame = _rankings_frame(dates, per_date, **over)
        monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: frame)
        return frame
    return _install


# The three, and how to break each one. Every entry names a column whose disappearance, whose
# nullity or whose scale change is a failure somebody downstream would otherwise absorb:
# `ecr` is the board's rank, `report_status` is the designation `hub.models.injury` fits on,
# `offense_pct` is the snap share `hub.models.panel` reads, and the `85.0` in that row is the
# unit change -- PFR publishing whole percents -- rather than an impossible number.
BREAKAGES = {
    "ff_rankings": (_rankings_frame, "ecr", "scrape_date", "ecr", 5000.0),
    "injuries": (_injuries_frame, "report_status", "gsis_id", "week", 99),
    "snap_counts": (_snaps_frame, "offense_pct", "game_id", "offense_pct", 85.0),
}

RAW_FETCHER = {"ff_rankings": "_raw_ff_rankings", "injuries": "_raw_injuries",
               "snap_counts": "_raw_snap_counts"}


def _load(source: str, frame: pl.DataFrame, monkeypatch, tmp_path) -> pl.DataFrame:
    """Put one frame through the real loader for its source.

    Through `load`, not through `CONTRACT.validate` -- a contract that holds and is applied
    nowhere is the thing `tests/contracts/test_every_contract_is_applied.py` exists for, and
    a refusal proved against the contract object alone would prove exactly that much.

    No as-of, deliberately. `_as_of_filter` drops rows whose scrape date is null before the
    contract ever sees them, which is right -- a row that cannot be placed in time must not
    survive a pinned load -- but it would also hide the null-column refusal below on the one
    source where the null column *is* the scrape date. Unpinned, the contract is what catches
    it, which is the path every non-pinned caller takes.
    """
    monkeypatch.setattr(nv, RAW_FETCHER[source], lambda keys: frame)
    keys = ["all"] if source == "ff_rankings" else [2024]
    return nv.load(source, seasons=keys, cache=tmp_path)


@pytest.mark.parametrize("source", sorted(BREAKAGES))
def test_each_new_source_loads_and_validates(source, monkeypatch, tmp_path):
    build = BREAKAGES[source][0]
    assert _load(source, build(), monkeypatch, tmp_path).height == 4


@pytest.mark.parametrize("source", sorted(BREAKAGES))
def test_a_missing_column_is_refused(source, monkeypatch, tmp_path):
    """The Week 7 failure this repo names: a column quietly renamed upstream."""
    build, drop, _, _, _ = BREAKAGES[source]
    with pytest.raises(ContractViolation, match="missing columns"):
        _load(source, build().drop(drop), monkeypatch, tmp_path)


@pytest.mark.parametrize("source", sorted(BREAKAGES))
def test_a_null_in_a_non_nullable_column_is_refused(source, monkeypatch, tmp_path):
    """Nullity, not absence. A column that is present and empty passes every name check and
    then produces nulls wherever it is read -- which for `scrape_date` means a row that
    cannot be placed in time and for `gsis_id` means a row that joins to nobody."""
    build, _, col, _, _ = BREAKAGES[source]
    df = build()
    holed = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
          .then(pl.lit(None, dtype=df.schema[col]))
          .otherwise(pl.col(col)).alias(col))
    assert holed[col].null_count() == 1, "the fake did not actually punch a hole"
    with pytest.raises(ContractViolation, match=f"{col} has 1 nulls"):
        _load(source, holed, monkeypatch, tmp_path)


@pytest.mark.parametrize("source", sorted(BREAKAGES))
def test_a_value_outside_its_plausible_range_is_refused(source, monkeypatch, tmp_path):
    """The dangerous case: structural breakage announces itself, a plausible-looking number
    does not. `offense_pct` at 85.0 is PFR switching from fractions to whole percents, which
    would multiply every snap share by a hundred in a column nothing prints."""
    build, _, _, col, bad = BREAKAGES[source]
    df = build()
    with pytest.raises(ContractViolation, match="range"):
        _load(source, df.with_columns(pl.lit(bad).cast(df.schema[col]).alias(col)),
              monkeypatch, tmp_path)


def test_the_new_sources_are_not_wide():
    """25, 17 and 16 columns. `WIDE` is for the two that cost a session to hand back whole,
    and listing a 16-column source there would make the refusal a formality nobody reads."""
    assert not set(BREAKAGES) & set(nv.WIDE)


# --- the rankings loader: a page type and an as-of ---------------------------

def test_the_rankings_loader_refuses_a_season_list():
    """The source is one table of every scrape, so a season list identifies nothing. It is
    refused rather than ignored: a caller who believes they have bounded a load to 2024 and
    has not is exactly the reader who would then publish a number from the whole archive."""
    with pytest.raises(nv.WideFrameRefused, match="not season-partitioned"):
        nv.load_rankings("all", seasons=[2024])


def test_a_season_list_is_refused_at_the_other_door_too(tmp_path):
    """`load_rankings` is a front door, not a fence. Reaching `load` directly with a year
    where the page belongs has to fail there as well, and before any network call."""
    with pytest.raises(nv.WideFrameRefused, match="one page type"):
        nv.load("ff_rankings", seasons=[2024], cache=tmp_path)


def test_an_unknown_rankings_page_is_refused_and_names_the_known_ones():
    """`week` is a real nflreadpy page and deliberately not served: its columns are a
    different table (`page_pos`, `player_name`, `rank`), so `FF_RANKINGS` would either have
    to be loosened to cover both or fail on every weekly load."""
    with pytest.raises(nv.WideFrameRefused) as e:
        nv.load_rankings("week")
    assert "draft" in str(e.value) and "all" in str(e.value)


def test_the_page_is_part_of_the_cache_key(fake_rankings, tmp_path, monkeypatch):
    """Two pages are two entries. Without that a caller asking for the archive would be
    served the last caller's draft page -- 5,850 rows of one scrape, on the same columns."""
    fake_rankings(per_date=4)
    assert nv.load_rankings("all", cache=tmp_path).height == 4
    monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: _rankings_frame(per_date=2))
    assert nv.load_rankings("draft", cache=tmp_path).height == 2
    assert nv.load_rankings("all", cache=tmp_path).height == 4, "the draft page overwrote it"


def test_a_rankings_load_at_an_as_of_is_filtered_not_merely_labelled(fake_rankings, tmp_path):
    """The distinction the whole pin rests on, on the source it was declared for."""
    fake_rankings(["2026-08-01", "2026-08-28", "2026-09-10"], per_date=2)
    got = nv.load_rankings("all", as_of="2026-08-28", cache=tmp_path)
    assert got.height == 4
    assert got["scrape_date"].max() == "2026-08-28", "the as-of is inclusive of its own day"


def test_a_grown_rankings_archive_still_yields_the_same_rows_at_one_as_of(
        fake_rankings, tmp_path, monkeypatch):
    """R2, end to end through the real source rather than through a substituted registry.

    The archive gains a week -- FantasyPros scrapes weekly -- and a gate re-run at the same
    as-of has to answer with the same rows and the same digest, or the number it published
    cannot be reproduced.
    """
    original = fake_rankings(["2026-08-01"], per_date=3)
    first = nv.load_rankings("all", as_of="2026-09-04", cache=tmp_path)
    grown = pl.concat([original, original.with_columns(
        pl.lit("2026-09-20").alias("scrape_date"))])
    monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: grown)
    second = nv.load_rankings("all", as_of="2026-09-04", cache=tmp_path, refresh=True)
    assert second.equals(first)
    pin = nv.data_pin("ff_rankings", ["all"], cache=tmp_path, as_of="2026-09-04")
    assert pin is not None
    assert pin.digest == nv.pin_digest("ff_rankings", "2026-09-04", first), (
        "the archive grew and the digest of the pinned frame moved with it")


def test_two_rankings_as_ofs_are_two_pins_and_two_digests(fake_rankings, tmp_path):
    """A pin is a claim about a date as well as about rows, so two as-ofs over one archive
    must not collide -- the filtered content differs and the digest has to say so."""
    fake_rankings(["2026-08-01", "2026-09-10"], per_date=2)
    early = nv.load_rankings("all", as_of="2026-08-15", cache=tmp_path)
    late = nv.load_rankings("all", as_of="2026-09-20", cache=tmp_path)
    assert (early.height, late.height) == (2, 4)
    pins = [nv.data_pin("ff_rankings", ["all"], cache=tmp_path, as_of=d)
            for d in ("2026-08-15", "2026-09-20")]
    assert all(p is not None for p in pins)
    assert pins[0].digest != pins[1].digest  # type: ignore[union-attr]


def test_a_rankings_load_with_no_as_of_still_returns_the_whole_archive(fake_rankings,
                                                                      tmp_path):
    """The unpinned path is unchanged: no as-of, no filter, everything the source sent.
    `hub.draft.board.consensus(None)` reads the small `draft` page this way."""
    fake_rankings(["2026-08-01", "2026-09-10"], per_date=2)
    assert nv.load_rankings("draft", cache=tmp_path).height == 4
