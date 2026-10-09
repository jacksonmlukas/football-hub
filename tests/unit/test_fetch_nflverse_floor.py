"""The plausibility floor on a completed season's cached read (#467).

The control (rule 18): the 2-row 2024 `player_stats` entry that sat in the cache for six weeks
and priced 2025 absence at zero. The contract checks shape, not size, so nothing said so.

The floor judges only the network's rows -- a Replay is a recorded set the caller built, and
a fixture of six rows is its point -- so these serve through a `Network` whose wire is a
function.
"""
from collections.abc import Sequence

import polars as pl

from hub.fetch import nflverse as nv
from hub.fetch.replay import serve

COLS = ["player_id", "player_display_name", "position", "season", "week", "season_type",
        "fantasy_points_ppr"]


def _stats(rows: int, season: int = 2024) -> pl.DataFrame:
    return pl.DataFrame({
        "player_id": [f"p{i}" for i in range(rows)],
        "player_display_name": [f"P {i}" for i in range(rows)],
        "position": ["WR"] * rows,
        "season": pl.Series([season] * rows, dtype=pl.Int32),
        "week": pl.Series([1 + i % 18 for i in range(rows)], dtype=pl.Int32),
        "season_type": ["REG"] * rows,
        "fantasy_points_ppr": [5.0] * rows,
    })


class Wire(nv.Network):
    def __init__(self, frame: pl.DataFrame | None = None, fail: bool = False) -> None:
        self.frame, self.fail, self.calls = frame, fail, 0

    def read(self, source: nv.Source, keys: Sequence[int | str]) -> pl.DataFrame:
        self.calls += 1
        if self.fail or self.frame is None:
            raise OSError("wire down")
        return self.frame


def plant(tmp_path, rows):
    """A cache entry as a truncated write leaves it: the right path, too few rows."""
    path = nv._cache_path("player_stats", [2024], COLS, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _stats(rows).write_parquet(path)
    return path


def test_a_truncated_completed_season_entry_is_refused_and_refetched(tmp_path, capsys):
    plant(tmp_path, 2)
    wire = Wire(_stats(6000))
    with nv.serving(wire, cache=tmp_path):
        got = nv.load("player_stats", [2024], cols=COLS)
    assert got.height == 6000 and wire.calls == 1
    assert "looks truncated" in capsys.readouterr().err
    # ... and the repair stuck: the next read is a plain hit on the full entry.
    again = Wire(fail=True)
    with nv.serving(again, cache=tmp_path):
        assert nv.load("player_stats", [2024], cols=COLS).height == 6000
    assert again.calls == 0


def test_a_full_completed_season_entry_is_served_from_cache(tmp_path, capsys):
    plant(tmp_path, 6000)
    wire = Wire(fail=True)
    with nv.serving(wire, cache=tmp_path):
        assert nv.load("player_stats", [2024], cols=COLS).height == 6000
    assert wire.calls == 0 and "truncated" not in capsys.readouterr().err


def test_a_truncated_entry_is_served_with_a_warning_when_the_refetch_fails(tmp_path, capsys):
    plant(tmp_path, 2)
    with nv.serving(Wire(fail=True), cache=tmp_path):
        got = nv.load("player_stats", [2024], cols=COLS)
    assert got.height == 2
    assert "serving the truncated cache entry" in capsys.readouterr().err


def test_a_short_wire_is_returned_but_not_cached(tmp_path, capsys):
    with nv.serving(Wire(_stats(2)), cache=tmp_path):
        got = nv.load("player_stats", [2024], cols=COLS)
    assert got.height == 2 and "not cached" in capsys.readouterr().err
    assert not nv._cache_path("player_stats", [2024], COLS, tmp_path).exists()


def test_the_current_season_has_no_floor(tmp_path):
    from hub.config import SEASON_AHEAD
    with nv.serving(Wire(_stats(3, SEASON_AHEAD)), cache=tmp_path):
        assert nv.load("player_stats", [SEASON_AHEAD], cols=COLS).height == 3


def test_a_multi_season_entry_that_lost_a_season_is_short(tmp_path):
    with nv.serving(Wire(), cache=tmp_path):
        assert nv._short_by("player_stats", [2023, 2024], 6000) is not None
        assert nv._short_by("player_stats", [2023, 2024], 10_000) is None


def test_a_replay_is_exempt(tmp_path):
    serve(player_stats=lambda seasons: _stats(2))
    assert nv._short_by("player_stats", [2024], 2) is None
    assert nv.load("player_stats", [2024], cols=COLS, cache=tmp_path).height == 2
