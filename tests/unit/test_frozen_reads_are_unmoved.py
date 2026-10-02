"""The three frozen modules read nflverse through a different door and print the same numbers.

#326 freezes `spread`, `injury` and `margin`: no number may move. #398 routes their reads through
`hub.fetch.nflverse.load`, which validates, caches and pins what it returns -- a change to *how*
the bytes arrive that is allowed to change no number a reader sees. "I checked it by eye" is not
a control for that, so this file is one, and it was written **before** the routing: the
`EXPECTED` digests below were measured on the code that read `nflreadpy` directly, from the
fixtures in this file, and the same fixtures must reproduce them on the code that does not.

What is digested is everything a reader of the CLI sees: every byte the command prints, and the
content of the frame it writes with `--out`. Not the cache entry or the pin the loader now
leaves behind -- those are the new thing, not the old one's output.

**The fixtures are served, not patched.** `_serve` is the only line that knows how the reads
are faked, so the door can change without the fixture or the expectation changing with it.
"""
from __future__ import annotations

import hashlib
import io
from contextlib import redirect_stdout

import numpy as np
import polars as pl
import pytest

from hub.fetch import nflverse as nv
from hub.fetch.replay import Replay
from hub.models import injury, margin, spread

# Measured 2026-10-01 on the code that imported `nflreadpy` directly, before #398 routed it.
EXPECTED = {
    "spread": "b68617dd0a7e3896",
    "injury": "88007c4ee7b94143",
    "margin": "9630773a09e5aa01",
}


def _stats_frame(rows, extra: dict[str, list] | None = None) -> pl.DataFrame:
    cols = ["season", "week", "player_id", "position", "fantasy_points_ppr"]
    df = pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)},
                      schema={"season": pl.Int32, "week": pl.Int32, "player_id": pl.Utf8,
                              "position": pl.Utf8, "fantasy_points_ppr": pl.Float64})
    return df.with_columns([pl.Series(k, v) for k, v in (extra or {}).items()])


def _spread_tables() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(1)
    rows, tgt, ay, tds = [], [], [], []
    for season in (2023, 2024, 2025):
        for pid in range(80):
            base = 12.0 + rng.normal(0, 2)
            for w in range(1, 13):
                rows.append((season, w, f"p{pid}", ("WR", "RB", "TE", "QB")[pid % 4],
                             float(max(0.0, base + rng.normal(0, 6)))))
                tgt.append(float(rng.uniform(0.05, 0.3)))
                ay.append(float(rng.uniform(0.0, 0.4)))
                tds.append(float(rng.integers(0, 2)))
    stats = _stats_frame(rows, {
        "season_type": ["REG"] * len(rows), "target_share": tgt, "air_yards_share": ay,
        "receiving_tds": tds, "rushing_tds": [0.0] * len(rows), "passing_tds": [0.0] * len(rows)})
    snap_rows = [(s, w, f"P{pid}", pid) for s in (2023, 2024, 2025) for pid in range(80)
                 for w in range(1, 13)]
    # Whole percents, the form nflverse has shipped them in: the Normalisation `SNAP_COUNTS`
    # declares is what turns these back into shares, and the digest moves if it stops.
    pct = [float(40 + (pid % 7) * 5 + w) for (_s, w, _p, pid) in snap_rows]
    snaps = pl.DataFrame({
        "game_id": [f"{s}_{w}_x" for (s, w, _p, _i) in snap_rows],
        "season": pl.Series([r[0] for r in snap_rows], dtype=pl.Int32),
        "week": pl.Series([r[1] for r in snap_rows], dtype=pl.Int32),
        "game_type": ["REG"] * len(snap_rows),
        "player": [f"name {r[2]}" for r in snap_rows],
        "pfr_player_id": [r[2] for r in snap_rows],
        "position": [("WR", "RB", "TE", "QB")[r[3] % 4] for r in snap_rows],
        "team": ["AAA"] * len(snap_rows), "opponent": ["BBB"] * len(snap_rows),
        "offense_snaps": [float(50)] * len(snap_rows),
        "offense_pct": pct, "defense_pct": [0.0] * len(snap_rows),
        "st_pct": [10.0] * len(snap_rows)})
    xw = pl.DataFrame({"pfr_id": [f"P{i}" for i in range(80)],
                       "gsis_id": [f"p{i}" for i in range(80)]})
    return {"player_stats": stats, "snap_counts": snaps, "ff_playerids": xw}


def _injury_tables() -> dict[str, pl.DataFrame]:
    rows_st, rows_inj = [], []
    designations = (("Questionable", "Limited", 6.0), ("Doubtful", "Did Not Participate", 2.5),
                    ("Out", "Did Not Participate", None), ("Questionable", "Full", 9.0))
    for season in (2023, 2024):
        for i, pid in enumerate("abcdefghijklmnop"):
            for w in range(1, 9):
                rows_st.append((season, w, pid, "WR", 10.0 + (w % 3) + i))
            for w in range(9, 18):
                status, practice, pts = designations[(i + w) % len(designations)]
                rows_inj.append((season, w, pid, "WR", status, practice))
                if pts is not None:
                    rows_st.append((season, w, pid, "WR", pts + (w % 2)))
    inj = pl.DataFrame(
        {"season": [r[0] for r in rows_inj], "week": [r[1] for r in rows_inj],
         "gsis_id": [r[2] for r in rows_inj], "position": [r[3] for r in rows_inj],
         "report_status": [r[4] for r in rows_inj], "practice_status": [r[5] for r in rows_inj]},
        schema={"season": pl.Int32, "week": pl.Int32, "gsis_id": pl.Utf8, "position": pl.Utf8,
                "report_status": pl.Utf8, "practice_status": pl.Utf8},
    ).with_columns(team=pl.lit("AAA"), game_type=pl.lit("REG"),
                   full_name=pl.col("gsis_id") + " name")
    return {"injuries": inj, "player_stats": _stats_frame(rows_st)}


def _margin_tables() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(7)
    rows = []
    for season in range(2010, 2021):
        for g in range(260):
            line = float(rng.choice([-10.5, -7.0, -3.0, -1.5, 1.5, 3.0, 7.0, 10.5]))
            rows.append((f"{season}_{g // 16 + 1:02d}_{g}", season, g // 16 + 1, "HHH", "AAA",
                         line, round(line + rng.normal(0, 13.5))))
    sched = pl.DataFrame(
        {"game_id": [r[0] for r in rows], "season": [r[1] for r in rows],
         "week": [r[2] for r in rows], "home_team": [r[3] for r in rows],
         "away_team": [r[4] for r in rows], "spread_line": [r[5] for r in rows],
         "result": [r[6] for r in rows]},
        schema={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                "home_team": pl.Utf8, "away_team": pl.Utf8, "spread_line": pl.Float64,
                "result": pl.Int32})
    return {"schedules": sched}


def _serve(tables: dict[str, pl.DataFrame], cache) -> None:
    """Make these frames what nflverse answers. The one line that knows how.

    Before #398 this patched `nflreadpy`'s functions, because that is where the modules read;
    now it selects a Replay as the nflverse adapter. The fixtures and the expectations did not
    change with it, which is the point of keeping this the only place that knows.
    """
    nv.select(Replay(tables), cache=cache)


def _observed(main, argv: list[str], out) -> str:
    """The digest of everything a reader of this command sees."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main([*argv, "--out", str(out)])
    assert code == 0
    # Floats to nine places: a `group_by` sum in `injury` adds in whatever order polars' threads
    # reach the rows, so its last digit differs run to run on identical input. That is the
    # module's own noise and not something routing may be blamed for or excused by.
    written = pl.read_parquet(out).with_columns(pl.col(pl.Float64).round(9))
    # The command names the file it wrote, and a temporary path is a different string each run.
    said = buf.getvalue().replace(str(out), "<out>")
    # Line order is not compared: `injury` prints a `group_by` in the order polars hands it
    # back, which varies run to run, and a control that flickers on its own is not one. Every
    # line, and so every number, still is.
    ordered = "\n".join(sorted(said.splitlines()))
    return hashlib.sha256(
        ordered.encode() + b"\n" + nv.content_digest(written, ()).encode()).hexdigest()[:16]


CASES = {
    "spread": (spread.main, ["--fit", "--seasons", "2023,2024,2025"], _spread_tables),
    "injury": (injury.main, ["--fit", "--seasons", "2023,2024"], _injury_tables),
    "margin": (margin.main, ["--fit", "--shape"], _margin_tables),
}


@pytest.mark.parametrize("module", sorted(CASES))
def test_a_frozen_module_prints_the_numbers_it_printed_before_it_was_routed(module, tmp_path):
    main, argv, tables = CASES[module]
    _serve(tables(), tmp_path / "raw")
    got = _observed(main, argv, tmp_path / "out.parquet")
    assert got == EXPECTED[module], (
        f"{module} printed a different thing from the same fixture ({got}). Under #326 no "
        f"number may move; a routing change that moves one is a modelling change.")


def test_the_unattributed_rows_the_loader_drops_were_already_invisible_to_injury():
    """`load` drops the 22-a-season `player_stats` rows with no player_id; the direct read kept
    them. Frozen means that must change nothing -- and it does not, because they carry no
    position and `observations` keeps drafted positions only. Asserted rather than argued, on
    the pure function, so it holds whichever door the rows came through."""
    st = _injury_tables()["player_stats"]
    orphans = pl.DataFrame(
        {"season": [2023, 2023], "week": [3, 4], "player_id": [None, None],
         "position": [None, None], "fantasy_points_ppr": [0.0, 0.0]},
        schema=st.schema)
    inj = _injury_tables()["injuries"]
    keys = ["season", "week", "gsis_id"]
    assert injury.observations(inj, pl.concat([st, orphans])).sort(keys).equals(
        injury.observations(inj, st).sort(keys))


def test_the_unattributed_rows_the_loader_drops_were_already_invisible_to_spread():
    stats = _spread_tables()["player_stats"]
    orphans = stats.head(2).with_columns(player_id=pl.lit(None, pl.Utf8),
                                         position=pl.lit(None, pl.Utf8),
                                         fantasy_points_ppr=pl.lit(0.0))
    assert spread.player_seasons(pl.concat([stats, orphans])).equals(
        spread.player_seasons(stats))
