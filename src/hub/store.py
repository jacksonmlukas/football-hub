"""Storage: parquet stays the format, DuckDB becomes the query layer.

This is additive, not a migration. DuckDB reads parquet natively, so nothing has to be
loaded or converted -- the catalog is a set of views over a Hive-partitioned tree that
polars keeps writing exactly as it does today.

Why DuckDB rather than SQLite: this workload is scans and aggregations over 20 weeks of
snapshots, which is OLAP. SQLite is row-oriented, single-threaded per query, and built for
point lookups. The decisive feature is ASOF JOIN -- matching each prediction to the closing
line that was live when it was made is an as-of problem, and doing it by hand in polars is
where silent lookahead bugs come from.

Why not Postgres: single user, no concurrent writers, no network. Nothing to buy.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
CATALOG = DATA / "hub.duckdb"

# Hive partitioning means DuckDB infers league/season/week as columns from the paths, and
# prunes whole directories on a WHERE clause instead of opening every file.
LAYOUT = "{table}/league={league}/season={season}/week={week:02d}/{name}.parquet"


def write(df: pl.DataFrame, table: str, league: str, season: int, week: int,
          name: str = "part", base: Path | None = None) -> Path:
    """Immutable dated partitions. Corrections write a new file; nothing is overwritten,
    so a backfill can always be reconstructed and audited.

    `base` exists so tests -- and any caller wanting a scratch tree -- can redirect the
    root without reassigning a module global. Every Phase 1 fetch module writes through
    here, so it needed an injection point that is not monkeypatching.
    """
    root = base or DATA
    p = root / LAYOUT.format(table=table, league=league, season=season, week=week, name=name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    return p


def connect(read_only: bool = False, base: Path | None = None) -> duckdb.DuckDBPyConnection:
    root = base or DATA
    root.mkdir(parents=True, exist_ok=True)
    catalog = (root / "hub.duckdb") if base else CATALOG
    con = duckdb.connect(str(catalog), read_only=read_only)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        # Discovered, not enumerated. This used to be a hardcoded list of four tables
        # while write() accepted any name, so `hub.fetch.nflverse --refresh` wrote 54,402
        # rows of pbp and ff_opportunity that the catalog then could not see. Write and
        # read have to agree on what a table is.
        if not d.name.isidentifier() or not any(d.rglob("*.parquet")):
            continue
        con.execute(f"""
            CREATE OR REPLACE VIEW {d.name} AS
            SELECT * FROM read_parquet('{d}/**/*.parquet', hive_partitioning := true)
        """)
    return con


def tables(base: Path | None = None) -> set[str]:
    """Which datasets the catalog can actually see.

    `connect` builds a view per directory that exists and holds parquet, so a store with no
    predictions in it has no `preds` view at all -- and querying one raises DuckDB's
    `CatalogException`, not an empty frame. That is the state of a fresh clone, and it is
    how `hub.models.conformal` and `hub.models.eval` came to answer a clean checkout with a
    stack trace. Callers that tolerate an absent dataset should ask first.

    Discovered the same way `connect` discovers them, so the two cannot disagree.
    """
    root = base or DATA
    if not root.exists():
        return set()
    return {d.name for d in root.iterdir()
            if d.is_dir() and d.name.isidentifier() and any(d.rglob("*.parquet"))}


def sql(query: str, params: Sequence[object] | None = None,
        base: Path | None = None) -> pl.DataFrame:
    """Run a query against the catalog.

    `params` is not optional in practice: AS_OF_LINES below carries a `?`, so without it
    the module's own canonical query could not be run through the module's own helper.
    """
    with connect(read_only=False, base=base) as con:
        return con.execute(query, list(params) if params else None).pl()


# The query that made DuckDB worth it. Matching a prediction to the line that was live at
# the moment it was made is an as-of join; approximating it with a normal join on week is
# exactly how lookahead sneaks into a backtest. tests/unit/test_store.py shows both, side
# by side, on a case built to break the naive version.
AS_OF_LINES = """
SELECT p.game_id, p.model, p.version, p.home_win_prob, p.predicted_at,
       l.close_spread, l.captured_at
FROM preds p
ASOF LEFT JOIN lines l
  ON p.game_id = l.game_id AND p.predicted_at >= l.captured_at
WHERE p.league = ?
"""


def verify(season: int = 2025, base: Path | None = None) -> int:
    """Exercise the whole path against real games and real closing lines.

    The unit tests prove the as-of semantics on a case built to break a naive join. This
    proves the same code survives real identifiers and real volume: every 2025 game with a
    published spread, written through `write`, read back through the catalog views, joined
    through `AS_OF_LINES`.

    Honest limitation: nflverse publishes one line per game -- the close -- so this cannot
    exercise intra-week line movement, and the predictions here are market-implied
    stand-ins rather than model output. Both arrive with `hub.fetch.odds` (1.4) and
    `MarketBaseline` (3.1). What it does establish is that the storage layer itself is not
    the thing standing between them.
    """
    import tempfile

    import nflreadpy as nfl

    root = base or Path(tempfile.mkdtemp(prefix="hub-store-verify-"))
    sched = (nfl.load_schedules()
             .filter((pl.col("season") == season) & pl.col("spread_line").is_not_null()))
    if sched.is_empty():
        print(f"  no {season} games with a published spread; nothing to verify")
        return 1

    kickoff = (pl.col("gameday").cast(pl.Utf8) + " " + pl.col("gametime").fill_null("13:00")
               ).str.to_datetime("%Y-%m-%d %H:%M", strict=False)
    lines = sched.select(
        pl.col("game_id"),
        pl.col("spread_line").cast(pl.Float64).alias("close_spread"),
        kickoff.alias("captured_at"),
    ).drop_nulls("captured_at")

    # Market-implied stand-in: a spread converted to a win probability by a logistic on
    # points, the same shape MarketBaseline will formalise. Timestamped an hour after the
    # close so there is a line for the as-of join to find.
    #
    # Sign convention matters and is easy to get backwards: in nflverse a POSITIVE
    # spread_line means the home team is favoured (2025_01_DAL_PHI is home PHI at +8.5,
    # and PHI won). Inverting it puts every home favourite below 50% -- a backtest would
    # still run, still look calibrated in aggregate, and be exactly wrong.
    preds = lines.select(
        pl.col("game_id"),
        pl.lit("market_implied").alias("model"),
        pl.lit("verify").alias("version"),
        (1.0 / (1.0 + (-pl.col("close_spread") / 7.0).exp())).alias("home_win_prob"),
        (pl.col("captured_at") + pl.duration(hours=1)).alias("predicted_at"),
    )

    store_dir = root
    write(lines, "lines", "nfl", season, 1, base=store_dir)
    write(preds, "preds", "nfl", season, 1, base=store_dir)
    got = sql(AS_OF_LINES, params=["nfl"], base=store_dir)

    matched = got.height - got["close_spread"].null_count()
    ahead = got.filter(pl.col("captured_at") > pl.col("predicted_at")).height
    print(f"  root: {store_dir}")
    print(f"  {lines.height} real {season} games with a closing spread, written and read back")
    print(f"  as-of join: {got.height} rows, {matched} matched a line, {ahead} looked ahead")
    print("  " + ("OK" if matched == got.height and ahead == 0 else "FAILED"))
    return 0 if matched == got.height and ahead == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="hub.store",
                                 description="Storage layer for predictions and lines.")
    ap.add_argument("--verify", action="store_true",
                    help="round-trip real games through the catalog and the as-of join")
    ap.add_argument("--season", type=int, default=2025)
    a = ap.parse_args(argv)
    if not a.verify:
        ap.print_help()
        return 0
    return verify(season=a.season)


if __name__ == "__main__":
    import sys
    sys.exit(main())
