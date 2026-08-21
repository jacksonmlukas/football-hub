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
          name: str = "part") -> Path:
    """Immutable dated partitions. Corrections write a new file; nothing is overwritten,
    so a backfill can always be reconstructed and audited."""
    p = DATA / LAYOUT.format(table=table, league=league, season=season, week=week, name=name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    return p


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CATALOG), read_only=read_only)
    for table in ("preds", "lines", "games", "ratings"):
        d = DATA / table
        if d.exists() and any(d.rglob("*.parquet")):
            con.execute(f"""
                CREATE OR REPLACE VIEW {table} AS
                SELECT * FROM read_parquet('{d}/**/*.parquet', hive_partitioning := true)
            """)
    return con


def sql(query: str) -> pl.DataFrame:
    with connect(read_only=False) as con:
        return con.execute(query).pl()


# The query that made DuckDB worth it. Matching a prediction to the line that was live at
# the moment it was made is an as-of join; approximating it with a normal join on week is
# exactly how lookahead sneaks into a backtest.
AS_OF_LINES = """
SELECT p.game_id, p.model, p.version, p.home_win_prob, p.predicted_at,
       l.close_spread, l.captured_at
FROM preds p
ASOF LEFT JOIN lines l
  ON p.game_id = l.game_id AND p.predicted_at >= l.captured_at
WHERE p.league = ?
"""
