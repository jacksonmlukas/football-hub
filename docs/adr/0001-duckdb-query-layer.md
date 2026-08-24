# DuckDB as the query layer, parquet stays the format

**Status:** accepted.

**Decision.** Keep writing parquet. Add a DuckDB catalog (`data/processed/hub.duckdb`) holding
views over a Hive-partitioned tree. Nothing migrates.

**Why not flat files alone.** They work fine until roughly Week 6, then "join every prediction
to the closing line that was live when it was made, grouped by model version" becomes a
100-line polars script that nobody trusts.

**Why not SQLite.** Wrong shape. SQLite is row-oriented and single-threaded per query, built for
transactional point lookups. This workload is scans and aggregations over 20 weeks of snapshots,
which is OLAP. DuckDB is columnar and vectorized, and reads parquet directly with no import step.

**The deciding feature is ASOF JOIN.** Matching a prediction to the line that was live at the
moment it was made is an as-of problem. Approximating it with a normal join on week is exactly
how lookahead sneaks into a backtest, and it produces results that look *better*, which is why
it goes unnoticed. DuckDB has ASOF JOIN natively; SQLite does not.

Secondary wins: `QUALIFY` for per-model top-N, zero-copy Arrow so polars frames query without a
copy, and reading polars DataFrames directly.

**Why not Postgres.** Single user, no concurrent writers, no network. Nothing to buy.

**Layout.**

```
data/processed/preds/league=nfl/season=2026/week=07/part.parquet
```

Hive partitioning means DuckDB infers `league`, `season`, `week` as columns from the path and
prunes entire directories on a `WHERE`, rather than opening every file.

**Partitions are immutable.** A correction writes a new file; nothing is overwritten. Backfills
stay reconstructible and auditable, which matters because the public track record's whole claim
is that predictions were not edited after the fact.

**Cost.** One dependency. The catalog is disposable and rebuilds from parquet in seconds, so it
is gitignored.
