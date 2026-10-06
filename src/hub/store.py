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

import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

from hub import atomic
from hub.cli import unavailable
from hub.config import SEASON_COMPLETED
from hub.paths import STATE_DIR

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
CATALOG = DATA / "hub.duckdb"

# Hive partitioning means DuckDB infers league/season/week as columns from the paths, and
# prunes whole directories on a WHERE clause instead of opening every file.
LAYOUT = "{table}/league={league}/season={season}/week={week:02d}/{name}.parquet"


def partition(table: str, league: str, season: int, week: int, name: str = "part",
              base: Path | None = None) -> Path:
    """Where one partition lives. `write` computes this and now so does one caller.

    Extracted rather than restated: the path is a pure function of the five keys and
    `LAYOUT`, and a second copy of that formatting is a second thing to keep in step with
    the zero-padding `week_key` exists for.
    """
    root = base or DATA
    return root / LAYOUT.format(table=table, league=league, season=season, week=week,
                                name=name)


def write(df: pl.DataFrame, table: str, league: str, season: int, week: int,
          name: str = "part", base: Path | None = None, *, replace: bool = False) -> Path:
    """Dated partitions. A caller that would destroy an existing one has to say so.

    This docstring used to promise that "nothing is overwritten, so a backfill can always be
    reconstructed and audited". It was not true and nothing enforced it: the path is a pure
    function of (table, league, season, week, name) and `write_parquet` truncates, so any
    caller reusing a name silently destroyed the previous partition. Three of the four call
    sites avoided that by passing a timestamp or a digest, each with a comment about the
    hazard; `fetch.nflverse` used the default `name="part"` and overwrote every week on every
    refresh, losing the record of any upstream stat correction.

    So the promise is now enforced rather than stated. `replace=False` refuses to overwrite a
    differing partition; a caller that genuinely means to replace passes `replace=True` and
    is visible in review. Re-writing identical bytes is a no-op rather than an error, because
    `make slate` re-runs must stay idempotent.

    `base` exists so tests -- and any caller wanting a scratch tree -- can redirect the
    root without reassigning a module global. Every Phase 1 fetch module writes through
    here, so it needed an injection point that is not monkeypatching.
    """
    p = partition(table, league, season, week, name, base)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not replace:
        # GUARD partition-not-silently-overwritten: a differing partition is never destroyed unasked
        try:
            unchanged = pl.read_parquet(p).equals(df)
        except Exception as e:
            # Named as what it is. This used to collapse into "different data" and the
            # instruction below, which is a data-loss instruction for a corruption (#258):
            # what the file held is not recoverable from here, and the operator deciding
            # that should be told so rather than told to replace a partition that differs.
            raise FileExistsError(
                f"{p} is unreadable ({type(e).__name__}: {e}). This is a corrupt "
                f"partition, not one that disagrees with the frame, and nothing in it can "
                f"be recovered from here. Move the file aside to keep its bytes for "
                f"inspection, or pass `replace=True` to discard them and write this frame "
                f"in its place.") from e
        if not unchanged:
            raise FileExistsError(
                f"{p} already holds different data. Pass a distinct `name=` to keep both "
                f"(what board, ratings and odds do), or `replace=True` to mean it.")
        # /GUARD
        return p
    atomic.write_parquet(df, p)
    return p


def is_table(d: Path) -> bool:
    """Whether a directory is a dataset the catalog can see: an identifier, holding parquet.

    One predicate, because there were two. `connect` and `tables` each carried a copy, and
    `tables`' own docstring claimed the two "cannot disagree" -- an invariant asserted in prose
    and enforced by nothing. Write and read have to agree on what a table is: a hardcoded list
    of four here, against a `write()` that accepted any name, is how 54,402 rows of pbp and
    ff_opportunity ended up in the store and invisible to the catalog.
    """
    return d.is_dir() and d.name.isidentifier() and any(d.rglob("*.parquet"))


def connect(read_only: bool = False, base: Path | None = None) -> duckdb.DuckDBPyConnection:
    root = base or DATA
    root.mkdir(parents=True, exist_ok=True)
    catalog = (root / "hub.duckdb") if base else CATALOG
    con = duckdb.connect(str(catalog), read_only=read_only)
    # Replacement scans off. DuckDB resolves a table name it does not know against Python
    # objects in the calling frame, so once this module had a function called `lines` a query on
    # a store with no `lines` table found *that* and failed with a message about functions,
    # where it must raise the `CatalogException` the fresh-clone guards read as "no archive".
    # A catalog that answers from whatever happens to be in scope is not one.
    con.execute("SET python_enable_replacements = false")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not is_table(d):
            continue
        # `union_by_name` because the store is append-only and its schemas evolve: a
        # partition written before `cfg_digest` existed has twelve columns where a later one
        # has fourteen, and without this DuckDB refuses the whole glob rather than filling
        # nulls. Found 2026-08-29 rehearsing `make slate`, where it took `publish --all` down
        # -- and only *after* #16 stopped `publish` swallowing it as "no scored predictions".
        # Schema evolution is the design here: "immutable dated partitions; corrections write
        # a new file", so a reader that cannot span two schemas cannot read this store.
        con.execute(f"""
            CREATE OR REPLACE VIEW {d.name} AS
            SELECT * FROM read_parquet('{d}/**/*.parquet',
                                       hive_partitioning := true,
                                       union_by_name := true)
        """)
    # The committed snapshots (#383), unioned under the same name. A TEMP view, so the
    # persistent catalog is never written with rows that live in git, and `lines` resolves to
    # it first. Local rows win a duplicate (game_id, captured_at).
    committed = committed_lines(base)
    if not committed.is_empty():
        con.register("committed_lines", committed)
        if is_table(root / "lines"):
            # The parquet is read inline rather than through the persistent `lines` view: a
            # temp view that is also named `lines` would bind to itself.
            local = (f"read_parquet('{root / 'lines'}/**/*.parquet', "
                     f"hive_partitioning := true, union_by_name := true)")
            con.execute(f"""
                CREATE OR REPLACE TEMP VIEW lines AS
                SELECT * FROM {local}
                UNION ALL BY NAME
                SELECT c.* FROM committed_lines c
                WHERE NOT EXISTS (SELECT 1 FROM {local} l
                                  WHERE l.game_id = c.game_id
                                    AND l.captured_at = c.captured_at)
            """)
        else:
            con.execute("CREATE OR REPLACE TEMP VIEW lines AS SELECT * FROM committed_lines")
    return con


def tables(base: Path | None = None) -> set[str]:
    """Which datasets the catalog can actually see.

    `connect` builds a view per directory that exists and holds parquet, so a store with no
    predictions in it has no `preds` view at all -- and querying one raises DuckDB's
    `CatalogException`, not an empty frame. That is the state of a fresh clone, and it is
    how `hub.models.conformal` and `hub.models.eval` came to answer a clean checkout with a
    stack trace. Callers that tolerate an absent dataset should ask first.

    Discovered by the same `is_table` predicate `connect` uses, so the two cannot disagree --
    which is now enforced by there being one of it rather than asserted here in prose.
    """
    root = base or DATA
    found = {d.name for d in root.iterdir() if is_table(d)} if root.exists() else set()
    # A checkout whose only captures are the committed ones has a `lines` view all the same
    # (`connect` builds it), so the guards that ask "is there an archive" must say yes (#383).
    if _has_committed(base):
        found.add("lines")
    return found


# The dtypes the partition keys are written with, so a read hands them back that way.
#
# Hive partitioning infers league/season/week from the *path*, and those inferred columns
# shadow the ones actually written: `week`, declared `Int32` in `PREDICTION_SCHEMA` and
# validated against it on the way in, came back as the zero-padded string `"01"`, and
# `season` as `Int64`. A schema that promises a type and a reader that believes it is the
# whole reason the schema is there -- `conformal.rolling_coverage` and `eval.compare` both
# treat week numerically and would have raised on real store output, latent only because
# each exits early for an unrelated reason today.
#
# Declared here rather than imported from `hub.models.base`: `LAYOUT` writes these keys and
# `week_key` pads them, so the round trip is this module's, and storage should not depend on
# the model layer to describe its own paths. `tests/unit/test_store.py` holds the two
# declarations against each other, which is where they are allowed to meet.
PARTITION_TYPES: dict[str, pl.DataType] = {
    "league": pl.Utf8(), "season": pl.Int32(), "week": pl.Int32(),
}


def _as_written(df: pl.DataFrame) -> pl.DataFrame:
    """Restore the partition keys to the dtypes they were written with.

    Only the keys, and only the ones present: every other column comes off the parquet with
    its own dtype intact, and casting more would be this module inventing a schema for data
    it does not declare.
    """
    return df.with_columns(pl.col(c).cast(t) for c, t in PARTITION_TYPES.items()
                           if c in df.columns)


def week_key(week: int) -> str:
    """The partition value for a week, zero-padded, as `LAYOUT` writes it.

    Part of the query interface rather than a convention. `publish` spelled this four ways in
    one 333-line file -- `f"preds_wk{week:02d}"` twice for filenames, `f"{week:02d}"` as a
    query parameter, and `int(got["w"][0])` reading it back -- because the format belonged to
    nobody. A partition key that a caller has to remember to pad is one a caller will
    eventually not pad.
    """
    return f"{week:02d}"


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


# The same as-of question asked forwards. `AS_OF_LINES` prices a prediction that already
# exists, which is the audit direction; a fit has to ask it *before* it has written anything,
# so the left side is the games themselves and the moment is supplied rather than read off a
# row. Same join, same lookahead guarantee, different direction -- and writing it as an ASOF
# JOIN rather than a window function keeps it the one idiom this module vouches for.
LINE_AS_OF = """
WITH asked AS (
    SELECT DISTINCT game_id, CAST(? AS TIMESTAMP) AS at
    FROM lines WHERE league = ? AND season = ?
)
SELECT a.game_id, l.close_spread, l.captured_at
FROM asked a
ASOF LEFT JOIN (SELECT game_id, close_spread, captured_at FROM lines
                WHERE league = ? AND season = ?) l
  ON a.game_id = l.game_id AND a.at >= l.captured_at
"""

LINE_SCHEMA = {"game_id": pl.Utf8, "close_spread": pl.Float64, "captured_at": pl.Datetime}


def lines_as_of(at: datetime, season: int, league: str = "nfl",
                base: Path | None = None) -> pl.DataFrame:
    """One line per game: the latest snapshot captured at or before `at`.

    A game whose only snapshots come *after* the moment is absent rather than null. It was
    not priced then, and returning its later line is the lookahead the as-of join exists to
    prevent -- a caller coalescing onto a fallback would take a number from the future and
    never see it happen.

    `season` has no default on purpose. Every other season-scoped helper here takes
    `SEASON_COMPLETED`, which is the right default for a backtest and the wrong one for a
    fit: a caller who forgets it would get last season's lines, find nothing matching this
    season's game ids, and silently fall back to the moving field for every game.

    An empty frame is returned when the store holds no lines at all, which is the state of a
    fresh clone: `connect` builds a view per directory that exists, so `lines` is not an
    empty table there but no table, and querying it raises `CatalogException`. Callers price
    from their fallback instead of dying, which is what the repo means by degrading.
    """
    if "lines" not in tables(base):
        return pl.DataFrame(schema=LINE_SCHEMA)
    got = sql(LINE_AS_OF, params=[at, league, season, league, season], base=base)
    return got.drop_nulls("close_spread")


# --- committed line snapshots (#383) ----------------------------------------------------
#
# `data/processed/` is gitignored, and an Actions runner starts with an empty one and discards
# it, so every capture the scheduled polls made was written where nothing could read it again
# (#383: the archive ended 2026-09-06). Each poll therefore also writes its validated frame to
# a *committed*, append-only file under `state/odds/`, one per (poll, week):
#
#     state/odds/<season>/wk<NN>/snap-<YYYYmmddTHHMMSS>.json
#
# **Reader design: the committed tree is unioned with the local store, deduped by
# (game_id, captured_at) -- option (ii), not a loader that rebuilds the parquet table (i).**
# (i) has to run *before* a reader, so every entry point (`lines`, `lines_as_of`,
# `staleness`, `noise_floor`, the starter-change study) would need to remember to call it, and
# its output is a second copy of the archive on disk that can disagree with the first. (ii)
# lives in `connect`, the one place every reader's SQL goes through, so a reader cannot
# forget it and nothing is materialised: `lines` is a view, as it always was, that now also
# spans the committed files. Local rows win a duplicate, which is the maintainer's own local
# run having written the same capture to both places (same `captured_at`, same game).
#
# The lookahead guarantee is untouched because the union happens *under* the `lines` name
# that `LINE_AS_OF`'s ASOF join already reads: a committed snapshot captured after `at` is
# just a row with `captured_at > at`, and the join never matches it.

SNAPSHOT_COLUMNS: dict[str, pl.DataType] = {
    "game_id": pl.Utf8(), "close_spread": pl.Float64(), "spread_price": pl.Float64(),
    "close_total": pl.Float64(), "total_price": pl.Float64(), "captured_at": pl.Datetime(),
    "polls_unmoved": pl.Int64(), "unmoved_since": pl.Datetime(),
}
_SNAPSHOT_TIMES = ("captured_at", "unmoved_since")
_SNAPSHOT_GLOB = "*/wk*/snap-*.json"


def snapshot_root(base: Path | None = None) -> Path:
    """Where committed snapshots live: `state/odds/`, or `<base>/_state/odds/` for a scratch store.

    A store rooted somewhere else (every test) carries its own state tree inside it rather
    than reading the repo's, so a test's `base` is a whole world and the snapshots a
    scheduled run has committed cannot leak into it.
    """
    return STATE_DIR / "odds" if base is None else Path(base) / "_state" / "odds"


def snapshot_path(season: int, week: int, when: datetime, base: Path | None = None) -> Path:
    return (snapshot_root(base) / str(season) / f"wk{week:02d}"
            / f"snap-{when:%Y%m%dT%H%M%S}.json")


def write_snapshot(df: pl.DataFrame, season: int, week: int, when: datetime,
                   base: Path | None = None) -> Path:
    """Append one poll's week to the committed tree. Never rewrites an earlier file.

    A second write to the same path with the same frame is a no-op (a re-run is idempotent);
    with a different frame it raises, because the file is a *capture* -- what the betting market
    said at `when` -- and a capture that changes is not one. Written through `hub.atomic`, so a
    runner killed mid-write leaves no partial for the next checkout to commit.
    """
    p = snapshot_path(season, week, when, base)
    body = {"season": season, "week": week, "captured_at": when.isoformat(),
            "rows": [{k: (v.isoformat() if k in _SNAPSHOT_TIMES and v is not None else v)
                      for k, v in row.items()} for row in df.to_dicts()]}
    text = json.dumps(body, indent=1, sort_keys=True) + "\n"
    if p.exists():
        if p.read_text(encoding="utf-8") == text:
            return p
        raise FileExistsError(
            f"{p} already holds a different capture. Snapshots are append-only; a poll "
            f"that disagrees with an earlier one is a second poll, with its own timestamp.")
    return atomic.write_text(p, text)


def committed_lines(base: Path | None = None) -> pl.DataFrame:
    """Every committed snapshot, as `lines` rows: the poll's columns plus league/season/week.

    `week` is the zero-padded string the Hive layout gives the parquet side, so a union of the
    two has one type for it; `lines()` casts it to an integer either way.
    """
    frames = []
    for f in sorted(snapshot_root(base).glob(_SNAPSHOT_GLOB)):
        doc = json.loads(f.read_text(encoding="utf-8"))
        schema = {k: (pl.Utf8() if k in _SNAPSHOT_TIMES else t)
                  for k, t in SNAPSHOT_COLUMNS.items()}
        part = pl.DataFrame(doc["rows"], schema=schema)
        frames.append(part.with_columns(pl.col(c).str.to_datetime() for c in _SNAPSHOT_TIMES)
                      .with_columns(league=pl.lit("nfl"),
                                    season=pl.lit(doc["season"], dtype=pl.Int32),
                                    week=pl.lit(week_key(doc["week"]), dtype=pl.Utf8)))
    return pl.concat(frames) if frames else pl.DataFrame()


def _has_committed(base: Path | None) -> bool:
    return any(snapshot_root(base).glob(_SNAPSHOT_GLOB))


# Every poll's columns, as `lines()` hands them back. `spread_price` is here although partitions
# written before #211 do not carry it: `lines()` is where that is answered, once.
POLLS_SCHEMA = {"game_id": pl.Utf8, "close_spread": pl.Float64, "spread_price": pl.Float64,
                "captured_at": pl.Datetime, "week": pl.Int64}


def lines(season: int, *, base: Path | None = None) -> pl.DataFrame:
    """Every poll the store holds for the season's NFL lines, as stored plus what a reader needs.

    The other question about this table: not "the price as of T, one row per game" -- that is
    `lines_as_of`, the lookahead guard -- but "every poll this season", which the staleness
    derivation, the noise floor and the quarterback study each need. Three readers used to ask
    it three ways outside the store, each repeating the fresh-clone guard and writing the empty
    frame by hand; this is the one owner of what the answer looks like.

    Read with `SELECT *` rather than by naming columns, because an archive written entirely
    before #211 has no `spread_price` column in any partition, and `union_by_name` unions what
    exists rather than what a contract now declares. A column in `POLLS_SCHEMA` the archive has
    never had is added as nulls, which `_quote_moved` reads as no evidence. `week` comes back an
    integer however the Hive partition spelled it (`01` is a string to the catalog); any other
    column the store holds rides along untouched, and a caller takes the projection it reads.

    An empty frame of `POLLS_SCHEMA` on a fresh clone, where `lines` is not an empty table but
    no table, and querying it raises `CatalogException`.
    """
    if "lines" not in tables(base):
        return pl.DataFrame(schema=POLLS_SCHEMA)
    got = sql("SELECT * FROM lines WHERE league = 'nfl' AND season = ?", params=[season],
              base=base)
    absent = [pl.lit(None, dtype=t).alias(c) for c, t in POLLS_SCHEMA.items()
              if c not in got.columns]
    return got.with_columns(absent).with_columns(pl.col("week").cast(pl.Utf8).cast(pl.Int64))


# --- the other archives ----------------------------------------------------------------
#
# Each one a name, so the table, its league filter and its fresh-clone answer are stated here
# and not in whichever module wanted the rows. The shared answer for a fresh clone is `None`:
# "no such archive", which is a different statement from an archive with nothing in the season
# (an empty frame), and the callers say different things for the two -- a CLI that names the
# command that writes the archive, a reader that degrades to an empty schema. Returning `None`
# keeps that decision with the caller who has a message to write, and the query with the store.
# The rows come back as stored, partition columns and all; a caller casts and selects what it
# reads, which is frame arithmetic and needs no catalog.

def _archive(table: str, season: int, week: int | None, base: Path | None) -> pl.DataFrame | None:
    if table not in tables(base):
        return None
    q = f"SELECT * FROM {table} WHERE league = 'nfl' AND season = ?"
    params: list[object] = [season]
    if week is not None:
        q += " AND week = ?"
        params.append(week_key(week))
    return sql(q, params=params, base=base)


def prop_lines(season: int, *, week: int | None = None,
               base: Path | None = None) -> pl.DataFrame | None:
    """Every prop poll stored for the season (one week of it if asked), or None if the store
    has no `prop_lines` archive -- which is `hub.fetch.odds --record-props` never having run."""
    return _archive("prop_lines", season, week, base)


def prop_log(season: int, *, base: Path | None = None) -> pl.DataFrame | None:
    """The season's prop decisions and their closes, or None if `hub.models.props --log` has
    never written one."""
    return _archive("prop_log", season, None, base)


def pool_state(season: int, *, week: int | None = None,
               base: Path | None = None) -> pl.DataFrame | None:
    """Every archived read of the survivor pool for the season, one row per entry per read,
    or None on a fresh clone. The grouping into reads and the contract are `hub.fetch.pool`'s."""
    return _archive("pool_state", season, week, base)


def journal(season: int, *, week: int | None = None,
            base: Path | None = None) -> pl.DataFrame | None:
    """The season's journalled decisions as stored, or None if none was ever recorded. Rows
    written before a column existed lack it; adding it as null is the journal's to say."""
    return _archive("journal", season, week, base)


def journal_outcomes(season: int, *, base: Path | None = None) -> pl.DataFrame | None:
    """The season's settled outcomes as stored, or None if none was ever settled."""
    return _archive("journal_outcome", season, None, base)


# One prediction per game, and the rule for choosing it.
#
# The store keeps every version on purpose -- two configurations both survive, which is
# `docs/foundation-plan.md` 3.5 -- and nothing that read it knew that. On the live store 2026
# week 1 held five fitted versions of the same sixteen games, and `publish`, `eval` and
# `conformal` each treated them as eighty independent rows: the site listed every game five
# times, `_paired` built a 25-fold cross product, and the calibration window counted each
# residual five times while reporting the inflated n as its own size.
#
# **The rule is the latest `predicted_at`, ties broken by version descending.** A later write
# for a game is a correction -- refreshed odds, a different price source -- so the most recent
# is what the model now says. The tie-break is not a preference between two versions, it is
# determinism: `ratings.fit` writes a partition per price source inside one run, so two rows
# for a game can share a timestamp, and a reader returning whichever DuckDB scanned first
# would publish a different page on every refresh.
#
# It answers "what does the model say", which is what all three callers ask. It is NOT the
# answer to "what was pre-registered": that is decided by which commit predates kickoff
# (`docs/track-record.md` rule 1), lives in git rather than here, and is why `track_record`
# counts it with `hub.prereg` (#422) rather than from rows.
LATEST_PREDICTIONS = """
SELECT * EXCLUDE (rn) FROM (
    SELECT *, row_number() OVER (
        PARTITION BY game_id, season, week
        ORDER BY predicted_at DESC, version DESC) AS rn
    FROM preds{where}
) WHERE rn = 1
ORDER BY season, week, game_id
"""


def predictions(league: str | None = None, season: int | None = None,
                week: int | None = None, model: str | None = None,
                base: Path | None = None, *, family: bool = False) -> pl.DataFrame:
    """Predictions from the store, one row per game, week and season.

    Every filter is optional and `None` means "do not narrow on this" -- a league default of
    "nfl" would silently drop college predictions the day the first one is written.

    **`model` is exact, and `family=True` widens it to the name and every variant spelled
    `name-<suffix>`** -- `market_baseline` and `market_baseline-qb`, the row the quarterback
    layer moved between #284 and #299, still one row per game. #299 pulled the adjustment
    and nothing writes the suffix now; the rows already written under it stay in the track
    record, which is why this reader keeps understanding it. The distinction is the
    reader's to declare: a reader scoring *what the module published* wants the family,
    because a game is published once and the adjusted rows were exactly the
    backup-quarterback games, so a reader on the exact name scores a subset biased by what
    left. `hub.models.conformal`
    and `hub.models.eval` read the family; a reader after one model's rows alone -- auditing
    what the adjustment did against what the baseline would have said -- asks for the exact
    name. The suffix convention is `hub.models.market`'s; a prefix match (`market_baseliner`)
    is not a variant and is not returned.

    `week` takes an int. The zero-padding `week_key` exists for is the store's business, and
    a caller passing 1 against a `week=01` partition matches nothing and reports it as an
    empty week -- which is the footgun `week_key`'s own docstring describes and which this
    keeps inside the module.

    **Typed as `PREDICTION_SCHEMA` declares**, not as Hive infers. The partition keys are read
    off the path, so `week` came back as the padded string `"01"` and `season` as `Int64` --
    and the two consumers that treat week numerically would have raised the first week either
    had enough data to do its job. `_as_written` puts them back.

    An empty frame when the store holds no predictions at all: a fresh clone has no `preds`
    view, and querying one raises `CatalogException` rather than returning nothing.

    Ordered, because the weekly artifact this feeds is committed to git and its value is
    entirely in the commit history: an unordered scan makes every republish a diff of the
    whole file, and a real change is then invisible inside the churn.

    To read *every* version -- comparing two configurations, auditing what a run wrote --
    query `preds` through `sql()` directly and say so. `tests/contracts` enforces that no
    module does it by accident.
    """
    if "preds" not in tables(base):
        return pl.DataFrame()
    clauses: list[str] = []
    params: list[object] = []
    for col, val in (("league", league), ("season", season)):
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    if model is not None:
        if family:
            clauses.append("(model = ? OR model LIKE ?)")
            params += [model, f"{model}-%"]
        else:
            clauses.append("model = ?")
            params.append(model)
    if week is not None:
        clauses.append("week = ?")
        params.append(week_key(week))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return _as_written(sql(LATEST_PREDICTIONS.format(where=where), params, base=base))


def latest_week(season: int, league: str | None = None,
                base: Path | None = None) -> int | None:
    """The highest week already predicted for a season, or None if none is.

    Here rather than in `hub.publish` because `preds` is this module's table and the padding
    is its convention: `max(week)` is a string comparison over a `week=01` partition key, and
    it gives the right answer only because `week_key` pads. A caller writing that query
    elsewhere has to know that, and a caller who writes weeks unpadded breaks it silently.
    """
    if "preds" not in tables(base):
        return None
    clauses: list[str] = ["season = ?"]
    params: list[object] = [season]
    if league is not None:
        clauses.append("league = ?")
        params.append(league)
    got = sql(f"SELECT max(week) AS w FROM preds WHERE {' AND '.join(clauses)}",
              params, base=base)
    return int(got["w"][0]) if got.height and got["w"][0] is not None else None


def schedules_for(season: int) -> pl.DataFrame:
    """The reach for nflverse, and the only part of `--verify` that is one.

    Separate so the CLI's guard can span it and nothing else. Everything `verify` does after
    this is the storage layer -- the write, the catalog views, the as-of join -- which is
    precisely what `--verify` exists to surface, and which it reports by *returning* rather
    than raising. So the exceptions that reach the CLI past this line are storage-layer
    defects, and calling them "the nflverse schedules unavailable" sends the operator to
    re-fetch data that is fine (issue #119).
    """
    from hub.fetch import nflverse
    return (nflverse.load("schedules", [season], refresh=True)
            .filter((pl.col("season") == season) & pl.col("spread_line").is_not_null()))


def verify(season: int = SEASON_COMPLETED, base: Path | None = None,
           sched: pl.DataFrame | None = None) -> int:
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

    root = base or Path(tempfile.mkdtemp(prefix="hub-store-verify-"))
    sched = schedules_for(season) if sched is None else sched
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
    ap.add_argument("--season", type=int, default=SEASON_COMPLETED)
    a = ap.parse_args(argv)
    if not a.verify:
        ap.print_help()
        return 0
    try:
        sched = schedules_for(a.season)
    except Exception as e:
        # The guard spans the reach for the source and nothing else.
        return unavailable("hub.store", f"the {a.season} nflverse schedules", e)
    try:
        return verify(season=a.season, sched=sched)
    except Exception as e:
        # And what is left is ours. A catalog error, a contract violation out of the write,
        # or an as-of join failure is the storage layer failing -- the thing this flag exists
        # to find. Reported in its own words rather than as a vendor being unreachable.
        print(f"hub.store: the storage layer failed verification: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
