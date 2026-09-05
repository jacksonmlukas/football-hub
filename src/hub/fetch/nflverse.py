"""nflverse fetch layer: narrow at the boundary, validate on the way out.

2025 play-by-play is 48,771 rows by **372 columns**. nflreadpy offers no column selection,
so anything that wants five of those columns downloads all 372 and hands them on. That is
the exact shape of the mistake CLAUDE.md rule 1 exists to prevent, and hoping every caller
remembers to narrow is not a control.

So this module refuses. Asking for a wide source without naming columns raises, and says
how wide it would have been. Narrow sources are returned as they are, because refusing
everything just relocates the friction.

Everything returned passes a `Contract` first. The failure this guards is not a crash --
it is nflverse renaming a column between releases and downstream numbers going quietly
wrong for weeks.

A load may also be *pinned*: given an as-of, it becomes its own cache entry and records a
digest over the bytes it actually loaded. The failure that buys is the one
`docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md` opens on -- no gate in the
tree re-runs to its own number while the archive it scored against is refetched live every
time. A pinned load answers with the same rows, or with a digest that says it could not.

    uv run python -m hub.fetch.nflverse --refresh --season 2025
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from hub import store
from hub.config import SEASON_COMPLETED, HubConfig, provenance
from hub.contracts import (
    FF_OPPORTUNITY,
    FTN_CHARTING,
    PARTICIPATION,
    PBP,
    PLAYER_STATS,
    SCHEDULES,
    Contract,
    ContractViolation,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "nflverse"

# Sources wide enough that handing one back whole is the mistake. Anything listed here
# must be asked for by column.
WIDE: dict[str, int] = {"pbp": 372, "player_stats": 150}

# The default slice `--refresh` takes of play-by-play. Named explicitly rather than
# defaulted inside load(), so the library API stays strict while the CLI stays usable:
# a caller writing code still has to decide what they want.
PBP_COLS: tuple[str, ...] = (
    "game_id", "season", "week", "posteam", "defteam",
    "play_type", "epa", "wp", "yards_gained", "success",
)


# The default slice of weekly player stats. 150 columns of box score, of which the weekly
# spread fit wants one: what he actually scored, in this league's scoring.
PLAYER_STATS_COLS: tuple[str, ...] = (
    "player_id", "player_display_name", "position", "season", "week", "season_type",
    "fantasy_points_ppr",
)


# Sources whose archive is append-only and carries the date each row was scraped, as
# {source: scrape-date column}. For these an as-of is a *filter* applied inside the loader,
# so a later fetch of an archive that has grown still yields the same rows; for everything
# else an as-of can only label a snapshot, and the pin says so.
#
# `ff_rankings` is the archive this exists for and has no loader yet -- it is page-typed
# rather than season-parameterised and needs a contract first (issue #33, U2 of
# docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md). The property belongs to
# the source rather than to its loader, and `hub.draft.tune.holdout` already bounds
# `scrape_date` on it by hand for the same reason this filter exists, so it is declared here
# and the loader reads the declaration when it lands.
#
# The consequence, stated rather than left for a reader to discover: nothing in this dict is
# in `SOURCES`, so the filter below and the `pinned_at is None` half of `Pin` are **not yet
# reachable through `load`**. `test_no_declared_append_only_source_can_be_loaded_yet` fails
# the day that stops being true, which is the day this paragraph and `Pin.pinned_at` need
# rewriting.
APPEND_ONLY: dict[str, str] = {"ff_rankings": "scrape_date"}


class WideFrameRefused(Exception):
    """Asked for a frame this module will not return in the shape requested."""


def _raw_pbp(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_pbp(seasons=list(seasons))


def _clean_ff_opportunity(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows that belong to no player.

    2025 ships 423 of 6,054 rows with player_id, position and full_name all null, still
    carrying up to 13.3 expected points. They are unattributed team-level residue: real
    numbers with nobody to assign them to.

    They go at the boundary rather than downstream, for two reasons. The FF_OPPORTUNITY
    contract declares player_id non-null, and it is right to -- weakening it to admit
    these would blind it to a genuine upstream break. And `expected_points()` already
    groups by player_id, so they were silently collapsing into a null bucket that joined
    to nothing. Dropping here makes a loss that was already happening visible.
    """
    return df.filter(pl.col("player_id").is_not_null())


def _raw_ff_opportunity(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_ff_opportunity(seasons=list(seasons), stat_type="weekly")
    clean = _clean_ff_opportunity(raw)
    if clean.height < raw.height:
        print(f"    ff_opportunity: dropped {raw.height - clean.height:,} unattributed "
              f"rows of {raw.height:,}")
    return clean


class UnattributedPoints(Exception):
    """Scoring rows that belong to no player -- an upstream break, not residue."""


def _clean_player_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows that belong to no player, and refuse if any of them scored.

    nflverse ships exactly 22 rows a season with player_id, position and name all null and
    zero fantasy points. They are residue, and the PLAYER_STATS contract declares player_id
    non-null, so they go here rather than by weakening the contract -- the same call
    `_clean_ff_opportunity` makes.

    The difference from that one: this refuses if an unattributed row carries points. There
    the residue genuinely held expected points with nobody to assign them to; here a null-id
    row that scored would mean nflverse had changed something, and silently dropping real
    points is how a projection goes quietly wrong for a month.
    """
    orphan = df.filter(pl.col("player_id").is_null())
    scoring = orphan.filter(pl.col("fantasy_points_ppr").fill_null(0.0) != 0.0)
    if scoring.height:
        raise UnattributedPoints(
            f"{scoring.height} rows with no player_id carry "
            f"{scoring['fantasy_points_ppr'].sum():.1f} fantasy points. Historically these "
            "rows are empty residue; points in them means the upstream shape changed.")
    return df.filter(pl.col("player_id").is_not_null())


def _raw_player_stats(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_player_stats(seasons=list(seasons), summary_level="week")
    clean = _clean_player_stats(raw)
    if clean.height < raw.height:
        print(f"    player_stats: dropped {raw.height - clean.height:,} unattributed "
              f"rows of {raw.height:,}")
    return clean


def _raw_schedules(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_schedules().filter(pl.col("season").is_in(list(seasons)))


def _raw_participation(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_participation(seasons=list(seasons))


def _raw_ftn_charting(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_ftn_charting(seasons=list(seasons))


SOURCES: dict[str, Contract | None] = {
    "pbp": PBP,
    "ff_opportunity": FF_OPPORTUNITY,
    "player_stats": PLAYER_STATS,
    "schedules": SCHEDULES,
    # The scheme layer. Neither is WIDE -- 26 and 29 columns -- so both come back whole.
    "participation": PARTICIPATION,
    "ftn_charting": FTN_CHARTING,
}


def _fetch(source: str, seasons: Sequence[int]) -> pl.DataFrame:
    """Dispatch by name at call time, not by binding function objects at import.

    A dict of function objects built at module scope captures whatever was defined then,
    so a test that replaces `_raw_ff_opportunity` is ignored and the call goes to the
    network instead. That is not only a testing problem: it makes the indirection a lie.
    """
    fetchers: dict[str, Callable[[Sequence[int]], pl.DataFrame]] = {
        "pbp": _raw_pbp,
        "ff_opportunity": _raw_ff_opportunity,
        "player_stats": _raw_player_stats,
        "schedules": _raw_schedules,
        "participation": _raw_participation,
        "ftn_charting": _raw_ftn_charting,
    }
    return fetchers[source](seasons)


@dataclass(frozen=True)
class Pin:
    """What one load recorded about the data behind it, so a gate can name it.

    `pinned_at` carries the honest half. It is set where the rows cannot be fetched again:
    `player_stats` and `pbp` mirror an upstream that revises in place, which `_write_by_week`
    says where it passes `replace=True`. A stamp there makes an unreproducible re-run
    *detectable* rather than silently assumed reproducible.

    It is None where they can -- an append-only archive filtered at its as-of reproduces from
    the as-of alone -- and **that state is not yet reachable through `load`**. `APPEND_ONLY`
    names `ff_rankings` and `SOURCES` does not, so a load of it is refused at the unknown-source
    check before the as-of filter is ever consulted, and every pin a caller writes today
    therefore carries a stamp. The loader that closes the gap is issue #33 (U2 of
    docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md); it needs a contract
    first, which is why it is not done here. Until it lands, a None here comes only from a
    test that substituted the registry, and
    `test_no_declared_append_only_source_can_be_loaded_yet` is the canary that will say so.
    """

    source: str
    as_of: str | None
    digest: str
    rows: int
    pinned_at: str | None = None


def content_digest(df: pl.DataFrame) -> str:
    """A hash of what the frame holds, reproducible in a fresh interpreter.

    Not `hash()`: it is salted per process, so a digest resting on it would change on every
    run and could name nothing.

    Not the Arrow IPC bytes either, which was the first attempt here. A frame and that same
    frame read back from the parquet this module writes serialise to *different* IPC bytes
    on polars 1.43 -- measured 2026-09-05, with identical schemas either side -- so the
    digest would have moved on a cache hit alone. The canonical form is instead the column
    names and dtypes followed by the frame's CSV bytes: text, exact for floats
    (`0.1 + 0.2` writes as `0.30000000000000004`), and distinguishing a null from an empty
    string. It is order-sensitive, which is right -- a reordered frame is not the frame a
    published number was computed on.
    """
    head = ";".join(f"{c}:{df.schema[c]}" for c in df.columns).encode()
    buf = io.BytesIO()
    df.write_csv(buf)
    return hashlib.sha256(head + b"\n" + buf.getvalue()).hexdigest()[:8]


def pin_digest(source: str, as_of: str | None, df: pl.DataFrame) -> str:
    """The published digest: content, folded with the source name and the as-of.

    Both halves. Hashing the labels alone would be invariant to exactly the drift this
    exists to catch -- two runs at one as-of that fetched different bytes would agree.
    Hashing content alone would make one archive that has not moved between two pins
    indistinguishable, and a pin is a claim about a date as well as about rows.

    Eight hex characters, the length `config_digest` and `fitted_digest` already use, since
    a gate output prints the two side by side.
    """
    return hashlib.sha256(
        f"{source}\n{as_of or ''}\n{content_digest(df)}".encode()).hexdigest()[:8]


def _as_of_date(as_of: str | date | None) -> date | None:
    """Normalise an as-of to a plain date, raising on anything that is not one."""
    if as_of is None:
        return None
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    return date.fromisoformat(as_of)


def _as_of_filter(source: str, df: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """Bound an append-only archive at the as-of, inside the loader.

    The reproducing half of the pin: the archive grows, and a load at the same as-of still
    yields the rows it yielded before. Inclusive of the day itself. A row whose scrape date
    is null cannot be placed in time, so it does not survive a pinned load.
    """
    col = APPEND_ONLY[source]
    if col not in df.columns:
        raise ContractViolation(
            f"{source} is declared append-only on {col!r} and that column is not there; "
            f"got {len(df.columns)} columns from upstream. The as-of cannot be applied, and "
            "returning the unfiltered archive would be the drift the pin exists to catch.")
    scraped = (pl.col(col).str.to_date(strict=False) if df.schema[col] == pl.Utf8
               else pl.col(col).cast(pl.Date))
    return df.filter(scraped <= pl.lit(as_of))


def _cache_path(source: str, seasons: Sequence[int], cols: Sequence[str] | None,
                cache: Path | None, as_of: date | None = None) -> Path:
    """One entry per (source, seasons, columns, as-of).

    The column set is part of the key. Without that, a caller asking for four columns
    would be served an earlier caller's three and never notice. The as-of is part of it for
    the same reason: two pins are two entries, and a gate re-run at one as-of must never be
    handed another's rows.

    With no as-of this is the undated path the module has always written, for read and for
    write both -- which is what leaves `make slate`, which drives `refresh=True` and passes
    no as-of, on exactly the file it used yesterday.
    """
    root = cache or RAW
    key = ",".join(sorted(cols)) if cols else "all"
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    stamp = "-".join(str(s) for s in sorted(seasons))
    dated = f"-asof-{as_of.isoformat()}" if as_of is not None else ""
    return root / source / f"{stamp}-{digest}{dated}.parquet"


def _pin_path(path: Path) -> Path:
    """The pin sits beside its cache entry, not in a registry of its own.

    Same call ADR-0006 makes for fitted constants and their provenance: a record kept away
    from the thing it describes is one that stops matching it. It also survives the process,
    so a load served from cache can still say what it is serving.
    """
    return path.with_suffix(".pin.json")


def data_pin(source: str, seasons: Sequence[int], cols: Sequence[str] | None = None,
             cache: Path | None = None, as_of: str | date | None = None) -> Pin | None:
    """The pin beside one cache entry, or None when nothing readable has been written there.

    None rather than a raise, three ways, all the same call: entries written before this
    existed have no pin beside them, an interrupted write leaves a file that is not JSON, and
    a caller asking about a cold tree should get an answer it can degrade on rather than an
    exception. CLAUDE.md's rule is that a module produces a usable answer with zero attention;
    a provenance lookup that takes down the gate asking it is the opposite of that.

    Fields this version does not know are dropped rather than raising. The pin tree outlives
    any one version of this module -- the sidecar sits beside a parquet file that survives a
    checkout, and U2 will add sources with more to record -- so a pin written by a later
    version has to degrade to the fields understood here. A record missing what *identifies*
    it is a different case and reads as nothing: a Pin with no digest names no data.
    """
    path = _pin_path(_cache_path(source, seasons, cols, cache, _as_of_date(as_of)))
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    known = {f.name for f in fields(Pin)}
    try:
        return Pin(**{k: v for k, v in raw.items() if k in known})
    except TypeError:
        return None


def load(source: str, seasons: Sequence[int], cols: Sequence[str] | None = None,
         refresh: bool = False, cache: Path | None = None,
         as_of: str | date | None = None) -> pl.DataFrame:
    """Fetch one nflverse source, narrowed and validated.

    Raises rather than returning a wide frame, because the caller who forgets to narrow
    is the caller this module exists for.

    `as_of` pins the load: its own cache entry, filtered where the source is append-only and
    labelled where it revises in place, with a `Pin` written beside the entry either way.
    Omitted, nothing about the call changes -- the same undated entry is read and written as
    before, `refresh=True` rewrites that undated entry, and the weekly slate path is
    untouched.

    Of the two, only labelling is reachable today: no source in `APPEND_ONLY` is in `SOURCES`,
    so the filtering branch is **not yet reachable** from here and every pin this writes
    carries a `pinned_at`. See `APPEND_ONLY` and `Pin.pinned_at` for what would change that.
    """
    if source not in SOURCES:
        raise WideFrameRefused(
            f"unknown source {source!r}. Known: {', '.join(sorted(SOURCES))}")

    # GUARD wide-frame-refused [unit/test_fetch_nflverse.py]: a wide source is never whole
    if source in WIDE and not cols:
        standard = {"pbp": "PBP_COLS", "player_stats": "PLAYER_STATS_COLS"}.get(source)
        raise WideFrameRefused(
            f"{source} is {WIDE[source]} columns wide; name the ones you need via cols=."
            + (f" For the standard slice use hub.fetch.nflverse.{standard}."
               if standard else ""))
    # /GUARD

    stamp = _as_of_date(as_of)
    path = _cache_path(source, seasons, cols, cache, stamp)
    if path.exists() and not refresh:
        return pl.read_parquet(path)

    contract = SOURCES[source]
    df = _fetch(source, seasons)

    # The as-of filters content where the source allows it and only labels a snapshot where
    # it does not. `pinned_at` on the pin below is what says which of the two happened, so
    # a re-run that cannot reproduce is detectable rather than assumed reproducible.
    reproducible = False
    if stamp is not None and source in APPEND_ONLY:
        df = _as_of_filter(source, df, stamp)
        reproducible = True

    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise WideFrameRefused(
                f"{source} has no column {missing!r}; "
                f"got {len(df.columns)} columns from upstream")
        df = df.select(list(cols))

    if contract is not None:
        contract.validate(df)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    iso = stamp.isoformat() if stamp is not None else None
    _pin_path(path).write_text(json.dumps(asdict(Pin(
        source=source,
        as_of=iso,
        digest=pin_digest(source, iso, df),
        rows=df.height,
        pinned_at=None if reproducible else datetime.now(UTC).isoformat(timespec="seconds"),
    )), indent=2, sort_keys=True) + "\n")
    return df


def _write_by_week(df: pl.DataFrame, table: str, season: int,
                   base: Path | None) -> tuple[int, int]:
    """Split a season frame into the store's week partitions.

    hub.store partitions on week, so a season-wide frame has to be broken up here rather
    than written as one blob -- otherwise every later query scans the whole season to read
    one week, which is the pruning the Hive layout exists for.
    """
    weeks = sorted(w for w in df["week"].unique().to_list() if w is not None)
    for wk in weeks:
        # `replace=True` deliberately: these partitions mirror an nflverse table that revises
        # in place, so the store mirrors it rather than accumulating a copy per refresh.
        # `make slate` re-fetches every week weekly, and appending would grow the tree without
        # bound and double-count anything that later queried it. The cost is real and worth
        # naming -- a stat correction overwrites the number it corrects, and the record of the
        # change lives with nflverse rather than here.
        store.write(df.filter(pl.col("week") == wk), table, "nfl", season, int(wk),
                    base=base, replace=True)
    return len(weeks), df.height


def refresh(season: int = SEASON_COMPLETED, cache: Path | None = None,
            base: Path | None = None) -> int:
    """Pull play-by-play and ff_opportunity, write both through the store, and say which bytes.

    Prints counts and digests only. A fetch path that prints rows is the failure it is meant
    to prevent, so the summary never names a column or a value.

    The last line is what made the pins worth writing. Until it, this module wrote a `Pin`
    beside every cache entry and nothing in `src/` ever read one back, so an archive that
    moved under the harness surfaced only as a different number downstream, with nothing to
    say the input had moved rather than the code. `docs/next.md` records that happening: two
    `--diagnose` runs at the same seed disagreed on pick 3 because `build()` refetched live
    ESPN ADP each time, and it took a flipped leader between two post-fix runs to notice.
    This is the path that actually fetches, so this is the path that has to name what it
    fetched.

    Three digests, not one, and `hub.config.data_digest` argues at length why the data digest
    sits beside the model version rather than inside it: this line moves on a Tuesday refetch,
    and `cfg` must not.
    """
    print(f"  nflverse refresh: season {season}")
    total_rows = 0
    pins: list[Pin] = []
    for table, cols, label in (("pbp", list(PBP_COLS), "play-by-play"),
                               ("ff_opportunity", None, "ff_opportunity")):
        df = load(table, seasons=[season], cols=cols, refresh=True, cache=cache)
        # Read back from the sidecar rather than re-digesting `df` here, so what is printed is
        # what a later reader of this cache entry will compute -- one path to the number.
        if (pin := data_pin(table, [season], cols=cols, cache=cache)) is not None:
            pins.append(pin)
        n_weeks, rows = _write_by_week(df, table, season, base)
        total_rows += rows
        print(f"    {label:<16} {rows:>7,} rows | {len(df.columns):>3} cols | "
              f"{n_weeks} week partitions")
    print(f"  wrote {total_rows:,} rows through hub.store")
    p = provenance(HubConfig(), pins)
    print(f"  cfg {p['cfg']} | fitted {p['fitted']} | data {p['data']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.fetch.nflverse",
        description="Fetch nflverse data, narrowed at the boundary and contract-checked.")
    ap.add_argument("--refresh", action="store_true",
                    help="pull this season's pbp and ff_opportunity into the store")
    ap.add_argument("--season", type=int, default=SEASON_COMPLETED)
    a = ap.parse_args(argv)
    if not a.refresh:
        ap.print_help()
        return 0
    return refresh(season=a.season, cache=RAW)


if __name__ == "__main__":
    sys.exit(main())
