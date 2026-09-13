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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic, store
from hub.cli import unavailable
from hub.config import (
    SEASON_COMPLETED,
    DataPin,
    UnpinnedRead,
    digests,
    pin_fold,
    resolved_config,
)
from hub.contracts import (
    FF_OPPORTUNITY,
    FF_RANKINGS,
    FTN_CHARTING,
    INJURIES,
    PARTICIPATION,
    PBP,
    PLAYER_STATS,
    SCHEDULES,
    SNAP_COUNTS,
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
# `ff_rankings` is the archive this exists for, and since #33 it has a loader: `load_rankings`
# below, which reaches `load` with the page type where a season-partitioned source passes a
# season list. So the filter below and the `pinned_at is None` half of `Pin` are reachable
# through `load` -- a pinned rankings load reproduces from its as-of alone, and its pin says
# so by carrying no stamp. `test_the_append_only_path_is_reachable_through_load` asserts that
# end to end, and `test_the_docstrings_match_what_a_caller_can_reach` ties this paragraph to
# it in both directions.
#
# The property belongs to the source rather than to its loader, which is why it was declared
# here before there was one, and why `hub.draft.tune.holdout` bounds `scrape_date` by hand
# for the same reason this filter exists.
APPEND_ONLY: dict[str, str] = {"ff_rankings": "scrape_date"}

# The FantasyPros ranking pages `load_rankings` will serve, and the reason the third is not
# among them. nflreadpy offers "draft" (the latest scrape, ~5,850 rows), "all" (the archive,
# 1.83M rows from 2019-12-27) and "week" -- and the first two are one table under two names
# while "week" is a different one: `page_pos` rather than `page_type`, `player_name` rather
# than `player`, and a `rank` column the other two do not carry. `FF_RANKINGS` describes the
# archive's shape, nothing in this repo reads the weekly page, and admitting it here would
# mean either a contract loose enough to cover both or one that fails on every weekly load.
# It gets its own source and contract on the day something needs it.
RANKINGS_PAGES: tuple[str, ...] = ("draft", "all")

# The columns `load_rankings` is asked for: the contract's own required set, which is also
# every column either routed reader takes. It lives here, beside the loader, because it is
# half of that loader's cache key -- `(source, page, columns, as-of)` -- and the two readers
# that name the key are in different packages. Stated in one of them, the other has to either
# import across a package boundary or retype it; retyping it is what `hub.draft.board` did,
# under a comment in `hub.models.panel` claiming it was stated once. The archive is 1.8M rows
# and two hand-written tuples drifting is two nearly-identical copies of it.
RANKINGS_COLS: tuple[str, ...] = tuple(FF_RANKINGS.required)


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


def _raw_injuries(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_injuries(seasons=list(seasons))


def _raw_snap_counts(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_snap_counts(seasons=list(seasons))


def _raw_ff_rankings(pages: Sequence[str]) -> pl.DataFrame:
    """The one source keyed by a page type instead of a season.

    `pages` is the partition key `load` was handed, which for this source is a single
    FantasyPros page. Refused rather than silently served if it is anything else: a caller
    who passed two pages would get one of them, and a board built from the wrong scale is
    the failure `hub.draft.board._select_consensus` already exists to prevent.
    """
    # GUARD rankings-partition-key: a key that is not one known page never reaches nflreadpy
    if len(pages) != 1 or pages[0] not in RANKINGS_PAGES:
        raise WideFrameRefused(
            f"ff_rankings is keyed by one page type, not by {list(pages)!r}. "
            f"Known: {', '.join(RANKINGS_PAGES)}; use load_rankings(page, as_of=...).")
    # /GUARD
    import nflreadpy as nfl
    # nflreadpy types the argument as a `Literal`, and the check above is what narrows it --
    # a membership test in a module-level tuple, which no type checker follows.
    return nfl.load_ff_rankings(pages[0])  # type: ignore[bad-argument-type]


SOURCES: dict[str, Contract | None] = {
    "pbp": PBP,
    "ff_opportunity": FF_OPPORTUNITY,
    "player_stats": PLAYER_STATS,
    "schedules": SCHEDULES,
    # The scheme layer. Neither is WIDE -- 26 and 29 columns -- so both come back whole.
    "participation": PARTICIPATION,
    "ftn_charting": FTN_CHARTING,
    # The three #33 added. None is WIDE -- 25, 17 and 16 columns -- so none needs `cols`,
    # though passing one still narrows. `ff_rankings` is keyed by page type rather than by
    # season and is reached through `load_rankings`.
    "ff_rankings": FF_RANKINGS,
    "injuries": INJURIES,
    "snap_counts": SNAP_COUNTS,
}


def _fetch(source: str, keys: Sequence[int | str]) -> pl.DataFrame:
    """Dispatch by name at call time, not by binding function objects at import.

    A dict of function objects built at module scope captures whatever was defined then,
    so a test that replaces `_raw_ff_opportunity` is ignored and the call goes to the
    network instead. That is not only a testing problem: it makes the indirection a lie.

    `keys` is the partition key set. For every source but one that is a list of seasons;
    `ff_rankings` is not season-partitioned and its key is the FantasyPros page type, which
    is why the annotation admits a string. The value type stays `Any`: a fetcher declared
    over `Sequence[int]` is not assignable to one declared over the wider type, and widening
    all six to say otherwise would be six lies told to make one true.
    """
    fetchers: dict[str, Callable[[Any], pl.DataFrame]] = {
        "pbp": _raw_pbp,
        "ff_opportunity": _raw_ff_opportunity,
        "player_stats": _raw_player_stats,
        "schedules": _raw_schedules,
        "participation": _raw_participation,
        "ftn_charting": _raw_ftn_charting,
        "injuries": _raw_injuries,
        "snap_counts": _raw_snap_counts,
        "ff_rankings": _raw_ff_rankings,
    }
    return fetchers[source](keys)


@dataclass(frozen=True)
class Pin:
    """What one load recorded about the data behind it, so a gate can name it.

    `pinned_at` carries the honest half. It is set where the rows cannot be fetched again:
    `player_stats` and `pbp` mirror an upstream that revises in place, which `_write_by_week`
    says where it passes `replace=True`. A stamp there makes an unreproducible re-run
    *detectable* rather than silently assumed reproducible.

    It is None where they can -- an append-only archive filtered at its as-of reproduces from
    the as-of alone. `ff_rankings` is that archive and #33 gave it a loader, so a caller
    writing `load_rankings("all", as_of=...)` gets a pin with no stamp on it, and the claim
    that pin makes is the strong one: these rows can be fetched again.

    Both states are reachable and both are asserted. The labelled half is taken through
    `player_stats` by the test that a source revising in place carries a stamp; this half is
    taken through a real `load_rankings` by
    `test_the_append_only_path_is_reachable_through_load`, rather than through a substituted
    registry -- which is what the canary it replaced was asking for.

    Before #33 the two were not symmetric: `APPEND_ONLY` named `ff_rankings`, `SOURCES` did
    not, and a load of it was refused at the unknown-source check before the as-of filter was
    ever consulted, so every pin a real caller could write carried a stamp.

    **`rescaled` is #140's open question, answered yes.** The ticket left it deliberately: a
    pin currently claims to describe what the source sent, and after a `Contract` repairs a
    declared upstream variation that is no longer quite true, so should the pin say so. It
    should, and the reason is the same one that made the silence worth fixing at all.

    The terminal line `hub.contracts._announce` writes is emitted once, by the fetch that
    repaired. Every later read of that entry is a cache hit -- `load` returns the parquet
    without re-validating -- so nothing prints again while the repaired frame is served from
    that entry for as long as it stands. The report answers "what just happened"; it cannot
    answer "what am I serving", and the second question is the one a gate asks in November
    about a refresh in September. The pin is the only record that lives beside the entry for
    as long as the entry does, and `refresh` already reads pins back to say what a run was
    computed on.

    `pinned_at` is the precedent and not merely a neighbour: it is already a field recording
    *how* a load was obtained rather than what it contains, for the same reason -- to keep a
    property of the fetch from being silently assumed by a later reader.

    **The argument against, and why it loses.** `digest` already moves when a repair fires,
    since it is computed on the frame that is written -- so a repair is not invisible to two
    pins compared side by side. But a moved digest is what a stat correction, a grown archive
    and a units change all look like; it says "not the bytes you had" and cannot say which.
    Naming the columns is the difference between that and "these three arrived in units the
    declaration did not expect, and were multiplied by a hundredth on the way in", which is
    the sentence somebody needs to go and check upstream.

    Empty is the ordinary state and means no declared repair fired -- not that nobody looked,
    because every load of a source with a contract asks. A pin written before this field
    existed reads back empty too, which understates by exactly one case and is the same
    degradation `data_pin` already documents for a field it has never heard of.
    """

    source: str
    as_of: str | None
    digest: str
    rows: int
    pinned_at: str | None = None
    rescaled: tuple[str, ...] = ()


def content_digest(df: pl.DataFrame, key: Sequence[str]) -> str:
    """A hash of what the frame holds, reproducible in a fresh interpreter and independent
    of the order the source happened to hand the rows over in.

    Not `hash()`: it is salted per process, so a digest resting on it would change on every
    run and could name nothing.

    Not the Arrow IPC bytes either, which was the first attempt here. A frame and that same
    frame read back from the parquet this module writes serialise to *different* IPC bytes
    on polars 1.43 -- measured 2026-09-05, with identical schemas either side -- so the
    digest would have moved on a cache hit alone. The canonical form is instead the column
    names and dtypes followed by the frame's CSV bytes: text, exact for floats
    (`0.1 + 0.2` writes as `0.30000000000000004`), and distinguishing a null from an empty
    string.

    Those bytes are order-sensitive, so the rows are put in a declared order before they are
    written. The argument that used to stand here -- that a reordered frame is not the frame
    a published number was computed on -- is true of a *published* frame and does not carry
    to one just read off the wire, where the row order is the source's whim. Left alone it
    reported drift that had not happened, which is the failure this hash exists to detect and
    therefore the one it must not manufacture.

    `key` is the contract's declared unique key; `pin_digest` reads it from `SOURCES`.
    Sorting on it alone would be canonical only while it really is unique, so every remaining
    column follows it as the tiebreak. Rows that tie through all of them are identical rows
    and write identical bytes whichever way the tie fell, so the digest is a function of the
    frame's content whether the key is unique, duplicated, or absent.

    Absent is the ordinary case rather than the corner: eight of the nine nflverse contracts
    declare no unique key, and a pinned load whose `cols=` omitted the key leaves a frame
    without it. Neither falls back to insertion order -- with no usable key every column is
    the key. `key` has no default so that each caller states which it has, rather than
    omitting the argument and getting the silent fallback this replaced.
    """
    # Declared key first, then the rest of the frame; with no usable key this is every
    # column, which is a total order over anything that is not a duplicate row.
    by = [c for c in key if c in df.columns]
    by += [c for c in df.columns if c not in by]
    ordered = df.sort(by) if by else df
    head = ";".join(f"{c}:{df.schema[c]}" for c in df.columns).encode()
    buf = io.BytesIO()
    ordered.write_csv(buf)
    return hashlib.sha256(head + b"\n" + buf.getvalue()).hexdigest()[:8]


def pin_digest(source: str, as_of: str | None, df: pl.DataFrame) -> str:
    """The published digest: content, folded with the source name and the as-of.

    The fold itself is `hub.config.pin_fold`, which argues there for what goes into it and
    what stays out. It lives there rather than here because `data_digest` folds a *set* of
    these the same way and `hub.config` may not import a fetch layer; two statements of one
    form, with a comment on each saying the other matched, is what this call replaced.

    Eight hex characters, the length `config_digest` and `fitted_digest` already use, since
    a gate output prints the two side by side.

    The order the rows hash in comes from the contract registered for `source`, which is the
    only place the unique key is declared -- so a caller cannot pass a key that disagrees
    with the one the frame was validated against. A source with no contract declares no key,
    and `content_digest` orders on the whole frame instead.
    """
    contract = SOURCES.get(source)
    key = () if contract is None else contract.unique
    return hashlib.sha256(
        pin_fold(source, as_of, content_digest(df, key)).encode()).hexdigest()[:8]


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
    # GUARD append-only-column-vanished: an as-of that cannot be applied refuses the archive
    if col not in df.columns:
        raise ContractViolation(
            f"{source} is declared append-only on {col!r} and that column is not there; "
            f"got {len(df.columns)} columns from upstream. The as-of cannot be applied, and "
            "returning the unfiltered archive would be the drift the pin exists to catch.")
    # /GUARD
    scraped = (pl.col(col).str.to_date(strict=False) if df.schema[col] == pl.Utf8
               else pl.col(col).cast(pl.Date))
    return df.filter(scraped <= pl.lit(as_of))


def _cache_path(source: str, seasons: Sequence[int | str], cols: Sequence[str] | None,
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


# What this process has actually read, keyed by the entry it came from. A gate folds these
# into the `data` digest it publishes, so an archive that moved shows up as a changed digest
# rather than as a silently different number -- which is the whole premise of the pinning
# layer and the one part of it nothing computed (issue #71).
#
# Recorded here rather than assembled by the gate, because this is the only place that knows
# what a run read. A gate naming its own sources would be a second list, free to drift from
# the loads that actually happened, and drift is the failure this digest exists to catch.
#
# Both paths record: a cache hit reads the sidecar beside the entry it served. A run that
# answered entirely from cache has read data and has to be able to say which.
#
# Every read, including one with no pin to read. An entry written before pinning existed, or
# one whose sidecar write was interrupted, used to contribute nothing here -- so a run that
# read three sources with two pinned published a digest over two, and nothing said so.
# `config.UnpinnedRead` is what such a read records instead, and `config.data_digest` turns
# the run's whole answer into the sentinel when it sees one.
_READ_THIS_RUN: dict[str, DataPin] = {}


@contextmanager
def reads_of_one_run() -> Iterator[None]:
    """Scope the reads below to one run, so a component's digest names its own bytes.

    The dict above is process-global and used never to be reset, which is right for a process
    that is one run and wrong the moment it is not: a build driving the board, the gate and
    the publisher in turn folded all three components' reads into every component's digest.
    Each then named bytes it had not read, and the digests of three different questions came
    out identical -- which is precisely the claim a digest exists to be able to deny.

    Reads made inside the block still reach the enclosing run on the way out. The scope
    narrows what a component *reports*, and must not become a way for a run to lose a read:
    a helper that opened a scope of its own would otherwise leave its caller's digest short by
    exactly the sources the helper loaded, which is the defect in this file's other half
    wearing a different hat. First read still wins, on both sides of the boundary.
    """
    global _READ_THIS_RUN
    outer = _READ_THIS_RUN
    _READ_THIS_RUN = {}
    try:
        yield
    finally:
        inner = _READ_THIS_RUN
        _READ_THIS_RUN = outer
        for entry, read in inner.items():
            _READ_THIS_RUN.setdefault(entry, read)


def pins_this_run() -> tuple[DataPin, ...]:
    """Every nflverse entry this run has loaded, in the order first read.

    A run rather than a process: `reads_of_one_run` above is what makes the difference real,
    and a caller that wants only its own reads opens one. Without a scope this is still every
    read the process has made, which is the right answer for the ordinary case of one run per
    process and the wrong one for a build running several components in turn.

    Empty is a truthful answer and `config.UNPINNED` is what a digest over it says: a run that
    loaded nothing pinned nothing. Callers fold this through `config.data_digest`.

    Entries whose pin could not be read come back as `config.UnpinnedRead` rather than not
    coming back, so the count here is the number of entries read and not the number that
    happened to have a sidecar.
    """
    return tuple(_READ_THIS_RUN.values())


def _remember(path: Path, read: DataPin) -> None:
    """Record what an entry held, once per entry. First read wins.

    A second load of the same entry in one process returns the same bytes -- the cache path is
    a function of the key -- so re-recording would only reorder the digest's inputs.

    `read` is not optional. It used to be, and a `None` meant nothing was recorded at all,
    which is how an entry with no readable pin left the digest looking complete. A caller with
    no pin passes `config.UnpinnedRead`, which says so.
    """
    _READ_THIS_RUN.setdefault(str(path), read)


def _pin_beside(path: Path) -> Pin | None:
    """The pin written next to one cache entry, or None if there is nothing readable there."""
    try:
        raw = json.loads(_pin_path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    known = {f.name for f in fields(Pin)}
    kept = {k: v for k, v in raw.items() if k in known}
    # JSON has no tuple, so the one field holding several column names comes back a list.
    # Closed here rather than left to the reader: `Pin` is frozen and therefore hashable,
    # and a list inside one is a pin that is a `Pin` right up until something hashes it.
    if isinstance(kept.get("rescaled"), list):
        kept["rescaled"] = tuple(kept["rescaled"])
    try:
        return Pin(**kept)
    except TypeError:
        return None


def data_pin(source: str, seasons: Sequence[int | str], cols: Sequence[str] | None = None,
             cache: Path | None = None, as_of: str | date | None = None) -> Pin | None:
    """The pin beside one cache entry, or None when nothing readable has been written there.

    None rather than a raise, three ways, all the same call: entries written before this
    existed have no pin beside them, an interrupted write leaves a file that is not JSON, and
    a caller asking about a cold tree should get an answer it can degrade on rather than an
    exception. CLAUDE.md's rule is that a module produces a usable answer with zero attention;
    a provenance lookup that takes down the gate asking it is the opposite of that.

    Fields this version does not know are dropped rather than raising. The pin tree outlives
    any one version of this module -- the sidecar sits beside a parquet file that survives a
    checkout, and a later unit may add sources with more to record -- so a pin written by a
    later version has to degrade to the fields understood here. A record missing what *identifies*
    it is a different case and reads as nothing: a Pin with no digest names no data.
    """
    return _pin_beside(_cache_path(source, seasons, cols, cache, _as_of_date(as_of)))


def load(source: str, seasons: Sequence[int | str], cols: Sequence[str] | None = None,
         refresh: bool = False, cache: Path | None = None,
         as_of: str | date | None = None) -> pl.DataFrame:
    """Fetch one nflverse source, narrowed and validated.

    Raises rather than returning a wide frame, because the caller who forgets to narrow
    is the caller this module exists for.

    `seasons` is the partition key. It is a season list for every source but `ff_rankings`,
    which is not season-partitioned and is keyed by FantasyPros page type instead -- reach
    that one through `load_rankings`, which takes the page and refuses a season list rather
    than quietly keying a cache entry on a year the source knows nothing about.

    `as_of` pins the load: its own cache entry, filtered where the source is append-only and
    labelled where it revises in place, with a `Pin` written beside the entry either way.
    Omitted, nothing about the call changes -- the same undated entry is read and written as
    before, `refresh=True` rewrites that undated entry, and the weekly slate path is
    untouched.

    Both halves are reachable. A pinned `ff_rankings` load is filtered at its as-of and its
    pin carries no `pinned_at`, because those rows reproduce from the as-of alone; every
    other source is labelled and stamped. See `APPEND_ONLY` and `Pin.pinned_at`.
    """
    # GUARD unknown-source-refused: a name the registry does not know reaches no fetcher
    if source not in SOURCES:
        raise WideFrameRefused(
            f"unknown source {source!r}. Known: {', '.join(sorted(SOURCES))}")
    # /GUARD

    # GUARD wide-frame-refused [unit/test_fetch_nflverse.py]: a wide source is never whole
    if source in WIDE and not cols:
        standard = {"pbp": "PBP_COLS", "player_stats": "PLAYER_STATS_COLS"}.get(source)
        raise WideFrameRefused(
            f"{source} is {WIDE[source]} columns wide; name the ones you need via cols=."
            + (f" For the standard slice use hub.fetch.nflverse.{standard}."
               if standard else ""))
    # /GUARD

    stamp = _as_of_date(as_of)
    iso = stamp.isoformat() if stamp is not None else None
    path = _cache_path(source, seasons, cols, cache, stamp)
    if path.exists() and not refresh:
        # A cache hit is still a read, and the run has to be able to say what it read --
        # including when there is no pin beside the entry to say it with. An entry written
        # before pinning existed, or one whose sidecar write was interrupted, is a read of
        # bytes this run cannot name, and `UnpinnedRead` is how it says that. Dropping it
        # instead left the digest short by one source and looking complete, which is the one
        # outcome the sentinel exists to prevent.
        served = _pin_beside(path)
        _remember(path, served if served is not None else UnpinnedRead(source, iso))
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
        # GUARD column-vanished-upstream: a column that stopped arriving is named, not selected
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise WideFrameRefused(
                f"{source} has no column {missing!r}; "
                f"got {len(df.columns)} columns from upstream")
        # /GUARD
        df = df.select(list(cols))

    rescaled: tuple[str, ...] = ()
    if contract is not None:
        # Asked of the frame as it arrived, and therefore before the line below repairs it:
        # `Contract.repairs` reads what it is handed, so the other order records "nothing was
        # rescaled" about a frame that was. `Pin.rescaled` says why this is written down at
        # all when the boundary has already said it on the terminal.
        # GUARD rescale-is-pinned [unit/test_fetch_nflverse.py]: a cache entry whose archive
        # was repaired on ingest carries that fact for as long as the entry stands
        rescaled = tuple(sorted(contract.repairs(df)))
        # /GUARD
        # The *returned* frame, because a declared repair only reaches anybody through it.
        # Dropping it here wrote whatever units the source happened to send, pinned them,
        # and served them from cache forever after -- and the frame passed, so nothing said
        # so. Refusing used to keep such a frame out of the cache entirely.
        df = contract.validate(df)

    atomic.write_parquet(df, path)
    pin = Pin(
        source=source,
        as_of=iso,
        digest=pin_digest(source, iso, df),
        rows=df.height,
        pinned_at=None if reproducible else datetime.now(UTC).isoformat(timespec="seconds"),
        rescaled=rescaled,
    )
    atomic.write_text(_pin_path(path), json.dumps(asdict(pin), indent=2, sort_keys=True) + "\n")
    _remember(path, pin)
    return df


def load_rankings(page: str = "draft", as_of: str | date | None = None,
                  cols: Sequence[str] | None = None, refresh: bool = False,
                  cache: Path | None = None, *,
                  seasons: object = None) -> pl.DataFrame:
    """The FantasyPros consensus archive, by page type and as-of.

    The one source here that is not season-partitioned. DynastyProcess republishes every
    scrape of every ranking page as one table -- 1.83M rows over 2019-12-27 to 2026-09-04,
    measured 2026-09-05 -- so what identifies a load is *which page* and *as of when*, and a
    season list identifies nothing. `seasons` exists only to say so: passing one raises
    rather than being silently ignored, because a caller who thinks they have bounded a load
    to 2024 and has not is the reader this refusal is for.

    Everything else is `load`: the same dated cache entry, the same `FF_RANKINGS` validation
    on the way out, the same `Pin` beside the entry -- and because `ff_rankings` is declared
    in `APPEND_ONLY`, an as-of here *filters* the archive rather than labelling it. Two loads
    at one as-of return the same rows however much the archive has grown between them, and
    the pin says so by carrying no `pinned_at`.

    The pin for a load from here reads back as `data_pin("ff_rankings", [page], ...)`: the
    page is the partition key, so it is the key the sidecar is filed under.

    47 ranking pages are stacked in the frame this returns, each on its own ECR scale --
    `hub.draft.board.CONSENSUS_PAGE` names the one this league drafts on, and blending them
    is the mistake `_select_consensus` refuses. This loader validates and caches; it does not
    choose a page for you.
    """
    # GUARD rankings-season-list-refused: a year cannot key an archive that is not by year
    if seasons is not None:
        raise WideFrameRefused(
            f"ff_rankings is not season-partitioned, so seasons={seasons!r} cannot key a "
            f"load of it: the archive is one table of every scrape of every page. Bound it "
            f"with as_of=, and filter `scrape_date` downstream for a lower bound.")
    # /GUARD
    # GUARD unknown-rankings-page: a page FF_RANKINGS does not describe never reaches a load
    if page not in RANKINGS_PAGES:
        raise WideFrameRefused(
            f"unknown rankings page {page!r}. Known: {', '.join(RANKINGS_PAGES)}. "
            f"nflreadpy also offers 'week', whose columns are a different table -- see "
            f"RANKINGS_PAGES.")
    # /GUARD
    return load("ff_rankings", seasons=[page], cols=cols, refresh=refresh, cache=cache,
                as_of=as_of)


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

    The `cfg` half comes from `resolved_config()` and not from a `HubConfig()` built here. A
    line that names a configuration has to name the one the run had; constructing the
    defaults instead is right only while `conf/` overrides nothing that diverges from one,
    and that is a coincidence this line would print straight past. The two agree today, so
    the difference is latent -- which is why the test on this line checks it against the
    defaults rather than against this call, and goes red the day they part.
    """
    print(f"  nflverse refresh: season {season}")
    total_rows = 0
    pins: list[DataPin] = []
    for table, cols, label in (("pbp", list(PBP_COLS), "play-by-play"),
                               ("ff_opportunity", None, "ff_opportunity")):
        df = load(table, seasons=[season], cols=cols, refresh=True, cache=cache)
        # Read back from the sidecar rather than re-digesting `df` here, so what is printed is
        # what a later reader of this cache entry will compute -- one path to the number.
        #
        # A sidecar that will not read is recorded rather than skipped. Skipping it printed a
        # `data` digest over one source while the line above said two had been fetched, and
        # nothing on the line said which of the two it named.
        pin = data_pin(table, [season], cols=cols, cache=cache)
        pins.append(pin if pin is not None else UnpinnedRead(table))
        n_weeks, rows = _write_by_week(df, table, season, base)
        total_rows += rows
        print(f"    {label:<16} {rows:>7,} rows | {len(df.columns):>3} cols | "
              f"{n_weeks} week partitions")
    print(f"  wrote {total_rows:,} rows through hub.store")
    p = digests(resolved_config(), pins)
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
    try:
        return refresh(season=a.season, cache=RAW)
    except Exception as e:
        return unavailable("hub.fetch.nflverse", f"the nflverse releases for {a.season}", e)


if __name__ == "__main__":
    sys.exit(main())
