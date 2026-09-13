"""The survivor pool's state, read off the pool host instead of retyped each week (#85).

`hub.season.pool` prices a pick against the field: how many entries are live, what the pot
is, and which teams each entry has already spent -- its **Ledger**, in `CONTEXT.md`'s sense,
one per entry. Until this module those three were typed on the command line from a browser
tab. This reads them from the host's JSON resource, replaces every member's identity with an
index, and keeps the last state it read under `data/processed/` so a host that is down on a
Sunday serves last week's field rather than no field.

**The payload shape is assumed, not documented.** Nothing in this repository records what
the host returns. The money-layer plan, U6 of
`docs/plans/2026-09-05-001-feat-survivor-money-layer-plan.md`, established only that the
resource is JSON behind a session cookie, answers 401 unauthenticated, and reveals each
week's picks after that week's deadline under Hidden Picks. #77 landed the environment
name, `POOL_SESSION`, ahead of this module so the secret scan could see the credential
before it existed. So the shape below is a guess written down in one place, held by
`tests/golden/fixtures/pool_payload.synthetic.json`, and the contract on what is stored
says `verified_against_live=False` for that reason. The first live run confirms or corrects
it, and a payload that does not match is *refused* -- see `parse_payload` -- never read
halfway.

    {"pool":    {"season": 2026, "current_week": 3, "field_size": 21, "pot": 420.0},
     "weeks":   [{"week": 1, "status": "final"}, ..., {"week": 3, "status": "open"}],
     "entries": [{"id": "...", "name": "...", "alive": true,
                  "picks": [{"week": 1, "team": "DAL", "result": "win"}, ...]}, ...]}

What the first live run must confirm, field by field: the four keys under `pool` and
whether `field_size` counts entries or people; that `weeks[].status` spells a completed
week as one of `FINAL_STATUSES`; that `entries[].picks[].team` is a nflverse abbreviation
rather than a full name, since a Ledger is compared against the board's `team` column and
no mapping is attempted here; that an entry's `alive` is a boolean; and the name the host's
cookie carries, which `POOL_COOKIE_NAME` sets and `scripts/preflight_public.sh` says it
cannot scan for until it is known. The URL is `POOL_URL`, read from the environment for the
same reason the cookie is: the host is a small operator's site, and naming it in a public
repository is not this module's decision to make.

**Which picks are spent.** A pick counts toward an entry's Ledger when its week is final and
not before. Under Hidden Picks a rival's current pick is absent from the payload anyway; our
own is present, and counting it would spend a team the week has not yet resolved. The line is
the week's status, not its number.

**Who is who.** Other members are real people, and this repository publishes artifacts. So
the parser numbers entries and drops the id and the name before anything is returned -- the
`PoolState` has no field that could hold either, the store never sees one, and the Ledger is
addressed by index. Two things make the index mean something. **Ours is 0**, always:
`hub.season.pool.Field.outcome` reads our Ledger at index 0, so our own host id is named in
`POOL_ENTRY_ID` and a payload without it is refused before anything is written -- whichever
id happened to sort first would otherwise be "us". **And an index is never reused.** A
member who drops out of a later payload would shift every index after theirs, and nothing
downstream could tell; so the store keeps an append-only map under `pool_entries.json`
from `sha256(host id)` to index, read before any index is assigned, with a newcomer taking
the next free one. The key is a hash of an opaque id, not a member's name: it identifies
nobody outside this store and is kept only so the store agrees with itself from one week
to the next.

**The cookie goes to the host and nowhere else.** It is read through `dotenv` the way every
credential here is, sent as the one cookie on the request, and scrubbed from every message
this module raises or prints: a transport exception quotes its own request, header and all,
and a 401 body may echo what it refused. `redact` is applied on both paths, and the chained
cause is dropped on purpose, because a traceback would print it.

**Every read is kept, and a decision names the one it was priced against** (#280). The
state file is the last-known state and is overwritten on every refresh; the archive under
`pool_state/league=nfl/season=/week=/snap-<moment>.parquet` -- the lines archive's layout --
keeps each read under the host's current week with its capture time, append-only, holding
exactly the rows the contract validated. `state_digest` names a read by what it says, the
decision journal carries that digest on the row (`journal.pool_state_digest`), and
`archived_state` resolves it back to the field as it stood, so a week can be re-derived
against the field it was priced against rather than the one the host reports now. And
the Ledger has one reading: `hub.season.survivor.prior_rows` takes our entry's `used` from
the last-known state beside the published plan's rows, and the money layer, the survivor
CLI and the published artifact all read that.

**Degradation.** `CLAUDE.md`'s rule: a failed fetch serves last-good state rather than
erroring. Every way a refresh can fail -- no host named, no cookie, the host unreachable, the
cookie refused, a payload that does not parse, a pool with nobody in it -- prints why on
stderr and then serves the last-known state, and only a fresh clone with nothing to serve
exits non-zero. `--payload FILE` ingests a payload saved from the browser with no network and
no cookie, which is how the first live run can be done by hand before the credential is
trusted to a runner.

Since #255 that policy, the stamp, the pytest network guard and the CLI's branch tree are
`hub.fetch.cached`'s, and this module is an adapter to it: the transport with its cookie,
the parser, the contract and the report, plus the index map and the archive, which are
this source's alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import quote, unquote

import polars as pl

from hub import atomic, store
from hub.cli import unavailable
from hub.config import SEASON_AHEAD
from hub.contracts import POOL_STATE, ContractViolation
from hub.fetch import cached
from hub.fetch.cached import LIVE_TEST_SUITE, PYTEST_NODE_ENV, LiveCallRefused  # noqa: F401
from hub.paths import PROCESSED

PROG = "hub.fetch.pool"

# The four environment names. `POOL_SESSION` is the one `scripts/preflight_public.sh`
# scans for and was chosen there (#77); `POOL_ENTRY_ID` -- our own entry on the host, the
# one that is index 0 -- has its own pattern there since this module named it; the other
# two carry no secret.
SESSION_ENV = "POOL_SESSION"
URL_ENV = "POOL_URL"
ENTRY_ENV = "POOL_ENTRY_ID"
COOKIE_NAME_ENV = "POOL_COOKIE_NAME"
DEFAULT_COOKIE_NAME = "session"

# What a scrubbed message says where the cookie was. Names the variable so a reader knows
# what was there without being shown it.
REDACTED = f"<{SESSION_ENV}>"

# The statuses that mean a week's picks are spent. Assumed; the first live run confirms the
# spelling. Anything else -- `open`, `locked`, `in_progress`, a spelling not listed -- is a
# week whose picks contribute nothing yet.
FINAL_STATUSES = frozenset({"final", "complete", "completed"})

STATE_FILE = "pool_state.json"
# The append-only map from `sha256(host id)` to index, beside the state. Never rewritten
# with a row missing, never with a row moved: see `write_index_map`.
INDEX_FILE = "pool_entries.json"
# Every read, kept (#280): a dated partition per read under the host's current week, in the
# store's Hive layout beside the lines archive. `write_state` appends one; `archived` reads
# them back; `state_digest` is how a journal row names one.
ARCHIVE_TABLE = "pool_state"
LEAGUE = "nfl"
# Ours, by construction, and `hub.season.pool.Field.outcome` reads it there.
OUR_INDEX = 0

_SHAPE_NOTE = (" The payload shape is assumed rather than documented -- see the module "
               "docstring of hub.fetch.pool for what the first live run must confirm.")


class AuthFailure(Exception):
    """The host refused the session cookie. Not an empty pool, and not the host being down."""


class EmptyPool(Exception):
    """The host answered, the cookie was accepted, and the pool has no entries in it."""


class HostUnreachable(OSError):
    """The reach for the host failed. Its message has already been scrubbed of the cookie."""


class Entry(NamedTuple):
    """One entry in the pool, by index. Deliberately no field for who it is."""
    index: int
    alive: bool
    used: tuple[str, ...]        # its Ledger: the teams spent in weeks that are final, sorted


class PoolState(NamedTuple):
    season: int
    week: int                    # the week the state is as of: the host's current week
    field_size: int
    pot: float
    entries: tuple[Entry, ...]   # in index order, so `entries[i].index == i`

    @property
    def alive(self) -> int:
        return sum(1 for e in self.entries if e.alive)


# --- the environment --------------------------------------------------------------------

def _cookie() -> str | None:
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get(SESSION_ENV) or None


def _url() -> str | None:
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get(URL_ENV) or None


def _entry_id() -> str | None:
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get(ENTRY_ENV) or None


def _cookie_name() -> str:
    return os.environ.get(COOKIE_NAME_ENV) or DEFAULT_COOKIE_NAME


def redact(text: str, cookie: str | None) -> str:
    """`text` with every spelling of the cookie replaced by `REDACTED`.

    Three spellings, because a session cookie is routinely url-encoded and a message may
    quote it either way: as sent, decoded, and encoded again. A cookie under eight
    characters is not scrubbed by substring, since that would eat ordinary words; nothing
    that short is a session.
    """
    if not cookie or len(cookie) < 8:
        return text
    for form in {cookie, unquote(cookie), quote(cookie, safe="")}:
        if len(form) >= 8:
            text = text.replace(form, REDACTED)
    return text


# --- the transport ----------------------------------------------------------------------

def _http_get(url: str, cookie: str) -> tuple[int, Any]:
    """One GET of the pool resource with the session cookie. Status and decoded body.

    A 401 or 403 returns its status and no body rather than raising, so the caller can name
    it as an auth failure; every other non-2xx raises. The body of a refusal is never
    returned: it is the one place the host might echo the cookie back.
    """
    # GUARD no-live-call-under-pytest [unit/test_fetch_pool.py]: a test outside
    # tests/golden/ never reaches the host -- the check is `hub.fetch.cached`'s
    cached.refuse_live_call("would reach the pool host with the session cookie",
                            patch="_http_get", tests="tests/unit/test_fetch_pool.py")
    # /GUARD
    return _transport(url, cookie)


def _transport(url: str, cookie: str) -> tuple[int, Any]:     # pragma: no cover - network
    import requests
    r = requests.get(url, timeout=30, cookies={_cookie_name(): cookie},
                     headers={"Accept": "application/json"})
    if r.status_code in (401, 403):
        return r.status_code, None
    r.raise_for_status()
    return r.status_code, r.json()


# --- the parser -------------------------------------------------------------------------

def _need(m: Any, key: str, where: str, kinds: tuple[type, ...]) -> Any:
    """`m[key]`, or a contract violation naming the field that is not there.

    A field that is absent, null, or of the wrong kind is one refusal: each would otherwise
    become a partial state one step later, and the criterion is that none of them does. `bool`
    is excluded from a numeric field explicitly, because Python says it is an `int`.
    """
    if not isinstance(m, Mapping) or key not in m or m[key] is None:
        raise ContractViolation(f"pool payload: {where} has no {key!r}." + _SHAPE_NOTE)
    v = m[key]
    if not isinstance(v, kinds) or (isinstance(v, bool) and bool not in kinds):
        want = "/".join(k.__name__ for k in kinds)
        raise ContractViolation(
            f"pool payload: {where}.{key} is {type(v).__name__}, expected {want}." + _SHAPE_NOTE)
    return v


def entry_key(ident: str) -> str:
    """The store's name for a host id: its sha256, which identifies nobody outside the store."""
    return hashlib.sha256(ident.encode()).hexdigest()


def assign_indices(idents: Sequence[str], *, ours: str,
                   index_map: Mapping[str, int] | None = None) -> dict[str, int]:
    """Every id in the payload mapped to its index: the one it already holds, or the next
    free one. Ours is `OUR_INDEX`, and a map that says otherwise -- somebody else at 0, or
    our id elsewhere -- was built under another `POOL_ENTRY_ID` and is refused rather than
    renumbered around. Newcomers are numbered in id order so two reads agree."""
    if ours not in idents:
        raise ContractViolation(
            f"pool payload: the entry named by {ENTRY_ENV} is not among the "
            f"{len(idents)} entries listed, so nothing can be index {OUR_INDEX}. Check the "
            f"id in .env against the host; nothing has been written.")
    out = dict(index_map or {})
    ours_key = entry_key(ours)
    holder = next((k for k, v in out.items() if v == OUR_INDEX), None)
    if (holder is not None and holder != ours_key) or out.get(ours_key, OUR_INDEX) != OUR_INDEX:
        raise ContractViolation(
            f"pool payload: the entry map under {INDEX_FILE} does not have {ENTRY_ENV}'s "
            f"entry at index {OUR_INDEX}; it was built for another entry id. Move the map "
            f"aside rather than renumbering the field around it.")
    out.setdefault(ours_key, OUR_INDEX)
    free = max(out.values()) + 1
    for ident in sorted(set(idents)):
        key = entry_key(ident)
        if key not in out:
            out[key] = free
            free += 1
    return out


def parse_payload(payload: Any, *, ours: str, season: int | None = None,
                  index_map: Mapping[str, int] | None = None
                  ) -> tuple[PoolState, dict[str, int]]:
    """The host's payload as a `PoolState` and the index map that numbers it -- or a
    `ContractViolation` and nothing.

    `ours` is our own host id and lands at `OUR_INDEX`; `index_map` is what the store
    already holds, and the map returned is that plus every newcomer, for `write_index_map`.

    Every field the state needs is required, and the refusals a reader is most likely to
    meet are the ones that would otherwise read as a smaller, emptier pool: an entry with no
    `picks` key is not an entry that has spent nothing, a `field_size` that disagrees with
    the entries listed is a truncated payload and not a smaller field, and a pick in a week
    the payload does not describe cannot be placed either side of the final/open line.

    `season`, when given, must match the payload's own: a stale URL pointing at last year's
    pool would otherwise write last year's field as this year's state.
    """
    pool = _need(payload, "pool", "the payload", (Mapping,))
    got_season = _need(pool, "season", "pool", (int,))
    if season is not None and got_season != season:
        raise ContractViolation(
            f"pool payload: the payload is season {got_season}, asked for {season}.")
    week = _need(pool, "current_week", "pool", (int,))
    field_size = _need(pool, "field_size", "pool", (int,))
    pot = float(_need(pool, "pot", "pool", (int, float)))

    statuses: dict[int, str] = {}
    for i, w in enumerate(_need(payload, "weeks", "the payload", (list,))):
        wk = _need(w, "week", f"weeks[{i}]", (int,))
        if wk in statuses:
            raise ContractViolation(f"pool payload: week {wk} is described twice.")
        statuses[wk] = _need(w, "status", f"weeks[{i}]", (str,)).strip().lower()

    raw = _need(payload, "entries", "the payload", (list,))
    if field_size != len(raw):
        raise ContractViolation(
            f"pool payload: field_size is {field_size} and {len(raw)} entries are listed. "
            f"A field smaller than its size is a truncated or paginated payload, and reading "
            f"it as the field would price the pick against people who are not there.")
    if not raw:
        raise EmptyPool("the pool host answered with no entries in the pool")

    keyed: list[tuple[str, bool, set[str]]] = []
    for i, e in enumerate(raw):
        where = f"entries[{i}]"
        ident = str(_need(e, "id", where, (str, int)))
        alive = _need(e, "alive", where, (bool,))
        used: set[str] = set()
        for j, p in enumerate(_need(e, "picks", where, (list,))):
            wk = _need(p, "week", f"{where}.picks[{j}]", (int,))
            if wk not in statuses:
                raise ContractViolation(
                    f"pool payload: {where}.picks[{j}] names week {wk}, which the payload "
                    f"does not describe, so it cannot be told final from open.")
            team = _need(p, "team", f"{where}.picks[{j}]", (str,)).strip().upper()
            if not team:
                raise ContractViolation(
                    f"pool payload: {where}.picks[{j}].team is empty in week {wk}.")
            # GUARD in-progress-week-contributes-nothing [unit/test_fetch_pool.py]: a pick
            # in a week that is not final is not spent, ours included -- the line is the
            # week's status, not its number
            if statuses[wk] not in FINAL_STATUSES:
                continue
            # /GUARD
            used.add(team)
        keyed.append((ident, alive, used))

    idents = [k[0] for k in keyed]
    if len(set(idents)) != len(idents):
        raise ContractViolation("pool payload: an entry id is listed twice.")
    # The map decides the number, then the id is dropped: from here an entry is its index
    # and nothing else. Index order, so ours is first and the store reads back the same.
    numbered = assign_indices(idents, ours=ours, index_map=index_map)
    entries = tuple(sorted(
        (Entry(index=numbered[entry_key(ident)], alive=alive, used=tuple(sorted(used)))
         for ident, alive, used in keyed), key=lambda e: e.index))
    return PoolState(season=got_season, week=week, field_size=field_size, pot=pot,
                     entries=entries), numbered


def ledgers(state: PoolState) -> list[set[str]]:
    """Every entry's Ledger in index order, ours first -- the `ledgers` that
    `hub.season.pool.Field.outcome` takes, one per entry present, positional gaps closed."""
    return [set(e.used) for e in state.entries]


# --- the store --------------------------------------------------------------------------

_SCHEMA = {"entry": pl.Int64, "alive": pl.Boolean, "used": pl.List(pl.Utf8),
           "season": pl.Int64, "week": pl.Int64, "field_size": pl.Int64, "pot": pl.Float64}


def to_frame(state: PoolState) -> pl.DataFrame:
    """The state as the flat frame `POOL_STATE` declares, dtypes stated by the writer."""
    return pl.DataFrame({
        "entry": [e.index for e in state.entries],
        "alive": [e.alive for e in state.entries],
        "used": [list(e.used) for e in state.entries],
        "season": [state.season] * len(state.entries),
        "week": [state.week] * len(state.entries),
        "field_size": [state.field_size] * len(state.entries),
        "pot": [state.pot] * len(state.entries),
    }, schema=_SCHEMA)


def _from_frame(df: pl.DataFrame) -> PoolState:
    rows = df.sort("entry").to_dicts()
    first = rows[0]
    return PoolState(
        season=int(first["season"]), week=int(first["week"]),
        field_size=int(first["field_size"]), pot=float(first["pot"]),
        entries=tuple(Entry(index=int(r["entry"]), alive=bool(r["alive"]),
                            used=tuple(r["used"])) for r in rows))


def state_path(base: Path | None = None) -> Path:
    return Path(base or PROCESSED) / STATE_FILE


def index_map_path(base: Path | None = None) -> Path:
    return Path(base or PROCESSED) / INDEX_FILE


def read_index_map(base: Path | None = None) -> dict[str, int]:
    """The map as stored, or empty on a fresh clone."""
    path = index_map_path(base)
    if not path.exists():
        return {}
    return {str(k): int(v) for k, v in json.loads(path.read_text()).items()}


def write_index_map(index_map: Mapping[str, int], base: Path | None = None) -> Path:
    """Append-only. A row already stored keeps its index whatever `index_map` says: a
    write that lost a row would hand the next newcomer a rival's Ledger, and one that
    moved a row would hand a rival ours. A row that disagrees is refused, not overwritten."""
    stored = read_index_map(base)
    moved = {k for k, v in index_map.items() if k in stored and stored[k] != v}
    if moved:
        raise ContractViolation(
            f"{INDEX_FILE}: {len(moved)} entries are already at another index; the map is "
            f"append-only and an index is never reassigned.")
    merged = {**stored, **{k: int(v) for k, v in index_map.items()}}
    path = index_map_path(base)
    atomic.write_text(path, json.dumps(dict(sorted(merged.items(), key=lambda kv: kv[1])),
                                       indent=2))
    return path


def write_state(state: PoolState, base: Path | None = None, *,
                when: datetime | None = None) -> Path:
    """Validate, then write. The file holds what `validate` handed back and nothing else.

    **And the archive keeps every read** (#280). The state file is the last-known state and
    is overwritten every refresh -- one capture time, no history -- so the field a week-3
    decision was priced against was gone the moment week 4 was read. Each write now also
    lands a dated partition under `ARCHIVE_TABLE`, in the layout the lines archive uses
    (`hub.store.write` under `pool_state/league=nfl/season=/week=/snap-<moment>`), holding
    the validated rows and the capture time. Append-only by the store's own rule: a partition
    is named by its moment and a differing rewrite of one is refused, never destroyed. The
    week is the host's current week, so a week has as many partitions as it had reads --
    and there is no retention: every `--refresh` adds one, as every odds poll adds a
    `lines` partition, and nothing prunes either. A partition here is a few rows, and the
    field a decision was priced against is exactly what a later pruning would lose.

    The privacy design is unchanged: the rows archived are exactly the rows validated --
    index, liveness, Ledger -- and no id or name exists on the state to be written.
    """
    df = POOL_STATE.validate(to_frame(state))
    when = when or datetime.now(UTC)
    path = state_path(base)
    # The partition before the file: a refused archive write leaves the last-known state as
    # it was, where the other order would serve a state the archive never recorded. Named
    # to the microsecond where the lines archive names to the second, because two reads a
    # second apart -- a saved payload ingested and then a refresh -- are two reads, and the
    # second would otherwise be refused as a rewrite of the first.
    moment = when.astimezone(UTC).replace(tzinfo=None)
    # The season and week are the partition's path, as the lines archive has them; the
    # rest of the validated frame is the file.
    store.write(df.drop("season", "week").with_columns(pl.lit(moment).alias("captured_at")),
                ARCHIVE_TABLE, LEAGUE, state.season, state.week, base=base,
                name=f"snap-{moment:%Y%m%dT%H%M%S%f}")
    return cached.write_stamp(
        path, when.isoformat(timespec="seconds"),
        season=state.season, week=state.week, field_size=state.field_size, pot=state.pot,
        entries=df.select("entry", "alive", "used").to_dicts())


def state_digest(state: PoolState) -> str:
    """Stable 8-char hash of what the state says: the field a decision was priced against.

    Over the validated frame as text, the way `hub.season.pool.grid_digest` hashes a board
    and for the same reason -- a frame and the frame read back from disk serialise to
    different IPC bytes and the same text. The capture time is not in it: two reads that
    found the identical field are the identical field, and a journal row naming this digest
    is re-run against that field whichever read is resolved to.
    """
    canon = to_frame(state).sort("entry").with_columns(pl.col("used").list.join(","))
    head = ",".join(f"{c}:{canon.schema[c]}" for c in canon.columns)
    return hashlib.sha256((head + "\n").encode()
                          + canon.write_csv().encode()).hexdigest()[:8]


def archived(season: int, *, week: int | None = None,
             base: Path | None = None) -> list[tuple[datetime, PoolState]]:
    """Every read archived for `season`, oldest first, as (captured at, state).

    Read through the store's catalog rather than by walking the tree, so the partition
    layout is the store's business; through the contract on the way out, as `read_state`
    is, so a drifted partition is refused rather than served as a field. Empty on a fresh
    clone or a season nothing has read.
    """
    if ARCHIVE_TABLE not in store.tables(base):
        return []
    q = f"SELECT * FROM {ARCHIVE_TABLE} WHERE league = ? AND season = ?"
    params: list[object] = [LEAGUE, season]
    if week is not None:
        q += " AND week = ?"
        params.append(week)
    rows = store.sql(q, params=params, base=base)
    if rows.is_empty():
        return []
    out = []
    for (when,), part in sorted(rows.group_by("captured_at"), key=lambda kv: kv[0]):
        df = POOL_STATE.validate(
            part.with_columns(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64))
            .select(list(_SCHEMA)))
        out.append((when, _from_frame(df)))
    return out


def archived_state(digest: str, *, season: int, base: Path | None = None) -> PoolState | None:
    """The archived field a journal row's `pool_state_digest` names, or None if none matches.

    The earliest read carrying the digest, which is the one the decision could have been
    priced against; later reads that found the same field are the same field.
    """
    return next((s for _, s in archived(season, base=base) if state_digest(s) == digest),
                None)


def _read_doc(base: Path | None) -> dict[str, Any] | None:
    """The state file as written, None on a fresh clone, and an empty record where the
    file will not parse -- which the contract then refuses by name, as it refuses a file
    that parses to the wrong shape, rather than the read handing back a traceback."""
    path = state_path(base)
    if not path.exists():
        return None
    return cached.read_stamp(path)


def captured_at(base: Path | None = None) -> str | None:
    """When the last-known state was read from the host, or None with no state."""
    doc = _read_doc(base)
    return None if doc is None else doc.get("captured_at")


def read_state(base: Path | None = None) -> PoolState | None:
    """The last-known state, validated on the way out; None on a fresh clone.

    Through the contract rather than trusted, because a cached file that has drifted from
    the declared shape is the one thing worse than no cache: it would be served as the field.
    """
    doc = _read_doc(base)
    if doc is None:
        return None
    rows = [{**{k: doc.get(k) for k in ("season", "week", "field_size", "pot")}, **e}
            for e in doc.get("entries", [])]
    try:
        frame = pl.DataFrame(rows, schema=_SCHEMA)
    except Exception as exc:
        # A field of the wrong kind -- `"alive": "yes"` -- fails the frame's construction
        # before the contract sees it, and polars' error is not a `ContractViolation`, so it
        # used to reach the operator as a traceback out of `--status` and `--refresh` alike.
        # It is the same fact the contract states about a drifted cache, and it is refused
        # in the contract's words: unavailable, beside the failure it would have covered.
        raise ContractViolation(
            f"{POOL_STATE.name}: {STATE_FILE} does not hold the declared shape "
            f"({type(exc).__name__}: {exc})") from exc
    df = POOL_STATE.validate(frame)
    return _from_frame(df)


# --- the refresh ------------------------------------------------------------------------

def refresh(*, season: int = SEASON_AHEAD, store: Path | None = None,
            now: datetime | None = None) -> PoolState:
    """One read of the host, parsed, written, returned. Raises rather than degrading: the
    entry point is what serves last-known state, and says which failure it is serving over.

    `AuthFailure` for a refused cookie, `EmptyPool` for a pool with nobody in it,
    `HostUnreachable` for a transport that failed, `ContractViolation` for a payload that
    does not parse or does not list our entry, `RuntimeError` for an environment with no
    host, no entry id or no cookie named. The cookie appears in none of them.
    """
    url = _url()
    if not url:
        raise RuntimeError(f"no {URL_ENV} set; the pool host's resource URL goes in .env")
    ours = _entry_id()
    if not ours:
        raise RuntimeError(f"no {ENTRY_ENV} set; our own entry's id on the host goes in .env")
    cookie = _cookie()
    if not cookie:
        raise RuntimeError(f"no {SESSION_ENV} set; the pool host's session cookie goes in .env")
    try:
        status, body = _http_get(url, cookie)
    except LiveCallRefused:
        raise
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        # GUARD cookie-never-printed [unit/test_fetch_pool.py]: a transport error quotes
        # its own request, cookie header and all, and is scrubbed before it becomes a
        # message anybody prints
        text = redact(text, cookie)
        # /GUARD
        # `from None`: the chained cause is the unscrubbed exception, and a traceback prints
        # the chain.
        raise HostUnreachable(text) from None
    if status in (401, 403):
        raise AuthFailure(
            f"the pool host answered HTTP {status}: the session cookie in {SESSION_ENV} was "
            f"refused. Copy a fresh one from the browser into .env.")
    state, index_map = parse_payload(body, ours=ours, season=season,
                                     index_map=read_index_map(store))
    write_index_map(index_map, store)
    write_state(state, store, when=now)
    return state


# --- the entry point --------------------------------------------------------------------

def report(state: PoolState, *, captured: str | None = None) -> list[str]:
    lines = [f"  survivor pool, {state.season} as of week {state.week}: {state.field_size} "
             f"entries, {state.alive} alive, pot ${state.pot:.2f}"
             + (f" (read {captured})" if captured else "")]
    for e in state.entries:
        spent = ", ".join(e.used) if e.used else "nothing spent"
        who = " (ours)" if e.index == OUR_INDEX else ""
        lines.append(f"  entry {e.index}{who}: {'alive' if e.alive else 'out'}, {spent}")
    # What the money layer reads from this state by default (#280); stated here so an
    # operator overriding it knows what the figures were read as.
    lines.append(f"  hub.season.pool reads this field: --entries {state.alive} "
                 f"--pot {state.pot:.2f}, state {state_digest(state)}")
    return lines


def _describe(exc: BaseException) -> str:
    """How a failed refresh is named in the last-known sentence. `EmptyPool` and
    `AuthFailure` are their own sentences; a refused payload says so; everything else --
    `HostUnreachable`, the unset-environment cases, anything the reach can raise -- is
    scrubbed again here, so no path prints what a message carries."""
    if isinstance(exc, (EmptyPool, AuthFailure)):
        return str(exc)
    if isinstance(exc, ContractViolation):
        return f"the payload was refused: {exc}"
    return redact(f"{type(exc).__name__}: {exc}", _cookie())


def _arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--payload", metavar="FILE", default=None,
                    help="ingest a payload saved from the browser; no network, no cookie")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)


def _ingest(ns: argparse.Namespace) -> int | None:
    """The `--payload FILE` branch, this source's alone: a payload saved from the browser,
    read from disk, no network and no cookie. None when the flag is absent."""
    if not ns.payload:
        return None
    try:
        ours = _entry_id()
        if not ours:
            raise RuntimeError(f"no {ENTRY_ENV} set; our own entry's id goes in .env")
        state, index_map = parse_payload(json.loads(Path(ns.payload).read_text()),
                                         ours=ours, season=ns.season,
                                         index_map=read_index_map(ns.cache))
        write_index_map(index_map, ns.cache)
        write_state(state, ns.cache)
    except Exception as exc:
        return unavailable(PROG, f"the saved payload {ns.payload}", exc)
    for line in report(state):
        print(line)
    return 0


# The adapter (#255): what this source supplies to `hub.fetch.cached`, which owns the
# last-known policy and the CLI. The nouns are the ones the sentences carried before; the
# report after a live read carries no capture time, as it did not.
ADAPTER: cached.Adapter[PoolState] = cached.Adapter(
    prog=PROG,
    description="The survivor pool's field, pot and every entry's spent teams, read "
                "from the pool host and cached under data/processed/. Members are "
                "indexed, never named.",
    what="the pool host's state",
    cached="the cached state",
    kept="the last-known pool state read",
    cache_flag="--store",
    cache_help="the processed store the state is written to and read from",
    refresh_help=f"read the host with the cookie in {SESSION_ENV}; on any failure "
                 f"serve the last-known state and say why",
    status_help="print the last-known state; reads nothing but the store",
    refresh_hint=f"run --refresh with {SESSION_ENV} set",
    cache_path=state_path,
    refresh=lambda ns: refresh(season=ns.season, store=ns.cache),
    read=read_state,
    captured_at=captured_at,
    report=report,
    describe=_describe,
    stamped_after_refresh=False,
    arguments=_arguments,
    before=_ingest,
)


def main(argv: Sequence[str] | None = None) -> int:
    return cached.run(ADAPTER, argv)


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
