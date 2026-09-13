"""The published quarterback ratings, consumed rather than refitted (#218).

`greerreNFL/nfeloqb` runs 538's quarterback-Elo method on nflfastR data and publishes
`qb_elos.csv` in 538's own schema, updated Tuesday and Thursday in season. #218's decision
was to consume it and build only the team layer, and this module is the consuming half: one
GET of the file, the contract on the columns the team layer reads, a cache under `data/raw/`
that a failed pull serves from, and a per-team **state** -- who starts, what he is worth,
the source's own adjustment for him on the latest row, and how long he has been the
starter -- of which the adjustment is all `hub.models.quarterback` needs to move a rating.

**What the first live pull confirmed, 2026-09-12, re-read under #283.** The shape the
hand-built fixture guessed is the shape the source publishes: the last 300 rows are frozen
as `tests/golden/fixtures/nfeloqb_qb_elos.json` and validated through `NFELOQB`, which now
says `verified_against_live=True`. What the pull found that the guess could not: 2,162 rows
before 1950, when the source's quarterback Elo begins, and a **twin row** for each played
game of the current week -- the source emits an Elo-only row (score and post-game Elo, no
quarterback) beside the quarterback row (starters, score, no post-game Elo). The first pull
read the twin as a game the source had not filled yet; it is the half of a pair this reader
does not read. Both kinds are dropped and counted by `parse` rather than refused, and a row
blank on *one* side loses that side alone. Field by field, what was confirmed and what the
re-read found:

* the URL. The raw-content path of the file at the repository's root, read off its README
  as of 2026-09-07 and answered on 2026-09-12. Since #271 `url()` names a commit (`COMMIT`)
  rather than the default branch, and the stamp records which; see the pin below.
* the schema. `NFELOQB` declares 538's names -- `team1`/`team2`, `qb1`/`qb2`,
  `qb1_value_pre`, `qb1_adj`, `score1` -- and the live file carries them. A rename is a
  contract refusal and the last-good file is served instead. Confirmed.
* that the coming week's games are listed with their expected starters and null scores.
  538's file did this, and it is the row the team layer wants most: it is where a backup is
  first named. Confirmed: the capture's fourteen unplayed rows (2026-09-13 and -14) every
  one name both starters and carry no score. Had nfeloqb listed only played games, the
  starter and his adjustment would have been last week's, stale by one week and said
  nowhere -- which is why this was the one to check.
* the team abbreviations. **This is the confirmation that failed.** 538 spelled Washington
  `WSH` and the Rams `LAR`, and the map carried those two. The live file spells Washington
  `WAS` already, the Rams `LAR`, and the Raiders `OAK` -- and `OAK` was unmapped, so `OAK`
  was in the state, `LV` was not, and every Raiders game was unadjustable on both sides
  with nothing downstream saying so beyond `unknown_teams`' sentence. `ABBREVIATIONS` maps
  all three now and `unknown_teams` reports empty on the capture against nflverse's 32;
  `hub.models.ratings` prints that sentence so a fourth spelling shows up as words rather
  than as a team that silently never adjusts.
* the sign and scale of `qb1_adj`. Read as 538 defined it: Elo points, positive when the
  starter is better than what the team's rolling value embeds. Confirmed on the capture:
  596 sides, median +0.9, range -125 to +50, and `qb_adj / 25` over the whole cached file
  reproduces the published 538 and nfelo magnitudes (#268).

**Degradation.** `CLAUDE.md`'s rule: a failed fetch serves last-good state rather than
erroring. The transport failing, a refused contract, a file that will not parse -- each
prints why on stderr and serves the cached file, validated again on the way out so a cache
that has drifted is refused rather than served as the field. Only a fresh clone with nothing
cached exits non-zero. Nothing here costs a credit: the file is public and unmetered.

Since #255 that policy, the stamp, the pytest network guard and the CLI's branch tree are
`hub.fetch.cached`'s, and this module is an adapter to it: the transport, the parser, the
contract and the report, plus the pin, which is this source's alone.

    uv run python -m hub.fetch.nfeloqb --refresh
    uv run python -m hub.fetch.nfeloqb --status
"""
from __future__ import annotations

import hashlib
import io
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic
from hub.contracts import NFELOQB, ContractViolation
from hub.declare import chosen
from hub.fetch import cached
from hub.fetch.cached import LIVE_TEST_SUITE, PYTEST_NODE_ENV, LiveCallRefused  # noqa: F401
from hub.paths import DATA

PROG = "hub.fetch.nfeloqb"

# The pin (#271). Two runs a week apart from the same pinned input produce the same number,
# so the URL names a *commit* of the source repository and not its default branch, and
# `PINNED_SHA256` is what the file at that commit hashes to. Advancing either is an edit to
# this file; `COMMIT` is declared `chosen`, so it is in `config_digest` and predictions
# under the new input carry a different version from predictions under the old, and a pull
# whose bytes do not match the pin is reported as a source change by `refresh`, the stamp
# and the fit's own sentence -- served, because the file validated, and never silently.
#
# `None` is unpinned: the default branch, whatever it held at pull time, which is what the
# URL always was until #271. Allowed, because a fetch must serve with zero attention, and
# said on every pull with the two values to set from the stamp it writes. The first pull
# (2026-09-12) recorded a hash and a row count and not the commit, so the pin could not be
# written from the record. Set 2026-09-13 from a `--refresh` (16,088 rows, sha256 below) and
# the source repository's `main` at that moment, checked by hashing the file at that commit.
REPO = "greerreNFL/nfeloqb"
BRANCH = "main"
COMMIT: str | None = chosen("2c95e5fc5e9aa289b2e160e5a4fef91f4a160ba8")
PINNED_SHA256: str | None = "9c7ec9f40e01621174040dd7af9ec1d8698d41e10e5d9e108db72faa776ee124"


def url(commit: str | None = None) -> str:
    """The raw-content path of the file at `commit`, or at the default branch with none.
    The repository publishes the file at its root (confirmed 2026-09-12)."""
    return f"https://raw.githubusercontent.com/{REPO}/{commit or BRANCH}/{FILE}"


FILE = "qb_elos.csv"
URL = url(COMMIT)

# Under `data/raw/`, beside the nflverse releases, and gitignored for the same reason: it is
# someone else's data. The stamp beside it is when the file was pulled and what it hashed to.
RAW = DATA / "raw" / "nfeloqb"
STAMP = "qb_elos.json"

# The source's spellings that nflverse spells differently, applied to both team columns; a
# spelling not listed passes through, and `unknown_teams` names it when it matches no
# schedule. `LAR` and `OAK` are what the live file spells (2026-09-12); `WSH` is 538's
# spelling, which the live file does not use -- it already says `WAS` -- and it is kept
# because mapping a spelling that never occurs costs nothing and a file that reverted to
# it would otherwise be a team that silently never adjusts. `OAK` was the missing one
# (#283): every Raiders game was unadjustable on both sides until it was mapped.
ABBREVIATIONS = {"WSH": "WAS", "LAR": "LA", "OAK": "LV"}

USER_AGENT = "football-hub/0.1 (+https://github.com/jacksonmlukas/football-hub)"

# The columns the team layer reads off one game row, per side. `_long` unpivots the
# two-sided row into one row per (team, game) under the unsuffixed names.
_SIDE = ("team", "qb", "value", "adj", "score")

STATE_SCHEMA = {"team": pl.Utf8, "qb": pl.Utf8, "qb_value": pl.Float64,
                "qb_adj": pl.Float64, "tenure": pl.Int64, "as_of": pl.Utf8}


# --- the transport ----------------------------------------------------------------------

def _http_get(url: str) -> bytes:
    """One GET of the published file. The bytes, or whatever `urllib` raises."""
    # GUARD no-live-call-under-pytest [unit/test_fetch_nfeloqb.py]: a test outside
    # tests/golden/ never reaches the network -- the check is `hub.fetch.cached`'s
    cached.refuse_live_call(f"would fetch {url}", patch="_http_get",
                            tests="tests/unit/test_fetch_nfeloqb.py")
    # /GUARD
    return _transport(url)


def _transport(url: str) -> bytes:                          # pragma: no cover - network
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


# --- the parser -------------------------------------------------------------------------

def parse(body: bytes) -> pl.DataFrame:
    """The file's rows, cast to the contract's dtypes and validated through it, less the
    rows with no quarterback on either side.

    **The twin-row structure the source emits.** A game the current week has played arrives
    as two rows (first pull, 2026-09-12): an Elo-only row with the score and the post-game
    Elo and nulls in every quarterback column, and a quarterback row with the starters, the
    score, and no post-game Elo. The twin names the starter; the Elo-only row is the half of
    the pair this reader does not read, dropped and counted here beside the 2,162 rows before
    1950 where the source's quarterback Elo has not begun. Unplayed games arrive as one row
    with both starters and no score, and played games of earlier weeks as one row -- the
    capture's 2025 rows carry no twins -- so the twin is how the current week's results land.
    A row blank on one side only is kept, said by team and date, and `_long` drops that side.

    Cast before the check rather than trusting inference: a short file whose values happen
    to be whole numbers infers `qb1_adj` as an integer column, and the contract would refuse
    a dtype the source never chose. A column the file does not carry is left for the
    contract to name, which is the refusal a rename should produce.
    """
    got = pl.read_csv(io.BytesIO(body), infer_schema_length=10000)
    casts = [pl.col(c).cast(t, strict=False) for c, t in NFELOQB.required.items()
             if c in got.columns and t is not pl.Utf8]
    got = got.with_columns(casts)
    if not set(_SIDE_COLUMNS) <= set(got.columns):
        return NFELOQB.validate(got)            # a rename: the contract names the column
    # Rows with no quarterback on either side are not rows this reader can use, and the live
    # file has two kinds (first pull, 2026-09-12): every game before 1950, when the source's
    # quarterback Elo begins -- 2,162 rows -- and the **Elo-only twin** of each played game
    # of the current week. The source emits two rows for those games: one carrying the score
    # and the post-game Elo and no quarterback, and one carrying the starters, the score and
    # no post-game Elo. Both twins carry the score, so the blank one is not a game the source
    # has not filled yet; it is the half of the pair this reader does not read. Neither kind
    # is a shape change, so neither is a refusal; both are dropped and counted here.
    both = pl.col("qb1").is_null() & pl.col("qb2").is_null()
    twins = got.filter(both)
    if twins.height:
        this = (twins.filter(pl.col("season") == got["season"].max()).height
                if "season" in got.columns else 0)
        print(f"  nfeloqb: dropped {twins.height} rows with no quarterback on either side "
              f"({this} of them this season, the Elo-only twin of a played game)")
        got = got.filter(~both)
    # A row blank on *one* side -- a quarterback the source has no prior for, a side it has
    # not named -- is kept, and `_long` drops that side alone (#283). The filter used to be
    # two-sided, so one team's blank dropped the opponent's game with it, and before that the
    # contract's null check refused the whole file for one such row. Said here by team and
    # date, because a starter this reader cannot see is a rating that will not move.
    half = got.filter(_blank("1") | _blank("2"))
    if half.height:
        named = [f"{r[f'team{n}']} {r['date']}" for r in half.iter_rows(named=True)
                 for n in ("1", "2") if any(r[c] is None for c in _side_columns(n))]
        print(f"  nfeloqb: {half.height} rows blank on one side; that side is dropped and the "
              f"other side's game kept: {', '.join(named)}")
    return NFELOQB.validate(got)


def _side_columns(n: str) -> tuple[str, str, str]:
    """The three columns a side must carry for the team layer to read it."""
    return f"qb{n}", f"qb{n}_value_pre", f"qb{n}_adj"


_SIDE_COLUMNS = (*_side_columns("1"), *_side_columns("2"))


def _blank(n: str) -> pl.Expr:
    """Side `n` of a row is blank: any of the three the team layer reads is null."""
    return pl.any_horizontal([pl.col(c).is_null() for c in _side_columns(n)])


def _long(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, game), every side that carries a quarterback, his value and his
    adjustment, in nflverse's spellings. A side blank in any of the three is dropped alone
    (#283); the other side of the row is a game the team layer can read."""
    sides = []
    for n in ("1", "2"):
        sides.append(rows.filter(~_blank(n)).select(
            pl.col("date"), pl.col("season"),
            pl.col(f"team{n}").replace(ABBREVIATIONS).alias("team"),
            pl.col(f"qb{n}").alias("qb"),
            pl.col(f"qb{n}_value_pre").alias("value"),
            pl.col(f"qb{n}_adj").alias("adj"),
            pl.col(f"score{n}").alias("score")))
    return pl.concat(sides).sort("team", "date")


# The source dates a row by the game's Eastern day, the way nflverse's `gameday` does;
# every kickoff and as-of in this repo is naive UTC. The two meet in `before`.
GAME_DAY_ZONE = "America/New_York"


def game_day(as_of: date | datetime) -> date:
    """The Eastern game day an as-of falls on. A `date` is taken as one already; a naive
    `datetime` is UTC, the way every moment in this repo is, and is converted -- a Thursday
    20:15 ET kickoff is 00:15 UTC Friday, and read as Friday it would keep the Thursday
    row, the game's own, on the wrong side of the split."""
    if isinstance(as_of, datetime):
        from zoneinfo import ZoneInfo
        return as_of.replace(tzinfo=UTC).astimezone(ZoneInfo(GAME_DAY_ZONE)).date()
    return as_of


def before(rows: pl.DataFrame, as_of: date | datetime) -> pl.DataFrame:
    """The split (#272): the rows dated strictly before the as-of's game day, and none
    dated on or after it. A fixture's own row is dated its game day, so a state built as of
    its kickoff is built from rows that hold nothing about that game or any later one.
    `docs/method.md` rule 2, and the only comparison against the date column here."""
    return rows.filter(pl.col("date") < game_day(as_of).isoformat())


def state(rows: pl.DataFrame, as_of: date | datetime | None = None) -> pl.DataFrame:
    """Per team: the current starter, his value and adjustment on the latest row, and his
    tenure -- as of a moment, or of the whole file.

    With `as_of`, the latest row is the latest *before* that moment's game day (`before`
    is the split), so a played fixture can be rated from what was knowable before it
    kicked off and a team with no row before the day is absent rather than served from its
    future. Without it, every row: the state as it stands, which is what every consumer
    read before #272 and what the live path still reads for the weeks ahead.

    Everything the team layer prices is on the latest row. `qb_adj` is the source's own
    adjustment there -- the gap between this starter and what the team's rolling value
    embeds, in Elo, already decayed by the source's update rule -- and
    `hub.models.quarterback.points` divides it by 25 and adds nothing (#268). `qb_value` is
    the starter as he stands now, reported and not priced.

    `tenure` is reported and not priced either. The *run* is the trailing rows of a team's
    games whose starter is the latest row's starter, and `tenure` counts the played rows
    in it: a starter named for the coming game and yet to start it has none. Until #268
    the state also carried the run's first row, and the estimator subtracted it from the
    latest and decayed the difference by `tenure` -- a quantity carrying the starter's own
    value drift, which is not the gap the module prices. Nothing reads an arrival row now.

    Rows are not filtered to a season on purpose. A run that began late last season is
    one run, and the tenure reported for it should say so.
    """
    long = _long(rows if as_of is None else before(rows, as_of))
    out = []
    for team, games in long.group_by("team", maintain_order=True):
        latest = games.row(-1, named=True)
        run = 0
        while run < games.height and games["qb"][games.height - 1 - run] == latest["qb"]:
            run += 1
        played = games.tail(run)["score"].is_not_null().sum()
        out.append({"team": team[0], "qb": latest["qb"], "qb_value": float(latest["value"]),
                    "qb_adj": float(latest["adj"]),
                    "tenure": int(played), "as_of": str(latest["date"])})
    return pl.DataFrame(out, schema=STATE_SCHEMA).sort("team")


# --- the cache --------------------------------------------------------------------------

def _paths(cache: Path | None) -> tuple[Path, Path]:
    root = Path(cache) if cache is not None else RAW
    return root / FILE, root / STAMP


def stamp(cache: Path | None = None) -> dict[str, Any]:
    """The record written beside the cached file, or an empty record with nothing cached
    or nothing readable: when it was pulled, from which URL and commit, what it hashed to,
    what the pin expected, whether the two matched, and how many rows it carried."""
    return cached.read_stamp(_paths(cache)[1])


def captured_at(cache: Path | None = None) -> str | None:
    """When the cached file was pulled, or None with nothing cached."""
    return stamp(cache).get("captured_at")


def source_change(cache: Path | None = None) -> str | None:
    """One sentence if the cached file's bytes did not match the pin when it was pulled,
    else None. Read by `hub.models.ratings.quarterback_rows`, so a fit run days after the
    pull repeats what the pull said rather than serving the change silently."""
    st = stamp(cache)
    if st.get("matches_pin") is not False:
        return None
    return (f"source change: the nfeloqb file pulled {st.get('captured_at')} at commit "
            f"{str(st.get('commit'))[:12]} hashed {str(st.get('sha256'))[:12]}, and the pin "
            f"expects {str(st.get('pinned_sha256'))[:12]}")


def read_rows(cache: Path | None = None) -> pl.DataFrame | None:
    """The cached file, validated on the way out; None on a fresh clone.

    Through the contract rather than trusted, because a cached file that has drifted from
    the declared shape is the one thing worse than no cache: it would move every rating
    without a live price.
    """
    path, _ = _paths(cache)
    if not path.exists():
        return None
    return parse(path.read_bytes())


def read_state(cache: Path | None = None) -> pl.DataFrame | None:
    """The per-team state off the cached file, or None with nothing cached."""
    rows = read_rows(cache)
    return None if rows is None else state(rows)


def refresh(*, cache: Path | None = None, now: datetime | None = None) -> pl.DataFrame:
    """One pull, validated, written, returned. Raises rather than degrading: the entry
    point is what serves last-good, and says which failure it is serving over."""
    where = url(COMMIT)
    body = _http_get(where)
    rows = parse(body)
    path, record = _paths(cache)
    atomic.write_bytes(path, body)
    when = (now or datetime.now(UTC)).replace(tzinfo=None, microsecond=0)
    sha = hashlib.sha256(body).hexdigest()
    # Three answers, and the stamp carries which: matched the pin, did not, or there was no
    # pin to match. The second is a source change and is said here and again by every
    # reader of the stamp; the third is said with the two values that would end it.
    matches = None if PINNED_SHA256 is None else sha == PINNED_SHA256
    cached.write_stamp(record, when.isoformat(timespec="seconds"), url=where, commit=COMMIT,
                       sha256=sha, pinned_sha256=PINNED_SHA256, matches_pin=matches,
                       rows=rows.height)
    if matches is False:
        print(f"{PROG}: {source_change(cache)}; served, because it validated -- advance "
              f"PINNED_SHA256 deliberately, or restore COMMIT", file=sys.stderr)
    elif COMMIT is None:
        print(f"{PROG}: unpinned: pulled {BRANCH} as it stood at {when.isoformat()}, "
              f"sha256 {sha[:12]}. To pin it set COMMIT to the commit {BRANCH} was at and "
              f"PINNED_SHA256 to {sha}", file=sys.stderr)
    return rows


# --- the entry point --------------------------------------------------------------------

def report(st: pl.DataFrame, *, captured: str | None = None) -> list[str]:
    lines = [f"  nfeloqb: {st.height} teams, latest game {st['as_of'].max()}"
             + (f" (pulled {captured})" if captured else "")]
    for r in st.iter_rows(named=True):
        lines.append(f"  {r['team']:<4} {r['qb']:<22} value {r['qb_value']:6.1f}  "
                     f"adj {r['qb_adj']:+7.1f}  tenure {r['tenure']:>3}")
    return lines


def unknown_teams(st: pl.DataFrame, games: pl.DataFrame) -> list[str]:
    """Teams the state carries that the season's games do not: a spelling `ABBREVIATIONS`
    does not map, which would otherwise be a team that silently never adjusts. Read by
    `hub.models.ratings.rated_games`, which has the games in hand and this module does not."""
    known = set(games["home_team"].to_list()) | set(games["away_team"].to_list())
    return sorted(set(st["team"].to_list()) - known)


def missing_teams(st: pl.DataFrame, games: pl.DataFrame) -> list[str]:
    """The mirror of `unknown_teams`: teams the season's games carry that the state does
    not -- a team the source has no usable row for -- whose games therefore never adjust.
    Silently, until `hub.models.ratings.rated_games` began printing this beside the other."""
    known = set(games["home_team"].to_list()) | set(games["away_team"].to_list())
    return sorted(known - set(st["team"].to_list()))


def _describe(exc: BaseException) -> str:
    """How a failed pull is named in the last-good sentence."""
    if isinstance(exc, ContractViolation):
        return f"the file was refused: {exc}"
    return f"{type(exc).__name__}: {exc}"


# The adapter (#255): what this source supplies to `hub.fetch.cached`, which owns the
# last-good policy and the CLI. The nouns are the ones the sentences carried before.
ADAPTER: cached.Adapter[pl.DataFrame] = cached.Adapter(
    prog=PROG,
    description="The published nfeloqb quarterback ratings, pulled into data/raw/ and "
                "read as a per-team state: starter, value, adjustment, tenure.",
    what="the nfeloqb ratings",
    cached="the cached file",
    kept="the last-good file pulled",
    cache_flag="--cache",
    cache_help=f"the directory the file is kept in (default {RAW})",
    refresh_help="pull the file; on any failure serve the last-good one and say why",
    status_help="print the state off the cached file; reads nothing but the cache",
    refresh_hint="run --refresh",
    cache_path=lambda cache: _paths(cache)[0],
    refresh=lambda ns: state(refresh(cache=ns.cache)),
    read=read_state,
    captured_at=captured_at,
    report=report,
    describe=_describe,
)


def main(argv: Sequence[str] | None = None) -> int:
    return cached.run(ADAPTER, argv)


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
