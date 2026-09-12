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

* the URL. `URL` is the raw-content path of the file at the repository's default branch,
  read off its README as of 2026-09-07 and answered on 2026-09-12.
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

    uv run python -m hub.fetch.nfeloqb --refresh
    uv run python -m hub.fetch.nfeloqb --status
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from hub.cli import unavailable
from hub.contracts import NFELOQB, ContractViolation
from hub.paths import DATA

PROG = "hub.fetch.nfeloqb"

# Assumed, see the module docstring. The repository publishes the file at its root.
URL = "https://raw.githubusercontent.com/greerreNFL/nfeloqb/main/qb_elos.csv"

# Under `data/raw/`, beside the nflverse releases, and gitignored for the same reason: it is
# someone else's data. The stamp beside it is when the file was pulled and what it hashed to.
RAW = DATA / "raw" / "nfeloqb"
FILE = "qb_elos.csv"
STAMP = "qb_elos.json"

# The source's spellings that nflverse spells differently, applied to both team columns; a
# spelling not listed passes through, and `unknown_teams` names it when it matches no
# schedule. `LAR` and `OAK` are what the live file spells (2026-09-12); `WSH` is 538's
# spelling, which the live file does not use -- it already says `WAS` -- and it is kept
# because mapping a spelling that never occurs costs nothing and a file that reverted to
# it would otherwise be a team that silently never adjusts. `OAK` was the missing one
# (#283): every Raiders game was unadjustable on both sides until it was mapped.
ABBREVIATIONS = {"WSH": "WAS", "LAR": "LA", "OAK": "LV"}

# The pytest node running right now, or nothing outside a test. Same guard as
# `hub.fetch.pool._http_get`: the suite patches the transport and should, and this is what
# holds for the test nobody remembered to patch.
PYTEST_NODE_ENV = "PYTEST_CURRENT_TEST"
LIVE_TEST_SUITE = "tests/golden/"

USER_AGENT = "football-hub/0.1 (+https://github.com/jacksonmlukas/football-hub)"

# The columns the team layer reads off one game row, per side. `_long` unpivots the
# two-sided row into one row per (team, game) under the unsuffixed names.
_SIDE = ("team", "qb", "value", "adj", "score")

STATE_SCHEMA = {"team": pl.Utf8, "qb": pl.Utf8, "qb_value": pl.Float64,
                "qb_adj": pl.Float64, "tenure": pl.Int64, "as_of": pl.Utf8}


class LiveCallRefused(Exception):
    """A test outside `tests/golden/` reached for the network."""


# --- the transport ----------------------------------------------------------------------

def _http_get(url: str) -> bytes:
    """One GET of the published file. The bytes, or whatever `urllib` raises."""
    # GUARD no-live-call-under-pytest [unit/test_fetch_nfeloqb.py]: a test outside
    # tests/golden/ never reaches the network
    node = os.environ.get(PYTEST_NODE_ENV, "")
    if node and not node.startswith(LIVE_TEST_SUITE):
        raise LiveCallRefused(
            f"{node.split(' ')[0]} would fetch {url}. Patch `_http_get`, the way "
            f"tests/unit/test_fetch_nfeloqb.py does; only {LIVE_TEST_SUITE} may reach the "
            f"network, and it is deselected by default.")
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


def state(rows: pl.DataFrame) -> pl.DataFrame:
    """Per team: the current starter, his value and adjustment on the latest row, and his
    tenure.

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
    long = _long(rows)
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


def captured_at(cache: Path | None = None) -> str | None:
    """When the cached file was pulled, or None with nothing cached."""
    _, stamp = _paths(cache)
    if not stamp.exists():
        return None
    try:
        return json.loads(stamp.read_text()).get("captured_at")
    except (OSError, ValueError):
        return None


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
    body = _http_get(URL)
    rows = parse(body)
    path, stamp = _paths(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    when = (now or datetime.now(UTC)).replace(tzinfo=None, microsecond=0)
    stamp.write_text(json.dumps({"captured_at": when.isoformat(timespec="seconds"),
                                 "url": URL, "sha256": hashlib.sha256(body).hexdigest(),
                                 "rows": rows.height}, indent=2))
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


def _serve_last_good(cache: Path | None, why: str) -> int:
    try:
        st = read_state(cache)
    except ContractViolation as exc:
        return unavailable(PROG, f"the nfeloqb ratings ({why}; and the cached file", exc)
    if st is None:
        return unavailable(PROG, "the nfeloqb ratings", RuntimeError(why))
    print(f"{PROG}: {why}; serving the last-good file pulled {captured_at(cache)}",
          file=sys.stderr)
    for line in report(st, captured=captured_at(cache)):
        print(line)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="The published nfeloqb quarterback ratings, pulled into data/raw/ and "
                    "read as a per-team state: starter, value, adjustment, tenure.")
    ap.add_argument("--refresh", action="store_true",
                    help="pull the file; on any failure serve the last-good one and say why")
    ap.add_argument("--status", action="store_true",
                    help="print the state off the cached file; reads nothing but the cache")
    ap.add_argument("--cache", type=Path, default=None,
                    help=f"the directory the file is kept in (default {RAW})")
    a = ap.parse_args(argv)

    if not a.refresh:
        try:
            st = read_state(a.cache)
        except ContractViolation as exc:
            return unavailable(PROG, "the cached nfeloqb file", exc)
        if st is None:
            return unavailable(PROG, "the nfeloqb ratings",
                               FileNotFoundError(f"nothing under {_paths(a.cache)[0]}; "
                                                 f"run --refresh"))
        for line in report(st, captured=captured_at(a.cache)):
            print(line)
        return 0

    try:
        rows = refresh(cache=a.cache)
    except ContractViolation as exc:
        return _serve_last_good(a.cache, f"the file was refused: {exc}")
    except Exception as exc:
        return _serve_last_good(a.cache, f"{type(exc).__name__}: {exc}")
    for line in report(state(rows), captured=captured_at(a.cache)):
        print(line)
    return 0


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
