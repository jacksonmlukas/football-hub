"""The published quarterback ratings, consumed rather than refitted (#218).

`greerreNFL/nfeloqb` runs 538's quarterback-Elo method on nflfastR data and publishes
`qb_elos.csv` in 538's own schema, updated Tuesday and Thursday in season. #218's decision
was to consume it and build only the team layer, and this module is the consuming half: one
GET of the file, the contract on the columns the team layer reads, a cache under `data/raw/`
that a failed pull serves from, and a per-team **state** -- who starts, what he is worth,
what the team's rating already embeds, and how long he has been the starter -- which is all
`hub.models.quarterback` needs to move a rating.

**What is assumed, and what the first live pull must confirm.** No pull has been made from
this repo: the test harness refuses the network and the fixture under
`tests/golden/fixtures/nfeloqb_qb_elos.synthetic.json` is hand-built, so `NFELOQB` says
`verified_against_live=False`. Field by field, the first live pull must confirm:

* the URL. `URL` is the raw-content path of the file at the repository's default branch;
  the file's name and location inside the repository are read off its README as of
  2026-09-07 and not off a response.
* the schema. `NFELOQB` declares 538's names -- `team1`/`team2`, `qb1`/`qb2`,
  `qb1_value_pre`, `qb1_adj`, `score1` -- and the file is described as "538's exact
  schema". A rename is a contract refusal and the last-good file is served instead.
* that the coming week's games are listed with their expected starters and null scores.
  538's file did this, and it is the row the team layer wants most: it is where a backup is
  first named. If nfeloqb lists only played games, `tenure` is still right and the starter
  is last week's, which is stale by one week and said nowhere -- so this is the one to check.
* the team abbreviations. 538 spelled Washington `WSH` and the Rams `LAR`; nflverse spells
  them `WAS` and `LA`, and every join in this repo is on nflverse's. `ABBREVIATIONS` maps
  the two known differences and `unknown_teams` names any team the state carries that the
  season's schedule does not -- `hub.models.ratings` prints it -- so a third spelling shows
  up as a sentence rather than as a team that silently never adjusts.
* the sign and scale of `qb1_adj`. Read as 538 defined it: Elo points, positive when the
  starter is better than what the team's rolling value embeds, 3.3 per value unit.

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

# 538's spellings that nflverse spells differently. Applied to both team columns; a
# spelling not listed passes through, and `report` says so when it matches no schedule.
ABBREVIATIONS = {"WSH": "WAS", "LAR": "LA"}

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
                "arrival_value": pl.Float64, "arrival_adj": pl.Float64,
                "tenure": pl.Int64, "as_of": pl.Utf8}


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
    """The file's rows, cast to the contract's dtypes and validated through it.

    Cast before the check rather than trusting inference: a short file whose values happen
    to be whole numbers infers `qb1_adj` as an integer column, and the contract would refuse
    a dtype the source never chose. A column the file does not carry is left for the
    contract to name, which is the refusal a rename should produce.
    """
    got = pl.read_csv(io.BytesIO(body), infer_schema_length=10000)
    casts = [pl.col(c).cast(t, strict=False) for c, t in NFELOQB.required.items()
             if c in got.columns and t is not pl.Utf8]
    return NFELOQB.validate(got.with_columns(casts))


def _long(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, game), both sides of every game, in nflverse's spellings."""
    sides = []
    for n in ("1", "2"):
        sides.append(rows.select(
            pl.col("date"), pl.col("season"),
            pl.col(f"team{n}").replace(ABBREVIATIONS).alias("team"),
            pl.col(f"qb{n}").alias("qb"),
            pl.col(f"qb{n}_value_pre").alias("value"),
            pl.col(f"qb{n}_adj").alias("adj"),
            pl.col(f"score{n}").alias("score")))
    return pl.concat(sides).sort("team", "date")


def state(rows: pl.DataFrame) -> pl.DataFrame:
    """Per team: the current starter, his value, the row he arrived on, and his tenure.

    The *run* is the trailing rows of a team's games whose starter is the latest row's
    starter. `tenure` counts the played rows in it -- a starter named for the coming game
    and yet to start it has a tenure of zero, which is what makes a fresh backup a fresh
    backup. `arrival_value` and `arrival_adj` are read off the run's first row: what the
    team rating embedded when he arrived is what the adjustment is relative to, and the
    latest row's own gap is already decayed by the source's rolling value, so reading it
    there and decaying it again would count the decay twice. `qb_value` is the latest row's,
    which is the starter as he stands now.

    Rows are not filtered to a season on purpose. A run that began late last season is
    one run, and the tenure that decays it should say so.
    """
    long = _long(rows)
    out = []
    for team, games in long.group_by("team", maintain_order=True):
        latest = games.row(-1, named=True)
        run = 0
        while run < games.height and games["qb"][games.height - 1 - run] == latest["qb"]:
            run += 1
        first = games.row(games.height - run, named=True)
        played = games.tail(run)["score"].is_not_null().sum()
        out.append({"team": team[0], "qb": latest["qb"], "qb_value": float(latest["value"]),
                    "arrival_value": float(first["value"]),
                    "arrival_adj": float(first["adj"]),
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
                     f"tenure {r['tenure']:>3}")
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
                    "read as a per-team state: starter, value, tenure.")
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
