"""The lineup each source would set, week by week, and what each one scored (#492).

**Why this exists.** ADR-0014 / issue #492: weekly consensus rank (ECR) is to set the lineup
provisionally from week 8, "until a forward-gated model beats it", and "the journal's forward
record" is what reopens that. So three sources are written down every week, side by side:

- `mu`   -- the current rule: `hub.season.roster.lock`'s best lineup by projected points.
- `ecr`  -- the best lineup by weekly expert consensus rank, from the committed pre-kickoff
            capture (`hub.fetch.consensus`; a capture taken at or after the deadline is not read).
- `espn` -- the best lineup by ESPN's own projection for the scoring period.

All three choose from the same pool `lock` does (a projected, startable, available player), so a
difference between lineups is a difference in *source*, not in who was eligible. Once the week is
scored, each lineup's realised PPR points are written beside it.

**Layout.** `state/lineup_journal/<season>/wk<NN>.json`, committed by the workflow that ran it,
like the consensus captures: a lookahead choice that is not kept cannot be recovered.

**Idempotent per week, first write wins.** `record` on a week that already has a file changes
nothing -- the entry is what the sources said when it was first written, not what a later sync
says. `settle` fills realised points once and leaves a scored week alone.

**A source that cannot answer says so.** No capture, no scoring period, no ESPN weekly numbers,
or too few ranked players to fill a legal lineup: that source's lineup is `null` with a reason,
and the other two are still journalled.

    uv run python -m hub.season.lineup_journal --record
    uv run python -m hub.season.lineup_journal --settle
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic
from hub.config import SEASON_AHEAD
from hub.names import player_key
from hub.paths import ROSTER_PARQUET, STATE_DIR

ROOT = STATE_DIR / "lineup_journal"
SOURCES = ("mu", "ecr", "espn")

# ECR is a rank (lower is better) and the lineup search maximises a total, so the search is
# handed a score that falls as the rank rises. The constant only has to exceed any rank.
_RANK_CEILING = 1000.0


def journal_path(season: int, week: int, base: Path | None = None) -> Path:
    return (ROOT if base is None else Path(base)) / str(season) / f"wk{week:02d}.json"


# --- the two weekly inputs, each with the reason it is absent -----------------

def ecr_ranks(season: int, week: int | None,
              base: Path | None = None) -> tuple[dict[str, float], str | None]:
    """`{player_key: ecr}` from the newest pre-deadline capture for `week`, or `({}, reason)`."""
    from hub.fetch import consensus
    if week is None:
        return {}, "no scoring period known for this roster"
    try:
        caps = consensus.read_captures(season, base)
    except Exception as e:                           # graceful degradation, per CLAUDE.md
        return {}, f"consensus captures unreadable ({type(e).__name__})"
    ok = [c for c in caps if c.week == week
          and not consensus.is_late(c.captured_at, c.first_game_day)]
    if not ok:
        late = sum(1 for c in caps if c.week == week)
        return {}, (f"no pre-kickoff consensus capture for week {week}"
                    + (f" ({late} found, all at or after the deadline)" if late else ""))
    cap = ok[-1]                                     # read_captures sorts earliest first
    out: dict[str, float] = {}
    for r in cap.rows.iter_rows(named=True):
        if r["ecr"] is not None:
            out.setdefault(player_key(r["player"]), float(r["ecr"]))
    return out, None


def espn_weekly(df: pl.DataFrame) -> tuple[dict[str, float], str | None]:
    """`{player_key: ESPN's weekly projection}` off the roster frame, or `({}, reason)`."""
    if "espn_week_proj" not in df.columns:
        return {}, "roster.parquet predates the ESPN weekly projection column"
    got = {r["key"] if "key" in df.columns else player_key(r["player"]): float(r["espn_week_proj"])
           for r in df.iter_rows(named=True) if r["espn_week_proj"] is not None}
    return got, (None if got else "ESPN carried no weekly projection for this roster")


def roster_week(df: pl.DataFrame) -> int | None:
    """The scoring period the roster was synced for, if it carries one."""
    if "espn_week" not in df.columns:
        return None
    weeks = df["espn_week"].drop_nulls()
    return int(weeks[0]) if len(weeks) else None


# --- the three lineups --------------------------------------------------------

def _pool(df: pl.DataFrame) -> pl.DataFrame:
    """The players `lock` would choose among: projected, startable, expected to play."""
    p = df.filter(pl.col("projected"))
    if "can_start" in p.columns:
        p = p.filter(pl.col("can_start"))
    if "available" in p.columns:
        p = p.filter(pl.col("available"))
    return p


def _key_of(df: pl.DataFrame) -> list[str]:
    return df["key"].to_list() if "key" in df.columns else [player_key(n) for n in df["player"]]


def _best(pool: pl.DataFrame, score: Mapping[str, float], name: str) -> dict[str, Any]:
    from hub.season.lineup import NoLegalLineup, TooManyLineups, best_by_points
    keys = _key_of(pool)
    have = pool.with_columns(pl.Series("_k", keys)).filter(pl.col("_k").is_in(list(score)))
    n_missing = pool.height - have.height
    if have.is_empty():
        return {"lineup": None, "reason": f"{name} ranks none of the {pool.height} candidates",
                "n_unranked": n_missing}
    scored = have.with_columns(
        pl.col("_k").replace_strict(dict(score), return_dtype=pl.Float64).alias("mu"),
        pl.lit(1.0).alias("sd"))
    try:
        starters = best_by_points(scored)["starters"]
    except (NoLegalLineup, TooManyLineups):
        return {"lineup": None, "n_unranked": n_missing,
                "reason": f"{name} ranks {have.height} of {pool.height} candidates, too few "
                          "to fill a legal lineup"}
    return {"lineup": sorted(starters["player"].to_list()), "reason": None,
            "n_unranked": n_missing}


def lineups(df: pl.DataFrame, ecr: Mapping[str, float], ecr_reason: str | None,
            espn: Mapping[str, float], espn_reason: str | None) -> dict[str, dict[str, Any]]:
    """What each source would set from this roster. A source with nothing says why."""
    from hub.season.roster import lock
    pool = _pool(df)
    lk = lock(df)
    out: dict[str, dict[str, Any]] = {
        "mu": ({"lineup": sorted(lk.best_lineup), "reason": None, "n_unranked": 0}
               if lk.best_lineup else
               {"lineup": None, "reason": "the current rule could not fill a legal lineup",
                "n_unranked": 0})}
    out["ecr"] = (_best(pool, {k: _RANK_CEILING - v for k, v in ecr.items()}, "ECR")
                  if ecr else {"lineup": None, "reason": ecr_reason or "no ECR",
                               "n_unranked": pool.height})
    out["espn"] = (_best(pool, espn, "ESPN's weekly projection")
                   if espn else {"lineup": None, "reason": espn_reason or "no ESPN weekly",
                                 "n_unranked": pool.height})
    return out


# --- the journal ----------------------------------------------------------------

def record(df: pl.DataFrame, *, season: int = SEASON_AHEAD, week: int | None = None,
           base: Path | None = None, captures: Path | None = None,
           now: datetime | None = None) -> tuple[Path | None, str]:
    """Write this week's entry, or say why not. Returns `(path, note)`.

    An existing entry is left exactly as it is (idempotent per week).
    """
    wk = week if week is not None else roster_week(df)
    if wk is None:
        return None, "roster carries no scoring period; nothing journalled"
    path = journal_path(season, wk, base)
    if path.exists():
        return path, f"week {wk} already journalled; left as it was"
    ecr, ecr_reason = ecr_ranks(season, wk, captures)
    espn, espn_reason = espn_weekly(df)
    got = lineups(df, ecr, ecr_reason, espn, espn_reason)
    sets = [tuple(v["lineup"]) for v in got.values() if v["lineup"]]
    doc = {
        "season": season, "week": wk,
        "recorded_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "sources": got,
        # More than one distinct lineup among the sources that answered.
        "disagree": len(set(sets)) > 1,
        "scored": False,
        "realised": dict.fromkeys(SOURCES),
    }
    atomic.write_text(path, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    n = sum(1 for v in got.values() if v["lineup"])
    return path, f"week {wk} journalled ({n} of 3 sources)"


def settle(season: int, week: int, points: Mapping[str, float],
           base: Path | None = None) -> str:
    """Write each lineup's realised points into the week's entry, once.

    `points` is `{player_key: PPR points}` for the played week. A rostered player absent from
    it scored nothing (did not play) and is counted in `n_no_row` so a name-matching break shows.
    """
    path = journal_path(season, week, base)
    if not path.exists():
        return f"week {week}: no journal entry"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("scored"):
        return f"week {week}: already scored; left as it was"
    if not points:
        return f"week {week}: no realised points available yet"
    realised: dict[str, Any] = {}
    for s in SOURCES:
        lineup = doc["sources"][s]["lineup"]
        if not lineup:
            realised[s] = None
            continue
        ks = [player_key(n) for n in lineup]
        realised[s] = {"points": round(sum(points.get(k, 0.0) for k in ks), 2),
                       "n_no_row": sum(1 for k in ks if k not in points)}
    doc["realised"], doc["scored"] = realised, True
    atomic.write_text(path, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return f"week {week}: scored"


def read(season: int, base: Path | None = None) -> list[dict[str, Any]]:
    root = (ROOT if base is None else Path(base)) / str(season)
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("wk*.json"))]


def _realised_points(season: int, week: int) -> dict[str, float]:  # pragma: no cover - network
    from hub.fetch import nflverse
    from hub.models.experiment import PLAYER_STATS_COLS, realised_ppg
    stats = nflverse.load("player_stats", [season], cols=list(PLAYER_STATS_COLS))
    got = realised_ppg(stats).filter(pl.col("week") == week)
    return {r["player"]: float(r["points"]) for r in got.iter_rows(named=True)}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser(prog="hub.season.lineup_journal")
    ap.add_argument("--record", action="store_true", help="journal this week's three lineups")
    ap.add_argument("--settle", action="store_true",
                    help="write realised points for journalled weeks that have finished")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not (a.record or a.settle):
        ap.error("say --record, --settle, or both")
    if not ROSTER_PARQUET.exists():
        print("hub.season.lineup_journal: no roster.parquet; nothing to do", file=sys.stderr)
        return 0
    df = pl.read_parquet(ROSTER_PARQUET)
    if a.record:
        print(f"  {record(df)[1]}")
    if a.settle:
        current = roster_week(df)
        for doc in read(SEASON_AHEAD):
            wk = int(doc["week"])
            if doc.get("scored") or current is None or wk >= current:
                continue          # a week is final only once the roster has moved past it
            try:
                print(f"  {settle(SEASON_AHEAD, wk, _realised_points(SEASON_AHEAD, wk))}")
            except Exception as e:                   # offline: the entry stays unscored
                print(f"  week {wk}: not settled ({type(e).__name__}: {e})", file=sys.stderr)
    return 0


if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main())
