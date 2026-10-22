"""The lineup each source would set, week by week, and what each one scored (#492).

**Why this exists.** ADR-0014 / issue #492: weekly consensus rank (ECR) is to set the lineup
provisionally from week 7, "until a forward-gated model beats it", and "the journal's forward
record" is what reopens that. So three sources are written down every week, side by side:

- `mu`   -- the current rule: `hub.season.roster.lock`'s best lineup by mu.
- `ecr`  -- the best lineup by weekly expert consensus rank, from the committed pre-kickoff
            capture (`hub.fetch.consensus`; a capture taken at or after the deadline is not read).
- `espn` -- the best lineup by ESPN's own projection for the scoring period.

Beside them, `rule` records what the rule chose when the entry was written, Wednesday
(`hub.season.lineup_rule`): `mu` through week 6, weekly ECR from week 7 as a provisional rule,
the capture week it was set on, and the reason whenever that was not the week's own ECR. That
block is part of the pre-kickoff comparison and is first-write-wins with the rest of it.

**What was actually published (#517) is a separate block, `published`.** The week's lineup is
set by whatever the roster publish decides, and the Friday lineup capture (and a late strict
capture) lands after the Wednesday entry is frozen, so the Saturday publish can decide
differently. Each roster publish up to the week's first Sunday kickoff therefore writes its own
decision -- source, rule, consensus week and capture, lineup, `decided_at`, and the players whose
game had kicked off (#516), when any had -- into `published`, replacing the previous publish's
block and touching nothing else. From the start of the first Sunday game day, Eastern
(`hub.fetch.consensus.lineup_deadline_for`, the lineup capture's own deadline), nothing is
written: the last decision before Sunday is the one that set the lineup. Once scored,
`realised.set` is the `published` lineup's points; a week journalled before `published`
existed falls back to `rule.lineup`, and `settle` says so.

All three choose from the same pool `lock` does (a projected, startable, available player not on
bye), so a difference between lineups is a difference in *source*, not in who was eligible. Once
the week is scored, each lineup's realised PPR points are written beside it.

**Layout.** `state/lineup_journal/<season>/wk<NN>.json`, committed by the workflow that ran it,
like the consensus captures: a lookahead choice that is not kept cannot be recovered.

**Idempotent per week, first write wins.** `record` on a week that already has its comparison
changes nothing -- the entry is what the sources said when it was first written, not what a
later sync says. (An entry holding only a `published` block, because a publish ran in a week
`record` could not, gains the comparison beside it.) `settle` fills realised points once and
leaves a scored week alone.

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
from typing import TYPE_CHECKING, Any

import polars as pl

from hub import atomic
from hub.config import SEASON_AHEAD
from hub.declare import not_an_input
from hub.names import player_key
from hub.paths import ROSTER_PARQUET, STATE_DIR

if TYPE_CHECKING:                                    # `lineup_rule` imports this module
    from hub.season.lineup_rule import Decision
    from hub.season.roster import Lock

ROOT = STATE_DIR / "lineup_journal"
SOURCES = ("mu", "ecr", "espn")

# ECR is a rank (lower is better) and the lineup search maximises a total, so the search is
# handed a score that falls as the rank rises. The constant only has to exceed any rank.
_RANK_CEILING = not_an_input(
    1000.0, "an offset turning a rank (lower is better) into a score to maximise; any value "
            "above the largest rank gives the same lineup")


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
    """The players `lock` would choose among: projected, startable, not on bye, expected to
    play -- the filters `hub.season.roster.lock` applies to its pool, in the same order."""
    p = df.filter(pl.col("projected"))
    if "can_start" in p.columns:
        p = p.filter(pl.col("can_start"))
    # GUARD journal-pool-excludes-bye: a team with no game this week supplies no candidate
    if "on_bye" in p.columns:
        p = p.filter(~pl.col("on_bye"))
    # /GUARD
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

# What the two checks read off the schedule: the consensus ask plus the final scores.
SCHEDULE_COLS = ("game_id", "season", "week", "game_type", "home_team", "away_team",
                 "gameday", "home_score", "away_score")


def _now() -> datetime:
    """The clock the CLI judges kickoff by; a seam so a test can pin it."""
    return datetime.now(UTC)


def _schedule(season: int) -> pl.DataFrame | None:  # pragma: no cover - network
    from hub.fetch import nflverse
    try:
        return nflverse.load("schedules", [season], cols=list(SCHEDULE_COLS))
    except Exception:                                # offline: the checks refuse, they don't guess
        return None


def refuse_after_kickoff(schedule: pl.DataFrame | None, week: int, now: datetime) -> str | None:
    """Why `week` may not be journalled now, or None if its first game has not started.

    `mu` is re-fitted on whatever has happened, so an entry written after the week's first
    game compares a pre-kickoff ranking with a number that has seen results. The boundary is
    `hub.fetch.consensus.deadline_for` -- midnight Eastern at the start of the first game
    day -- the same one a consensus capture is held to. A schedule that cannot be read refuses
    too: the entry is the forward record, and an unverifiable one is not a pre-kickoff one.
    """
    from hub.fetch import consensus
    if schedule is None:
        return f"week {week}: no schedule to check kickoff against; not journalled"
    first = consensus.first_game_days(schedule).get(week)
    if first is None:
        return f"week {week}: not in the schedule; not journalled"
    if now >= consensus.deadline_for(first):
        return (f"week {week}: first game day {first} has started; a lineup journalled now "
                "is not a pre-kickoff one, so nothing was written")
    return None


def refuse_after_first_sunday(schedule: pl.DataFrame | None, week: int,
                              now: datetime) -> str | None:
    """Why the published decision for `week` may not be written now, or None if it may.

    The lineup is set by the last publish before Sunday's games, so the decision journalled is
    frozen from the start of the week's first Sunday game day, Eastern --
    `hub.fetch.consensus.lineup_deadline_for`, the deadline a lineup capture is held to, so the
    two cannot disagree about when the week's lineup was settled. A schedule that cannot be
    read refuses: a decision that cannot be placed before Sunday is not written as one.
    """
    from hub.fetch import consensus
    if schedule is None:
        return (f"week {week}: no schedule to check the first Sunday against; the published "
                "decision was not journalled")
    sunday = consensus.first_sundays(schedule).get(week)
    if sunday is None:
        return (f"week {week}: no Sunday game in the schedule; the published decision was "
                "not journalled")
    if now >= consensus.lineup_deadline_for(sunday):
        return (f"week {week}: first Sunday {sunday} has started; the published decision is "
                "kept as it was")
    return None


def week_incomplete(schedule: pl.DataFrame | None, week: int) -> str | None:
    """Why `week` cannot be scored yet, or None if every scheduled game has a final score."""
    if schedule is None:
        return f"week {week}: no schedule to check completeness against; left unscored"
    games = schedule.filter((pl.col("game_type") == "REG") & (pl.col("week") == week))
    if games.is_empty():
        return f"week {week}: not in the schedule; left unscored"
    open_games = games.filter(pl.col("home_score").is_null() | pl.col("away_score").is_null())
    if open_games.height:
        return (f"week {week}: {open_games.height} of {games.height} games not final; "
                "left unscored")
    return None


def record(df: pl.DataFrame, *, season: int = SEASON_AHEAD, week: int | None = None,
           base: Path | None = None, captures: Path | None = None,
           now: datetime | None = None,
           schedule: pl.DataFrame | None = None) -> tuple[Path | None, str]:
    """Write this week's entry, or say why not. Returns `(path, note)`.

    An existing entry is left exactly as it is (idempotent per week). A week whose first game
    has started is refused (`refuse_after_kickoff`); `now` is the clock, so a test can pin it.
    """
    wk = week if week is not None else roster_week(df)
    if wk is None:
        return None, "roster carries no scoring period; nothing journalled"
    path = journal_path(season, wk, base)
    old = _read_entry(path)
    if old is not None and "sources" in old:
        return path, f"week {wk} already journalled; left as it was"
    now = now or datetime.now(UTC)
    # GUARD no-record-after-kickoff: a week already under way is never journalled
    if why := refuse_after_kickoff(schedule, wk, now):
        return None, why
    # /GUARD
    ecr, ecr_reason = ecr_ranks(season, wk, captures)
    espn, espn_reason = espn_weekly(df)
    got = lineups(df, ecr, ecr_reason, espn, espn_reason)
    sets = [tuple(v["lineup"]) for v in got.values() if v["lineup"]]
    # What was actually published (#492, ADR-0014's "what was actually done"): the rule the
    # week fell under, the source that set it, why if it was not the week's own ECR, and the
    # lineup itself -- the same `lineup_rule.decide` the publisher calls, on the same capture.
    from hub.season import lineup_rule
    set_lock, decision = lineup_rule.decide(df, season, wk, captures, ecr=ecr,
                                            ecr_reason=ecr_reason)
    rule = {**decision.as_dict(), "lineup": sorted(set_lock.best_lineup) or None}
    doc = {
        "season": season, "week": wk,
        "recorded_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "sources": got,
        "rule": rule,
        # More than one distinct lineup among the sources that answered.
        "disagree": len(set(sets)) > 1,
        "scored": False,
        "realised": dict.fromkeys(SOURCES),
    }
    if old is not None and "published" in old:       # a publish ran first; its block stays
        doc["published"] = old["published"]
    atomic.write_text(path, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    n = sum(1 for v in got.values() if v["lineup"])
    return path, f"week {wk} journalled ({n} of 3 sources)"


def _read_entry(path: Path) -> dict[str, Any] | None:
    """The week's entry, or None when there is none."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def published_block(lk: Lock, decision: Decision, decided_at: datetime) -> dict[str, Any]:
    """What one roster publish decided (#517), from the `Lock` and `Decision` it published.

    The decision's own fields less its fixed `basis`/`caveat` text (the `rule` block carries
    those), the lineup, when, and -- only when a game had kicked off -- who was locked (#516)."""
    out = {k: v for k, v in decision.as_dict().items() if k not in ("basis", "caveat")}
    out["lineup"] = sorted(lk.best_lineup) or None
    out["decided_at"] = decided_at.astimezone(UTC).isoformat()
    if lk.locked:
        out["locked"] = list(lk.locked)
    return out


def record_published(season: int, week: int | None, lk: Lock, decision: Decision, *,
                     base: Path | None = None, schedule: pl.DataFrame | None = None,
                     now: datetime | None = None) -> tuple[Path | None, str]:
    """Write the decision a roster publish made into the week's `published` block (#517).

    Called by every roster publish. Replaces the previous publish's `published` block and
    nothing else: the Wednesday comparison (`sources`, `rule`) is first-write-wins and stays.
    Refused from the week's first Sunday (`refuse_after_first_sunday`), and on a scored week.
    With no entry yet (the week's `record` did not run) the block is written into a new one,
    and `record` adds the comparison beside it if it still can.
    """
    if week is None:
        return None, "roster carries no scoring period; no published decision journalled"
    now = now or datetime.now(UTC)
    # GUARD no-published-write-after-first-sunday: nothing is journalled once Sunday starts
    if why := refuse_after_first_sunday(schedule, week, now):
        return None, why
    # /GUARD
    path = journal_path(season, week, base)
    doc = _read_entry(path)
    if doc is None:
        doc = {"season": season, "week": week, "scored": False,
               "realised": dict.fromkeys(SOURCES)}
    elif doc.get("scored"):
        return path, f"week {week}: already scored; the published decision left as it was"
    doc["published"] = published_block(lk, decision, now)
    atomic.write_text(path, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return path, (f"week {week}: published decision journalled "
                  f"({decision.source}, {len(lk.best_lineup)} starters)")


def settle(season: int, week: int, points: Mapping[str, float],
           base: Path | None = None, schedule: pl.DataFrame | None = None) -> str:
    """Write each lineup's realised points into the week's entry, once.

    Only a complete week is scored: every scheduled game must be final in `schedule`
    (`week_incomplete`). nflverse loads a week as it goes, so non-empty points can still be
    missing Monday night, and a week frozen short is the forward record wrong for good.

    `points` is `{player_key: PPR points}` for the played week. A rostered player absent from
    it scored nothing (did not play) and is counted in `n_no_row` so a name-matching break shows.
    """
    path = journal_path(season, week, base)
    if not path.exists():
        return f"week {week}: no journal entry"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("scored"):
        return f"week {week}: already scored; left as it was"
    # GUARD settle-only-complete-weeks: a week with games still to be final stays unscored
    if why := week_incomplete(schedule, week):
        return why
    # /GUARD
    if not points:
        return f"week {week}: no realised points available yet"
    realised: dict[str, Any] = {}
    sources = doc.get("sources") or {}              # absent on an entry only a publish wrote
    for s in SOURCES:
        lineup = (sources.get(s) or {}).get("lineup")
        if not lineup:
            realised[s] = None
            continue
        ks = [player_key(n) for n in lineup]
        realised[s] = {"points": round(sum(points.get(k, 0.0) for k in ks), 2),
                       "n_no_row": sum(1 for k in ks if k not in points)}
    # The lineup that was set, scored beside the three sources: the last publish's decision
    # before Sunday (#517). An entry from before `published` existed falls back to the
    # Wednesday `rule` block and says so; one from before #492's switch carries neither and
    # gains no `set`.
    set_lineup, set_from = (doc.get("rule") or {}).get("lineup"), "rule"
    # GUARD settle-scores-the-published-lineup: `set` is the published lineup when recorded
    if "published" in doc:
        set_lineup, set_from = (doc["published"] or {}).get("lineup"), "published"
    # /GUARD
    note = ("; set lineup scored from the Wednesday `rule` block -- no published decision "
            "was journalled for this week (#517)" if set_lineup and set_from == "rule" else "")
    if set_lineup:
        ks = [player_key(n) for n in set_lineup]
        realised["set"] = {"points": round(sum(points.get(k, 0.0) for k in ks), 2),
                           "n_no_row": sum(1 for k in ks if k not in points)}
        doc["set_scored_from"] = set_from            # "published", or the "rule" fallback
    doc["realised"], doc["scored"] = realised, True
    atomic.write_text(path, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return f"week {week}: scored{note}"


def read(season: int, base: Path | None = None) -> list[dict[str, Any]]:
    root = (ROOT if base is None else Path(base)) / str(season)
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("wk*.json"))]


def _realised_points(season: int, week: int) -> dict[str, float]:  # pragma: no cover - network
    from hub.fetch import nflverse
    from hub.models.experiment import PLAYER_STATS_COLS, realised_ppg
    stats = nflverse.load("player_stats", [season], cols=list(PLAYER_STATS_COLS))
    got = realised_ppg(stats).filter(pl.col("week") == week)
    return {r["player"]: float(r["points"]) for r in got.iter_rows(named=True)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hub.season.lineup_journal")
    ap.add_argument("--record", action="store_true", help="journal this week's three lineups")
    ap.add_argument("--settle", action="store_true",
                    help="write realised points for journalled weeks that have finished")
    ap.add_argument("--roster", default=None, help="override the roster parquet read")
    ap.add_argument("--out", default=None, help="override the journal directory")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not (a.record or a.settle):
        ap.error("say --record, --settle, or both")
    src = Path(a.roster) if a.roster else ROSTER_PARQUET
    base = Path(a.out) if a.out else None
    if not src.exists():
        print(f"hub.season.lineup_journal: no roster at {src}; run "
              "`python -m hub.season.roster --write` first", file=sys.stderr)
        return 1
    df = pl.read_parquet(src)
    sched = _schedule(SEASON_AHEAD)
    if a.record:
        print(f"  {record(df, base=base, schedule=sched, now=_now())[1]}")
    if a.settle:
        current = roster_week(df)
        for doc in read(SEASON_AHEAD, base):
            wk = int(doc["week"])
            if doc.get("scored") or current is None or wk >= current:
                continue          # a week is final only once the roster has moved past it
            try:
                got = settle(SEASON_AHEAD, wk, _realised_points(SEASON_AHEAD, wk), base,
                             schedule=sched)
                print(f"  {got}")
            except Exception as e:                   # offline: the entry stays unscored
                print(f"  week {wk}: not settled ({type(e).__name__}: {e})", file=sys.stderr)
    return 0


if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main())
