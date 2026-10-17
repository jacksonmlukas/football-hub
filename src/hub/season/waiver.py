"""Which free agent to pick up, against the player he would replace (#493).

**This is a provisional rule (ADR-0014), shown with its basis, and it is not a measured edge.**
No gate has run on it. It is journalled (`record`, `settle`) so one can.

**What is ranked.** For each free agent, *value* is

    his ESPN projected points per game  minus  that of the rostered player he would replace

where "the rostered player he would replace" is the lowest-projected player at his own position
on our roster that is not on injured reserve (starter or bench alike; the ticket's "lowest
starter or bench player"). **Per game, not per rest of season (#491's lesson):** ESPN's
`projected_total_points` covers only the games a team has left, so a player whose bye is still
ahead has one game fewer -- some 6-7% of a season -- and a total would mark him down for it.
The rate is `espn_avg`, or the total over that team's remaining games
(`WeekGames.remaining`) when the average is absent. The rest-of-season totals and `bye_ahead`
are shown beside each row and used for nothing. A free agent who is not better than the player
he would replace has value <= 0 and is never recommended.

**Which projection, and why that one.** Not `mu` (the board blend), not the weekly ECR. `mu` is
a per-week number for the players the board scoped, and a free agent is by construction mostly
outside that scope, so a subtraction of `mu` would compare two different instruments. The
weekly ECR (which #492 decided sets lineups provisionally from week 8) is a rank, not points,
and cannot be subtracted: it is shown beside each name and never sorted on. ESPN's projection
exists for every player the league lists and is the only rest-of-season number on one basis for
both sides.

**Unshrunk.** The ticket asks for a shrunk estimate "see the empirical-Bayes ticket" (#497).
That model does not exist yet, so the projection is ESPN's own, unshrunk, and the published
block says so. A top-of-the-pool free agent is selected *for* a high projection, so the value
here is biased upward by exactly the selection a shrinkage would remove; read it as a ranking
and not as a forecast of the gain.

**Shown, never sorted on (ADR-0013, ADR-0016).** The snap-share trend (`snap_trend`), the
weekly ECR, ESPN's weekly projection and the percent owned sit beside each row. None enters the
value or the order.

**Bye and playoff weeks.** Each candidate carries his team's bye week and its opponent in each
of weeks 15-17 (`calendar`), because a rest-of-season total counts a bye as nothing and a
fantasy playoff week against a bye is a week without him.

**FAAB.** The ticket asks for no bid sizing and none is made. A bid is a price, and nothing
here measures what the room would pay.

    uv run python -m hub.season.waiver            # print this week's ranking
    uv run python -m hub.season.waiver --record   # and journal it (first write wins)
    uv run python -m hub.season.waiver --settle   # score journalled weeks that have finished
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic
from hub.config import SEASON_AHEAD
from hub.paths import FREE_AGENTS_PARQUET, ROSTER_PARQUET, STATE_DIR

ROOT = STATE_DIR / "waiver_journal"

LABEL = "provisional"
BASIS = ("value = the free agent's ESPN projected points per game minus that of the lowest-"
         "projected non-IR rostered player at his position (ESPN's own points, unshrunk); "
         "rest-of-season totals and bye_ahead are shown, not used")
SHRINKAGE = ("unshrunk: the empirical-Bayes player model (#497) is not built, so a top-of-the-"
             "pool projection is biased upward by the selection that put it there")
SHOWN_NOT_SORTED = ("snap_delta, ecr, week_proj and percent_owned are shown beside each row and "
                    "never enter the value or the order (ADR-0013, ADR-0016)")
CAVEAT = ("provisional rule (ADR-0014), not a measured edge: no gate has run on this ordering; "
          "it is journalled so one can")
NO_FAAB = "no bid is sized: nothing here measures what the room would pay"

# What ESPN calls a player who cannot be started: not recommended, but still shown.
BLOCKED_STATUS = frozenset({"OUT", "INJURY_RESERVE", "SUSPENSION"})
PLAYOFF_WEEKS = (15, 16, 17)
REGULAR_SEASON_WEEKS = 18
TOP_PER_POSITION = 5
# A pool older than this is served with a staleness note: waivers turn over weekly.
STALE_AFTER = timedelta(days=4)
# The snap screen compared weeks w-1..w with 1..w-2, and said it needed about eight weeks.
SNAP_ENOUGH_WEEKS = 8
# ADR-0013's horizon: the points a claim is scored on are the next three weeks'.
HORIZON = 3
INJURED_RESERVE = "IR"
# Positions nflverse's weekly player stats do not carry, so a realised score cannot be read.
UNSCORABLE = frozenset({"K", "D/ST"})


def _team(code: str | None) -> str | None:
    from hub.draft.playoff_sos import canon_team
    return canon_team(code)


def _moment(at: datetime | None) -> datetime:
    return at or datetime.now(UTC).replace(tzinfo=None)


def _proj(value: Any) -> float:
    """A projection as points; None, NaN and non-positive all read as ESPN carrying none."""
    ok = isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    return float(value) if ok else 0.0


# --- the replacement ---------------------------------------------------------------------

def per_game(row: Mapping[str, Any], remaining: Mapping[str, int] | None) -> float:
    """ESPN's projected points per game, 0.0 when it carries none.

    `espn_avg` is already per game. Without it the rest-of-season total is divided by the games
    his own team has left (`WeekGames.remaining`, #491): a total counts only the games still to
    be played, so a player whose bye is ahead has one fewer and a total is not like for like.
    """
    avg = _proj(row.get("espn_avg"))
    if avg > 0:
        return avg
    team = _team(row.get("nfl_team"))
    left = (remaining or {}).get(team) if team else None
    total = _proj(row.get("espn_total"))
    return total / left if total > 0 and left else 0.0


def replacements(roster: pl.DataFrame,
                 remaining: Mapping[str, int] | None = None) -> dict[str, dict[str, Any]]:
    """Per position, the rostered player a free agent there would replace.

    The lowest ESPN per-game projection among players not on injured reserve; ties go to the
    name that sorts first so the answer does not depend on row order. A player ESPN carries no
    projection for counts as zero, and is flagged so the page can say that the value is against
    an empty spot rather than against a weak player. The rest-of-season total rides along for
    display only.
    """
    out: dict[str, dict[str, Any]] = {}
    for r in roster.iter_rows(named=True):
        if r.get("slot") == INJURED_RESERVE or not r.get("pos"):
            continue
        pg = per_game(r, remaining)
        cand = {"player": r["player"], "pg": pg, "proj": _proj(r.get("espn_total")),
                "week_proj": r.get("espn_week_proj"),
                "key": r.get("key"), "no_projection": pg == 0.0}
        cur = out.get(r["pos"])
        if cur is None or (pg, r["player"]) < (cur["pg"], cur["player"]):
            out[r["pos"]] = cand
    return out


# --- the calendar: bye weeks and the fantasy playoff weeks --------------------------------

def calendar(games: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Per canonical team: its bye week and its opponent (or None: a bye) in weeks 15-17.

    The bye is a regular-season week 1-18 in which the team is absent from the schedule; a
    team the schedule does not carry is absent from the result rather than guessed at.
    """
    reg = games.filter(pl.col("week") <= REGULAR_SEASON_WEEKS)
    opp: dict[str, dict[int, str]] = {}
    for g in reg.iter_rows(named=True):
        home, away = _team(g["home_team"]), _team(g["away_team"])
        if home is None or away is None:
            continue
        opp.setdefault(home, {})[int(g["week"])] = away
        opp.setdefault(away, {})[int(g["week"])] = home
    weeks = sorted({int(w) for w in reg["week"].drop_nulls().to_list()})
    out: dict[str, dict[str, Any]] = {}
    for team, played in opp.items():
        byes = [w for w in weeks if w not in played]
        out[team] = {"bye_week": byes[0] if byes else None,
                     "playoff": {str(w): played.get(w) for w in PLAYOFF_WEEKS}}
    return out


# --- the snap-share trend, shown and never sorted on --------------------------------------

def snap_trend(snaps: pl.DataFrame, week: int) -> dict[str, dict[str, Any]]:
    """`{player_key: {delta, recent, baseline}}` as of the last finished week before `week`.

    The screen's windows (docs/snap-trend-signal.md): offence snap share averaged over weeks
    w-1..w against weeks 1..w-2, with `w = week - 1`. A key two different PFR ids share is
    ambiguous and is left out, because a trend belonging to the other player is worse than none.
    """
    from hub.names import player_key
    anchor = week - 1
    reg = snaps.filter(pl.col("game_type") == "REG") if "game_type" in snaps.columns else snaps
    reg = reg.filter(pl.col("week") <= anchor).with_columns(
        pl.col("player").map_elements(player_key, return_dtype=pl.Utf8).alias("_key"))
    ids = reg.group_by("_key").agg(pl.col("pfr_player_id").n_unique().alias("_n"))
    unique = set(ids.filter(pl.col("_n") == 1)["_key"].to_list())
    out: dict[str, dict[str, Any]] = {}
    for key, grp in reg.group_by("_key"):
        k = key[0]
        if k not in unique:
            continue
        recent = grp.filter(pl.col("week") >= anchor - 1)["offense_pct"].drop_nulls()
        base = grp.filter(pl.col("week") <= anchor - 2)["offense_pct"].drop_nulls()
        if recent.is_empty() or base.is_empty():
            continue
        r, b = float(recent.mean()), float(base.mean())   # type: ignore[arg-type]
        out[str(k)] = {"delta": r - b, "recent": r, "baseline": b}
    return out


def _snap_reason(week: int | None, have: bool) -> str | None:
    if week is None:
        return "no week is open, so no snap-share window is defined"
    if not have:
        return "snap counts unavailable"
    if week - 1 < SNAP_ENOUGH_WEEKS:
        return (f"only {max(week - 1, 0)} weeks of snaps: the screen needed about "
                f"{SNAP_ENOUGH_WEEKS} before the trend said anything")
    return None


# --- the ranking --------------------------------------------------------------------------

def _row(fa: Mapping[str, Any], repl: Mapping[str, Any], cal: Mapping[str, Any],
         bye_now: frozenset[str], snaps: Mapping[str, Mapping[str, Any]],
         ecr: Mapping[str, float], remaining: Mapping[str, int] | None = None,
         week: int | None = None) -> dict[str, Any]:
    team = _team(fa.get("nfl_team"))
    c = cal.get(team) if team else None
    snap = snaps.get(fa["key"])
    fa_proj = _proj(fa.get("espn_total"))
    fa_pg = per_game(fa, remaining)
    wk, rwk = fa.get("espn_week_proj"), repl.get("week_proj")
    status = fa.get("injury_status") or ""
    # GUARD waiver-value-is-net-of-the-replacement [unit/test_waiver.py]: the subtraction
    # that makes a pickup a gain over his roster spot and not merely a good player; per game,
    # so a bye ahead (one fewer game in a rest-of-season total) is not a penalty (#491)
    value = fa_pg - repl["pg"] if fa_pg > 0 else None
    # /GUARD
    bye = c["bye_week"] if c else None
    return {
        "player": fa["player"], "pos": fa["pos"], "nfl_team": fa.get("nfl_team"),
        "injury_status": status, "percent_owned": fa.get("percent_owned"),
        # `value` is points per game; the totals and `bye_ahead` are display only
        "proj": fa_proj if fa_proj > 0 else None,
        "proj_per_game": fa_pg if fa_pg > 0 else None,
        "value": value,
        "value_total": fa_proj - repl["proj"] if fa_proj > 0 else None,
        "bye_ahead": bool(bye is not None and week is not None and bye >= week),
        "replaces": repl["player"], "replaces_proj": repl["proj"],
        "replaces_per_game": repl["pg"],
        "replaces_no_projection": repl["no_projection"],
        "week_proj": wk, "week_value": (wk - rwk if wk is not None and rwk is not None
                                        else None),
        "ecr": ecr.get(fa["key"]), "replaces_ecr": ecr.get(repl["key"]) if repl.get("key")
        else None,
        "snap_delta": snap["delta"] if snap else None,
        "snap_recent": snap["recent"] if snap else None,
        "snap_baseline": snap["baseline"] if snap else None,
        "bye_week": c["bye_week"] if c else None,
        "on_bye_now": bool(team and team in bye_now),
        "playoff": c["playoff"] if c else None,
        "blocked": status if status in BLOCKED_STATUS else None,
        "missing": None if value is not None else "ESPN carries no rest-of-season projection",
    }


def build(fa: pl.DataFrame, roster: pl.DataFrame, *, week: int | None = None,
          bye_now: frozenset[str] = frozenset(),
          cal: Mapping[str, Any] | None = None,
          snaps: Mapping[str, Mapping[str, Any]] | None = None,
          ecr: Mapping[str, float] | None = None,
          top: int = TOP_PER_POSITION,
          remaining: Mapping[str, int] | None = None) -> dict[str, Any]:
    """The published `waiver` object: the best `top` free agents per position, ranked by value.

    `value` is ESPN points per game over the replacement's, so players with different numbers
    of games left compare like for like; `remaining` is `WeekGames.remaining`.

    `recommended` names those with value > 0 and no blocking injury designation; everyone else
    on a row is still shown, so a reader sees that a position has nobody worth the move.
    """
    repl = replacements(roster, remaining)
    rows, no_spot = [], 0
    for r in fa.iter_rows(named=True):
        if r["pos"] not in repl:
            no_spot += 1
            continue
        rows.append(_row(r, repl[r["pos"]], cal or {}, bye_now, snaps or {}, ecr or {},
                         remaining, week))
    ranked = sorted((r for r in rows if r["value"] is not None),
                    key=lambda r: (-r["value"], r["player"]))
    kept: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for i, r in enumerate(ranked, start=1):
        if seen.get(r["pos"], 0) >= top:
            continue
        seen[r["pos"]] = seen.get(r["pos"], 0) + 1
        kept.append({**r, "rank": i,
                     "recommended": r["value"] > 0 and r["blocked"] is None})
    return {
        "label": LABEL, "caveat": CAVEAT, "basis": BASIS, "shrinkage": SHRINKAGE,
        "shown_not_sorted": SHOWN_NOT_SORTED, "faab": NO_FAAB, "week": week,
        "rows": kept,
        "recommended": [r["player"] for r in kept if r["recommended"]],
        "unprojected": sum(1 for r in rows if r["value"] is None),
        "no_roster_spot": no_spot,
        "replaced_by_position": {p: {"player": v["player"], "proj": v["proj"],
                                     "per_game": v["pg"]}
                                 for p, v in sorted(repl.items())},
    }


def unavailable(reason: str) -> dict[str, Any]:
    """The `waiver` object when it cannot be built: labelled, empty, and saying why."""
    return {"label": LABEL, "caveat": CAVEAT, "basis": BASIS, "shrinkage": SHRINKAGE,
            "shown_not_sorted": SHOWN_NOT_SORTED, "faab": NO_FAAB, "week": None,
            "rows": [], "recommended": [], "missing": reason}


def _load_snaps(season: int) -> pl.DataFrame:  # pragma: no cover - network
    from hub.fetch import nflverse
    return nflverse.load("snap_counts", [season])


def for_roster(roster: pl.DataFrame, season: int = SEASON_AHEAD, at: datetime | None = None,
               base: Path | None = None, fa_path: Path | None = None) -> dict[str, Any]:
    """The ranking for the open week, degrading to `unavailable` on any failure.

    The free agents are read from last-good (`hub.fetch.free_agents`), never fetched here: the
    publisher must work offline, and a fetch that failed is the reason the last good pool is
    what is served, which `last_error` and `stale` carry onto the page.
    """
    from hub import schedule
    from hub.fetch import free_agents
    from hub.season import lineup_journal
    from hub.season.roster import week_games
    moment = _moment(at)
    fa, meta = free_agents.read(fa_path)
    if fa is None:
        return unavailable("no free agents fetched yet -- run "
                           "`python -m hub.fetch.free_agents`")
    if roster.is_empty():
        return unavailable("no roster to compare the free agents against")
    reasons: dict[str, str | None] = {"calendar_reason": None, "ecr_reason": None}
    cal: dict[str, Any] = {}
    week: int | None = None
    bye: frozenset[str] = frozenset()
    remaining: dict[str, int] = {}
    try:
        games = schedule.priced_games(season, at=moment, base=base)
        cal = calendar(games)
        wk = week_games(games, moment)
        if wk is not None:
            week, bye, remaining = wk.week, wk.bye, dict(wk.remaining)
        else:
            reasons["calendar_reason"] = "no week is open in the schedule"
    except Exception as e:                       # unattended: say which failure, never raise
        reasons["calendar_reason"] = (f"the schedule was unavailable ({type(e).__name__}): "
                                      "bye and playoff weeks are not shown")
    snaps: dict[str, Any] = {}
    snap_note: str | None
    try:
        if week is None:
            raise LookupError("no open week")
        snaps = snap_trend(_load_snaps(season), week)
        snap_note = _snap_reason(week, bool(snaps))
    except Exception as e:
        snap_note = _snap_reason(week, False) if week is not None else (
            f"snap trend not computed ({type(e).__name__})")
    ecr, ecr_reason = ({}, "no week is open") if week is None else lineup_journal.ecr_ranks(
        season, week, base)
    got = build(fa, roster, week=week, bye_now=bye, cal=cal, snaps=snaps, ecr=ecr,
                remaining=remaining)
    as_of = meta.get("fetched_at")
    stale = None
    if isinstance(as_of, str):
        try:
            age = moment.replace(tzinfo=UTC) - datetime.fromisoformat(as_of)
            stale = f"the pool is {age.days} days old" if age > STALE_AFTER else None
        except ValueError:
            stale = "the pool's fetch time is unreadable"
    return {**got, **reasons, "ecr_reason": ecr_reason, "snap_reason": snap_note,
            "as_of": as_of, "pool_size": fa.height, "capped": bool(meta.get("capped")),
            "last_error": meta.get("last_error"), "stale": stale}


# --- the journal --------------------------------------------------------------------------

def journal_path(season: int, week: int, base: Path | None = None) -> Path:
    return (ROOT if base is None else Path(base)) / str(season) / f"wk{week:02d}.json"


def record(waiver: Mapping[str, Any], season: int = SEASON_AHEAD, base: Path | None = None,
           now: datetime | None = None) -> tuple[Path | None, str]:
    """Journal this week's recommendations, once. First write wins; no week, no entry.

    ADR-0014 requirement 4: the rule's recommendation, what was actually claimed, and why if
    they differ. The recommendation is written here; `claimed` and `why_different` are left
    null for the manager to fill, because no replay can recover why a claim was or was not
    made. Points scored afterwards are not typed in: `settle` reads them from nflverse.
    """
    wk = waiver.get("week")
    if wk is None:
        return None, f"nothing journalled: {waiver.get('missing') or 'no week is open'}"
    path = journal_path(season, int(wk), base)
    if path.exists():
        return path, f"week {wk} already journalled; left as it was"
    picks = [{k: r[k] for k in ("player", "pos", "nfl_team", "proj", "value", "replaces",
                                "replaces_proj", "rank", "snap_delta", "ecr")}
             for r in waiver.get("rows") or [] if r.get("recommended")]
    doc = {"season": season, "week": int(wk),
           "recorded_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
           "label": LABEL, "basis": waiver["basis"], "shrinkage": waiver["shrinkage"],
           "as_of": waiver.get("as_of"), "picks": picks,
           "claimed": None, "why_different": None,
           "scored": False, "outcome": None}
    atomic.write_text(path, json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return path, f"week {wk} journalled ({len(picks)} recommended)"


def settle(season: int, week: int, points_by_week: Mapping[int, Mapping[str, float]],
           base: Path | None = None, last_final: int | None = None) -> str:
    """Score a journalled week once: each pick's realised points over the next three weeks,
    against the player he would have replaced.

    `points_by_week` is `{week: {player_key: PPR points}}`. Only a settled horizon is scored:
    `last_final`, the latest week every game of which is final, must reach `week + HORIZON - 1`.
    A name absent from a week scored nothing there and is counted in `n_no_row`, so a
    name-matching break (and kickers and defences, which nflverse's player stats do not carry)
    shows instead of reading as a zero.
    """
    from hub.names import player_key
    path = journal_path(season, week, base)
    if not path.exists():
        return f"week {week}: no journal entry"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("scored"):
        return f"week {week}: already scored; left as it was"
    horizon = range(week, week + HORIZON)
    # GUARD waiver-settle-only-a-finished-horizon [unit/test_waiver.py]: a claim is scored
    # on three finished weeks or not at all
    if last_final is None or last_final < horizon[-1]:
        return f"week {week}: weeks {horizon[0]}-{horizon[-1]} are not all final; left unscored"
    # /GUARD

    def total(name: str) -> dict[str, Any]:
        k = player_key(name)
        return {"points": round(sum(points_by_week.get(w, {}).get(k, 0.0) for w in horizon), 2),
                "n_no_row": sum(1 for w in horizon if k not in points_by_week.get(w, {}))}

    scored: list[dict[str, Any]] = []
    for p in doc["picks"]:
        # GUARD waiver-settle-leaves-k-and-dst-unscored [unit/test_waiver.py]: nflverse player
        # stats carry no kickers or defences, so scoring them reads 0 and drags the mean down
        if p.get("pos") in UNSCORABLE:
            scored.append({"player": p["player"], "replaces": p["replaces"], "scored": False,
                           "reason": f"{p['pos']} is not in nflverse player stats, so no "
                                     "realised points exist to score it against"})
            continue
        # /GUARD
        got_in, got_out = total(p["player"]), total(p["replaces"])
        scored.append({"player": p["player"], "replaces": p["replaces"],
                       "in_points": got_in["points"], "out_points": got_out["points"],
                       "net": round(got_in["points"] - got_out["points"], 2),
                       "n_no_row": got_in["n_no_row"] + got_out["n_no_row"]})
    nets: list[float] = [s["net"] for s in scored if "net" in s]
    doc["outcome"] = {"horizon_weeks": list(horizon), "picks": scored,
                      "mean_net": round(sum(nets) / len(nets), 2) if nets else None,
                      "n_unscored": len(scored) - len(nets)}
    doc["scored"] = True
    atomic.write_text(path, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return f"week {week}: scored"


def last_final_week(schedule: pl.DataFrame | None) -> int | None:
    """The latest week up to which every regular-season game has a final score, or None.

    Counted from week 1 and stopping at the first week with an open game, so a finished
    week 9 behind an unfinished week 8 does not count: `settle` needs three consecutive weeks.
    Uses `lineup_journal.week_incomplete`, the one definition of "this week is not final".
    """
    from hub.season.lineup_journal import week_incomplete
    last = None
    for w in range(1, REGULAR_SEASON_WEEKS + 1):
        if week_incomplete(schedule, w) is not None:
            break
        last = w
    return last


def read(season: int, base: Path | None = None) -> list[dict[str, Any]]:
    root = (ROOT if base is None else Path(base)) / str(season)
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("wk*.json"))]


# --- the command line ---------------------------------------------------------------------

def render(waiver: Mapping[str, Any]) -> str:
    """A plain-text summary: the label first, then each position's best free agents."""
    lines = [f"  [{waiver['label']}] {waiver['caveat']}"]
    if waiver.get("missing"):
        lines.append(f"  no ranking: {waiver['missing']}")
        return "\n".join(lines)
    lines.append(f"  week {waiver['week']}  {waiver['basis']}")
    lines.append(f"  {waiver['shrinkage']}")
    for k in ("last_error", "stale"):
        if waiver.get(k):
            lines.append(f"  NOTE ({k}): {waiver[k]}")
    for pos in sorted({r["pos"] for r in waiver["rows"]}):
        lines.append(f"\n  {pos}")
        for r in (x for x in waiver["rows"] if x["pos"] == pos):
            mark = "*" if r["recommended"] else " "
            snap = "   n/a" if r["snap_delta"] is None else f"{r['snap_delta'] * 100:+5.1f}"
            bye = "-" if r["bye_week"] is None else str(r["bye_week"])
            lines.append(f"   {mark} {r['player']:<24} {r['proj_per_game']:5.1f}/g"
                         f" vs {r['replaces']:<22}"
                         f" {r['value']:+6.2f}/g snap {snap}pp  bye {bye}"
                         + (f"  [{r['blocked']}]" if r["blocked"] else ""))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network
    ap = argparse.ArgumentParser(
        prog="hub.season.waiver",
        description="Rank the league's free agents against our replacement level (provisional).")
    ap.add_argument("--record", action="store_true", help="journal this week's ranking")
    ap.add_argument("--settle", action="store_true",
                    help="score journalled weeks whose next three weeks are final")
    ap.add_argument("--roster", default=None, help="override the roster parquet read")
    ap.add_argument("--free-agents", default=None, help="override the free-agent parquet read")
    ap.add_argument("--out", default=None, help="override the journal directory")
    a = ap.parse_args(list(argv) if argv is not None else None)
    src = Path(a.roster) if a.roster else ROSTER_PARQUET
    if not src.exists():
        print(f"hub.season.waiver: no roster at {src}; run "
              "`python -m hub.season.roster --write` first", file=sys.stderr)
        return 1
    base = Path(a.out) if a.out else None
    fa_path = Path(a.free_agents) if a.free_agents else FREE_AGENTS_PARQUET
    got = for_roster(pl.read_parquet(src), fa_path=fa_path)
    print(render(got))
    if a.record:
        print(f"  {record(got, base=base)[1]}")
    if a.settle:
        from hub.fetch import nflverse
        from hub.models.experiment import PLAYER_STATS_COLS, realised_ppg
        from hub.season import lineup_journal
        try:
            stats = realised_ppg(nflverse.load("player_stats", [SEASON_AHEAD],
                                               cols=list(PLAYER_STATS_COLS)))
            by_week = {int(w): {r["player"]: float(r["points"]) for r in g.iter_rows(named=True)}
                       for (w,), g in stats.group_by("week")}
            final = last_final_week(lineup_journal._schedule(SEASON_AHEAD))
            for doc in read(SEASON_AHEAD, base):
                if not doc.get("scored"):
                    print(f"  {settle(SEASON_AHEAD, int(doc['week']), by_week, base, final)}")
        except Exception as e:                   # offline: the entries stay unscored
            print(f"  not settled ({type(e).__name__}: {e})", file=sys.stderr)
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
