"""Which kicker and which defence to start or pick up, from the betting market (#494).

**This is a heuristic, shown with its basis, and it is not a measured edge (ADR-0014).** No
gate has run on it. The ranking is an ordering of games by one number each, and the page that
shows it must say "heuristic" beside it. What is claimed is only the arithmetic below; whether
streaming by it beats what the manager does anyway is the question the journal exists to
answer later (`record`), and until a gate says so it is not an answer.

The two rules, each a single number read off the betting market's own price for the game:

* **D/ST: lowest opponent implied total first.** A defence facing an offence the market expects
  to score little is the cheap stream the ticket names.
* **K: highest own implied total first.** A kicker on an offence the market expects to score a
  lot has more drives ending in a kick.

**Implied total** is the market's total split by its spread: with `s` the home spread (positive
when the home side is favoured -- nflverse's `spread_line`, the same sign `market_row` turns
into a home-win probability) and `T` the total, home implies `(T + s) / 2` and away
`(T - s) / 2`. Inputs and the result are all published per row, so the page can show the
working.

**Not used: dome and weather.** The ticket allows them "if available". Nothing in this repo
carries them for a game (`schedule` does not load the roof or wind columns), so the K rule is
the implied total alone and the basis string says so. Nothing is invented.

**Availability is not known here.** The free-agent pool is not fetched yet (#493), so the
ranking is of *every* team's unit that has a priced game this week, with the units the
manager rosters flagged beside them. A reader must check that a ranked unit is actually on the
waiver wire; `available_known` is `false` and says why.

**Missing is null with a reason, never a number.** A team on its bye, a game that has already
kicked off, a game no pre-kickoff capture priced, a capture that carried no total: each is an
unranked row naming the cause. The price is read through `hub.market_row.market_fields`, which
holds #471/#482's guard -- the last capture at or before kickoff, never an in-play one.

Logged weekly (`record`) to `state/stream_journal/<season>/wk<NN>.json`, first write wins,
carrying the inputs and the ranks so a later gate can settle them against realised points.

    uv run python -m hub.season.stream            # print this week's ranking
    uv run python -m hub.season.stream --record   # and journal it
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic
from hub.config import SEASON_AHEAD
from hub.paths import ROSTER_PARQUET, STATE_DIR

ROOT = STATE_DIR / "stream_journal"

LABEL = "heuristic"
BASIS = {
    "dst": "lowest opponent implied total first",
    "k": "highest own implied total first",
}
FORMULA = ("implied total: home (total + spread) / 2, away (total - spread) / 2, from the last "
           "pre-kickoff capture; spread is positive when the home side is favoured")
NOT_USED = "dome and weather are not used: no source for them is loaded"
AVAILABILITY_REASON = ("the free-agent pool is not fetched yet (#493), so every team with a "
                       "priced game is listed and the rostered ones are flagged; check the "
                       "waiver wire before acting on a rank")
CAVEAT = ("heuristic, not a measured edge: no gate has run on this ordering (ADR-0014); it "
          "is journalled so one can")


def _team(code: str | None) -> str | None:
    from hub.draft.playoff_sos import canon_team
    return canon_team(code)


def _moment(at: datetime | None) -> datetime:
    return at or datetime.now(UTC).replace(tzinfo=None)


def implied(spread: float, total: float) -> tuple[float, float]:
    """`(home, away)` implied team totals from the home spread and the game total."""
    return (total + spread) / 2.0, (total - spread) / 2.0


def _game_lines(game: Mapping[str, Any], seen: Mapping[Any, Mapping[str, Any]],
                at: datetime) -> dict[str, Any]:
    """One game's market inputs, or the single reason it has none to rank on."""
    from hub.market_row import market_fields
    kickoff = game.get("kickoff")
    done = game.get("result") is not None
    if done or (kickoff is not None and kickoff <= at):
        return {"missing": "the game has already kicked off"}
    got = market_fields(game, kickoff, "the schedule carries no kickoff for this game", seen)
    spread, total = got["market_spread"], got["market_total"]
    if spread is None:
        return {"missing": got["market_missing"] or "no pre-kickoff price"}
    if total is None:
        return {"missing": "the capture carried no total"}
    home, away = implied(spread, total)
    return {"spread": spread, "total": total, "home_implied": home, "away_implied": away,
            "kickoff": got["kickoff"], "priced_at": game.get("priced_at"),
            "price_source": game.get("price_source"), "missing": None}


def team_lines(games: pl.DataFrame, week: int, bye: frozenset[str],
               seen: Mapping[Any, Mapping[str, Any]],
               at: datetime | None = None) -> dict[str, dict[str, Any]]:
    """Per canonical team: its opponent, side, and the market inputs for `week`.

    A team on `bye` has a row with a reason and no opponent; a team in a game that cannot be
    priced has the opponent and the reason. Every team is keyed by `playoff_sos.canon_team`'s
    vocabulary, so a roster row joins through the same canon the bye check uses.
    """
    moment = _moment(at)
    out: dict[str, dict[str, Any]] = {t: {"team": t, "opp": None, "home": None,
                                          "missing": "bye week"} for t in sorted(bye)}
    for g in games.filter(pl.col("week") == week).iter_rows(named=True):
        home, away = _team(g["home_team"]), _team(g["away_team"])
        if home is None or away is None:
            continue
        ln = _game_lines(g, seen, moment)
        for team, opp, is_home in ((home, away, True), (away, home, False)):
            row: dict[str, Any] = {"team": team, "opp": opp, "home": is_home,
                                   "game_id": g.get("game_id"),
                                   "kickoff": ln.get("kickoff"), "spread": None,
                                   "total": None, "own_implied": None, "opp_implied": None,
                                   "priced_at": None, "price_source": g.get("price_source"),
                                   "missing": ln["missing"]}
            if ln["missing"] is None:
                own, other = (ln["home_implied"], ln["away_implied"]) if is_home else (
                    ln["away_implied"], ln["home_implied"])
                row.update(spread=ln["spread"], total=ln["total"], own_implied=own,
                           opp_implied=other,
                           priced_at=_iso(ln["priced_at"]))
            out[team] = row
    return out


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def rank(lines: Mapping[str, Mapping[str, Any]], key: str, descending: bool,
         rostered: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    """Rows ordered best first by `key`, ties sharing a rank; unrankable rows last, rank null.

    `rostered` is team -> the names of the manager's players at this position on that team.
    """
    ok = [r for r in lines.values() if r.get(key) is not None and math.isfinite(r[key])]
    bad = [r for r in lines.values() if r not in ok]
    ok.sort(key=lambda r: (-r[key] if descending else r[key], r["team"]))
    rows: list[dict[str, Any]] = []
    for r in ok:
        better = sum(1 for o in ok if (o[key] > r[key] if descending else o[key] < r[key]))
        rows.append({**r, "rank": better + 1, "rostered": rostered.get(r["team"], [])})
    for r in sorted(bad, key=lambda r: r["team"]):
        rows.append({**r, "rank": None, "rostered": rostered.get(r["team"], [])})
    return rows


def build(games: pl.DataFrame, roster: pl.DataFrame | None, week: int, bye: frozenset[str],
          seen: Mapping[Any, Mapping[str, Any]], at: datetime | None = None) -> dict[str, Any]:
    """The published `stream` object: both rankings, the basis and what is not known."""
    lines = team_lines(games, week, bye, seen, at)
    held: dict[str, dict[str, list[str]]] = {"D/ST": {}, "K": {}}
    unplaced: list[dict[str, str]] = []
    if roster is not None and not roster.is_empty():
        for r in roster.filter(pl.col("pos").is_in(["D/ST", "K"])).iter_rows(named=True):
            team = _team(r.get("nfl_team"))
            if team is None or team not in lines:
                unplaced.append({"player": r["player"], "pos": r["pos"],
                                 "reason": "his team is not in this week's schedule"})
                continue
            held[r["pos"]].setdefault(team, []).append(r["player"])
    return {
        "label": LABEL, "caveat": CAVEAT, "week": week,
        "basis": dict(BASIS), "formula": FORMULA, "not_used": NOT_USED,
        "available_known": False, "availability_reason": AVAILABILITY_REASON,
        "dst": rank(lines, "opp_implied", False, held["D/ST"]),
        "k": rank(lines, "own_implied", True, held["K"]),
        "unplaced": unplaced,
    }


def unavailable(reason: str) -> dict[str, Any]:
    """The `stream` object when it cannot be built: labelled, empty, and saying why."""
    return {"label": LABEL, "caveat": CAVEAT, "week": None, "basis": dict(BASIS),
            "formula": FORMULA, "not_used": NOT_USED, "available_known": False,
            "availability_reason": AVAILABILITY_REASON, "dst": [], "k": [], "unplaced": [],
            "missing": reason}


def for_roster(roster: pl.DataFrame, season: int = SEASON_AHEAD, at: datetime | None = None,
               base: Path | None = None) -> dict[str, Any]:
    """The ranking for the open week, degrading to `unavailable` on any failure."""
    from hub import schedule
    from hub.market_row import captures
    from hub.season.roster import week_games
    try:
        moment = _moment(at)
        games = schedule.priced_games(season, at=moment, base=base)
        wk = week_games(games, moment)
        if wk is None:
            return unavailable("no week is open in the schedule")
        return build(games, roster, wk.week, wk.bye, captures(season, base), moment)
    except Exception as e:  # unattended network call: say which failure, never raise
        return unavailable(f"the schedule or the odds store was unavailable ({type(e).__name__})")


def journal_path(season: int, week: int, base: Path | None = None) -> Path:
    return (ROOT if base is None else Path(base)) / str(season) / f"wk{week:02d}.json"


def record(stream: Mapping[str, Any], season: int = SEASON_AHEAD,
           base: Path | None = None, now: datetime | None = None) -> tuple[Path | None, str]:
    """Journal this week's ranking, once. First write wins; a week with no ranking is not written.

    The entry is what the market said when it was first written, not what a later run says; the
    captures it was read from are committed, so a gate can recompute any later state.
    """
    wk = stream.get("week")
    if wk is None:
        return None, f"nothing journalled: {stream.get('missing', 'no week')}"
    path = journal_path(season, int(wk), base)
    if path.exists():
        return path, f"week {wk} already journalled; left as it was"
    doc = {"season": season, "week": int(wk),
           "recorded_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
           "label": LABEL, "basis": stream["basis"], "formula": stream["formula"],
           "dst": stream["dst"], "k": stream["k"], "scored": False}
    atomic.write_text(path, json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return path, f"week {wk} journalled"


def read(season: int, base: Path | None = None) -> list[dict[str, Any]]:
    root = (ROOT if base is None else Path(base)) / str(season)
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("wk*.json"))]


def render(stream: Mapping[str, Any], top: int = 5) -> str:
    """A plain-text summary: the label first, then each ranking's best rows and the rostered."""
    lines = [f"  [{stream['label']}] {stream['caveat']}"]
    if stream.get("missing"):
        lines.append(f"  no ranking: {stream['missing']}")
        return "\n".join(lines)
    lines.append(f"  week {stream['week']}  {stream['formula']}")
    for name, key in (("D/ST", "dst"), ("K", "k")):
        lines.append(f"\n  {name}: {stream['basis'][key]}")
        rows = stream[key]
        shown = [r for r in rows if r["rank"] is not None and (r["rank"] <= top or r["rostered"])]
        for r in shown:
            imp = r["opp_implied"] if key == "dst" else r["own_implied"]
            mine = f"  <- {', '.join(r['rostered'])}" if r["rostered"] else ""
            lines.append(f"    {r['rank']:>2}  {r['team']:<4} vs {r['opp']:<4} {imp:5.1f}{mine}")
        for r in rows:
            if r["rank"] is None and r["rostered"]:
                lines.append(f"     -  {r['team']:<4} {r['missing']}  <- {', '.join(r['rostered'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network
    ap = argparse.ArgumentParser(
        prog="hub.season.stream",
        description="Rank D/ST and K by the betting market's implied totals (a heuristic).")
    ap.add_argument("--record", action="store_true", help="journal this week's ranking")
    ap.add_argument("--roster", default=None, help="override the roster parquet read")
    ap.add_argument("--out", default=None, help="override the journal directory")
    a = ap.parse_args(list(argv) if argv is not None else None)
    src = Path(a.roster) if a.roster else ROSTER_PARQUET
    roster = pl.read_parquet(src) if src.exists() else pl.DataFrame(
        schema={"player": pl.Utf8, "pos": pl.Utf8, "nfl_team": pl.Utf8})
    got = for_roster(roster)
    print(render(got))
    if a.record:
        print(f"  {record(got, base=Path(a.out) if a.out else None)[1]}")
    return 0


if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main())
