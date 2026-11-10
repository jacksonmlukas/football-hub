"""Score a Move (GLOSSARY.md) over the rest of the season (#501).

An add/drop or a trade, judged by what it does to our odds of making the playoffs, earning a
bye and winning the title, read off the weekly league capture (`hub.fetch.league`, #545):

    uv run python -m hub.season.ros --add "Some Back" --drop "Bench Guy"
    uv run python -m hub.season.ros --trade "Our A,Our B" "Their C"        # give, then get
    uv run python -m hub.season.ros --capture state/league/2026/wk10.json  # default: newest

Each remaining week every team scores Normal(mean, spread), floored at zero. A team starts its
best lineup by that week's projection (`hub.league.starting_lineup`), the mean is the starters'
summed projection, and the spread is `hub.models.predict.group_sd` over them with the per-
position weekly spread `hub.models.predict.moments` carries -- the model `hub.season.lineup`
uses. Seeding is played on the captured remaining schedule from the captured record (a tie is
half a win) and regular-season points-for; the bracket is `hub.draft.season.champion`, unchanged.

**One set of draws prices both worlds.** The standard normals are drawn once and shared by the
before and the after, so only the teams the Move touches differ and the paired difference is
reported with its own standard error (20,000 draws, a few seconds). A gain within two standard
errors prints "can't tell apart" instead of a number to act on.

Assumptions the output states, because each moves the answer:

- **Other teams keep their current rosters and start their best lineup by projection.** No
  manager adds, drops, trades, or starts the wrong player, which flatters weak managers: the
  teams we are racing never get better, and the bottom of the table never gets worse.
- **Remaining weeks are flat projections unless the capture holds one.** A week's mean is the
  player's captured `week_proj` for that week; otherwise his current-week projection (the
  season's projected per-game average when there is none), zeroed on his NFL bye. The bye weeks
  come from `hub.draft.season.bye_weeks`, the nflverse schedule through its validated loader
  (cached under data/raw). When that cannot be read the byes are not zeroed, and the output says
  so. Check the first real capture's `week_proj` coverage before leaning on the late weeks.
- Weekly scores are Normal and independent across players except teammates in a lineup
  (`group_sd`); no injury beyond the IR slot, no skew.

**Degradation.** No capture under `state/league/` is a sentence and exit 0, because the capture
is the slate's optional step. A capture named with `--capture` that is not there is exit 1.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import polars as pl

from hub.contracts import ContractViolation, check_league_capture
from hub.draft import season as draft_season
from hub.draft.playoff_sos import canon_team
from hub.fetch.league import STATE_LEAGUE
from hub.league import PLAYOFF_ROUNDS, PLAYOFF_TEAMS, starting_lineup
from hub.models.predict import group_sd, moments
from hub.names import player_key

DRAWS = 20_000
STALE_DAYS = 7
IR_SLOT = "IR"
# `seed_table`'s key: a win outweighs any gap in points, a half win included.
WIN_WEIGHT = 10_000


class MoveError(ValueError):
    """The Move cannot be scored as asked."""


class UnknownPlayer(MoveError):
    """Names that are on no roster (or in no pool) where the Move says they are."""


@dataclass(frozen=True)
class Move:
    """Players we give away and players we get: an add/drop from the pool, or a trade."""
    give: tuple[str, ...]
    get: tuple[str, ...]


class Estimate(NamedTuple):
    delta: float
    se: float


@dataclass(frozen=True)
class Counterparty:
    name: str
    playoffs: Estimate


@dataclass(frozen=True)
class MoveResult:
    n_draws: int
    team: str
    before: dict[str, float]
    after: dict[str, float]
    playoffs: Estimate
    bye: Estimate
    title: Estimate
    indistinguishable: dict[str, bool]
    counterparty: Counterparty | None
    flat_share: float
    warnings: list[str]


# --- the weekly mean -----------------------------------------------------------------------

def weekly_means(player: Mapping[str, Any], weeks: Sequence[int],
                 byes: Mapping[str, int] | None) -> tuple[np.ndarray, int]:
    """A player's mean for each of `weeks`, and how many of them are flat projections.

    The captured `week_proj` for the week if there is one; otherwise the current-week
    projection (the season's projected per-game average when that is absent), zero on his NFL
    bye. A bye is not counted as a flat projection: it is a known zero.
    """
    wp = {int(k): v for k, v in (player.get("week_proj") or {}).items()}
    cur, avg = player.get("espn_week_proj"), player.get("espn_avg")
    flat = float(cur) if cur is not None and cur > 0 else float(avg or 0.0)
    bye = (byes or {}).get(canon_team(player.get("nfl_team")) or "", 0)
    out, n_flat = [], 0
    for w in weeks:
        if wp.get(w) is not None:
            out.append(float(wp[w]))
        elif w == bye:
            out.append(0.0)
        else:
            out.append(flat)
            n_flat += 1
    return np.array(out), n_flat


def _team_moments(roster: Sequence[Mapping[str, Any]], weeks: Sequence[int],
                  byes: Mapping[str, int] | None) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(mean, spread) of the team's best-projection lineup for each week, and the count of
    flat and of all player-weeks behind them."""
    rows = [p for p in roster if p.get("slot") != IR_SLOT]
    mu = np.zeros((len(rows), len(weeks)))
    flat = 0
    for i, p in enumerate(rows):
        mu[i], f = weekly_means(p, weeks, byes)
        flat += f
    pos = [str(p.get("pos") or "") for p in rows]
    teams = [canon_team(p.get("nfl_team")) for p in rows]
    sd = moments(pl.DataFrame({"pos": np.repeat(pos, len(weeks)).tolist(),
                               "proj_blend": mu.ravel().tolist()}))["sd"].to_numpy()
    sd = sd.reshape(mu.shape)
    team_mu, team_sd = np.zeros(len(weeks)), np.zeros(len(weeks))
    for j in range(len(weeks)):
        start = starting_lineup(pos, mu[:, j])
        team_mu[j] = float(mu[start, j].sum())
        team_sd[j] = group_sd((sd[k, j], pos[k], teams[k]) for k in start)
    return team_mu, team_sd, flat, mu.size


# --- seeding -------------------------------------------------------------------------------

def seed_standings(pts: np.ndarray, schedule: Sequence[tuple[int, int, int]], *,
                   wins0: np.ndarray, points0: np.ndarray, reg_weeks: int,
                   playoff_teams: int = PLAYOFF_TEAMS) -> tuple[np.ndarray, np.ndarray]:
    """Wins and the playoff field from the remaining schedule, starting at the captured record.

    `pts` is (sims, weeks, teams) indexed by week from 1, as `hub.draft.season.seed_table`
    reads it; `schedule` is (week, team, opponent) index triples, each game once or from both
    sides. `wins0` already counts a tie as half a win and `points0` is regular-season points-for
    through the week before the first simulated one: the in-progress week is not counted, and its
    simulated points are added from `pts` instead.
    Seeds on wins, ties on regular-season points (`pts[:, :reg_weeks]` plus `points0`) --
    with no record and a round-robin schedule this orders the table as `seed_table` does.
    """
    n_sims = pts.shape[0]
    wins = np.tile(np.asarray(wins0, dtype=float), (n_sims, 1))
    games = {(w, min(a, b), max(a, b)) for w, a, b in schedule}
    for w, a, b in sorted(games):
        diff = pts[:, w - 1, a] - pts[:, w - 1, b]
        wins[:, a] += (diff > 0) + 0.5 * (diff == 0)
        wins[:, b] += (diff < 0) + 0.5 * (diff == 0)
    key = wins * WIN_WEIGHT + np.asarray(points0, dtype=float) + pts[:, :reg_weeks, :].sum(axis=1)
    return wins, np.argsort(-key, axis=1)[:, :playoff_teams]


# --- scoring -------------------------------------------------------------------------------

def _utc(stamp: str) -> datetime:
    """An ISO timestamp as an aware UTC datetime; a naive one is read as UTC."""
    t = datetime.fromisoformat(stamp)
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _paired(after: np.ndarray, before: np.ndarray) -> Estimate:
    d = after.astype(float) - before.astype(float)
    n = d.size
    return Estimate(float(d.mean()), float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0)


def _apart(e: Estimate) -> bool:
    """True when the gain is within two standard errors: nothing here tells it from zero."""
    return abs(e.delta) <= 2 * e.se


def _find(rows: Sequence[Mapping[str, Any]], name: str) -> int | None:
    k = player_key(name)
    for i, r in enumerate(rows):
        if r.get("key") == k or player_key(str(r.get("player"))) == k:
            return i
    return None


def _resolve(capture: Mapping[str, Any], move: Move):
    """(our index, counterparty index or None, our roster after, their roster after)."""
    teams = capture["teams"]
    me = next(i for i, t in enumerate(teams) if t["is_mine"])
    mine = list(teams[me]["roster"])
    pool = list((capture.get("free_agents") or {}).get("rows") or [])
    found = [(n, _find(mine, n)) for n in move.give]
    unknown = [n for n, i in found if i is None]
    give_at: list[tuple[str, int]] = []
    for n, i in found:
        if i is not None:
            give_at.append((n, i))
    owner: dict[int, list[tuple[str, int]]] = {}
    from_pool: list[tuple[str, int]] = []
    for n in move.get:
        hit = next(((t, j) for t, team in enumerate(teams) if t != me
                    if (j := _find(team["roster"], n)) is not None), None)
        if hit is not None:
            owner.setdefault(hit[0], []).append((n, hit[1]))
        elif (j := _find(pool, n)) is not None:
            from_pool.append((n, j))
        else:
            unknown.append(n)
    if unknown:
        raise UnknownPlayer("not on the roster or in the pool where the Move puts them: "
                            + ", ".join(unknown))
    if len(owner) > 1 or (owner and from_pool):
        raise MoveError("the players we get must all come from one team (or all from the "
                        "free-agent pool)")
    gone = {i for _, i in give_at}
    after_mine = [r for i, r in enumerate(mine) if i not in gone]
    given = [dict(mine[i], slot="BE") for _, i in give_at]
    if owner:
        cp, hits = next(iter(owner.items()))
        theirs = list(teams[cp]["roster"])
        out = {i for _, i in hits}
        got = [dict(theirs[i], slot="BE") for i in sorted(out)]
        return me, cp, after_mine + got, [r for i, r in enumerate(theirs) if i not in out] + given
    got = [dict(pool[j], slot="BE", week_proj={}) for _, j in from_pool]
    return me, None, after_mine + got, None


def score_move(capture: Mapping[str, Any], move: Move, *, byes: Mapping[str, int] | None,
               n_draws: int, seed: int, now: datetime | None = None) -> MoveResult:
    """Our change in P(playoffs), P(bye) and P(title) from `move`, each with a paired SE, and
    the counterparty's change in P(playoffs) for a trade. Pure: `capture` is a loaded dict."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    s = capture["settings"]
    reg, first = int(s["reg_season_weeks"]), int(capture["week"])
    pw = list(s["playoff_weeks"])
    if (int(s["playoff_team_count"]) != PLAYOFF_TEAMS or len(pw) != PLAYOFF_ROUNDS
            or len(capture["teams"]) < PLAYOFF_TEAMS):
        raise MoveError("only the six-team, three-round bracket `hub.draft.season.champion` "
                        "plays can be scored")
    if pw != [reg + 1, reg + 2, reg + 3]:
        raise MoveError(f"playoff weeks {pw} are not the {[reg + 1, reg + 2, reg + 3]} "
                        "that `champion` reads straight after the regular season")
    if first > reg:
        raise MoveError(f"week {first} is past the regular season ({reg} weeks); the field "
                        "is set and a Move no longer changes the seeding")
    last = max(pw)
    weeks = list(range(first, last + 1))
    teams = capture["teams"]
    index = {t["team_id"]: i for i, t in enumerate(teams)}
    me, cp, mine_after, theirs_after = _resolve(capture, move)

    flat = total = 0
    mu = np.zeros((len(weeks), len(teams)))
    sd = np.zeros_like(mu)
    for i, t in enumerate(teams):
        mu[:, i], sd[:, i], f, n = _team_moments(t["roster"], weeks, byes)
        flat, total = flat + f, total + n
    mu2, sd2 = mu.copy(), sd.copy()
    for i, roster in ((me, mine_after), (cp, theirs_after)):
        if i is not None and roster is not None:
            mu2[:, i], sd2[:, i], _, _ = _team_moments(roster, weeks, byes)

    shocks = np.random.default_rng(seed).standard_normal((n_draws, len(weeks), len(teams)))
    # Only games still to play: a pair before `first` would score 0-0 and hand both teams a half
    # win, and one past the regular season indexes beyond the array.
    sched = [(p["week"], index[p["team_id"]], index[p["opponent_id"]])
             for p in capture["schedule"] if first <= p["week"] <= reg]
    wins0 = np.array([t["wins"] + 0.5 * t["ties"] for t in teams], dtype=float)
    points0 = np.array([t["points_for"] for t in teams], dtype=float)

    def world(mean, spread):
        pts = np.zeros((n_draws, last, len(teams)))
        pts[:, first - 1:, :] = np.maximum(mean[None] + spread[None] * shocks, 0.0)
        _, seeds = seed_standings(pts, sched, wins0=wins0, points0=points0, reg_weeks=reg)
        champ = np.fromiter((draft_season.champion(pts, seeds, k, reg_weeks=reg)
                             for k in range(n_draws)), dtype=int, count=n_draws)
        return seeds, champ

    seeds0, champ0 = world(mu, sd)
    seeds1, champ1 = world(mu2, sd2)

    def outcomes(seeds, champ, i):
        return {"playoffs": (seeds == i).any(axis=1), "bye": (seeds[:, :2] == i).any(axis=1),
                "title": champ == i}

    before_hit, after_hit = outcomes(seeds0, champ0, me), outcomes(seeds1, champ1, me)
    est = {k: _paired(after_hit[k], before_hit[k]) for k in before_hit}
    counter = None
    if cp is not None:
        counter = Counterparty(
            teams[cp]["name"],
            _paired(outcomes(seeds1, champ1, cp)["playoffs"],
                    outcomes(seeds0, champ0, cp)["playoffs"]))

    warnings: list[str] = []
    captured = _utc(capture["captured_at"])
    if now - captured > timedelta(days=STALE_DAYS):
        warnings.append(f"the capture is stale: taken {captured:%Y-%m-%d}, "
                        f"{(now - captured).days} days ago (over {STALE_DAYS})")
    deadline = s.get("trade_deadline")
    if cp is not None and deadline and now > _utc(deadline):
        warnings.append(f"the trade deadline passed on {deadline[:10]}; ESPN will not process "
                        "this trade")
    if byes is None:
        warnings.append("NFL bye weeks could not be read, so flat projections are not zeroed "
                        "on byes")
    return MoveResult(
        n_draws=n_draws, team=teams[me]["name"],
        before={k: float(v.mean()) for k, v in before_hit.items()},
        after={k: float(v.mean()) for k, v in after_hit.items()},
        playoffs=est["playoffs"], bye=est["bye"], title=est["title"],
        indistinguishable={k: _apart(e) for k, e in est.items()},
        counterparty=counter, flat_share=flat / total if total else 0.0, warnings=warnings)


# --- the command line ----------------------------------------------------------------------

def newest_capture(root: Path) -> Path | None:
    """The latest `<season>/wk<NN>.json` under `root`, by season then week; None if none."""
    found = []
    for p in root.glob("*/wk*.json"):
        try:
            found.append((int(p.parent.name), int(p.stem[2:]), p))
        except ValueError:
            continue
    return max(found)[2] if found else None


def render(res: MoveResult) -> list[str]:
    def row(label: str, key: str, e: Estimate) -> str:
        head = f"  {label:<12}{res.before[key]:6.1%} -> {res.after[key]:6.1%}"
        if res.indistinguishable[key]:
            return f"{head}   {e.delta * 100:+.1f} +/- {e.se * 100:.1f} pts: can't tell apart"
        return f"{head}   {e.delta * 100:+.1f} +/- {e.se * 100:.1f} pts"

    out = [f"  {res.team}, {res.n_draws:,} paired draws over the rest of the season",
           row("P(playoffs)", "playoffs", res.playoffs), row("P(bye)", "bye", res.bye),
           row("P(title)", "title", res.title)]
    if res.counterparty:
        c = res.counterparty
        out.append(f"  {c.name} P(playoffs) {c.playoffs.delta * 100:+.1f} +/- "
                   f"{c.playoffs.se * 100:.1f} pts"
                   + ("  (can't tell apart)" if _apart(c.playoffs) else ""))
    out.append("  Other teams keep their current rosters and start their best lineup by "
               "projection; that flatters weak managers.")
    out.append(f"  Remaining weeks are a flat projection for {res.flat_share:.0%} of "
               "player-weeks (the rest are captured week projections), zeroed on NFL byes.")
    out += [f"  WARNING: {w}" for w in res.warnings]
    return out


def _names(text: str) -> tuple[str, ...]:
    return tuple(n.strip() for n in text.split(",") if n.strip())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.season.ros",
        description="Score an add/drop or a trade by our odds of the playoffs, a bye and the "
                    "title over the rest of the season.")
    ap.add_argument("--add", action="append", metavar="NAME", default=[],
                    help="a free agent to add (with --drop)")
    ap.add_argument("--drop", action="append", metavar="NAME", default=[],
                    help="one of ours to drop")
    ap.add_argument("--trade", nargs=2, metavar=("GIVE", "GET"),
                    help='comma-separated names: ours to give, theirs to get ("A,B" "C")')
    ap.add_argument("--capture", default=None,
                    help="a league capture; default the newest state/league/<season>/wk*.json")
    ap.add_argument("--league-dir", default=str(STATE_LEAGUE), help="override state/league")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-byes", action="store_true",
                    help="do not read the NFL schedule; flat projections are not zeroed on byes")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.trade and (a.add or a.drop):
        ap.error("--trade and --add/--drop are separate Moves")
    if a.trade:
        move = Move(give=_names(a.trade[0]), get=_names(a.trade[1]))
    elif a.add and a.drop:
        move = Move(give=tuple(a.drop), get=tuple(a.add))
    else:
        ap.error("give --add NAME --drop NAME, or --trade \"A,B\" \"C\"")
    if not move.give or not move.get:
        ap.error("a Move gives and gets at least one player")

    if a.capture:
        path = Path(a.capture)
        if not path.is_file():
            print(f"hub.season.ros: no capture at {path}", file=sys.stderr)
            return 1
    else:
        newest = newest_capture(Path(a.league_dir))
        if newest is None:
            print(f"  no league capture under {a.league_dir}; run `python -m hub.fetch.league "
                  "--write` once the cookie is set. Nothing scored.")
            return 0
        path = newest
    try:
        capture = json.loads(path.read_text())
        check_league_capture(capture)
    except (OSError, ValueError, ContractViolation) as e:
        print(f"hub.season.ros: {path} is not a usable capture: {e}", file=sys.stderr)
        return 1

    byes = None
    if not a.no_byes:
        try:
            byes = draft_season.bye_weeks(int(capture["season"]))
        except Exception as e:                       # graceful degradation, per CLAUDE.md
            print(f"hub.season.ros: bye weeks unavailable ({type(e).__name__}: {e})",
                  file=sys.stderr)
    try:
        res = score_move(capture, move, byes=byes, n_draws=a.draws, seed=a.seed)
    except MoveError as e:
        print(f"hub.season.ros: {e}", file=sys.stderr)
        return 1
    print(f"  {path.name}: week {capture['week']}")
    print("\n".join(render(res)))
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
